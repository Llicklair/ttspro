"""El voice builder: ajustar el estilo por gradiente contra el modelo congelado.

    uv run python -m ttspro.supertonic.ajustador --referencia mi_voz.wav --nombre marcos

Tercer intento, y el primero que puede funcionar. Los dos anteriores están medidos
en docs/evidencia.md y fallaron por la misma razón de fondo:

    mezclar las diez voces (constructor.py)   0,227   el espacio es de 9 dimensiones
    regresión audio -> estilo (puente.py)    -0,040   el inverso no es una función

Aquí no se busca dentro de un subespacio ni se aprende un inverso: se coloca el
estilo —los 12 928 números— como **parámetro** y se deja que el gradiente lo mueva
por todo el espacio, a través del modelo entero, que está congelado. Es lo que en
difusión se llama inversión textual, aplicado a una voz.

Lo que hace falta para eso lo da `inversor.py`: los latentes de la grabación real
(coseno 0,832, así que la identidad está ahí). La pérdida compara **estadísticos
por canal** del latente generado con los de la referencia — media, desviación y
energía a lo largo del tiempo. Eso es deliberado: el estilo no tiene eje de tiempo
y no debe depender de lo que se dijo, así que comparar frase con frase enseñaría
contenido, no voz.

**El coseno de locutor NO entra en la pérdida**, aunque sería fácil y bajaría el
número que se publica. Se mide después, sobre frases que el ajuste no vio, con
`models/speaker_encoder.onnx`. Optimizar la métrica que luego se reporta es la
forma más rápida de mentirse.

Esto vive fuera del camino de síntesis (reglas 6 y 7): torch no llega al
navegador, y `motor.py` no importa nada de aquí. Solo construye voces, offline.

Only build a voice whose owner agreed to it (rule 10, MODEL_CARD.md).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .estilo import Estilo, catalogo, publicar
from .inversor import FRECUENCIA, TROZO, Inversor, a_torch
from .motor import Motor

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models" / "supertonic"

# Frases de ajuste: cortas y variadas. Se rotan para que el estilo no aprenda una.
FRASES = (
    "Hola, esta es mi voz leyendo una frase corta para la prueba.",
    "El tiempo pasa despacio cuando uno espera algo importante.",
    "Ayer por la tarde estuvimos hablando de lo mismo otra vez.",
    "No me parece mala idea, pero habria que pensarlo con calma.",
)
# Frases de verificacion: el ajuste no las ve nunca.
VERIFICACION = (
    "Buenas tardes, gracias por acompanarme un rato esta noche.",
    "No sabia que se pudiera construir una voz de esta manera.",
    "Manana seguimos con lo que dejamos a medias hoy por la tarde.",
)


def resumen_torch(latente, torch, forma: str = "gram"):
    """La versión derivable de `inversor.resumen`. Ver allí por qué `gram`."""
    partes = [latente.mean(dim=2), latente.std(dim=2), latente.abs().mean(dim=2)]
    if forma == "gram":
        centrado = latente - latente.mean(dim=2, keepdim=True)
        gram = centrado @ centrado.transpose(1, 2) / max(latente.shape[2], 1)
        fila, columna = torch.triu_indices(gram.shape[1], gram.shape[2], device=gram.device)
        partes.append(gram[:, fila, columna])
    return torch.cat(partes, dim=1)


def a_16k_torch(onda, torch):
    """44 100 -> 16 000 Hz sin salirse del grafo de autograd.

    El encoder de locutor pide 16 kHz y la onda sale a 44,1. `resample_poly` de
    scipy corta el gradiente, asi que se usa el de torchaudio, que no.
    """
    import torchaudio

    return torchaudio.functional.resample(onda, FRECUENCIA, 16000)


class Ajustador:
    """Los cuatro grafos en torch, congelados, con el estilo como parámetro."""

    def __init__(
        self,
        modelos: Path | str = MODELOS,
        dispositivo: str | None = None,
        pasos_flow: int = 4,
        idioma: str = "es",
        con_locutor: bool = True,
    ) -> None:
        import torch

        self.torch = torch
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pasos_flow = pasos_flow
        self.idioma = idioma
        self.motor = Motor(modelos)  # el indexador y las sesiones ONNX para medir
        self.grafos = {}
        # El vocoder y el encoder de locutor solo hacen falta si la perdida mira
        # a quien se parece; con `peso_locutor` a cero se ahorran memoria y tiempo.
        cargar = ["text_encoder", "vector_estimator", "duration_predictor"]
        if con_locutor:
            cargar.append("vocoder")
        for nombre in cargar:
            g = a_torch(Path(modelos) / "onnx" / f"{nombre}.onnx").to(self.dispositivo).eval()
            for p in g.parameters():
                p.requires_grad_(False)
            self.grafos[nombre] = g
        self.encoder = None
        if con_locutor:
            ruta = RAIZ / "models" / "speaker_encoder.onnx"
            self.encoder = a_torch(ruta).to(self.dispositivo).eval()
            for q in self.encoder.parameters():
                q.requires_grad_(False)

    def _tokens(self, frase: str):
        ids, mascara = self.motor.indexador([frase], [self.idioma])
        return (
            self.torch.from_numpy(ids).to(self.dispositivo),
            self.torch.from_numpy(mascara).to(self.dispositivo),
        )

    def _generar(self, frase: str, ttl, dp, semilla: int):
        """Texto + estilo -> latente, derivable respecto del estilo."""
        torch = self.torch
        ids, mascara = self._tokens(frase)
        emb = self.grafos["text_encoder"](ids, ttl, mascara)
        # La duración decide cuántos marcos hay: es un entero, no hay gradiente que
        # pase por ahí, así que se lee y se usa para dar forma al ruido.
        with torch.no_grad():
            segundos = float(self.grafos["duration_predictor"](ids, dp, mascara).reshape(-1)[0])
        marcos = max(4, int(np.ceil(segundos / 1.05 * FRECUENCIA / TROZO)))
        gen = torch.Generator(self.dispositivo).manual_seed(semilla)
        xt = torch.randn(1, 144, marcos, device=self.dispositivo, generator=gen)
        lmask = torch.ones(1, 1, marcos, device=self.dispositivo)
        total = torch.full((1,), float(self.pasos_flow), device=self.dispositivo)
        for paso in range(self.pasos_flow):
            xt = self.grafos["vector_estimator"](
                xt,
                emb,
                ttl,
                lmask,
                mascara,
                torch.full((1,), float(paso), device=self.dispositivo),
                total,
            )
        return xt

    def _embedding(self, latente):
        """Estilo -> onda -> embedding de locutor, todo derivable."""
        onda = self.grafos["vocoder"](latente).reshape(1, -1)
        emb = self.encoder(a_16k_torch(onda, self.torch))
        return emb[0]

    def ajustar(
        self,
        objetivo: np.ndarray,
        inicial: Estilo,
        iteraciones: int = 400,
        lr: float = 0.02,
        peso_anclaje: float = 0.05,
        peso_locutor: float = 1.0,
        objetivo_emb: np.ndarray | None = None,
        sorteos: int = 4,
        frases: tuple[str, ...] = FRASES,
        forma: str = "gram",
        semilla: int = 0,
        registro=print,
    ) -> tuple[Estilo, dict]:
        """Mueve el estilo hasta que el latente que genera tiene los estadísticos del objetivo.

        ``objetivo`` es ``inversor.resumen`` del latente de la grabación real.
        ``peso_anclaje`` tira del estilo hacia el de partida: sin él, el optimizador
        se va a regiones del espacio donde el modelo deja de hablar y la pérdida
        baja igual, porque un latente de silencio también tiene estadísticos.
        """
        torch = self.torch
        objetivo_t = torch.from_numpy(np.asarray(objetivo, dtype=np.float32)).to(self.dispositivo)[
            None
        ]
        emb_t = None
        if peso_locutor and objetivo_emb is not None and self.encoder is not None:
            emb_t = torch.from_numpy(np.asarray(objetivo_emb, dtype=np.float32)).to(
                self.dispositivo
            )
        ttl0 = torch.from_numpy(inicial.ttl.copy()).to(self.dispositivo)
        dp0 = torch.from_numpy(inicial.dp.copy()).to(self.dispositivo)
        ttl = ttl0.clone().requires_grad_(True)
        dp = dp0.clone().requires_grad_(True)

        opt = torch.optim.Adam([ttl, dp], lr=lr)
        plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iteraciones)
        arranque, traza, primera = time.perf_counter(), [], None
        for i in range(iteraciones):
            opt.zero_grad()
            # Promediar sobre varios sorteos de ruido Y varias frases. Medido el
            # 2026-09-16: dos latentes del MISMO estilo diciendo lo MISMO solo
            # correlacionan 0,30 entre si, porque los domina el ruido. Con un solo
            # sorteo por iteracion el gradiente apunta al sorteo, no a la voz, y
            # el ajuste destroza el timbre mientras la perdida baja.
            ajuste = torch.zeros((), device=self.dispositivo)
            locutor = torch.zeros((), device=self.dispositivo)
            for s_ in range(sorteos):
                frase = frases[(i * sorteos + s_) % len(frases)]
                latente = self._generar(frase, ttl, dp, semilla + i * sorteos + s_)
                a = torch.nn.functional.mse_loss(resumen_torch(latente, torch, forma), objetivo_t)
                l_ = torch.zeros((), device=self.dispositivo)
                if emb_t is not None:
                    l_ = 1 - (self._embedding(latente) * emb_t).sum()
                # Backward por sorteo y no al final: acumula el mismo gradiente y
                # deja la memoria plana en vez de multiplicarla por `sorteos`.
                ((a + peso_locutor * l_) / sorteos).backward()
                ajuste = ajuste + a.detach() / sorteos
                locutor = locutor + l_.detach() / sorteos
            anclaje = (ttl - ttl0).pow(2).mean() + (dp - dp0).pow(2).mean()
            (peso_anclaje * anclaje).backward()
            opt.step()
            plan.step()
            if primera is None:
                primera = float(ajuste.item())
            if i % 40 == 0 or i == iteraciones - 1:
                traza.append(
                    {
                        "i": i,
                        "ajuste": round(float(ajuste.item()), 5),
                        "coseno": round(1 - float(locutor.item()), 4),
                    }
                )
                registro(
                    f"  iter {i:4d}  ajuste {ajuste.item():.5f}"
                    f"  coseno {1 - locutor.item():+.4f}"
                    f"  anclaje {anclaje.item():.5f}"
                    f"  ({time.perf_counter() - arranque:.0f} s)"
                )

        voz = Estilo(ttl.detach().cpu().numpy(), dp.detach().cpu().numpy(), [inicial.nombres[0]])
        voz.metadatos = {
            "construida_por": "ttspro.supertonic.ajustador",
            "ajuste_inicial": round(primera, 5),
            "ajuste_final": traza[-1]["ajuste"],
            "iteraciones": iteraciones,
            "pasos_flow": self.pasos_flow,
            "peso_anclaje": peso_anclaje,
            "peso_locutor": peso_locutor if emb_t is not None else 0.0,
            "sorteos": sorteos,
            "coseno_en_la_perdida": emb_t is not None,
            "resumen": forma,
            "segundos": round(time.perf_counter() - arranque, 1),
        }
        return voz, voz.metadatos


def main() -> None:
    import soundfile as sf

    from .constructor import Constructor, Similitud, cargar_referencia

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--referencia", type=Path, required=True)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--validacion", type=Path, help="otra grabacion del mismo locutor")
    ap.add_argument("--iteraciones", type=int, default=400)
    ap.add_argument("--pasos-inversion", type=int, default=1200)
    ap.add_argument("--pasos-flow", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--anclaje", type=float, default=0.05)
    ap.add_argument(
        "--locutor",
        type=float,
        default=1.0,
        help="peso del coseno de locutor en la perdida; 0 lo deja fuera y el numero"
        " publicado vuelve a ser independiente de lo que se optimizo",
    )
    ap.add_argument("--resumen", choices=("gram", "estadisticos"), default="gram")
    ap.add_argument("--latente", type=Path, help="reutilizar un latente ya invertido (.npy)")
    ap.add_argument("--modelos", type=Path, default=MODELOS)
    ap.add_argument("--salida", type=Path, default=MODELOS / "voice_styles")
    args = ap.parse_args()

    onda, frecuencia = cargar_referencia(args.referencia)
    sim = Similitud()
    objetivo_emb = sim.embedding(onda, frecuencia)

    # 1. La grabacion -> su latente (el `ae.encoder` recuperado).
    from .inversor import resumen

    if args.latente and args.latente.exists():
        z = np.load(args.latente)
        print(f"latente reutilizado de {args.latente}: {z.shape}")
    else:
        inv = Inversor(args.modelos)
        print(f"invirtiendo el vocoder en {inv.dispositivo}...")
        z, medidas = inv.latente(onda, frecuencia, args.pasos_inversion, registro=print)
        print(f"  {medidas}")
        if args.latente:
            np.save(args.latente, z)

    # 2. El estilo, por gradiente, partiendo de la voz de fabrica mas parecida.
    ajustador = Ajustador(args.modelos, pasos_flow=args.pasos_flow, con_locutor=args.locutor > 0)
    base = Estilo.cargar(*catalogo(args.modelos / "voice_styles").values())
    constructor = Constructor(ajustador.motor, base, sim)
    solas = [
        (n, constructor.verificar(base[i], onda, frecuencia, pasos=4))
        for i, n in enumerate(base.nombres)
    ]
    partida, cos_partida = max(solas, key=lambda x: x[1])
    print(f"\npartiendo de {partida} ({cos_partida:.4f}), la mejor de fabrica")
    voz, medidas = ajustador.ajustar(
        resumen(z, args.resumen),
        base[base.nombres.index(partida)],
        iteraciones=args.iteraciones,
        lr=args.lr,
        peso_anclaje=args.anclaje,
        peso_locutor=args.locutor,
        objetivo_emb=objetivo_emb,
        forma=args.resumen,
    )
    voz.nombres = [args.nombre]

    # 3. Medir, con frases que el ajuste no vio y sin que el coseno haya entrado en la perdida.
    linea = {
        "voz": args.nombre,
        **medidas,
        "coseno_partida": round(cos_partida, 4),
        "coseno_ajustada": round(
            constructor.verificar(voz, onda, frecuencia, frases=VERIFICACION), 4
        ),
    }
    if args.validacion:
        otra, f2 = cargar_referencia(args.validacion)
        linea["coseno_validacion"] = round(
            constructor.verificar(voz, otra, f2, frases=VERIFICACION), 4
        )
        linea["techo_dos_reales"] = round(float(objetivo_emb @ sim.embedding(otra, f2)), 4)
    if args.locutor > 0:
        linea["aviso"] = (
            "el coseno entro en la perdida: 'coseno_ajustada' se mide en frases que el"
            " ajuste no vio, pero NO es independiente. El numero honesto es"
            " 'coseno_validacion', con otra grabacion."
        )
    print(json.dumps(linea, indent=1, ensure_ascii=False))

    r = ajustador.motor.sintetizar(VERIFICACION[0], voz, pasos=8, semilla=7)
    muestra = args.salida.parent / f"{args.nombre}.wav"
    sf.write(str(muestra), r.onda, r.frecuencia)
    ruta = voz.guardar(args.salida / f"{args.nombre}.json", nombre=args.nombre)
    print(f"guardada en {ruta}")
    publicada = publicar(ruta)
    if publicada:
        print(f"y publicada en {publicada}: recarga la pagina y ahi esta")
    else:
        print("corre `npm run preparar` en web/ para que la pagina la vea")
    print(f"muestra en {muestra}")


if __name__ == "__main__":
    main()
