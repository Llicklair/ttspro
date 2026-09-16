"""The voice builder that Supertone did not ship, learned from the model itself.

    uv run python -m ttspro.supertonic.puente muestrear --n 4000
    uv run python -m ttspro.supertonic.puente entrenar
    uv run python -m ttspro.supertonic.puente clonar --referencia mi_voz.wav --nombre marcos

Why this exists, in one paragraph. Supertonic takes the speaker's identity as an
input (``estilo.py``), but the only thing that could turn a *recording* into that
input was their hosted Voice Builder, and it went dark on 2026-08-31. A pretrained
speaker encoder cannot be dropped in instead: WeSpeaker, ECAPA and the rest each
live in their own space, learned in their own training run, and Supertonic's
``style_ttl`` was learned jointly with its own encoder. There is no fixed matrix
between two such spaces, any more than between word2vec and GloVe.

What there *is* is a way to learn the bridge, and the model supplies its own
training data. Run it forwards — a style, an audio, an embedding — and you have a
labelled pair, for free, with no corpus and nobody's consent to ask for:

    estilo aleatorio --Supertonic--> audio --WeSpeaker--> embedding
                     ^________________ se aprende esta flecha ______|

Then cloning is one forward pass: embed the recording, run the bridge, get a
style. Milliseconds, and small enough to ship to the browser.

The bridge is a ridge regression onto the principal components of style space,
not a deep net, and that is deliberate: with a few thousand samples the linear map
is the one that does not hallucinate confidently off-distribution, and its error
is a number you can write down. ``constructor.py`` then refines the result per
voice by direct search, which is slower but can only improve the score.

What this cannot do: invent timbre the checkpoint has never produced. The bridge
only reaches what sampling reached. That is why ``--validacion`` scores against a
second recording the fitting never saw, and why the honest number is that one.

Only build a voice whose owner agreed to it (rule 10, MODEL_CARD.md).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constructor import Similitud, cargar_referencia
from .estilo import Estilo, catalogo
from .inversor import resumen
from .motor import Motor

RAIZ = Path(__file__).resolve().parents[3]
MUESTRAS = RAIZ / ".scratch" / "puente_muestras.npz"
PESOS = RAIZ / "models" / "supertonic" / "puente.npz"

# Frases de sondeo: cortas, fonéticamente variadas, y siempre las mismas, para que
# el embedding de una muestra y el de una grabación real se midan igual.
SONDAS = (
    "Hola, esta es mi voz leyendo una frase corta para la prueba.",
    "El tiempo pasa despacio cuando uno espera algo importante.",
)


# -- 1. muestrear ------------------------------------------------------------


def muestrear(
    motor: Motor,
    base: Estilo,
    n: int,
    similitud: Similitud,
    lote: int = 16,
    pasos: int = 4,
    idioma: str = "es",
    semilla: int = 0,
    registro=print,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk style space, listen to what comes out, write down both.

    The distribution matters more than the count. Three sources, mixed:
    Dirichlet mixtures of the voices you have (safe, always speaks), the same
    with per-token noise (leaves the straight lines between speakers), and
    Gaussian draws around the mean with the measured covariance, stretched — that
    last one is what reaches timbres none of the ten has, and also what produces
    the occasional unusable take, which is why bad samples are dropped below.
    """
    rng = np.random.default_rng(semilla)
    k, t = base.ttl.shape[0], base.ttl.shape[1]
    media = base.vector.mean(axis=0)
    desviacion = base.vector.std(axis=0) + 1e-6

    estilos: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    resumenes: list[np.ndarray] = []
    descartadas = 0
    arranque = time.perf_counter()
    while len(estilos) < n:
        pesos_globales = rng.dirichlet(np.full(k, 0.6), size=lote).astype(np.float32)
        vectores = np.empty((lote, base.vector.shape[1]), dtype=np.float32)
        for i in range(lote):
            fuente = i % 3
            if fuente == 0:  # mezcla limpia
                w = np.tile(pesos_globales[i], (t, 1))
            elif fuente == 1:  # mezcla con ruido por token
                w = np.clip(
                    pesos_globales[i] + 0.25 * rng.standard_normal((t, k)), -0.4, 1.4
                ).astype(np.float32)
            else:  # gaussiana estirada alrededor de la media
                escala = rng.uniform(0.6, 1.8)
                v = media + escala * desviacion * rng.standard_normal(media.shape).astype(
                    np.float32
                )
                vectores[i] = v
                continue
            ttl = np.einsum("tk,ktd->td", w.astype(np.float32), base.ttl)
            dp = np.einsum("k,kjd->jd", w.mean(axis=0).astype(np.float32), base.dp)
            vectores[i] = np.concatenate([ttl.reshape(-1), dp.reshape(-1)])

        candidatos = Estilo.desde_vector(vectores)
        sonda = SONDAS[len(estilos) // lote % len(SONDAS)]
        resultados = motor.lote(
            [sonda] * lote,
            candidatos,
            idioma,
            pasos=pasos,
            semilla=semilla + len(estilos),
            con_latente=True,
        )
        for i, r in enumerate(resultados):
            if not _usable(r.onda, r.segundos):
                descartadas += 1
                continue
            estilos.append(vectores[i])
            embeddings.append(similitud.embedding(r.onda, r.frecuencia))
            # El latente es la representacion a la que `inversor.py` lleva una
            # grabacion REAL, asi que un puente ajustado sobre el se puede usar
            # con audio de fuera; el embedding solo, no llega (evidencia).
            resumenes.append(resumen(r.latente))
        if len(estilos) % (lote * 10) < lote:
            ritmo = (time.perf_counter() - arranque) / max(len(estilos), 1)
            registro(
                f"  {len(estilos)}/{n} muestras  ({descartadas} descartadas, "
                f"{ritmo:.2f} s/muestra, faltan {(n - len(estilos)) * ritmo / 60:.0f} min)"
            )
    return np.stack(estilos[:n]), np.stack(embeddings[:n]), np.stack(resumenes[:n])


def _usable(onda: np.ndarray, segundos: float) -> bool:
    """Drop takes that are silence, clipping or a runaway duration.

    A style drawn far from the mean sometimes produces a whisper or a scream, and
    a pair whose audio is not speech teaches the bridge nothing true.
    """
    if segundos < 1.0 or segundos > 20.0:
        return False
    rms = float(np.sqrt(np.mean(np.square(onda))))
    return 0.005 < rms < 0.5 and float(np.max(np.abs(onda))) < 0.999


# -- 2. entrenar -------------------------------------------------------------


@dataclass
class Puente:
    """embedding de locutor (256) -> estilo (12 928), pasando por la PCA."""

    media_estilo: np.ndarray  # [12928]
    componentes: np.ndarray  # [C, 12928]
    media_emb: np.ndarray  # [256]
    pesos: np.ndarray  # [257, C]  (la última fila es el término independiente)
    error: dict

    def __call__(self, embedding: np.ndarray, nombre: str = "clonada") -> Estilo:
        x = np.concatenate([np.asarray(embedding, dtype=np.float32) - self.media_emb, [1.0]])
        coef = x @ self.pesos
        vector = self.media_estilo + coef @ self.componentes
        e = Estilo.desde_vector(vector[None], [nombre])
        e.metadatos = {"construida_por": "ttspro.supertonic.puente", **self.error}
        return e

    def guardar(self, ruta: Path | str = PESOS) -> Path:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ruta,
            media_estilo=self.media_estilo,
            componentes=self.componentes,
            media_emb=self.media_emb,
            pesos=self.pesos,
            error=json.dumps(self.error),
        )
        return ruta

    @classmethod
    def cargar(cls, ruta: Path | str = PESOS) -> Puente:
        d = np.load(Path(ruta), allow_pickle=False)
        return cls(
            d["media_estilo"],
            d["componentes"],
            d["media_emb"],
            d["pesos"],
            json.loads(str(d["error"])),
        )

    def exportar_web(self, carpeta: Path | str, encoder: Path | str | None = None) -> Path:
        """Write the bridge as a header plus one flat float32 blob for the browser.

        JSON would spend 8 MB of text on 857 000 numbers and parse them one at a
        time; a `.bin` read into a Float32Array is one fetch and no parsing. The
        speaker encoder is copied next to it so the page has a single folder to
        fetch from, and cloning in the browser is then this: encode, multiply, done.
        """
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        piezas = {
            "media_estilo": self.media_estilo,
            "componentes": self.componentes,
            "media_emb": self.media_emb,
            "pesos": self.pesos,
        }
        cabecera, desplazamiento, crudo = {}, 0, bytearray()
        for nombre, a in piezas.items():
            a = np.ascontiguousarray(a, dtype=np.float32)
            cabecera[nombre] = {"forma": list(a.shape), "desde": desplazamiento, "n": int(a.size)}
            crudo += a.tobytes()
            desplazamiento += int(a.size)
        (carpeta / "puente.bin").write_bytes(bytes(crudo))
        (carpeta / "puente.json").write_text(
            json.dumps(
                {"piezas": cabecera, "floats": desplazamiento, "error": self.error}, indent=1
            ),
            encoding="utf-8",
        )
        encoder = Path(encoder or RAIZ / "models" / "speaker_encoder.onnx")
        if encoder.exists():
            (carpeta / "speaker_encoder.onnx").write_bytes(encoder.read_bytes())
        return carpeta / "puente.json"


def _direcciones(centrado: np.ndarray, componentes: int) -> tuple[np.ndarray, float]:
    """Las `componentes` direcciones principales de un `[N, 12928]`, por la matriz de Gram.

    Un SVD completo de 3 600 x 12 928 tarda medio minuto y calcula 3 600
    direcciones para tirar todas menos 64. La matriz de Gram es `[N, N]`, y de sus
    autovectores salen las mismas: `X = U S V^T` implica `X X^T = U S^2 U^T`, y
    `V^T = S^-1 U^T X`. Son segundos, y la varianza explicada sale de los mismos
    autovalores, asi que no se pierde el diagnostico.
    """
    gram = centrado @ centrado.T
    valores, vectores = np.linalg.eigh(gram)
    orden = np.argsort(-valores)
    valores, vectores = valores[orden], vectores[:, orden]
    k = min(componentes, int(np.sum(valores > 1e-8)))
    escala = np.sqrt(np.maximum(valores[:k], 1e-12))
    direcciones = ((vectores[:, :k] / escala).T @ centrado).astype(np.float32)
    # eigh no garantiza norma unidad tras el cambio de base; la ridge la asume.
    direcciones /= np.linalg.norm(direcciones, axis=1, keepdims=True) + 1e-12
    explicado = float(np.sum(valores[:k]) / max(np.sum(np.maximum(valores, 0)), 1e-12))
    return direcciones, explicado


def entrenar(
    estilos: np.ndarray,
    embeddings: np.ndarray,
    componentes: int = 64,
    alpha: float = 1.0,
    validacion: float = 0.1,
) -> Puente:
    """Ridge onto the first ``componentes`` principal directions of style space.

    Predicting 12 928 numbers from 256 is asking for overfitting; the PCA cuts the
    target to the directions that carry the variance and the ridge term handles
    the rest. Both numbers are reported as held-out error, not training error.
    """
    n = len(estilos)
    corte = int(n * (1 - validacion))
    media_estilo = estilos[:corte].mean(axis=0)
    centrado = estilos[:corte] - media_estilo
    componentes_v, explicado = _direcciones(centrado, componentes)

    media_emb = embeddings[:corte].mean(axis=0)

    def diseno(e):
        return np.concatenate([e - media_emb, np.ones((len(e), 1), dtype=np.float32)], axis=1)

    x = diseno(embeddings[:corte])
    y = centrado @ componentes_v.T
    pesos = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1], dtype=np.float32), x.T @ y).astype(
        np.float32
    )

    def error_en(desde: int, hasta: int) -> float:
        if hasta <= desde:
            return float("nan")
        pred = media_estilo + (diseno(embeddings[desde:hasta]) @ pesos) @ componentes_v
        real = estilos[desde:hasta]
        return float(
            np.mean(
                np.linalg.norm(pred - real, axis=1) / np.linalg.norm(real - media_estilo, axis=1)
            )
        )

    error = {
        "muestras": int(n),
        "componentes": int(componentes),
        "varianza_explicada": round(explicado, 4),
        "error_relativo_entrenamiento": round(error_en(0, corte), 4),
        "error_relativo_validacion": round(error_en(corte, n), 4),
    }
    return Puente(
        media_estilo.astype(np.float32), componentes_v, media_emb.astype(np.float32), pesos, error
    )


def entrada(muestras, desde: str) -> np.ndarray:
    """Lo que ve el puente. Tres opciones, para poder comparar cual carga la voz.

    ``emb`` son los 256 numeros de WeSpeaker, entrenados para llevar identidad de
    locutor y nada mas. ``latente`` son los 432 estadisticos por canal del latente
    del propio modelo, que es el idioma en el que el checkpoint piensa. ``ambos``
    los concatena, que es lo que tiene mas informacion y lo que la ridge sabe
    podar sola.
    """
    if desde == "emb":
        return muestras["embeddings"]
    if desde == "latente":
        return muestras["resumenes"]
    return np.concatenate([muestras["embeddings"], muestras["resumenes"]], axis=1)


# -- 3. clonar ---------------------------------------------------------------


def clonar(
    puente: Puente,
    similitud: Similitud,
    ruta: Path,
    nombre: str,
    modelos: Path | None = None,
    pasos_inversion: int = 1500,
    registro=print,
) -> Estilo:
    """Grabacion -> voz. Si el puente mira el latente, hay que invertir el vocoder.

    Ese es el paso caro (minutos en GPU) y el que hace que esto funcione con audio
    real: sin el, lo unico que se le puede dar al puente es el embedding, y el
    embedding solo no basta.
    """
    onda, frecuencia = cargar_referencia(ruta)
    piezas = []
    desde = puente.error.get("desde", "emb")
    if desde in ("emb", "ambos"):
        piezas.append(similitud.embedding(onda, frecuencia))
    if desde in ("latente", "ambos"):
        from .inversor import Inversor

        inv = Inversor(modelos or (RAIZ / "models" / "supertonic"))
        registro(f"  invirtiendo el vocoder en {inv.dispositivo} ({pasos_inversion} pasos)...")
        z, medidas = inv.latente(onda, frecuencia, pasos_inversion, registro=registro)
        registro(f"  inversion: {medidas}")
        piezas.append(resumen(z))
    return puente(np.concatenate(piezas), nombre)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="orden", required=True)

    m = sub.add_parser("muestrear", help="genera pares (estilo, embedding) con el propio modelo")
    m.add_argument("--n", type=int, default=4000)
    m.add_argument("--lote", type=int, default=16)
    m.add_argument("--pasos", type=int, default=4)
    m.add_argument("--salida", type=Path, default=MUESTRAS)
    m.add_argument("--semilla", type=int, default=0)

    e = sub.add_parser("entrenar", help="ajusta el puente sobre las muestras")
    e.add_argument("--muestras", type=Path, default=MUESTRAS)
    e.add_argument(
        "--desde",
        choices=("emb", "latente", "ambos"),
        default="ambos",
        help="que se le da al puente: el embedding de locutor, el resumen del latente, o los dos",
    )
    e.add_argument("--componentes", type=int, default=64)
    e.add_argument("--alpha", type=float, default=1.0)
    e.add_argument("--salida", type=Path, default=PESOS)
    e.add_argument(
        "--web",
        type=Path,
        default=RAIZ / "models" / "supertonic",
        help="carpeta donde dejar puente.bin y el encoder para el navegador",
    )

    c = sub.add_parser("clonar", help="grabacion -> voz, en un forward")
    c.add_argument("--referencia", type=Path, required=True)
    c.add_argument("--nombre", required=True)
    c.add_argument(
        "--validacion", type=Path, help="otra grabacion del mismo locutor, para puntuar sin trampa"
    )
    c.add_argument("--puente", type=Path, default=PESOS)
    c.add_argument("--afinar", action="store_true", help="refina con la busqueda de constructor.py")
    c.add_argument("--salida", type=Path, default=RAIZ / "models" / "supertonic" / "voice_styles")

    ap.add_argument("--modelos", type=Path, default=RAIZ / "models" / "supertonic")
    args = ap.parse_args()

    if args.orden == "muestrear":
        motor = Motor(args.modelos)
        base = Estilo.cargar(*catalogo(args.modelos / "voice_styles").values())
        estilos, embeddings, resumenes = muestrear(
            motor, base, args.n, Similitud(), args.lote, args.pasos, semilla=args.semilla
        )
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.salida, estilos=estilos, embeddings=embeddings, resumenes=resumenes
        )
        print(f"{len(estilos)} muestras en {args.salida}")

    elif args.orden == "entrenar":
        d = np.load(args.muestras)
        puente = entrenar(d["estilos"], entrada(d, args.desde), args.componentes, args.alpha)
        puente.error["desde"] = args.desde
        print(json.dumps(puente.error, indent=1))
        print(f"guardado en {puente.guardar(args.salida)}")
        print(f"para el navegador en {puente.exportar_web(args.web)}")

    else:
        similitud = Similitud()
        puente = Puente.cargar(args.puente)
        voz = clonar(puente, similitud, args.referencia, args.nombre, args.modelos)
        motor = Motor(args.modelos)
        from .constructor import Constructor

        base = Estilo.cargar(*catalogo(args.modelos / "voice_styles").values())
        constructor = Constructor(motor, base, similitud)
        referencia, frecuencia = cargar_referencia(args.referencia)
        linea = {
            "voz": args.nombre,
            "puente": round(constructor.verificar(voz, referencia, frecuencia), 4),
        }
        if args.validacion:
            otra, f2 = cargar_referencia(args.validacion)
            linea["puente_validacion"] = round(constructor.verificar(voz, otra, f2), 4)
        if args.afinar:
            c = constructor.construir(referencia, frecuencia)
            if c.similitud > linea["puente"]:
                voz = c.voz
            linea["afinada"] = round(c.similitud, 4)
        print(json.dumps(linea, indent=1))
        print(f"guardada en {voz.guardar(args.salida / f'{args.nombre}.json', nombre=args.nombre)}")


if __name__ == "__main__":
    main()
