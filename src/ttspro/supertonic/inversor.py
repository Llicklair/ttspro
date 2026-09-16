"""El `ae.encoder` que Supertone no publicó, recuperado invirtiendo el vocoder.

    uv run python -m ttspro.supertonic.inversor --audio mi_voz.wav --salida latente.npy

Supertonic publica el **decoder** del autoencoder de voz (`vocoder.onnx`: latente
de 144 canales → onda a 44,1 kHz) y se guarda el encoder. Pero un decoder se puede
recorrer al revés: el grafo es diferenciable, así que en vez de preguntar "¿qué
latente tiene este audio?" se busca, por descenso de gradiente, el latente cuyo
audio **es** este audio.

Medido el 2026-09-15 sobre `web/e2e/referencia.wav` (VCTK p225, 6 s):

    paridad torch vs onnxruntime      max |dif| 1,25e-6
    pérdida multi-STFT                11,38 -> 5,54 en 1 500 pasos (374 s, GTX 1070)
    coseno de locutor original/recon  0,832

Ese 0,832 es lo que importa: la identidad del hablante **sobrevive** al viaje de
ida y vuelta, así que el latente es un sitio legítimo desde el que construir una
voz. Compárese con 0,227 (mezclar las diez voces) y 0,338 (el conversor antiguo).

Esto vive fuera del camino de síntesis y solo lo usa el voice builder: la regla 7
(la inferencia no depende del entrenamiento) se mantiene porque `motor.py` no
importa nada de aquí, y torch no llega jamás al navegador.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models" / "supertonic"
FRECUENCIA = 44100
# ae.base_chunk_size * ttl.chunk_compress_factor, y ttl.latent_dim * el mismo factor.
TROZO = 3072
CANALES = 144


def _parchear_clip() -> None:
    """Un `Clip` con limite opcional vacio rompe onnx2torch 1.5.15.

    En ONNX, una entrada opcional que no se usa viaja como la cadena vacia, no
    como ausente. Su conversor la trata como un nombre real, busca una constante
    que no existe y se rinde con "Dynamic value of min/max is not implemented".
    `text_encoder.onnx` tiene uno de esos (un ReLU escrito como Clip con minimo y
    sin maximo), asi que sin esto no hay gradiente que valga.
    """
    from onnx2torch.node_converters.clip import _create_torch_module
    from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
    from onnx2torch.utils.common import (
        OnnxMapping,
        OperationConverterResult,
        get_const_value,
    )

    def conversor(node, graph):
        nombres = [n if n else None for n in list(node.input_values) + [None, None]]
        limites = []
        for nombre in nombres[1:3]:
            limites.append(float(get_const_value(nombre, graph)) if nombre else None)
        return OperationConverterResult(
            torch_module=_create_torch_module(min_val=limites[0], max_val=limites[1]),
            onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values),
        )

    for version in (11, 12, 13):
        _CONVERTER_REGISTRY[OperationDescription("", "Clip", version)] = conversor


def a_torch(ruta: Path):
    """ONNX -> torch, esquivando dos cosas de onnx2torch 1.5.15.

    1. `convert(ruta)` escribe un temporal JUNTO al modelo y lo reabre, que en
       Windows es `PermissionError`. Con el `ModelProto` ya cargado, la inferencia
       de formas ocurre en memoria y no toca el disco.
    2. Le faltan conversores para versiones altas de ops que **no cambiaron de
       semántica** (Shape 19 solo añadió tipos; Cast, Pad, Reshape y Constant,
       igual). Bajar el opset del grafo entero sí podría cambiar números, así que
       en su lugar se reutiliza el conversor de la versión registrada más alta por
       debajo — y se dice en voz alta cuál, porque es una suposición.
    """
    import onnx
    from onnx import defs
    from onnx2torch import convert
    from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription

    _parchear_clip()
    modelo = onnx.load(str(ruta))
    alias = []
    for nodo in modelo.graph.node:
        dominio = nodo.domain or ""
        try:
            version = defs.get_schema(
                nodo.op_type, domain=dominio, max_inclusive_version=19
            ).since_version
        except Exception:
            continue
        clave = OperationDescription(dominio, nodo.op_type, version)
        if clave in _CONVERTER_REGISTRY:
            continue
        previas = [
            d
            for d in _CONVERTER_REGISTRY
            if d.operation_type == nodo.op_type and d.domain == dominio and d.version < version
        ]
        if previas:
            mejor = max(previas, key=lambda d: d.version)
            _CONVERTER_REGISTRY[clave] = _CONVERTER_REGISTRY[mejor]
            alias.append(f"{nodo.op_type} {mejor.version}->{version}")
    if alias:
        print(f"  conversores reutilizados: {', '.join(sorted(set(alias)))}")
    return convert(modelo)


def resumen(latente: np.ndarray, forma: str = "gram") -> np.ndarray:
    """Un latente `[1, 144, T]` -> un vector que no depende de lo que se dijo.

    El estilo no tiene eje de tiempo: describe una voz, no una frase. Así que lo
    que entra al ajuste no puede ser el latente crudo, que es tan largo como el
    audio, sino un resumen a lo largo del tiempo. Cuál, importa:

    ``estadisticos`` (432) son media, desviación y energía por canal. Medido el
    2026-09-16: ajustar el estilo contra esto baja la pérdida 13 veces y **no
    mueve el coseno de locutor ni una milésima** (0,0729 → 0,0719). Muchas voces
    distintas comparten esos 432 números; el resumen es demasiado pobre.

    ``gram`` (20 880) añade la matriz de correlaciones **entre canales**, que es
    donde la transferencia de estilo lleva desde 2015 diciendo que vive la
    textura: no en cuánta energía tiene cada canal, sino en cuáles se encienden
    juntos. Es la opción por defecto por esa medición, no por elegancia.
    """
    x = np.asarray(latente, dtype=np.float32).reshape(latente.shape[-2], latente.shape[-1])
    partes = [x.mean(axis=1), x.std(axis=1), np.abs(x).mean(axis=1)]
    if forma == "gram":
        centrado = x - x.mean(axis=1, keepdims=True)
        gram = (centrado @ centrado.T) / max(x.shape[1], 1)
        indices = np.triu_indices(gram.shape[0])
        partes.append(gram[indices])
    return np.concatenate(partes).astype(np.float32)


class Inversor:
    """Mantiene el vocoder en torch y devuelve el latente de cualquier grabación."""

    def __init__(self, modelos: Path | str = MODELOS, dispositivo: str | None = None) -> None:
        import torch

        self.torch = torch
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocoder = a_torch(Path(modelos) / "onnx" / "vocoder.onnx").to(self.dispositivo).eval()
        for p in self.vocoder.parameters():
            p.requires_grad_(False)

    # -- la pérdida ----------------------------------------------------------

    def _stft(self, x, n: int):
        return self.torch.abs(
            self.torch.stft(
                x,
                n,
                hop_length=n // 4,
                win_length=n,
                window=self.torch.hann_window(n, device=x.device),
                return_complex=True,
            )
        )

    def _perdida(self, a, b, escalas=(512, 1024, 2048)):
        """Multi-resolution STFT, en magnitud lineal y logarítmica.

        Es la pérdida con la que se entrenan los vocoders, así que es la que este
        vocoder sabe satisfacer; una L2 en la onda perseguiría la fase, que el
        oído no oye y el modelo no controla.
        """
        total = 0.0
        for n in escalas:
            sa, sb = self._stft(a, n), self._stft(b, n)
            total = total + self.torch.norm(sa - sb, p="fro") / (
                self.torch.norm(sb, p="fro") + 1e-7
            )
            total = total + self.torch.nn.functional.l1_loss(
                self.torch.log(sa + 1e-5), self.torch.log(sb + 1e-5)
            )
        return total

    # -- invertir ------------------------------------------------------------

    def latente(
        self,
        onda: np.ndarray,
        frecuencia: int = FRECUENCIA,
        pasos: int = 1500,
        lr: float = 0.05,
        registro=None,
    ) -> tuple[np.ndarray, dict]:
        """La grabación -> su latente, por descenso de gradiente sobre el latente."""
        torch = self.torch
        onda = a_44k(onda, frecuencia)
        largo = (len(onda) // TROZO) * TROZO
        if largo < TROZO * 8:
            raise ValueError(
                f"hacen falta al menos {TROZO * 8 / FRECUENCIA:.1f} s de audio;"
                f" llegaron {len(onda) / FRECUENCIA:.1f}"
            )
        objetivo = torch.from_numpy(onda[:largo]).to(self.dispositivo)[None]
        marcos = largo // TROZO

        z = (
            torch.zeros(1, CANALES, marcos, device=self.dispositivo)
            .normal_(0, 0.1)
            .requires_grad_(True)
        )
        opt = torch.optim.Adam([z], lr=lr)
        plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pasos)
        arranque, primera = time.perf_counter(), None
        for paso in range(pasos):
            opt.zero_grad()
            p = self._perdida(self.vocoder(z).reshape(1, -1)[:, :largo], objetivo)
            p.backward()
            opt.step()
            plan.step()
            if primera is None:
                primera = float(p.item())
            if registro and (paso % 150 == 0 or paso == pasos - 1):
                transcurrido = time.perf_counter() - arranque
                registro(f"  paso {paso:4d}  perdida {p.item():.4f}  ({transcurrido:.0f} s)")
        with torch.no_grad():
            final = float(self._perdida(self.vocoder(z).reshape(1, -1)[:, :largo], objetivo).item())
        return z.detach().cpu().numpy(), {
            "perdida_inicial": round(primera, 4),
            "perdida_final": round(final, 4),
            "segundos": round(time.perf_counter() - arranque, 1),
            "marcos": marcos,
            "pasos": pasos,
        }

    def reconstruir(self, latente: np.ndarray) -> np.ndarray:
        """El audio que ese latente produce, para poder oír y medir la inversión."""
        with self.torch.no_grad():
            z = self.torch.from_numpy(np.asarray(latente, dtype=np.float32)).to(self.dispositivo)
            return self.vocoder(z).reshape(-1).cpu().numpy()


def a_44k(onda: np.ndarray, frecuencia: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    onda = np.asarray(onda, dtype=np.float32)
    if onda.ndim > 1:
        onda = onda.mean(axis=tuple(range(onda.ndim - 1)))
    if frecuencia == FRECUENCIA:
        return onda
    d = gcd(FRECUENCIA, frecuencia)
    return resample_poly(onda, FRECUENCIA // d, frecuencia // d).astype(np.float32)


def main() -> None:
    import json

    import soundfile as sf

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--salida", type=Path, help="donde dejar el latente .npy")
    ap.add_argument("--reconstruida", type=Path, help="donde dejar el wav reconstruido, para oírlo")
    ap.add_argument("--pasos", type=int, default=1500)
    ap.add_argument("--modelos", type=Path, default=MODELOS)
    args = ap.parse_args()

    onda, frecuencia = sf.read(str(args.audio), dtype="float32", always_2d=True)
    onda = onda.mean(axis=1)
    inv = Inversor(args.modelos)
    print(f"invirtiendo {args.audio.name} ({len(onda) / frecuencia:.1f} s) en {inv.dispositivo}")
    z, medidas = inv.latente(onda, frecuencia, args.pasos, registro=print)

    from .constructor import Similitud

    sim = Similitud()
    recon = inv.reconstruir(z)
    original = a_44k(onda, frecuencia)[: len(recon)]
    medidas["coseno_original_reconstruida"] = round(
        float(sim.embedding(original, FRECUENCIA) @ sim.embedding(recon, FRECUENCIA)), 4
    )
    print(json.dumps(medidas, indent=1))
    if args.salida:
        np.save(args.salida, z)
        print(f"latente en {args.salida}")
    if args.reconstruida:
        sf.write(str(args.reconstruida), recon, FRECUENCIA)
        print(f"reconstruida en {args.reconstruida}")


if __name__ == "__main__":
    main()
