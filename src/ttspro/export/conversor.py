"""Export the tone-colour converter to ONNX (ADR 0007): two more graphs.

    uv run python -m ttspro.export.conversor

- `voz.onnx`: reference wave (22 050 Hz) -> voice vector (1, 256, 1). Runs ONCE
  per voice, and it is the one that carries the GRU, which WebGPU has no kernel
  for — so it is meant to run on the `wasm` provider and nobody notices.
- `conversor.onnx`: base wave + source voice + target voice + noise + tau ->
  wave in the target voice. This is the one that runs per sentence, and it stays
  inside the WebGPU op set.

The spectrogram lives inside both graphs (`SpecConv`), so the browser only ever
passes audio. The noise is an input (rule 5) and is tiled to whatever length the
audio needs, so the caller never has to compute frame counts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from ttspro.export.ops_ort_web import ops_fuera_de_webgpu
from ttspro.model.conversor import Conversor, cargar
from ttspro.model.fbank import SpecConv

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models"
SR = 22050
OPSET = 17


class GrafoVoz(nn.Module):
    """Reference wave -> voice vector."""

    def __init__(self, conversor: Conversor) -> None:
        super().__init__()
        self.spec = SpecConv()
        self.ref_enc = conversor.ref_enc

    def forward(self, onda: torch.Tensor) -> torch.Tensor:
        return self.ref_enc(self.spec(onda))


class GrafoConversor(nn.Module):
    """Base wave + two voice vectors -> converted wave."""

    def __init__(self, conversor: Conversor) -> None:
        super().__init__()
        self.spec = SpecConv()
        self.conversor = conversor

    def forward(
        self,
        onda: torch.Tensor,
        voz_origen: torch.Tensor,
        voz_destino: torch.Tensor,
        ruido: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        spec = self.spec(onda)
        frames = spec.size(2)
        repeticiones = (frames + ruido.size(2) - 1) // ruido.size(2)
        ruido = ruido.repeat(1, 1, repeticiones)[:, :, :frames]
        longitudes = torch.full((spec.size(0),), frames, dtype=torch.int64, device=spec.device)
        return self.conversor(spec, longitudes, voz_origen, voz_destino, ruido, tau)


def exportar(
    modulo: nn.Module, ejemplo: tuple, ruta: Path, entradas: list[str], ejes: dict
) -> onnx.ModelProto:
    torch.onnx.export(
        modulo.eval(),
        ejemplo,
        str(ruta),
        opset_version=OPSET,
        input_names=entradas,
        output_names=["salida"],
        dynamic_axes=ejes,
        dynamo=False,
    )
    m = onnx.load(str(ruta))
    onnx.checker.check_model(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=MODELOS)
    ap.add_argument("--segundos", type=float, default=4.0)
    args = ap.parse_args()

    conversor, informe = cargar()
    from ttspro.model.comun import remove_weight_norm

    remove_weight_norm(conversor)
    args.salida.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    muestras = int(SR * args.segundos) // 256 * 256
    onda = (torch.randn(1, muestras) * 0.2).clamp(-1, 1)

    g_voz = GrafoVoz(conversor).eval()
    with torch.no_grad():
        voz = g_voz(onda)
    m_voz = exportar(g_voz, (onda,), args.salida / "voz.onnx", ["onda"], {"onda": {1: "muestras"}})

    g_conv = GrafoConversor(conversor).eval()
    ruido = torch.randn(1, conversor.inter_channels, 400)
    tau = torch.tensor(0.3)
    ejemplo = (onda, voz, voz, ruido, tau)
    with torch.no_grad():
        esperado = g_conv(*ejemplo).numpy()
    m_conv = exportar(
        g_conv,
        ejemplo,
        args.salida / "conversor.onnx",
        ["onda", "voz_origen", "voz_destino", "ruido", "tau"],
        {"onda": {1: "muestras"}, "ruido": {2: "frames"}, "salida": {2: "muestras_salida"}},
    )

    from onnxruntime.transformers.float16 import convert_float_to_float16

    for nombre in ("voz", "conversor"):
        fp32 = args.salida / f"{nombre}.onnx"
        onnx.save(
            convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True),
            str(fp32.with_suffix(".fp16.onnx")),
        )

    import onnxruntime as ort

    hechos = {
        **informe,
        "voz": {
            "MB": round((args.salida / "voz.onnx").stat().st_size / 1e6, 1),
            "ops": sorted({n.op_type for n in m_voz.graph.node}),
            "ops_fuera_de_webgpu": ops_fuera_de_webgpu(m_voz),
            "salida": list(voz.shape),
        },
        "conversor": {
            "MB": round((args.salida / "conversor.onnx").stat().st_size / 1e6, 1),
            "ops_fuera_de_webgpu": ops_fuera_de_webgpu(m_conv),
        },
    }
    sesion = ort.InferenceSession(
        str(args.salida / "conversor.onnx"), providers=["CPUExecutionProvider"]
    )
    feed = {
        "onda": onda.numpy(),
        "voz_origen": voz.numpy(),
        "voz_destino": voz.numpy(),
        "ruido": ruido.numpy(),
        "tau": tau.numpy(),
    }
    sesion.run(None, feed)
    tiempos = []
    for _ in range(5):
        t0 = time.perf_counter()
        (salida,) = sesion.run(None, feed)
        tiempos.append(time.perf_counter() - t0)
    hechos["conversor"]["max_abs_diff_vs_torch"] = float(np.abs(salida - esperado).max())
    hechos["conversor"]["ms_mediana_cpu"] = round(statistics.median(tiempos) * 1000)
    hechos["conversor"]["rtf_cpu"] = round(statistics.median(tiempos) / (salida.shape[2] / SR), 3)

    for nombre, comprobar in (("voz", False), ("conversor", True)):
        ruta16 = args.salida / f"{nombre}.fp16.onnx"
        hechos[nombre]["MB_fp16"] = round(ruta16.stat().st_size / 1e6, 1)
        s16 = ort.InferenceSession(str(ruta16), providers=["CPUExecutionProvider"])
        if comprobar:
            (salida16,) = s16.run(None, feed)
            hechos[nombre]["fp16_max_abs_diff_vs_fp32"] = float(np.abs(salida16 - salida).max())
        else:
            (v16,) = s16.run(None, {"onda": onda.numpy()})
            hechos[nombre]["fp16_coseno_vs_fp32"] = float(
                np.dot(v16.reshape(-1), voz.numpy().reshape(-1))
                / (np.linalg.norm(v16) * np.linalg.norm(voz.numpy()) + 1e-9)
            )

    (args.salida / "conversor.export.json").write_text(
        json.dumps(hechos, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in hechos.items() if k not in ("mios_sin_cargar", "suyos_sin_destino")},
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
