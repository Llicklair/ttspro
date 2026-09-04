"""Export the synthesizer to `models/tts.onnx` against the contract, and measure.

    uv run python -m ttspro.export.tts                      # random weights: size + ops only
    uv run python -m ttspro.export.tts --checkpoint runs/x/G_1000.pt

Facts printed (docs/evidencia.md): parameter count per block, fp32/fp16 size,
ops outside ORT Web's WebGPU set (rule 4), torch vs onnxruntime parity on the
waveform, latency in onnxruntime CPU for a 10-word sentence. With random
weights the parity and latency are still real facts about the GRAPH; only the
audio is meaningless.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import torch

from ttspro.export.ops_ort_web import ops_fuera_de_webgpu
from ttspro.model.config import ConfigSintetizador
from ttspro.model.sintetizador import Sintetizador, SintetizadorExport

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models"
CONTRATO = json.loads((MODELOS / "contrato.json").read_text(encoding="utf-8"))
FIRMA = CONTRATO["grafos"]["tts"]
NOMBRES_IN = [e["nombre"] for e in FIRMA["entradas"]]
NOMBRES_OUT = [s["nombre"] for s in FIRMA["salidas"]]


def config_por_defecto() -> ConfigSintetizador:
    return ConfigSintetizador(
        n_symbols=len(CONTRATO["simbolos"]["tabla"]),
        n_langs=len(CONTRATO["idiomas"]),
        gin_channels=CONTRATO["embedding_locutor"]["dim"],
    )


def cargar(checkpoint: Path | None, cfg: ConfigSintetizador) -> Sintetizador:
    """A checkpoint that carries its `cfg` decides the architecture (the model
    ported from coqui is base-sized; the default config is the reduced one)."""
    if checkpoint is not None:
        estado = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "cfg" in estado:
            for k, v in estado["cfg"].items():
                setattr(cfg, k, v)
        modelo = Sintetizador(cfg)
        modelo.load_state_dict(estado["modelo"] if "modelo" in estado else estado)
        return modelo
    return Sintetizador(cfg)


def entradas_ejemplo(cfg: ConfigSintetizador, longitud: int = 60, frames: int = 400):
    torch.manual_seed(0)
    return (
        torch.randint(1, cfg.n_symbols, (1, longitud)),
        torch.tensor([longitud]),
        torch.nn.functional.normalize(torch.randn(1, cfg.gin_channels), dim=1),
        torch.tensor([0]),
        torch.randn(1, cfg.inter_channels, frames),
        torch.randn(1, 2, longitud),
        torch.tensor([0.667, 0.8, 1.0]),
    )


def exportar(modelo: Sintetizador, ruta: Path, cfg: ConfigSintetizador) -> onnx.ModelProto:
    modelo.preparar_export()
    # torch.onnx.export restores the wrapper's ORIGINAL mode when done; a wrapper
    # born in train mode would drag the model back to train (dropout 0.5 in the
    # duration predictor) for any reference computed afterwards. Measured
    # 2026-09-04: 169 vs 186 frames on the same inputs.
    envoltorio = SintetizadorExport(modelo).eval()
    torch.onnx.export(
        envoltorio,
        entradas_ejemplo(cfg),
        str(ruta),
        opset_version=FIRMA["opset"],
        input_names=NOMBRES_IN,
        output_names=NOMBRES_OUT,
        dynamic_axes={
            "tokens": {1: "longitud"},
            "ruido_flow": {2: "frames"},
            "ruido_duracion": {2: "longitud"},
            "onda": {2: "muestras"},
        },
        dynamo=False,
    )
    m = onnx.load(str(ruta))
    onnx.checker.check_model(m)
    return m


def parametros_por_bloque(modelo: Sintetizador) -> dict[str, float]:
    return {
        nombre: round(sum(p.numel() for p in bloque.parameters()) / 1e6, 2)
        for nombre, bloque in [
            ("enc_p", modelo.enc_p),
            ("flow", modelo.flow),
            ("dec", modelo.dec),
            ("dp", modelo.dp),
            ("enc_q (solo entrenamiento)", modelo.enc_q),
        ]
    }


def medir(ruta: Path, entradas, esperado: np.ndarray | None) -> dict:
    import onnxruntime as ort

    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    feed = {n: e.numpy() for n, e in zip(NOMBRES_IN, entradas, strict=True)}
    sesion.run(None, feed)
    tiempos = []
    for _ in range(5):
        t0 = time.perf_counter()
        (onda,) = sesion.run(None, feed)
        tiempos.append(time.perf_counter() - t0)
    hechos = {
        "fichero": ruta.name,
        "MB": round(ruta.stat().st_size / 1e6, 1),
        "muestras": int(onda.shape[2]),
        "segundos_audio": round(onda.shape[2] / 22050, 2),
        "ms_mediana_cpu": round(statistics.median(tiempos) * 1000),
    }
    if esperado is not None:
        hechos["max_abs_diff_vs_torch"] = float(np.abs(onda - esperado).max())
        hechos["rtf_cpu"] = round(statistics.median(tiempos) / (onda.shape[2] / 22050), 3)
    return hechos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--salida", type=Path, default=MODELOS / "tts.onnx")
    ap.add_argument(
        "--upsample-initial", type=int, default=None, help="override decoder width to measure size"
    )
    ap.add_argument(
        "--espacio",
        default=None,
        help="speaker space the graph expects (voces.json must match); auto: "
        "coqui-emb_g-vctk for the un-finetuned port, wespeaker-resnet34-LM otherwise",
    )
    args = ap.parse_args()

    cfg = config_por_defecto()
    if args.upsample_initial:
        cfg.upsample_initial_channel = args.upsample_initial
    modelo = cargar(args.checkpoint, cfg)
    bloques = parametros_por_bloque(modelo)
    exportables = sum(v for k, v in bloques.items() if "entrenamiento" not in k)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    grafo = exportar(modelo, args.salida, cfg)
    entradas = entradas_ejemplo(cfg)
    modelo.eval()
    with torch.no_grad():
        esperado = modelo.inferir(*entradas).numpy()

    from onnxruntime.transformers.float16 import convert_float_to_float16

    salida16 = args.salida.with_suffix(".fp16.onnx")
    onnx.save(
        convert_float_to_float16(onnx.load(str(args.salida)), keep_io_types=True), str(salida16)
    )

    espacio = args.espacio
    if espacio is None:
        estado = (
            torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if args.checkpoint
            else {}
        )
        espacio = (
            "coqui-emb_g-vctk"
            if "origen" in estado and "paso" not in estado
            else "wespeaker-resnet34-LM"
        )
    hechos = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "espacio_locutor": espacio,
        "config": dataclasses.asdict(cfg),
        "parametros_M": bloques,
        "parametros_exportados_M": round(exportables, 2),
        "ops": sorted({n.op_type for n in grafo.graph.node}),
        "ops_fuera_de_webgpu": ops_fuera_de_webgpu(grafo),
        "ort": {
            "fp32": medir(args.salida, entradas, esperado),
            "fp16": medir(salida16, entradas, esperado),
        },
    }
    # The done criterion (tests/terminado) reads this to know which checkpoint
    # the graph came from; without one the graph is random and criterion 2 fails.
    args.salida.with_suffix(".export.json").write_text(
        json.dumps(hechos, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(hechos, indent=1))


if __name__ == "__main__":
    main()
