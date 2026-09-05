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
from ttspro.model.sintetizador import (
    Sintetizador,
    SintetizadorExport,
    SintetizadorExportVozFija,
)

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models"
CONTRATO = json.loads((MODELOS / "contrato.json").read_text(encoding="utf-8"))
FIRMA = CONTRATO["grafos"]["tts"]
NOMBRES_IN = [e["nombre"] for e in FIRMA["entradas"]]
# The full signature `inferir` has. The contract may declare FEWER, because the
# exporter prunes what a fixed-voice model ignores (ADR 0008); this list is what
# `entradas_ejemplo` produces, in order, and never changes.
ENTRADAS_COMPLETAS = [
    "tokens",
    "longitud_tokens",
    "embedding",
    "idioma",
    "ruido_flow",
    "ruido_duracion",
    "escala_ruido",
]
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
        pesos = estado["modelo"] if "modelo" in estado else estado
        # A ported voice (Piper, ADR 0008) is saved with weight norm already folded,
        # so the fresh model has to be folded too before the keys line up.
        if not any("parametrizations" in k for k in pesos):
            from ttspro.model.comun import remove_weight_norm

            remove_weight_norm(modelo)
        modelo.load_state_dict(pesos, strict=False)
        return modelo
    return Sintetizador(cfg)


def entradas_ejemplo(cfg: ConfigSintetizador, longitud: int = 60, frames: int = 400):
    """The graph keeps the contract signature even for a single-voice model: the
    `embedding` and `idioma` inputs are then accepted and IGNORED (ADR 0008), so
    the browser code and the contract stay the same across base voices."""
    torch.manual_seed(0)
    dim = cfg.gin_channels or CONTRATO["embedding_locutor"]["dim"]
    return (
        torch.randint(1, cfg.n_symbols, (1, longitud)),
        torch.tensor([longitud]),
        torch.nn.functional.normalize(torch.randn(1, dim), dim=1),
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
    voz_fija = not modelo.usa_locutor and not modelo.usa_idioma
    envoltorio = (SintetizadorExportVozFija if voz_fija else SintetizadorExport)(modelo).eval()
    ejemplo = entradas_ejemplo(cfg)
    nombres = [n for n in ENTRADAS_COMPLETAS if not (voz_fija and n in ("embedding", "idioma"))]
    if voz_fija:
        ejemplo = tuple(e for n, e in zip(ENTRADAS_COMPLETAS, ejemplo, strict=True) if n in nombres)
    torch.onnx.export(
        envoltorio,
        ejemplo,
        str(ruta),
        opset_version=FIRMA["opset"],
        input_names=nombres,
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


def medir(
    ruta: Path, entradas, esperado: np.ndarray | None, nombres: list[str] | None = None
) -> dict:
    import onnxruntime as ort

    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    feed = {n: e.numpy() for n, e in zip(nombres or NOMBRES_IN, entradas, strict=True)}
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
        "--contrato",
        type=Path,
        default=MODELOS / "contrato.json",
        help="contract to sync and write; a voice pack keeps its own (copied from models/)",
    )
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

    # A ported voice carries its own symbol table (Piper has its own map and wraps
    # the sentence in ^ … $). The contract must describe what is IN models/, so the
    # export syncs it — and says so, because it changes what the frontend emits.
    ruta_contrato = args.contrato
    if not ruta_contrato.exists():
        ruta_contrato.parent.mkdir(parents=True, exist_ok=True)
        ruta_contrato.write_text(
            (MODELOS / "contrato.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
    if args.checkpoint is not None:
        estado_previo = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "simbolos" in estado_previo:
            contrato = json.loads(ruta_contrato.read_text(encoding="utf-8"))
            cambios = {}
            if contrato["simbolos"] != estado_previo["simbolos"]:
                cambios["simbolos"] = estado_previo["simbolos"]
            if "idiomas" in estado_previo and contrato["idiomas"] != estado_previo["idiomas"]:
                cambios["idiomas"] = estado_previo["idiomas"]
            if cambios:
                contrato.update(cambios)
                ruta_contrato.write_text(
                    json.dumps(contrato, ensure_ascii=False, indent=1), encoding="utf-8"
                )
                print(f"contrato: {list(cambios)} actualizado desde {args.checkpoint}")
        del estado_previo

    cfg = config_por_defecto()
    if args.upsample_initial:
        cfg.upsample_initial_channel = args.upsample_initial
    modelo = cargar(args.checkpoint, cfg)
    bloques = parametros_por_bloque(modelo)
    exportables = sum(v for k, v in bloques.items() if "entrenamiento" not in k)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    grafo = exportar(modelo, args.salida, cfg)
    # torch.onnx.export PRUNES inputs the model ignores, so a single-voice base TTS
    # has no `embedding` and a monolingual one has no `idioma`. The contract has to
    # say what the graph really takes, and the browser builds its feed from it.
    reales = {i.name for i in grafo.graph.input}
    contrato = json.loads(ruta_contrato.read_text(encoding="utf-8"))
    catalogo = {e["nombre"]: e for e in CONTRATO["grafos"]["tts"]["entradas"]}
    # Shapes come from the GRAPH, not from the catalogue: a smaller voice (an x_low
    # Piper voice has 96 flow channels, not 192) declares its own `ruido_flow`.
    formas = {
        i.name: [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
        for i in grafo.graph.input
    }
    declaradas = [
        {**catalogo[n], "forma": formas[n]}
        for n in ENTRADAS_COMPLETAS
        if n in reales and n in catalogo
    ]
    if contrato["grafos"]["tts"]["entradas"] != declaradas:
        contrato["grafos"]["tts"]["entradas"] = declaradas
        ruta_contrato.write_text(
            json.dumps(contrato, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"contrato: firma de tts actualizada a {[e['nombre'] for e in declaradas]}")
    nombres_in = [e["nombre"] for e in declaradas]
    entradas = tuple(
        e for n, e in zip(ENTRADAS_COMPLETAS, entradas_ejemplo(cfg), strict=True) if n in reales
    )
    modelo.eval()
    with torch.no_grad():
        esperado = modelo.inferir(*entradas_ejemplo(cfg)).numpy()

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
        "entradas": nombres_in,
        "ops": sorted({n.op_type for n in grafo.graph.node}),
        "ops_fuera_de_webgpu": ops_fuera_de_webgpu(grafo),
        "ort": {
            "fp32": medir(args.salida, entradas, esperado, nombres_in),
            "fp16": medir(salida16, entradas, esperado, nombres_in),
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
