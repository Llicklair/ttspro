"""Export the speaker encoder to ONNX and measure everything the ADRs ask for.

    uv run python -m ttspro.export.speaker_encoder --backbone wespeaker-resnet34
    uv run python -m ttspro.export.speaker_encoder --backbone ecapa-speechbrain

Every backbone ends up as ONE graph `onda (1, muestras) -> embedding (1, dim)`
with the features inside (ADR 0002) and unit norm at the end, composed from
three graphs with `onnx.compose`: our fbank (fp32), the backbone (fp32 or
fp16), our normalize (fp32). The fbank never goes to fp16: Kaldi features scale
the wave by 32768 and square it, and those intermediates overflow half
precision (measured 2026-09-04: NaN embeddings, docs/evidencia.md). Blocking
nodes in the converter does not help - it still stores the tensors between
blocked nodes in fp16 - so precision is decided per graph, before composing.

Backbones are integrated by reference (rule 11): a PyTorch one (speechbrain
ECAPA) is exported from features to embedding; a published ONNX one (WeSpeaker)
is used as is.

Printed facts (for docs/evidencia.md): feature parity with the backbone's own
pipeline, ops outside ORT Web's WebGPU set (rule 4), fp32/fp16 sizes, cosine
vs the reference pipeline, latency on 5 s of audio in onnxruntime CPU. The
contract signature (models/contrato.json) is applied at export.
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
from onnx import compose, version_converter
from torch import nn

from ttspro.export.ops_ort_web import ops_fuera_de_webgpu
from ttspro.model.fbank import FbankConv, FbankKaldiConv

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models"
CONTRATO = json.loads((MODELOS / "contrato.json").read_text(encoding="utf-8"))
FIRMA = CONTRATO["grafos"]["speaker_encoder"]
NOMBRE_IN = FIRMA["entradas"][0]["nombre"]
NOMBRE_OUT = FIRMA["salidas"][0]["nombre"]
OPSET = FIRMA["opset"]
FEATS = "feats"
EMBS = "embs"


class _Normalizar(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x.reshape(x.shape[0], -1), dim=1)


def _exportar_torch(
    modulo: nn.Module,
    ejemplo: torch.Tensor,
    ruta: Path,
    nombre_in: str,
    nombre_out: str,
    ejes: dict,
) -> onnx.ModelProto:
    torch.onnx.export(
        modulo.eval(),
        (ejemplo,),
        str(ruta),
        opset_version=OPSET,
        input_names=[nombre_in],
        output_names=[nombre_out],
        dynamic_axes={nombre_in: ejes},
        dynamo=False,
    )
    return onnx.load(str(ruta))


def _renombrar_io(g: onnx.ModelProto, entrada: str, salida: str) -> None:
    viejo_in, viejo_out = g.graph.input[0].name, g.graph.output[0].name
    g.graph.input[0].name, g.graph.output[0].name = entrada, salida
    for n in g.graph.node:
        n.input[:] = [entrada if x == viejo_in else x for x in n.input]
        n.output[:] = [salida if x == viejo_out else x for x in n.output]


def _ordenar(g: onnx.ModelProto) -> onnx.ModelProto:
    """Topological sort. The fp16 converter appends its boundary Casts at the end
    of the node list, and the checker (rightly) refuses that."""
    disponibles = {i.name for i in g.graph.input} | {t.name for t in g.graph.initializer} | {""}
    pendientes = list(g.graph.node)
    ordenados = []
    while pendientes:
        listos = [n for n in pendientes if all(x in disponibles for x in n.input)]
        if not listos:
            raise ValueError(
                "grafo con dependencias irresolubles: " + ", ".join(n.name for n in pendientes[:5])
            )
        for n in listos:
            ordenados.append(n)
            disponibles.update(n.output)
        pendientes = [n for n in pendientes if n not in listos]
    del g.graph.node[:]
    g.graph.node.extend(ordenados)
    return g


def componer(
    g_fbank: onnx.ModelProto, g_backbone: onnx.ModelProto, g_norm: onnx.ModelProto
) -> onnx.ModelProto:
    """onda -> [fbank] -> feats -> [backbone] -> embs -> [normalize] -> embedding."""
    partes = [
        _ordenar(version_converter.convert_version(g, OPSET)) for g in (g_fbank, g_backbone, g_norm)
    ]
    ir = max(g.ir_version for g in partes)
    for g in partes:
        g.ir_version = ir  # compose refuses mixed IR versions; the highest reads all
    g_fbank, g_backbone, g_norm = partes
    grafo = compose.merge_models(
        g_fbank,
        g_backbone,
        io_map=[(FEATS, g_backbone.graph.input[0].name)],
        prefix1="fb_",
        prefix2="bb_",
    )
    grafo = compose.merge_models(
        grafo, g_norm, io_map=[("bb_" + g_backbone.graph.output[0].name, EMBS)], prefix2="nm_"
    )
    _renombrar_io(grafo, NOMBRE_IN, NOMBRE_OUT)
    onnx.checker.check_model(grafo)
    return grafo


# ----------------------------------------------------------------------------- backbones
# Each returns (g_backbone feats->embs, fbank module, feats_ref, emb_ref unit-norm, hechos).


def ecapa_speechbrain(onda: torch.Tensor, tmp: Path):
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(tmp / "hf"),
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )
    enc.eval()
    fbank = FbankConv().eval()
    with torch.no_grad():
        feats_ref = enc.mods.mean_var_norm(enc.mods.compute_features(onda), torch.ones(1))
        emb_ref = nn.functional.normalize(enc.encode_batch(onda).reshape(1, -1), dim=1)
    g_backbone = _exportar_torch(
        enc.mods.embedding_model, feats_ref, tmp / "backbone.onnx", FEATS, EMBS, {1: "frames"}
    )
    hechos = {
        "backbone": "speechbrain/spkrec-ecapa-voxceleb (Apache-2.0), exportado desde torch",
        "parametros_M": round(
            sum(p.numel() for p in enc.mods.embedding_model.parameters()) / 1e6, 2
        ),
    }
    return g_backbone, fbank, feats_ref, emb_ref, hechos


def wespeaker_resnet34(onda: torch.Tensor, tmp: Path):
    import onnxruntime as ort
    import torchaudio
    from huggingface_hub import hf_hub_download

    repo = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
    g_backbone = onnx.load(
        hf_hub_download(repo, "voxceleb_resnet34_LM.onnx", cache_dir=str(tmp / "hf"))
    )
    feats_ref = torchaudio.compliance.kaldi.fbank(
        onda * 32768,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
        window_type="hamming",
        use_energy=False,
    )
    feats_ref = (feats_ref - feats_ref.mean(dim=0, keepdim=True)).unsqueeze(0)
    sesion = ort.InferenceSession(
        g_backbone.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (emb_ref,) = sesion.run(None, {sesion.get_inputs()[0].name: feats_ref.numpy()})
    emb_ref = nn.functional.normalize(torch.from_numpy(emb_ref), dim=1)
    hechos = {
        "backbone": f"{repo} (CC-BY-4.0), ONNX publicado, usado tal cual",
        "parametros_M": round(
            sum(int(np.prod(t.dims)) for t in g_backbone.graph.initializer) / 1e6, 2
        ),
    }
    return g_backbone, FbankKaldiConv().eval(), feats_ref, emb_ref, hechos


BACKBONES = {"ecapa-speechbrain": ecapa_speechbrain, "wespeaker-resnet34": wespeaker_resnet34}


# ----------------------------------------------------------------------------- main


def medir(ruta: Path, onda: torch.Tensor, emb_ref: torch.Tensor) -> dict:
    import onnxruntime as ort

    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    entrada = {NOMBRE_IN: onda.numpy()}
    sesion.run(None, entrada)
    tiempos = []
    for _ in range(10):
        t0 = time.perf_counter()
        (emb,) = sesion.run(None, entrada)
        tiempos.append(time.perf_counter() - t0)
    return {
        "fichero": ruta.name,
        "MB": round(ruta.stat().st_size / 1e6, 1),
        "dim": int(emb.shape[1]),
        "norma": round(float(np.linalg.norm(emb[0])), 4),
        "coseno_vs_referencia": round(float(np.dot(emb[0], emb_ref[0].numpy())), 5),
        "ms_mediana_5s_cpu": round(statistics.median(tiempos) * 1000),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="wespeaker-resnet34", choices=sorted(BACKBONES))
    ap.add_argument("--segundos", type=float, default=5.0)
    ap.add_argument("--salida", default=str(MODELOS / "speaker_encoder.onnx"))
    args = ap.parse_args()

    torch.manual_seed(0)
    onda = (torch.randn(1, int(16000 * args.segundos)) * 0.5).clamp(-1, 1)
    tmp = RAIZ / ".scratch" / args.backbone
    tmp.mkdir(parents=True, exist_ok=True)
    g_backbone, fbank, feats_ref, emb_ref, hechos = BACKBONES[args.backbone](onda, tmp)
    with torch.no_grad():
        hechos["fbank_max_abs_diff_vs_backbone"] = float((feats_ref - fbank(onda)).abs().max())

    g_fbank = _exportar_torch(fbank, onda, tmp / "fbank.onnx", NOMBRE_IN, FEATS, {1: "muestras"})
    g_norm = _exportar_torch(
        _Normalizar(), torch.randn(1, emb_ref.shape[1]), tmp / "norm.onnx", EMBS, NOMBRE_OUT, {}
    )

    from onnxruntime.transformers.float16 import convert_float_to_float16

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida16 = salida.with_suffix(".fp16.onnx")
    grafo32 = componer(g_fbank, g_backbone, g_norm)
    onnx.save(grafo32, str(salida))
    grafo16 = componer(g_fbank, convert_float_to_float16(g_backbone, keep_io_types=True), g_norm)
    onnx.save(grafo16, str(salida16))

    print(
        json.dumps(
            {
                **hechos,
                "opset": [(o.domain, o.version) for o in grafo32.opset_import],
                "ops": sorted({n.op_type for n in grafo32.graph.node}),
                "ops_fuera_de_webgpu": ops_fuera_de_webgpu(grafo32),
                "ort": {
                    "fp32": medir(salida, onda, emb_ref),
                    "fp16": medir(salida16, onda, emb_ref),
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
