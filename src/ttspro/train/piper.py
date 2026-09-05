"""Port a Piper voice (MIT) into `Sintetizador`, straight from its ONNX.

    uv run python -m ttspro.train.piper --onnx .scratch/piper/es_ES-davefx-medium.onnx \
        --salida runs/piper_es/G_0.pt

Piper publishes ONNX, not PyTorch checkpoints, but the export keeps the original
parameter names (`enc_p.encoder.attn_layers.0.conv_q.weight`, `flow.flows.2.…`),
so the initializers map onto our modules the same way the coqui and OpenVoice
ports did. The architecture is read from the tensor SHAPES, not guessed.

What is deliberately absent from a Piper inference graph, and why it is fine:

- `enc_q` (posterior encoder): training only, never used to synthesize.
- `dp.flows.1`: VITS drops one flow at inference (`flows[:-2] + [flows[-1]]`),
  and so does our implementation.
- `dp.flows.0.logs`: zeros, folded away by the exporter; our init is zeros too.
- speaker and language embeddings: these voices are single-speaker and
  monolingual, so the ported model runs with `gin_channels=0` and `n_langs=1`.

The symbol table travels with the checkpoint: Piper has its own phoneme id map
and wraps the sentence in `^ … $`, which is not our convention, and guessing it
would produce audio that is almost right — the worst kind of wrong.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from ttspro.model.config import ConfigSintetizador
from ttspro.model.sintetizador import Sintetizador

RAIZ = Path(__file__).resolve().parents[3]


# Two things Piper's exporter renames, which have to be undone by hand:
# - the symbol embedding becomes `sid`;
# - every weight-normed convolution loses its name and becomes `onnx::Conv_NNNN`,
#   recoverable because the NODE that consumes it keeps the module path
#   (`/flow/flows.6/enc/in_layers.0/Conv`).
RENOMBRES = {"sid": "enc_p.emb.weight"}
CON_PESO = {"Conv": 1, "ConvTranspose": 1, "Gemm": 1, "MatMul": 1}


def pesos(ruta: Path) -> dict[str, np.ndarray]:
    m = onnx.load(str(ruta))
    bruto = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    nombres = dict(RENOMBRES)
    for nodo in m.graph.node:
        idx = CON_PESO.get(nodo.op_type)
        if idx is None or len(nodo.input) <= idx or not nodo.name.startswith("/"):
            continue
        entrada = nodo.input[idx]
        if entrada in bruto and entrada.startswith("onnx::"):
            ruta_modulo = nodo.name.lstrip("/").rsplit("/", 1)[0].replace("/", ".")
            nombres[entrada] = f"{ruta_modulo}.weight"
    return {nombres.get(k, k): v for k, v in bruto.items()}


def resblocks_desde_grafo(ruta: Path) -> tuple[str, list[int], list[list[int]]]:
    """Kernel sizes and dilations of the decoder, read off the Conv nodes.

    Also tells ResBlock1 from ResBlock2: `convs1`/`convs2` versus a single `convs`.
    """
    m = onnx.load(str(ruta))
    por_bloque: dict[int, list[tuple[int, int]]] = {}
    tipo = "2"
    for nodo in m.graph.node:
        if nodo.op_type != "Conv" or "/dec/resblocks." not in nodo.name:
            continue
        if "convs1." in nodo.name or "convs2." in nodo.name:
            tipo = "1"
        indice = int(nodo.name.split("resblocks.")[1].split("/")[0])
        kernel = next((list(a.ints)[0] for a in nodo.attribute if a.name == "kernel_shape"), 0)
        dilatacion = next((list(a.ints)[0] for a in nodo.attribute if a.name == "dilations"), 1)
        por_bloque.setdefault(indice, []).append((kernel, dilatacion))
    kernels, dilataciones, vistos = [], [], set()
    for indice in sorted(por_bloque):
        kernel = por_bloque[indice][0][0]
        if kernel in vistos:
            continue
        vistos.add(kernel)
        kernels.append(kernel)
        dilataciones.append([d for _, d in por_bloque[indice]])
    return tipo, kernels, dilataciones


def config_desde_pesos(
    w: dict[str, np.ndarray], onnx_ruta: Path, n_symbols: int, sample_rate: int
) -> ConfigSintetizador:
    """Everything read off the shapes and the graph; nothing assumed."""
    hidden = w["enc_p.emb.weight"].shape[1]
    inter = w["enc_p.proj.weight"].shape[0] // 2
    n_layers = len({k.split(".")[3] for k in w if k.startswith("enc_p.encoder.attn_layers.")})
    filtro = w["enc_p.encoder.ffn_layers.0.conv_1.weight"].shape[0]
    kernel = w["enc_p.encoder.ffn_layers.0.conv_1.weight"].shape[2]
    # `emb_rel_k` is (heads_share ? 1 : n_heads, 2*window+1, hidden/n_heads): the first
    # axis is 1 when the heads share the table, so the head count comes from the LAST.
    rel = w["enc_p.encoder.attn_layers.0.emb_rel_k"].shape
    heads = hidden // rel[2]
    ventana = (rel[1] - 1) // 2
    ups = sorted(
        (int(k.split(".")[2]), w[k].shape)
        for k in w
        if k.startswith("dec.ups.") and k.endswith(".weight")
    )
    tipo, kernels_res, dilataciones = resblocks_desde_grafo(onnx_ruta)
    capas_flow = len(
        {int(k.split(".")[5]) for k in w if k.startswith("flow.flows.0.enc.in_layers.")}
    )
    n_flows = len({int(k.split(".")[2]) for k in w if k.startswith("flow.flows.") and ".pre." in k})
    return ConfigSintetizador(
        n_symbols=n_symbols,
        n_langs=1,
        gin_channels=0,
        sample_rate=sample_rate,
        hop_length=int(np.prod([f[2] // 2 for _, f in ups])),
        hidden_channels=hidden,
        inter_channels=inter,
        filter_channels=filtro,
        n_heads=heads,
        n_layers=n_layers,
        kernel_size=kernel,
        window_size=ventana,
        flow_n_flows=n_flows,
        flow_kernel=w["flow.flows.0.enc.in_layers.0.weight"].shape[2],
        flow_layers=capas_flow,
        upsample_initial_channel=ups[0][1][0],
        upsample_rates=[f[2] // 2 for _, f in ups],
        upsample_kernel_sizes=[f[2] for _, f in ups],
        resblock_kernel_sizes=kernels_res,
        resblock_dilation_sizes=dilataciones,
        resblock=tipo,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()

    cfg_piper = json.loads(Path(str(args.onnx) + ".json").read_text(encoding="utf-8"))
    w = pesos(args.onnx)
    cfg = config_desde_pesos(
        w, args.onnx, cfg_piper["num_symbols"], cfg_piper["audio"]["sample_rate"]
    )
    modelo = Sintetizador(cfg)
    # Piper's ONNX carries weight-norm already folded, so materialize ours first.
    from ttspro.model.comun import remove_weight_norm

    remove_weight_norm(modelo)
    mio = modelo.state_dict()

    cargados, sin_destino = {}, []
    for k, v in w.items():
        if k not in mio:
            sin_destino.append(k)
        elif tuple(mio[k].shape) != tuple(v.shape):
            sin_destino.append(f"{k}: {v.shape} vs {tuple(mio[k].shape)}")
        else:
            cargados[k] = torch.from_numpy(v.copy())
    faltan = sorted(set(mio) - set(cargados))
    modelo.load_state_dict(cargados, strict=False)

    tabla = [None] * cfg_piper["num_symbols"]
    for simbolo, ids in cfg_piper["phoneme_id_map"].items():
        for i in ids:
            tabla[i] = simbolo
    simbolos = {
        "pad": cfg_piper["phoneme_id_map"]["_"][0],
        "bos": cfg_piper["phoneme_id_map"]["^"][0],
        "eos": cfg_piper["phoneme_id_map"]["$"][0],
        "blank_entre_tokens": True,
        "blank_al_inicio": False,
        # Piper never saw the Spanish opening marks (its punctuation set is
        # !"#$'(),-.:;?), so they are DECLARED as dropped instead of silently
        # ignored: the closing ? and ! already carry the intonation.
        "equivalencias": {"¿": "", "¡": ""},
        "tabla": [s if s is not None else "" for s in tabla],
        "nota": "Tabla de Piper tal cual (ADR 0008). Secuencia: bos, cada simbolo seguido"
        " de pad, y eos.",
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    origen = {
        "onnx": str(args.onnx),
        "licencia": "MIT (rhasspy/piper-voices)",
        "dataset": cfg_piper.get("dataset"),
        "espeak": cfg_piper.get("espeak"),
        "inference": cfg_piper.get("inference"),
        "tensores_suyos": len(w),
        "cargados": len(cargados),
        "mios_sin_cargar": faltan,
        "suyos_sin_destino": sin_destino,
    }
    # A ported voice is monolingual: the contract must not offer languages the
    # model would answer with noise.
    voz_espeak = (cfg_piper.get("espeak") or {}).get("voice", "es")
    idioma = "en" if voz_espeak.startswith("en") else voz_espeak.split("-")[0]
    torch.save(
        {
            "modelo": modelo.state_dict(),
            "cfg": cfg.__dict__,
            "simbolos": simbolos,
            "idiomas": [idioma],
            "voz": args.onnx.stem,
            "origen": origen,
        },
        args.salida,
    )
    print(
        json.dumps(
            {
                **{
                    k: v
                    for k, v in origen.items()
                    if k not in ("mios_sin_cargar", "suyos_sin_destino")
                },
                "sin_cargar": len(faltan),
                "sin_destino": sin_destino,
                "config": {
                    k: cfg.__dict__[k]
                    for k in (
                        "n_symbols",
                        "sample_rate",
                        "hidden_channels",
                        "filter_channels",
                        "n_layers",
                        "n_heads",
                        "window_size",
                        "upsample_rates",
                        "upsample_kernel_sizes",
                        "upsample_initial_channel",
                        "resblock_kernel_sizes",
                        "resblock_dilation_sizes",
                        "resblock",
                        "hop_length",
                        "flow_layers",
                    )
                },
                "salida": str(args.salida),
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
