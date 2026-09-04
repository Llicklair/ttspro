"""Initialize `Sintetizador` from a coqui-ai VITS checkpoint (by reference, rule 11).

    uv run python -m ttspro.train.inicializar --coqui <carpeta con config.json y model_file.pth> \
        --salida runs/init/G_0.pt

Measured 2026-09-04: `tts_models--en--vctk--vits` (Apache 2.0) is the same
network as our BASE config tensor for tensor — same blocks, same audio front
end, espeak IPA phonemes, 256-channel speaker conditioning. What differs and
how it is handled:

- block names: text_encoder→enc_p, posterior_encoder→enc_q, flow→flow,
  duration_predictor→dp, waveform_decoder→dec;
- old-style weight_norm (`weight_g`/`weight_v`) → torch parametrizations
  (`parametrizations.weight.original0/1`);
- the symbol table: rows of `text_encoder.emb` are copied CHARACTER BY
  CHARACTER into our table (models/contrato.json); symbols we have and they
  do not keep their random init and are reported;
- `emb_g` (their per-speaker table) is dropped: our speaker vector comes from
  speaker_encoder.onnx. The cond layers that consume it load unchanged;
- `enc_p.emb_lang` (ours only) stays at its small random init;
- coqui trained with `add_blank`: a blank (pad, id 0) between every two
  symbols. The frontend replicates that (`simbolos.blank_entre_tokens`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttspro.model.config import ConfigSintetizador
from ttspro.model.sintetizador import Sintetizador

RAIZ = Path(__file__).resolve().parents[3]
CONTRATO = json.loads((RAIZ / "models" / "contrato.json").read_text(encoding="utf-8"))

BLOQUES = {
    "text_encoder.": "enc_p.",
    "posterior_encoder.": "enc_q.",
    "flow.": "flow.",
    "duration_predictor.": "dp.",
    "waveform_decoder.": "dec.",
}


def config_base_coqui(
    cfg_coqui: dict, n_symbols: int, n_langs: int, gin: int
) -> ConfigSintetizador:
    ma = cfg_coqui["model_args"]
    return ConfigSintetizador(
        n_symbols=n_symbols,
        n_langs=n_langs,
        gin_channels=gin,
        hidden_channels=ma["hidden_channels"],
        filter_channels=ma["hidden_channels_ffn_text_encoder"],
        n_heads=ma["num_heads_text_encoder"],
        n_layers=ma["num_layers_text_encoder"],
        kernel_size=ma["kernel_size_text_encoder"],
        flow_layers=ma["num_layers_flow"],
        flow_kernel=ma["kernel_size_flow"],
        flow_dilation=ma["dilation_rate_flow"],
        posterior_layers=ma["num_layers_posterior_encoder"],
        upsample_initial_channel=ma["upsample_initial_channel_decoder"],
        upsample_rates=ma["upsample_rates_decoder"],
        upsample_kernel_sizes=ma["upsample_kernel_sizes_decoder"],
        resblock_kernel_sizes=ma["resblock_kernel_sizes_decoder"],
        resblock_dilation_sizes=ma["resblock_dilation_sizes_decoder"],
    )


def vocabulario_coqui(cfg_coqui: dict) -> list[str]:
    """VitsCharacters: [pad] + punctuations + characters + phonemes, in that order."""
    ch = cfg_coqui["characters"]
    return [ch["pad"]] + list(ch["punctuations"]) + list(ch["characters"]) + list(ch["phonemes"])


def _reindexar(clave: str, prefijo: str, indice_mio) -> str:
    """`prefijo.<k>.resto` -> `prefijo.<indice_mio(k)>.resto`."""
    if not clave.startswith(prefijo + "."):
        return clave
    resto = clave[len(prefijo) + 1 :]
    k, _, cola = resto.partition(".")
    return f"{prefijo}.{indice_mio(int(k))}.{cola}"


def traducir_clave(clave: str) -> str | None:
    for suyo, mio in BLOQUES.items():
        if clave.startswith(suyo):
            clave = mio + clave[len(suyo) :]
            break
    else:
        return None  # emb_g and anything unknown
    # coqui applies the channel flip inline, we have Flip modules: coupling k -> 2k,
    # and in the duration predictor ElementwiseAffine 0 -> 0, ConvFlow j -> 2j - 1.
    clave = _reindexar(clave, "flow.flows", lambda k: 2 * k)
    for lista in ("dp.flows", "dp.post_flows"):
        clave = _reindexar(clave, lista, lambda k: 0 if k == 0 else 2 * k - 1)
    clave = clave.replace(".translation", ".m").replace(".log_scale", ".logs")
    if clave.startswith("dec.cond_layer."):
        clave = "dec.cond." + clave[len("dec.cond_layer.") :]
    if clave.endswith(".weight_g"):
        clave = clave[: -len(".weight_g")] + ".parametrizations.weight.original0"
    elif clave.endswith(".weight_v"):
        clave = clave[: -len(".weight_v")] + ".parametrizations.weight.original1"
    return clave


def estado_desde_coqui(carpeta: Path, tabla: list[str]) -> tuple[ConfigSintetizador, dict, dict]:
    cfg_coqui = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    cfg = config_base_coqui(
        cfg_coqui, len(tabla), len(CONTRATO["idiomas"]), CONTRATO["embedding_locutor"]["dim"]
    )
    fichero = next(carpeta.glob("model_file.pth*"))
    suyo = torch.load(fichero, map_location="cpu", weights_only=False)
    suyo = suyo["model"] if "model" in suyo else suyo
    modelo = Sintetizador(cfg)
    mio = modelo.state_dict()

    traducidas = {}
    sin_destino = []
    for clave, tensor in suyo.items():
        nueva = traducir_clave(clave)
        if nueva is None:
            sin_destino.append(clave)
        elif nueva not in mio:
            sin_destino.append(f"{clave} -> {nueva} (no existe)")
        elif nueva == "enc_p.emb.weight":
            continue
        elif tuple(mio[nueva].shape) != tuple(tensor.shape):
            sin_destino.append(f"{clave}: {tuple(tensor.shape)} vs {tuple(mio[nueva].shape)}")
        else:
            traducidas[nueva] = tensor

    # symbol embeddings, character by character
    vocab = vocabulario_coqui(cfg_coqui)
    suyo_emb = suyo["text_encoder.emb.weight"]
    emb = mio["enc_p.emb.weight"].clone()
    copiados, nuevos = 0, []
    posicion = {c: i for i, c in enumerate(vocab)}
    for i, c in enumerate(tabla):
        if c in posicion:
            emb[i] = suyo_emb[posicion[c]]
            copiados += 1
        else:
            nuevos.append(c)
    traducidas["enc_p.emb.weight"] = emb

    faltan = sorted(set(mio) - set(traducidas))
    informe = {
        "checkpoint": str(fichero),
        "licencia": cfg_coqui.get("license") or "(ver .models.json de coqui)",
        "tensores_suyos": len(suyo),
        "cargados": len(traducidas),
        "mios_sin_cargar": faltan,
        "suyos_sin_destino": sin_destino,
        "simbolos_copiados": copiados,
        "simbolos_nuevos": nuevos,
        "add_blank": bool(cfg_coqui.get("add_blank", False)),
        "config": cfg.__dict__,
    }
    return cfg, traducidas, informe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coqui", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()
    tabla = CONTRATO["simbolos"]["tabla"]
    cfg, estado, informe = estado_desde_coqui(args.coqui, tabla)
    modelo = Sintetizador(cfg)
    incompatibles = modelo.load_state_dict(estado, strict=False)
    assert not incompatibles.unexpected_keys, incompatibles.unexpected_keys
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"modelo": modelo.state_dict(), "cfg": cfg.__dict__, "origen": informe}, args.salida)
    print(
        json.dumps(
            {k: v for k, v in informe.items() if k != "config"}, ensure_ascii=False, indent=1
        )
    )


if __name__ == "__main__":
    main()
