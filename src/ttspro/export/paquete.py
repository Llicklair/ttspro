"""Voice packs: a base voice the browser can download on demand (ADR 0011).

    uv run python -m ttspro.export.paquete --voz es_MX-claude-high --voz es_ES-carlfm-x_low
    uv run python -m ttspro.export.paquete --todas es        # every Piper voice of that family

A pack is three flat files, so any static host works (a GitHub release, a
Hugging Face repo, a folder):

    <clave>.json          meta + the voice's OWN contract + its base-voice vector
    <clave>.tts.onnx      fp32
    <clave>.tts.fp16.onnx fp16

plus one `indice.json` listing every pack. Each voice ships its own contract
because the symbol table differs between Piper voices (carlfm has 130 symbols,
davefx 256, claude adds five): the frontend must tokenize with the table of the
voice that is about to speak, never with a global one.

The base-voice vector is what the converter needs as `voz_origen`: measured with
the exported graph itself and models/voz.onnx, the same two files the browser
runs, over the three fixed sentences of `ttspro.export.voces`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ttspro.export.modelos import REPO_PIPER, descargar_voz
from ttspro.export.voces import FRASES_BASE

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models"
CALIDAD = {"x_low": "muy baja", "low": "baja", "medium": "media", "high": "alta"}


def voces_disponibles(familia: str) -> list[str]:
    """Single-speaker Piper voices of a language family, from the published index."""
    from huggingface_hub import hf_hub_download

    ruta = hf_hub_download(REPO_PIPER, "voices.json", cache_dir=str(RAIZ / ".scratch" / "hf"))
    indice = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return sorted(
        k
        for k, v in indice.items()
        if v["language"]["family"] == familia and v.get("num_speakers", 1) == 1
    )


def paso(argumentos: list[str]) -> None:
    r = subprocess.run([sys.executable, "-m", *argumentos], cwd=RAIZ, check=False)
    if r.returncode != 0:
        raise SystemExit(f"fallo en {argumentos[0]} (codigo {r.returncode})")


def vector_base(tts_onnx: Path, contrato: dict, idioma: str) -> list[float]:
    """Mean voice vector of the exported graph's own speech, via models/voz.onnx."""
    import onnxruntime as ort
    import torch

    from ttspro.frontend import tokenizar

    tts = ort.InferenceSession(str(tts_onnx), providers=["CPUExecutionProvider"])
    voz = ort.InferenceSession(str(MODELOS / "voz.onnx"), providers=["CPUExecutionProvider"])
    nombres = [e["nombre"] for e in contrato["grafos"]["tts"]["entradas"]]
    inter = next(e for e in contrato["grafos"]["tts"]["entradas"] if e["nombre"] == "ruido_flow")
    canales = inter["forma"][1]
    vectores = []
    for i, texto in enumerate(FRASES_BASE.get(idioma, FRASES_BASE["es"])):
        _, sec = tokenizar(texto, idioma, contrato["simbolos"])
        g = torch.Generator().manual_seed(i)
        n = len(sec)
        feed = {
            "tokens": np.array([sec], dtype=np.int64),
            "longitud_tokens": np.array([n], dtype=np.int64),
            "ruido_flow": torch.randn(1, canales, n * 12, generator=g).numpy(),
            "ruido_duracion": torch.randn(1, 2, n, generator=g).numpy(),
            "escala_ruido": np.array([0.667, 0.5, 1.0], dtype=np.float32),
        }
        (onda,) = tts.run(None, {k: feed[k] for k in nombres})
        (v,) = voz.run(None, {"onda": onda.reshape(1, -1).astype(np.float32)})
        vectores.append(v.reshape(-1))
    return np.mean(vectores, axis=0).tolist()


def empaquetar(clave: str, salida: Path) -> dict:
    onnx_piper = descargar_voz(clave, RAIZ / ".scratch" / "piper")
    meta_piper = json.loads(Path(str(onnx_piper) + ".json").read_text(encoding="utf-8"))
    checkpoint = RAIZ / "runs" / clave / "G_0.pt"
    salida.mkdir(parents=True, exist_ok=True)
    tts_onnx = salida / f"{clave}.tts.onnx"
    contrato_ruta = salida / f"{clave}.contrato.tmp.json"

    paso(["ttspro.export.piper", "--onnx", str(onnx_piper), "--salida", str(checkpoint)])
    paso(
        [
            "ttspro.export.tts",
            "--checkpoint",
            str(checkpoint),
            "--salida",
            str(tts_onnx),
            "--contrato",
            str(contrato_ruta),
        ]
    )
    contrato = json.loads(contrato_ruta.read_text(encoding="utf-8"))
    contrato_ruta.unlink()
    hechos = json.loads(tts_onnx.with_suffix(".export.json").read_text(encoding="utf-8"))
    idioma = contrato["idiomas"][0]
    lengua = meta_piper["language"]
    nombre = f"{meta_piper['dataset']} · {lengua['name_english']} ({lengua['country_english']})"
    meta = {
        "clave": clave,
        "nombre": nombre,
        "idioma": idioma,
        "region": lengua["code"],
        "calidad": CALIDAD.get(meta_piper.get("audio", {}).get("quality", ""), ""),
        "licencia": "MIT (rhasspy/piper-voices)",
        "parametros_M": hechos["parametros_exportados_M"],
        "MB": {"fp32": hechos["ort"]["fp32"]["MB"], "fp16": hechos["ort"]["fp16"]["MB"]},
        "ms_frase_cpu": hechos["ort"]["fp32"]["ms_mediana_cpu"],
        "ficheros": {"fp32": f"{clave}.tts.onnx", "fp16": f"{clave}.tts.fp16.onnx"},
    }
    paquete = {
        **meta,
        # only what the browser needs to drive THIS graph: its symbols and signature
        "contrato": {
            "simbolos": contrato["simbolos"],
            "idiomas": contrato["idiomas"],
            "fonemizador": contrato.get("fonemizador"),
            "grafos": {"tts": contrato["grafos"]["tts"]},
        },
        "voz_base": vector_base(tts_onnx, contrato, idioma),
    }
    (salida / f"{clave}.json").write_text(json.dumps(paquete, ensure_ascii=False), encoding="utf-8")
    mb = meta["MB"]
    print(f"paquete {clave}: {mb['fp32']} / {mb['fp16']} MB, {meta['ms_frase_cpu']} ms/frase")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voz", action="append", default=[], help="clave de Piper, repetible")
    ap.add_argument("--todas", default=None, help="familia de idioma, p. ej. es")
    ap.add_argument("--salida", type=Path, default=RAIZ / "paquetes")
    ap.add_argument(
        "--solo-nuevas", action="store_true", help="saltar las que ya estan en el indice"
    )
    args = ap.parse_args()

    claves = list(args.voz)
    if args.todas:
        claves += voces_disponibles(args.todas)
    if not claves:
        raise SystemExit("di --voz <clave> o --todas <familia>")
    indice_ruta = args.salida / "indice.json"
    indice = json.loads(indice_ruta.read_text(encoding="utf-8")) if indice_ruta.exists() else {}
    for clave in dict.fromkeys(claves):
        if args.solo_nuevas and clave in indice:
            print(f"paquete {clave}: ya en el indice, saltado")
            continue
        indice[clave] = empaquetar(clave, args.salida)
        indice_ruta.write_text(
            json.dumps(dict(sorted(indice.items())), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    print(f"{len(indice)} paquete(s) en {args.salida}")


if __name__ == "__main__":
    main()
