"""Publish the voice packs on Hugging Face, so anyone downloads them from there.

    uv run hf auth login                       # once; token from https://huggingface.co/settings/tokens
    uv run python -m ttspro.export.publicar --repo <usuario>/ttspro-voces

Uploads everything in `paquetes/` (built by `ttspro.export.paquete`) to a public
model repo, creating it if needed, and prints the URL the page and the library
take as `vocesBase`: `https://huggingface.co/<repo>/resolve/main/`. Hugging Face
answers cross-origin fetches with the right CORS headers along its redirect chain
(measured 2026-09-05, docs/evidencia.md), which GitHub releases do not.

Files are flat (`indice.json`, `<clave>.json`, `<clave>.tts.onnx`, `<clave>.tts.fp16.onnx`),
so the same folder works on any host: the URL is the only thing that changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]

TARJETA = """---
license: mit
language:
- es
tags:
- text-to-speech
- onnx
- onnxruntime-web
- piper
- ttspro
---

# ttspro · voice packs

Base voices for [ttspro](https://github.com/Llicklair/ttspro), a Spanish text-to-speech with
voice cloning that runs entirely in the browser with ONNX Runtime Web. Each pack is a
[Piper](https://github.com/OHF-Voice/piper1-gpl) voice (MIT) ported onto ttspro's modules and
exported within ONNX Runtime Web's WebGPU operator set, with its own contract (symbol table,
graph signature) and its base-voice vector for the tone-colour converter.

Files: `indice.json` lists the packs; `<clave>.json` is meta + contract + vector;
`<clave>.tts.onnx` (fp32) and `<clave>.tts.fp16.onnx` are the graphs.

Use: open the ttspro page with `?voces=https://huggingface.co/{repo}/resolve/main/`, or
`TTS.cargar({{ vocesBase: "https://huggingface.co/{repo}/resolve/main/" }})` in the library.
Built by `uv run python -m ttspro.export.paquete --todas es`.

| Pack | fp32 / fp16 | ms per sentence (CPU) |
|---|---|---|
{filas}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="<usuario>/<nombre> en Hugging Face")
    ap.add_argument("--carpeta", type=Path, default=RAIZ / "paquetes")
    ap.add_argument("--privado", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    indice_ruta = args.carpeta / "indice.json"
    if not indice_ruta.exists():
        raise SystemExit(f"no hay {indice_ruta}: construye los paquetes con ttspro.export.paquete")
    indice = json.loads(indice_ruta.read_text(encoding="utf-8"))
    filas = "\n".join(
        f"| {m['clave']} | {m['MB']['fp32']} / {m['MB']['fp16']} MB | {m['ms_frase_cpu']} |"
        for m in indice.values()
    )
    (args.carpeta / "README.md").write_text(
        TARJETA.format(repo=args.repo, filas=filas), encoding="utf-8"
    )

    api = HfApi()
    quien = api.whoami()
    print(f"sesión: {quien.get('name')}")
    api.create_repo(args.repo, repo_type="model", private=args.privado, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.carpeta),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=["indice.json", "README.md", "*.json", "*.onnx"],
        ignore_patterns=["*.export.json"],
        commit_message=f"voice packs: {len(indice)} voces ({', '.join(indice)})",
    )
    url = f"https://huggingface.co/{args.repo}/resolve/main/"
    print(f"publicado: https://huggingface.co/{args.repo}")
    print(f"vocesBase: {url}")
    print(f"página:    http://localhost:5173/?voces={url}")


if __name__ == "__main__":
    main()
