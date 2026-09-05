"""Build every graph in `models/` from public weights, with no corpus and no training.

    uv run python -m ttspro.export.modelos

This is the path a person who just cloned the repo takes. It downloads the base
voice from Piper and the tone-colour converter from OpenVoice v2, ports both onto
this repo's own modules and exports them to ONNX (ADR 0007 and 0008). Nothing here
touches `ttspro.train`: the models that ship are ported, not trained.

What it does NOT rebuild is `models/voces.json`, the preset voices. Those vectors
were measured over VCTK and OpenSLR es (~12 GB of audio), so the file travels in
git instead; `ttspro.export.voces` regenerates it for whoever has the corpora.

Steps, each one a module you can also run on its own:

    piper            rhasspy/piper-voices -> runs/<voz>/G_0.pt      (MIT)
    export.tts       that checkpoint      -> models/tts.onnx        (+ fp16)
    export.conversor myshell-ai/OpenVoiceV2 -> models/conversor.onnx, voz.onnx
    export.speaker_encoder  WeSpeaker     -> models/speaker_encoder.onnx

The speaker encoder never reaches the browser: it is how similarity is MEASURED
(criterion 4), so `--sin-encoder` skips it when you only want to run the demo.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models"
REPO_PIPER = "rhasspy/piper-voices"


def descargar_voz(clave: str, destino: Path) -> Path:
    """Fetch a Piper voice (the .onnx and its .json) into `destino`.

    The key is `<code>-<dataset>-<quality>` and the repo lays the files out as
    `<family>/<code>/<dataset>/<quality>/<key>.onnx`, so the path is derivable
    and no hardcoded table can rot.
    """
    from huggingface_hub import hf_hub_download

    codigo, dataset, calidad = clave.split("-")
    carpeta = f"{codigo.split('_')[0]}/{codigo}/{dataset}/{calidad}"
    destino.mkdir(parents=True, exist_ok=True)
    for sufijo in (".onnx", ".onnx.json"):
        origen = hf_hub_download(
            REPO_PIPER, f"{carpeta}/{clave}{sufijo}", cache_dir=str(RAIZ / ".scratch" / "hf")
        )
        ruta = destino / f"{clave}{sufijo}"
        if not ruta.exists() or ruta.stat().st_size != Path(origen).stat().st_size:
            ruta.write_bytes(Path(origen).read_bytes())
    return destino / f"{clave}.onnx"


def paso(titulo: str, argumentos: list[str]) -> None:
    """Run one export as its own process, so a failure names the step that failed."""
    print(f"\n=== {titulo} ===", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", *argumentos], cwd=RAIZ, check=False)
    if r.returncode != 0:
        raise SystemExit(f"{titulo}: fallo (codigo {r.returncode})")
    print(f"--- {titulo}: {time.perf_counter() - t0:.1f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voz", default="es_ES-davefx-medium", help="clave de la voz de Piper")
    ap.add_argument("--sin-encoder", action="store_true", help="no exportar el speaker encoder")
    args = ap.parse_args()

    onnx_piper = descargar_voz(args.voz, RAIZ / ".scratch" / "piper")
    checkpoint = RAIZ / "runs" / args.voz / "G_0.pt"
    print(f"voz base: {onnx_piper.relative_to(RAIZ)}")

    paso(
        "port de la voz base",
        ["ttspro.export.piper", "--onnx", str(onnx_piper), "--salida", str(checkpoint)],
    )
    paso("export del tts", ["ttspro.export.tts", "--checkpoint", str(checkpoint)])
    paso("export del conversor", ["ttspro.export.conversor"])
    if not args.sin_encoder:
        paso("export del speaker encoder", ["ttspro.export.speaker_encoder"])

    print("\nmodels/:")
    for f in sorted(MODELOS.glob("*.onnx")):
        print(f"  {f.name:28} {f.stat().st_size / 1e6:7.1f} MB")
    if not (MODELOS / "voces.json").exists():
        print("\nfalta models/voces.json: el demo se queda sin voces preset")


if __name__ == "__main__":
    main()
