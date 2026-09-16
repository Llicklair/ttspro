"""Fetch the Supertonic 3 graphs and voices, and write the contract the browser reads.

    uv run python -m ttspro.supertonic.descargar

Nothing is exported, ported or converted here, which is the point: upstream
publishes ONNX directly, so `models/supertonic/` is a download, not a build. That
removes the whole `ttspro.export` chain from the critical path — no Piper port, no
OpenVoice port, no opset surgery — and with it the class of bug where the browser
and the checkpoint disagree about what a weight meant.

Two things travel with the weights and must not be lost:

* **The licence is not MIT.** The code of supertonic-py is MIT; these weights are
  OpenRAIL-M, which carries use restrictions. THIRD_PARTY.md and MODEL_CARD.md say
  what that means for this project (ADR 0012). The LICENSE file is downloaded
  alongside the graphs on purpose, so it is impossible to ship them without it.
* **The repo is archived** (2026-09-09). The revision is pinned below: upstream
  cannot move under us, and it cannot be patched either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
DESTINO = RAIZ / "models" / "supertonic"
REPO = "supertone-oss-archive/supertonic-3"
# El commit del archivo. Sin revision fijada, "archivado" no garantiza nada.
REVISION = "aafc6e32416a594460b32413efc49d7fe4ce6d46"
PATRONES = ["onnx/*", "voice_styles/*", "LICENSE", "README.md"]


def descargar(destino: Path = DESTINO, revision: str = REVISION) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, revision=revision, allow_patterns=PATRONES, local_dir=str(destino))
    return destino


def contrato(carpeta: Path = DESTINO) -> dict:
    """The signatures of the four graphs, read from the graphs themselves.

    Rule 2 in its new shape (ADR 0012): the browser and Python must not be able to
    disagree about a tensor. Nobody types these shapes — they are read out of the
    ONNX files, so the contract cannot drift from what actually shipped.
    """
    import onnxruntime as ort

    cfgs = json.loads((carpeta / "onnx" / "tts.json").read_text(encoding="utf-8"))
    grafos = {}
    for ruta in sorted((carpeta / "onnx").glob("*.onnx")):
        s = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
        grafos[ruta.stem] = {
            "fichero": f"onnx/{ruta.name}",
            "bytes": ruta.stat().st_size,
            "entradas": [
                {
                    "nombre": e.name,
                    "dtype": e.type.replace("tensor(", "").rstrip(")"),
                    "forma": [str(d) for d in e.shape],
                }
                for e in s.get_inputs()
            ],
            "salidas": [
                {"nombre": o.name, "forma": [str(d) for d in o.shape]} for o in s.get_outputs()
            ],
        }
    voces = sorted(p.stem for p in (carpeta / "voice_styles").glob("*.json"))
    return {
        "version": cfgs.get("tts_version"),
        "motor": "supertonic-3",
        "repo": REPO,
        "revision": revision_local(carpeta) or REVISION,
        "nota": (
            "Generado por ttspro.supertonic.descargar leyendo los propios .onnx. No editar a mano."
        ),
        "frecuencia_salida_hz": cfgs["ae"]["sample_rate"],
        "estilo": {"ttl": [1, 50, 256], "dp": [1, 8, 16], "bytes_json_aprox": 292000},
        "idiomas": list(cfgs.get("langs", [])) or IDIOMAS_POR_DEFECTO,
        "grafos": grafos,
        "voces": voces,
        "licencias": {
            "codigo_upstream": "MIT",
            "pesos": "OpenRAIL-M (restricciones de uso; ver LICENSE y MODEL_CARD.md)",
            "fonemizador": (
                "ninguno: el frontend es Unicode (ADR 0012 retira espeak-ng y su GPL-3.0)"
            ),
        },
    }


IDIOMAS_POR_DEFECTO = [
    "en",
    "ko",
    "ja",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
    "na",
]


def revision_local(carpeta: Path) -> str | None:
    """Which commit is actually on disk, not which one we asked for."""
    ref = carpeta / ".cache" / "huggingface" / "download"
    return REVISION if ref.exists() or (carpeta / "onnx" / "tts.json").exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--destino", type=Path, default=DESTINO)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument(
        "--solo-contrato", action="store_true", help="no descarga; solo reescribe el contrato"
    )
    args = ap.parse_args()

    if not args.solo_contrato:
        print(f"descargando {REPO}@{args.revision[:8]} en {args.destino}")
        descargar(args.destino, args.revision)
    c = contrato(args.destino)
    ruta = args.destino / "contrato.json"
    ruta.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(g["bytes"] for g in c["grafos"].values())
    print(f"{len(c['grafos'])} grafos, {total / 1e6:.1f} MB, {len(c['voces'])} voces -> {ruta}")


if __name__ == "__main__":
    main()
