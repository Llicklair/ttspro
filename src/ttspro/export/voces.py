"""Voice presets for the demo: `models/voces.json`, speaker vectors with a label.

    uv run python -m ttspro.export.voces --coqui <carpeta del VITS de coqui>   # the ported model
    uv run python -m ttspro.export.voces --cache cache/openslr_es cache/vctk --por-idioma 12

A preset is only valid for a model that expects the SAME speaker space, so the
file carries `espacio` and the demo refuses a mismatch:

- `coqui-emb_g-vctk`: rows of coqui's `emb_g` (109 VCTK speakers). What the
  ported, un-finetuned model understands.
- `wespeaker-resnet34-LM`: mean embedding per speaker computed by
  models/speaker_encoder.onnx over the prepared cache. What the fine-tuned model
  understands, and the same space the mic/file cloning produces.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models"

PAISES = {
    "ar": "Argentina",
    "cl": "Chile",
    "co": "Colombia",
    "pe": "Perú",
    "pr": "Puerto Rico",
    "ve": "Venezuela",
}


def desde_coqui(carpeta: Path) -> tuple[str, list[dict]]:
    fichero = next(carpeta.glob("model_file.pth*"))
    emb_g = torch.load(fichero, map_location="cpu", weights_only=False)["model"]["emb_g.weight"]
    ids = json.loads((carpeta / "speaker_ids.json").read_text(encoding="utf-8"))
    info = {}
    speaker_info = next(RAIZ.glob("data/raw/vctk/**/speaker-info.txt"), None)
    if speaker_info:
        for linea in speaker_info.read_text(encoding="utf-8").splitlines()[1:]:
            partes = linea.split()
            if len(partes) >= 4:
                info[partes[0]] = {"genero": partes[2], "acento": " ".join(partes[3:5])}
    voces = []
    for nombre, indice in sorted(ids.items(), key=lambda kv: kv[0]):
        extra = info.get(nombre, {})
        etiqueta = f"{nombre} · {extra.get('genero', '?')} · {extra.get('acento', 'English')}"
        voces.append(
            {"id": nombre, "nombre": etiqueta, "idioma": "en", "embedding": emb_g[indice].tolist()}
        )
    return "coqui-emb_g-vctk", voces


def desde_cache(caches: list[Path], por_idioma: int) -> tuple[str, list[dict]]:
    por_locutor: dict[tuple[str, str], list] = defaultdict(list)
    segundos: dict[tuple[str, str], float] = defaultdict(float)
    for cache in caches:
        for e in torch.load(cache / "indice.pt"):
            clave = (e["idioma"], e["locutor"])
            por_locutor[clave].append(e["embedding"].numpy())
            segundos[clave] += e["segundos"]
    info_vctk = _info_vctk()
    voces = []
    for idioma in sorted({k[0] for k in por_locutor}):
        candidatos = sorted((k for k in por_locutor if k[0] == idioma), key=lambda k: -segundos[k])
        # Alternate genders: the speakers with most audio in VCTK are all female,
        # and a preset list of twelve identical-sounding voices is not a preset list.
        colas: dict[str, list] = {"F": [], "M": [], "?": []}
        for clave in candidatos:
            colas[_genero(clave[0], clave[1], info_vctk)].append(clave)
        elegidos = []
        while len(elegidos) < por_idioma and any(colas.values()):
            for g in ("F", "M", "?"):
                if colas[g] and len(elegidos) < por_idioma:
                    elegidos.append(colas[g].pop(0))
        for _, locutor in elegidos:
            media = np.mean(por_locutor[(idioma, locutor)], axis=0)
            media = media / (np.linalg.norm(media) + 1e-8)
            voces.append(
                {
                    "id": locutor,
                    "nombre": _etiqueta(idioma, locutor, segundos[(idioma, locutor)], info_vctk),
                    "idioma": idioma,
                    "embedding": media.astype(np.float32).tolist(),
                }
            )
    return "wespeaker-resnet34-LM", voces


def _info_vctk() -> dict[str, str]:
    """speaker-info.txt: `p225  23  F  English  Southern England`."""
    fichero = next(RAIZ.glob("data/raw/vctk/**/speaker-info.txt"), None)
    if fichero is None:
        return {}
    info = {}
    for linea in fichero.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        partes = linea.split()
        if len(partes) >= 4:
            info[partes[0]] = f"{partes[2]} · {' '.join(partes[3:5])}"
    return info


def _genero(idioma: str, locutor: str, info_vctk: dict[str, str]) -> str:
    if idioma == "es" and "_" in locutor and len(locutor.split("_")[0]) == 3:
        return {"f": "F", "m": "M"}.get(locutor.split("_")[0][2], "?")
    etiqueta = info_vctk.get(locutor, "")
    return etiqueta.split(" · ")[0] if etiqueta[:1] in ("F", "M") else "?"


def _etiqueta(idioma: str, locutor: str, seg: float, info_vctk: dict[str, str]) -> str:
    # OpenSLR es ids look like "arf_00295": country + gender letter
    if idioma == "es" and "_" in locutor and len(locutor.split("_")[0]) == 3:
        pref = locutor.split("_")[0]
        pais = PAISES.get(pref[:2], pref[:2])
        genero = {"f": "F", "m": "M"}.get(pref[2], "?")
        return f"{locutor} · {genero} · español ({pais}) · {seg / 60:.0f} min"
    if locutor in info_vctk:
        return f"{locutor} · {info_vctk[locutor]} · {seg / 60:.0f} min"
    return f"{locutor} · {idioma} · {seg / 60:.0f} min"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coqui", type=Path, default=None)
    ap.add_argument("--cache", type=Path, nargs="*", default=None)
    ap.add_argument("--por-idioma", type=int, default=12)
    ap.add_argument("--salida", type=Path, default=MODELOS / "voces.json")
    args = ap.parse_args()
    if args.coqui:
        espacio, voces = desde_coqui(args.coqui)
    elif args.cache:
        espacio, voces = desde_cache(args.cache, args.por_idioma)
    else:
        ap.error("--coqui o --cache")
    args.salida.write_text(
        json.dumps(
            {"espacio": espacio, "dim": len(voces[0]["embedding"]), "voces": voces},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "espacio": espacio,
                "voces": len(voces),
                "por_idioma": {
                    i: sum(v["idioma"] == i for v in voces) for i in {v["idioma"] for v in voces}
                },
                "fichero": str(args.salida),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
