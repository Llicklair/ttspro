"""`models/voces.json` for the demo: the base voice and the preset voices.

    uv run python -m ttspro.export.voces --cache cache/openslr_es cache/vctk --por-idioma 12

With cloning as post-processing (ADR 0007) a "voice" is no longer a speaker
embedding for the synthesizer: it is a **converter vector**, and the file carries
two things the browser cannot compute on its own:

- `base`: the speaker embedding the TTS is driven with (its fixed voice) AND the
  converter vector of that voice, measured on the TTS's OWN output. That second
  one is what the converter needs as source, and measuring it on synthetic audio
  rather than on the original speaker matters: the converter sees what the TTS
  produces, not what the human sounded like.
- `voces`: for each preset, the converter vector of a real recording of that
  speaker, plus a label.

The vectors are computed with the very same graphs the browser runs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models"
SR = 22050

PAISES = {
    "ar": "Argentina",
    "cl": "Chile",
    "co": "Colombia",
    "pe": "Perú",
    "pr": "Puerto Rico",
    "ve": "Venezuela",
}
FRASES_BASE = {
    "es": [
        "La casa tiene un jardín muy grande con árboles y flores",
        "Mañana por la mañana iremos a la playa bien temprano",
        "El libro que me prestaste la semana pasada era interesante",
    ],
    "en": [
        "The house has a very large garden with trees and flowers",
        "Tomorrow morning we will go to the beach quite early",
        "The book you lent me last week was really interesting",
    ],
}


def _info_vctk() -> dict[str, str]:
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


def _etiqueta(idioma: str, locutor: str, info_vctk: dict[str, str]) -> str:
    if idioma == "es" and "_" in locutor and len(locutor.split("_")[0]) == 3:
        pref = locutor.split("_")[0]
        pais = PAISES.get(pref[:2], pref[:2])
        genero = {"f": "F", "m": "M"}.get(pref[2], "?")
        return f"{locutor} · {genero} · español ({pais})"
    if locutor in info_vctk:
        return f"{locutor} · {info_vctk[locutor]}"
    return f"{locutor} · {idioma}"


def _onda(entrada: dict) -> torch.Tensor:
    onda, sr = sf.read(entrada["wav"], dtype="float32", always_2d=True)
    onda = torch.from_numpy(onda.mean(axis=1))
    if sr != SR:
        onda = torchaudio.functional.resample(onda, sr, SR)
    return onda.unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, nargs="+", required=True)
    ap.add_argument("--por-idioma", type=int, default=12)
    ap.add_argument("--referencias", type=int, default=5, help="audios por voz preset")
    ap.add_argument("--checkpoint", type=Path, default=None, help="TTS para medir la voz base")
    ap.add_argument("--salida", type=Path, default=MODELOS / "voces.json")
    args = ap.parse_args()

    from ttspro.export.tts import cargar as cargar_tts
    from ttspro.export.tts import config_por_defecto
    from ttspro.frontend import tokenizar
    from ttspro.model.conversor import cargar as cargar_conversor
    from ttspro.model.fbank import SpecConv

    conversor, _ = cargar_conversor()
    # The in-graph spectrogram, not the training one: this must be bit-for-bit what
    # the browser feeds voz.onnx (and rule 7 forbids importing ttspro.train here).
    spec = SpecConv().eval()

    def vector_voz(ondas: list[torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            vs = [conversor.voz(spec(o)) for o in ondas]
        return torch.stack(vs).mean(0).reshape(-1).numpy()

    por_locutor: dict[tuple[str, str], list] = defaultdict(list)
    segundos: dict[tuple[str, str], float] = defaultdict(float)
    for cache in args.cache:
        for e in torch.load(cache / "indice.pt"):
            por_locutor[(e["idioma"], e["locutor"])].append(e)
            segundos[(e["idioma"], e["locutor"])] += e["segundos"]

    info_vctk = _info_vctk()
    voces = []
    for idioma in sorted({k[0] for k in por_locutor}):
        candidatos = sorted(
            (k for k in por_locutor if k[0] == idioma and len(por_locutor[k]) > args.referencias),
            key=lambda k: -segundos[k],
        )
        colas: dict[str, list] = {"F": [], "M": [], "?": []}
        for clave in candidatos:
            colas[_genero(clave[0], clave[1], info_vctk)].append(clave)
        elegidos: list[tuple[str, str]] = []
        while len(elegidos) < args.por_idioma and any(colas.values()):
            for g in ("F", "M", "?"):
                if colas[g] and len(elegidos) < args.por_idioma:
                    elegidos.append(colas[g].pop(0))
        for _, locutor in elegidos:
            ondas = [_onda(e) for e in por_locutor[(idioma, locutor)][: args.referencias]]
            voces.append(
                {
                    "id": locutor,
                    "nombre": _etiqueta(idioma, locutor, info_vctk),
                    "idioma": idioma,
                    "voz": vector_voz(ondas).tolist(),
                }
            )
        print(f"{idioma}: {len(elegidos)} voces", flush=True)

    # The base voice: the TTS speaks with one fixed speaker embedding, and the
    # converter must be told what THAT sounds like, measured on synthetic audio.
    checkpoint = args.checkpoint
    if checkpoint is None:
        hechos = MODELOS / "tts.export.json"
        checkpoint = Path(json.loads(hechos.read_text(encoding="utf-8"))["checkpoint"])
    cfg = config_por_defecto()
    nombre_voz = torch.load(checkpoint, map_location="cpu", weights_only=False).get("voz")
    tts = cargar_tts(checkpoint, cfg).preparar_export()
    cfg = tts.cfg
    idiomas = {"es": 0, "en": 1}
    candidatas_es = [k for k in por_locutor if k[0] == "es"]
    base_clave = (
        max(candidatas_es, key=lambda k: segundos[k]) if candidatas_es else next(iter(por_locutor))
    )
    emb_base = por_locutor[base_clave][0]["embedding"].unsqueeze(0)
    ondas_base = []
    for i, texto in enumerate(FRASES_BASE.get(base_clave[0], FRASES_BASE["es"])):
        _, sec = tokenizar(texto, base_clave[0])
        t = torch.tensor([sec])
        n = t.shape[1]
        g = torch.Generator().manual_seed(i)
        with torch.no_grad():
            onda = tts.inferir(
                t,
                torch.tensor([n]),
                emb_base,
                torch.tensor([idiomas[base_clave[0]]]),
                torch.randn(1, cfg.inter_channels, n * 12, generator=g),
                torch.randn(1, 2, n, generator=g),
                torch.tensor([0.667, 0.5, 1.0]),
            )
        ondas_base.append(onda[0])

    base = {
        "locutor": nombre_voz or base_clave[1],
        "idioma": base_clave[0],
        "checkpoint": str(checkpoint),
        "voz": vector_voz(ondas_base).tolist(),
    }
    # A single-voice base TTS (a ported Piper voice, ADR 0008) takes no speaker
    # vector, and shipping one the browser would then feed to a graph that does not
    # declare it is exactly the kind of lie the contract exists to prevent.
    if cfg.gin_channels > 0:
        base["embedding_tts"] = emb_base.reshape(-1).tolist()
    salida = {"espacio": "openvoice-v2", "dim": len(voces[0]["voz"]), "base": base, "voces": voces}
    args.salida.write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "fichero": str(args.salida),
                "MB": round(args.salida.stat().st_size / 1e6, 2),
                "voces": len(voces),
                "base": base_clave,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
