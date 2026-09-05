"""Objective quality of a checkpoint: WER (Whisper) and speaker similarity (SECS).

    uv run python -m ttspro.train.evaluar --checkpoint runs/x/G_20000.pt \
        --cache cache/openslr_es cache/vctk --locutores 10 --frases 4

Held-out speakers only: the reference embedding comes from one utterance and
the synthesized sentences are OTHER utterances of that speaker, so nothing is
repeated from training conditioning.

Three numbers, not one. A SECS of 0.70 means nothing on its own, so every run
also reports the ceiling and the floor measured on the SAME audio:

- `secs_sintesis`: cosine(reference, synthesized) — what the model achieves;
- `secs_real_mismo`: cosine(reference, another REAL utterance of that speaker) —
  the ceiling this encoder can give;
- `secs_real_distinto`: cosine(reference, a real utterance of ANOTHER speaker) —
  the floor, i.e. what "no cloning at all" looks like.

Whisper and jiwer come from the `eval` extra; the speaker encoder is the same
one the browser uses (ADR 0002).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ttspro.export.tts import cargar, config_por_defecto
from ttspro.model.config import ConfigSintetizador

RAIZ = Path(__file__).resolve().parents[3]
IDIOMA_WHISPER = {"es": "es", "en": "en"}


def normalizar_para_wer(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9ñ ]+", " ", texto).split())


def sintetizar(
    modelo,
    cfg: ConfigSintetizador,
    tokens,
    embedding,
    idioma: int,
    semilla: int,
    noise_w: float = 0.4,
):
    g = torch.Generator().manual_seed(semilla)
    n = tokens.shape[1]
    with torch.no_grad():
        onda = modelo.inferir(
            tokens,
            torch.tensor([n]),
            embedding,
            torch.tensor([idioma]),
            torch.randn(1, cfg.inter_channels, n * 12, generator=g),
            torch.randn(1, 2, n, generator=g),
            torch.tensor([0.667, noise_w, 1.0]),
        )
    return onda[0, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cache", type=Path, nargs="+", required=True)
    ap.add_argument("--locutores", type=int, default=10, help="por idioma")
    ap.add_argument("--frases", type=int, default=4, help="por locutor")
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--salida", type=Path, default=None, help="carpeta donde dejar los wav")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument(
        "--noise-w",
        type=float,
        default=0.4,
        help="ruido del predictor de duración; 0.8 es el defecto de VITS y con un SDP poco "
        "entrenado dispara la duración (evidencia 2026-09-05)",
    )
    args = ap.parse_args()

    t0 = time.time()
    import soundfile as sf
    import torchaudio
    import whisper
    from jiwer import wer

    from ttspro.train.consistencia import ConsistenciaLocutor

    cfg = config_por_defecto()
    modelo = cargar(args.checkpoint, cfg).preparar_export()
    cfg = modelo.cfg
    consistencia = ConsistenciaLocutor()
    asr = whisper.load_model(args.whisper)
    idiomas = {"es": 0, "en": 1}

    por_locutor: dict[tuple[str, str], list] = defaultdict(list)
    for cache in args.cache:
        for e in torch.load(cache / "indice.pt"):
            por_locutor[(e["idioma"], e["locutor"])].append(e)
    rng = random.Random(args.semilla)

    filas = []
    for idioma in sorted({k[0] for k in por_locutor}):
        candidatos = sorted(
            k for k in por_locutor if k[0] == idioma and len(por_locutor[k]) >= args.frases + 2
        )
        rng.shuffle(candidatos)
        otros = [k for k in candidatos]
        for clave in candidatos[: args.locutores]:
            ejemplos = list(por_locutor[clave])
            rng.shuffle(ejemplos)
            referencia, real_mismo, *resto = ejemplos
            otro_locutor = rng.choice([o for o in otros if o != clave])
            real_distinto = rng.choice(por_locutor[otro_locutor])
            emb_ref = referencia["embedding"].unsqueeze(0)
            for ejemplo in resto[: args.frases]:
                onda = sintetizar(
                    modelo,
                    cfg,
                    ejemplo["tokens"].unsqueeze(0),
                    emb_ref,
                    idiomas[idioma],
                    args.semilla,
                    args.noise_w,
                )
                onda16 = torchaudio.functional.resample(onda, cfg.sample_rate, 16000).numpy()
                hipotesis = asr.transcribe(onda16, language=IDIOMA_WHISPER[idioma], fp16=False)[
                    "text"
                ]
                referencia_txt = normalizar_para_wer(ejemplo["texto"])
                hipotesis_txt = normalizar_para_wer(hipotesis)
                with torch.no_grad():
                    emb_sint = consistencia.embedding(onda.unsqueeze(0))[0]
                filas.append(
                    {
                        "idioma": idioma,
                        "locutor": clave[1],
                        "texto": ejemplo["texto"],
                        "hipotesis": hipotesis.strip(),
                        "wer": float(wer(referencia_txt, hipotesis_txt))
                        if referencia_txt
                        else None,
                        "secs_sintesis": float(np.dot(emb_sint.numpy(), emb_ref[0].numpy())),
                        "secs_real_mismo": float(
                            np.dot(real_mismo["embedding"].numpy(), emb_ref[0].numpy())
                        ),
                        "secs_real_distinto": float(
                            np.dot(real_distinto["embedding"].numpy(), emb_ref[0].numpy())
                        ),
                        "segundos": len(onda) / cfg.sample_rate,
                    }
                )
                if args.salida:
                    args.salida.mkdir(parents=True, exist_ok=True)
                    sf.write(
                        str(args.salida / f"{idioma}_{clave[1]}_{len(filas):03d}.wav"),
                        onda.numpy(),
                        cfg.sample_rate,
                    )

    resumen = {
        "checkpoint": str(args.checkpoint),
        "frases": len(filas),
        "segundos": round(time.time() - t0),
        "noise_w": args.noise_w,
    }
    for idioma in sorted({f["idioma"] for f in filas}):
        sub = [f for f in filas if f["idioma"] == idioma]
        resumen[idioma] = {
            "n": len(sub),
            "wer_medio": round(statistics.mean(f["wer"] for f in sub if f["wer"] is not None), 4),
            "wer_mediana": round(
                statistics.median(f["wer"] for f in sub if f["wer"] is not None), 4
            ),
            "secs_sintesis": round(statistics.mean(f["secs_sintesis"] for f in sub), 4),
            "secs_real_mismo": round(statistics.mean(f["secs_real_mismo"] for f in sub), 4),
            "secs_real_distinto": round(statistics.mean(f["secs_real_distinto"] for f in sub), 4),
        }
    destino = args.checkpoint.with_suffix(".evaluacion.json")
    destino.write_text(
        json.dumps({"resumen": resumen, "filas": filas}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
