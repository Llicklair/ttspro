"""Turn a manifest into a training index: token ids, speaker embedding, lengths.

    uv run python -m ttspro.data.preparar --manifiesto data/manifests/x.tsv --salida cache/x

Per utterance: the text goes through the SAME frontend the browser uses
(`ttspro.frontend`, batched per language) and the audio through the SAME
exported `models/speaker_encoder.onnx` the browser uses, so training sees
exactly what inference will feed the model. Utterances with symbols outside
the contract table, or outside the length window, are dropped and COUNTED —
the summary says how many and why, never silently.

Output: `<salida>/indice.pt` (list of dicts) + `<salida>/resumen.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from ttspro.data.manifiesto import Frase, leer
from ttspro.frontend import ids
from ttspro.frontend.fonemas import fonemizar_lotes
from ttspro.frontend.normalizar import normalizar
from ttspro.frontend.trocear import trocear

RAIZ = Path(__file__).resolve().parents[3]
SR_MODELO = 22050
SR_ENCODER = 16000


def fonemas_por_lotes(frases: list[Frase], tam_lote: int = 200) -> list[str | None]:
    """Same assembly as ttspro.frontend.tokens.fonemas_de_frase, many sentences per process."""
    salida: list[str | None] = [None] * len(frases)
    por_idioma: dict[str, list[int]] = {}
    for i, f in enumerate(frases):
        por_idioma.setdefault(f.idioma, []).append(i)
    for idioma, indices in por_idioma.items():
        for inicio in range(0, len(indices), tam_lote):
            lote = indices[inicio : inicio + tam_lote]
            trozos_por_frase = [trocear(normalizar(frases[i].texto)) for i in lote]
            textos = [[t.valor for t in trozos if t.tipo == "texto"] for trozos in trozos_por_frase]
            fonemas = fonemizar_lotes(textos, idioma)
            for i, trozos, fon in zip(lote, trozos_por_frase, fonemas, strict=True):
                it = iter(fon)
                elementos = []
                for t in trozos:
                    if t.tipo == "texto":
                        f_ = next(it)
                        if f_:
                            elementos.append(f_)
                    else:
                        elementos.append(t.valor)
                salida[i] = " ".join(elementos)
    return salida


class EncoderLocutor:
    def __init__(self, ruta: Path) -> None:
        import onnxruntime as ort

        self.sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
        self.entrada = self.sesion.get_inputs()[0].name

    def __call__(self, onda_16k: np.ndarray) -> np.ndarray:
        (emb,) = self.sesion.run(None, {self.entrada: onda_16k[None].astype(np.float32)})
        return emb[0]


def cargar_onda(ruta: Path) -> tuple[np.ndarray, int]:
    onda, sr = sf.read(str(ruta), dtype="float32", always_2d=True)
    return onda.mean(axis=1), sr


def remuestrear(onda: np.ndarray, sr: int, objetivo: int) -> np.ndarray:
    if sr == objetivo:
        return onda
    return torchaudio.functional.resample(torch.from_numpy(onda), sr, objetivo).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifiesto", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--encoder", type=Path, default=RAIZ / "models" / "speaker_encoder.onnx")
    ap.add_argument("--min-segundos", type=float, default=1.0)
    ap.add_argument("--max-segundos", type=float, default=12.0)
    ap.add_argument(
        "--wav-cache",
        type=Path,
        default=None,
        help="write 22.05 kHz mono wavs here (skips resampling at train time)",
    )
    args = ap.parse_args()

    t0 = time.time()
    frases = leer(args.manifiesto)
    print(f"{len(frases)} frases en {args.manifiesto}")
    fonemas = fonemas_por_lotes(frases)
    encoder = EncoderLocutor(args.encoder)
    descartes: Counter[str] = Counter()
    simbolos_fuera: Counter[str] = Counter()
    indice = []
    if args.wav_cache:
        args.wav_cache.mkdir(parents=True, exist_ok=True)
    for n, (frase, fon) in enumerate(zip(frases, fonemas, strict=True)):
        if not fon:
            descartes["sin fonemas"] += 1
            continue
        try:
            tokens = ids(fon)
        except ValueError:
            for c in fon:
                if c not in ids.__globals__["tabla"]():
                    simbolos_fuera[c] += 1
            descartes["símbolo fuera de la tabla"] += 1
            continue
        if not frase.wav.exists():
            descartes["wav no existe"] += 1
            continue
        onda, sr = cargar_onda(frase.wav)
        segundos = len(onda) / sr
        if not (args.min_segundos <= segundos <= args.max_segundos):
            descartes["fuera de la ventana de duración"] += 1
            continue
        emb = encoder(remuestrear(onda, sr, SR_ENCODER))
        ruta_wav = frase.wav
        if args.wav_cache:
            onda22 = remuestrear(onda, sr, SR_MODELO)
            ruta_wav = args.wav_cache / f"{n:07d}.wav"
            sf.write(str(ruta_wav), onda22, SR_MODELO)
        indice.append(
            {
                "wav": str(ruta_wav),
                "sr": SR_MODELO if args.wav_cache else sr,
                "segundos": round(segundos, 3),
                "idioma": frase.idioma,
                "locutor": frase.locutor,
                "texto": frase.texto,
                "fonemas": fon,
                "tokens": torch.tensor(tokens, dtype=torch.int64),
                "embedding": torch.from_numpy(emb),
            }
        )
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(frases)}  ({time.time() - t0:.0f} s)")

    args.salida.mkdir(parents=True, exist_ok=True)
    torch.save(indice, args.salida / "indice.pt")
    resumen = {
        "manifiesto": str(args.manifiesto),
        "frases": len(frases),
        "aceptadas": len(indice),
        "descartadas": dict(descartes),
        "simbolos_fuera_de_tabla": dict(simbolos_fuera.most_common(20)),
        "horas": round(sum(e["segundos"] for e in indice) / 3600, 2),
        "locutores": len({e["locutor"] for e in indice}),
        "por_idioma": dict(Counter(e["idioma"] for e in indice)),
        "segundos_de_preparacion": round(time.time() - t0),
    }
    (args.salida / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
