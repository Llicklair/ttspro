"""Turn a manifest into a training index: token ids, speaker embedding, lengths.

    uv run python -m ttspro.data.preparar --manifiesto data/manifests/x.tsv --salida cache/x \
        --wav-cache data/wav22/x --workers 6

Per utterance: the text goes through the SAME frontend the browser uses
(`ttspro.frontend`, batched per language) and the audio through the SAME
exported `models/speaker_encoder.onnx` the browser uses, so training sees
exactly what inference will feed the model. Utterances with symbols outside
the contract table, or outside the length window, are dropped and COUNTED —
the summary says how many and why, never silently.

Audio work (decode, resample, embedding, wav cache) is CPU bound — measured
2026-09-04: 151 ms encoder + 89 ms resample per 5 s utterance in one process,
so it runs in a process pool; each worker owns one single-threaded ORT session.

Output: `<salida>/indice.pt` (list of dicts) + `<salida>/resumen.json`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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
from ttspro.frontend.tokens import tabla
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
    def __init__(self, ruta: Path, hilos: int | None = None) -> None:
        import onnxruntime as ort

        opciones = ort.SessionOptions()
        if hilos:
            opciones.intra_op_num_threads = hilos
            opciones.inter_op_num_threads = 1
        self.sesion = ort.InferenceSession(str(ruta), opciones, providers=["CPUExecutionProvider"])
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


# ----------------------------------------------------------------------------- worker

_encoder: EncoderLocutor | None = None


def _iniciar_worker(ruta_encoder: str) -> None:
    global _encoder
    torch.set_num_threads(1)
    _encoder = EncoderLocutor(Path(ruta_encoder), hilos=1)


def _procesar_audio(tarea: tuple) -> dict:
    """(n, wav, min_s, max_s, wav_cache) -> {"ok": True, ...} or {"ok": False, "motivo": ...}."""
    n, wav, min_s, max_s, wav_cache = tarea
    wav = Path(wav)
    if not wav.exists():
        return {"n": n, "ok": False, "motivo": "wav no existe"}
    try:
        onda, sr = cargar_onda(wav)
    except Exception as e:  # a corrupt file is data, not a crash of the whole run
        return {"n": n, "ok": False, "motivo": f"no se pudo leer ({type(e).__name__})"}
    segundos = len(onda) / sr
    if not (min_s <= segundos <= max_s):
        return {"n": n, "ok": False, "motivo": "fuera de la ventana de duración"}
    assert _encoder is not None
    emb = _encoder(remuestrear(onda, sr, SR_ENCODER))
    ruta_wav, sr_final = wav, sr
    if wav_cache:
        ruta_wav = Path(wav_cache) / f"{n:07d}.wav"
        if not ruta_wav.exists():
            sf.write(str(ruta_wav), remuestrear(onda, sr, SR_MODELO), SR_MODELO)
        sr_final = SR_MODELO
    return {
        "n": n,
        "ok": True,
        "wav": str(ruta_wav),
        "sr": sr_final,
        "segundos": round(segundos, 3),
        "embedding": emb.astype(np.float32),
    }


# ----------------------------------------------------------------------------- main


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
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    t0 = time.time()
    frases = leer(args.manifiesto)
    print(f"{len(frases)} frases en {args.manifiesto}", flush=True)
    fonemas = fonemas_por_lotes(frases)
    print(f"fonemas listos en {time.time() - t0:.0f} s", flush=True)

    descartes: Counter[str] = Counter()
    simbolos_fuera: Counter[str] = Counter()
    tokens_por_n: dict[int, list[int]] = {}
    tareas = []
    for n, (frase, fon) in enumerate(zip(frases, fonemas, strict=True)):
        if not fon:
            descartes["sin fonemas"] += 1
            continue
        try:
            tokens_por_n[n] = ids(fon)
        except ValueError:
            for c in fon:
                if c not in tabla():
                    simbolos_fuera[c] += 1
            descartes["símbolo fuera de la tabla"] += 1
            continue
        tareas.append(
            (
                n,
                str(frase.wav),
                args.min_segundos,
                args.max_segundos,
                str(args.wav_cache) if args.wav_cache else "",
            )
        )
    if args.wav_cache:
        args.wav_cache.mkdir(parents=True, exist_ok=True)

    resultados: dict[int, dict] = {}
    with mp.Pool(args.workers, initializer=_iniciar_worker, initargs=(str(args.encoder),)) as pool:
        for i, r in enumerate(pool.imap_unordered(_procesar_audio, tareas, chunksize=8)):
            if r["ok"]:
                resultados[r["n"]] = r
            else:
                descartes[r["motivo"]] += 1
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(tareas)}  ({time.time() - t0:.0f} s)", flush=True)

    indice = []
    for n, frase in enumerate(frases):
        r = resultados.get(n)
        if r is None:
            continue
        indice.append(
            {
                "wav": r["wav"],
                "sr": r["sr"],
                "segundos": r["segundos"],
                "idioma": frase.idioma,
                "locutor": frase.locutor,
                "texto": frase.texto,
                "fonemas": fonemas[n],
                "tokens": torch.tensor(tokens_por_n[n], dtype=torch.int64),
                "embedding": torch.from_numpy(r["embedding"]),
            }
        )

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
        "workers": args.workers,
        "segundos_de_preparacion": round(time.time() - t0),
    }
    (args.salida / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
