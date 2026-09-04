"""Download public corpora and write their manifests (the coqui-style recipes).

    uv run python -m ttspro.data.descargar openslr_es      # es, ~170 speakers, 38 h, CC BY-SA 4.0
    uv run python -m ttspro.data.descargar libritts_r      # en, 247 speakers, 54 h, CC BY 4.0
    uv run python -m ttspro.data.descargar ljspeech        # en, 1 speaker, 24 h, public domain
    uv run python -m ttspro.data.descargar vctk            # en, 110 speakers, 44 h, CC BY 4.0

Files land in `data/raw/<corpus>/`, manifests in `data/manifests/<corpus>.tsv`.
Downloads resume (HTTP Range) and are verified by size, never re-downloaded
when complete. Every corpus here has a row in data/README.md (rule 10).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

from ttspro.data.manifiesto import Frase, escribir

RAIZ = Path(__file__).resolve().parents[3]
RAW = RAIZ / "data" / "raw"
MANIFESTS = RAIZ / "data" / "manifests"

OPENSLR_ES = {
    "61": ["es_ar_female", "es_ar_male"],
    "71": ["es_cl_female", "es_cl_male"],
    "72": ["es_co_female", "es_co_male"],
    "73": ["es_pe_female", "es_pe_male"],
    "74": ["es_pr_female"],
    "75": ["es_ve_female", "es_ve_male"],
}
LIBRITTS_R = "https://www.openslr.org/resources/141/train_clean_100.tar.gz"
LJSPEECH = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
# The bitstream URL serves an HTML page (4 KB); the "download" endpoint is the real 11.7 GB zip.
VCTK = "https://datashare.ed.ac.uk/download/DS_10283_3443.zip"


# ----------------------------------------------------------------------------- download / extract


def descargar(url: str, destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    cabecera = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(cabecera, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
    if destino.exists() and total and destino.stat().st_size == total:
        print(f"  ya está: {destino.name} ({total / 1e9:.2f} GB)")
        return destino
    parcial = destino.with_suffix(destino.suffix + ".part")
    desde = parcial.stat().st_size if parcial.exists() else 0
    peticion = urllib.request.Request(url, headers={"Range": f"bytes={desde}-"} if desde else {})
    t0 = time.time()
    with (
        urllib.request.urlopen(peticion, timeout=60) as r,
        parcial.open("ab" if desde else "wb") as f,
    ):
        if desde and r.status != 206:
            f.seek(0)
            f.truncate()
            desde = 0
        bajado = desde
        ultimo = time.time()
        while True:
            trozo = r.read(1 << 20)
            if not trozo:
                break
            f.write(trozo)
            bajado += len(trozo)
            if time.time() - ultimo > 15:
                velocidad = (bajado - desde) / max(time.time() - t0, 1) / 1e6
                progreso = f"{bajado / 1e9:.2f}/{total / 1e9:.2f} GB"
                print(f"  {destino.name}: {progreso}  {velocidad:.1f} MB/s", flush=True)
                ultimo = time.time()
    if total and parcial.stat().st_size != total:
        raise RuntimeError(f"{destino.name}: {parcial.stat().st_size} bytes, esperaba {total}")
    parcial.rename(destino)
    print(f"  descargado: {destino.name} ({total / 1e9:.2f} GB en {time.time() - t0:.0f} s)")
    return destino


def extraer(fichero: Path, carpeta: Path, marca: str) -> None:
    hecho = carpeta / f".extraido_{marca}"
    if hecho.exists():
        return
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"  extrayendo {fichero.name} …", flush=True)
    if fichero.suffix == ".zip":
        with zipfile.ZipFile(fichero) as z:
            z.extractall(carpeta)
    else:
        with tarfile.open(fichero) as t:
            t.extractall(carpeta, filter="data")
    hecho.touch()


# ----------------------------------------------------------------------------- recipes


def receta_openslr_es() -> list[Frase]:
    carpeta = RAW / "openslr_es"
    frases: list[Frase] = []
    for recurso, nombres in OPENSLR_ES.items():
        for nombre in nombres:
            zip_ = descargar(
                f"https://www.openslr.org/resources/{recurso}/{nombre}.zip",
                carpeta / f"{nombre}.zip",
            )
            extraer(zip_, carpeta / nombre, nombre)
            indice = next((carpeta / nombre).rglob("line_index.tsv"))
            base = indice.parent
            with indice.open(encoding="utf-8") as f:
                for fila in csv.reader(f, delimiter="\t"):
                    if len(fila) < 2:
                        continue
                    ident, texto = fila[0].strip(), fila[-1].strip()
                    wav = base / f"{ident}.wav"
                    if wav.exists():
                        locutor = "_".join(ident.split("_")[:2])  # e.g. arf_00295
                        frases.append(Frase(wav, "es", locutor, texto))
    return frases


def receta_libritts_r() -> list[Frase]:
    carpeta = RAW / "libritts_r"
    tar = descargar(LIBRITTS_R, carpeta / "train_clean_100.tar.gz")
    extraer(tar, carpeta, "train_clean_100")
    frases: list[Frase] = []
    for txt in sorted(carpeta.rglob("*.normalized.txt")):
        wav = txt.with_name(txt.name.replace(".normalized.txt", ".wav"))
        if wav.exists():
            locutor = wav.name.split("_")[0]
            frases.append(Frase(wav, "en", locutor, txt.read_text(encoding="utf-8").strip()))
    return frases


def receta_ljspeech() -> list[Frase]:
    carpeta = RAW / "ljspeech"
    tar = descargar(LJSPEECH, carpeta / "LJSpeech-1.1.tar.bz2")
    extraer(tar, carpeta, "ljspeech")
    meta = next(carpeta.rglob("metadata.csv"))
    frases = []
    with meta.open(encoding="utf-8") as f:
        for fila in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            wav = meta.parent / "wavs" / f"{fila[0]}.wav"
            if wav.exists():
                frases.append(Frase(wav, "en", "ljspeech", fila[2].strip() or fila[1].strip()))
    return frases


def receta_vctk() -> list[Frase]:
    carpeta = RAW / "vctk"
    # datashare serves a WRAPPER zip (README.txt + VCTK-Corpus-0.92.zip). Naming the
    # wrapper after the inner file made extraction overwrite the archive being read
    # (2026-09-04: EOFError, then a 0-byte file, 11.7 GB re-downloaded). Keep the names.
    envoltorio = descargar(VCTK, carpeta / "DS_10283_3443.zip")
    extraer(envoltorio, carpeta, "envoltorio")
    interior = next(carpeta.rglob("VCTK-Corpus-0.92.zip"))
    extraer(interior, carpeta / "VCTK-Corpus-0.92", "vctk")
    frases = []
    for txt in sorted(carpeta.rglob("txt/*/*.txt")):
        locutor = txt.parent.name
        corpus = txt.parents[2]  # .../VCTK-Corpus-0.92/txt/p225/p225_001.txt
        flac = corpus / "wav48_silence_trimmed" / locutor / f"{txt.stem}_mic1.flac"
        if flac.exists():
            frases.append(Frase(flac, "en", locutor, txt.read_text(encoding="utf-8").strip()))
    return frases


RECETAS = {
    "openslr_es": receta_openslr_es,
    "libritts_r": receta_libritts_r,
    "ljspeech": receta_ljspeech,
    "vctk": receta_vctk,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=sorted(RECETAS), nargs="+")
    args = ap.parse_args()
    for nombre in args.corpus:
        print(f"== {nombre}")
        frases = RECETAS[nombre]()
        salida = MANIFESTS / f"{nombre}.tsv"
        escribir(salida, frases)
        locutores = len({f.locutor for f in frases})
        print(
            f"  manifiesto {salida.relative_to(RAIZ)}: {len(frases)} frases, {locutores} locutores"
        )
    if shutil.which("espeak-ng") is None and sys.platform != "win32":
        print("aviso: espeak-ng no está en PATH; ttspro.data.preparar lo necesita")


if __name__ == "__main__":
    main()
