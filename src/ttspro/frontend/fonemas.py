"""IPA phonemes from espeak-ng, through its command-line executable (ADR 0004).

The executable is integrated by reference (rule 11): found at runtime, never
bundled. If it is missing this module raises — it does not fall back to
characters, because a phoneme-trained model fed letters produces audio that
"almost" works, which is the worst kind of failure.

Why the CLI and not the C API (measured 2026-09-04, docs/evidencia.md):
``espeak_TextToPhonemes`` skips the intonation pass that synthesis runs, and
that pass rewrites stress marks on function words ("and" -> "ˈænd" through the
CLI, "ænd" through the API; 3 of 24 fixture sentences differed). The WASM in
``web/`` is a CLI build with no API exported, so the CLI is the only path both
sides can share.

Protocol, identical in ``web/src/frontend/fonemas.ts``: every text chunk goes on
its own line with " ." appended, so espeak closes exactly one clause per line
and output lines align with input chunks. Many sentences can share one process
(``fonemizar_lotes``) — the cost is the process, not the text.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

VERSION_ESPERADA = "1.52"  # same major.minor as the WASM pinned in web/package.json

# Language id used by the model -> espeak-ng voice. Same table in fonemas.ts.
VOCES = {"es": "es", "en": "en-us"}


class EspeakNoDisponible(RuntimeError):
    pass


def _candidatos() -> list[Path]:
    out: list[Path] = []
    if env := os.environ.get("ESPEAK_NG_EXE"):
        out.append(Path(env))
    if exe := shutil.which("espeak-ng"):
        out.append(Path(exe))
    if sys.platform == "win32":
        for base in (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "eSpeak NG",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "eSpeak NG",
        ):
            out.append(base / "espeak-ng.exe")
    return out


@lru_cache(maxsize=1)
def ejecutable() -> Path:
    for ruta in _candidatos():
        if ruta.is_file():
            return ruta
    raise EspeakNoDisponible(
        "espeak-ng no está instalado o no se encuentra (ADR 0004). Instálalo por su canal "
        "oficial (Windows: `winget install eSpeak-NG.eSpeak-NG`; Debian: `apt install espeak-ng`) "
        "o apunta ESPEAK_NG_EXE al ejecutable. Rutas probadas: "
        + ", ".join(str(r) for r in _candidatos())
    )


@lru_cache(maxsize=1)
def version() -> str:
    salida = subprocess.run(
        [str(ejecutable()), "--version"], capture_output=True, text=True, encoding="utf-8"
    ).stdout
    # "eSpeak NG text-to-speech: 1.52.0  Data at: ..."
    return salida.split(":", 1)[1].split()[0] if ":" in salida else salida.strip()


def _ejecutar(flujo: str, voz: str) -> list[str]:
    """One espeak process over `flujo`; returns its non-empty output lines.

    Text goes through STDIN, never through `-f`: espeak-ng 1.52.0 on Windows
    appends a garbage clause when reading a file (14 of 20 runs; 0 of 20 via
    stdin or argv — docs/evidencia.md 2026-09-04). The WASM side passes the same
    text as an argument; both are memory buffers on the same path inside espeak.
    """
    with tempfile.TemporaryDirectory() as tmp:
        salida = Path(tmp) / "salida.txt"
        proceso = subprocess.run(
            [str(ejecutable()), "-q", "--ipa", "-v", voz, "--stdin", "--phonout", str(salida)],
            input=flujo,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if proceso.returncode != 0:
            raise EspeakNoDisponible(
                f"espeak-ng falló (voz {voz!r}, código {proceso.returncode}): {proceso.stderr}"
            )
        return [ln.strip() for ln in salida.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _flujo(trozos: list[str]) -> str:
    return "".join(f"{t} ." + chr(10) for t in trozos)


def _fonemizar_trozos_alineados(trozos: list[str], voz: str) -> list[str]:
    """One output line per chunk. When espeak breaks the protocol (a chunk that
    yields no line, or two consecutive chunks that merge), bisect until the
    culprit is alone and take its real output — empty or joined — instead of
    failing the whole batch. Measured 2026-09-04 on OpenSLR es: 1 line short in
    a batch of 224, and no chunk of that batch misbehaves on its own."""
    if not trozos:
        return []
    lineas = _ejecutar(_flujo(trozos), voz)
    if len(lineas) == len(trozos):
        return lineas
    if len(trozos) == 1:
        return [" ".join(lineas)]
    mitad = len(trozos) // 2
    return _fonemizar_trozos_alineados(trozos[:mitad], voz) + _fonemizar_trozos_alineados(
        trozos[mitad:], voz
    )


def fonemizar_lotes(lotes: list[list[str]], idioma: str) -> list[list[str]]:
    """IPA for the text chunks of many sentences in ONE espeak process.

    ``lotes[i]`` are the chunks of sentence *i*; the result has the same shape.
    """
    trozos = [t for lote in lotes for t in lote]
    if not trozos:
        return [[] for _ in lotes]
    lineas = _fonemizar_trozos_alineados(trozos, VOCES.get(idioma, idioma))
    resultado: list[list[str]] = []
    pos = 0
    for lote in lotes:
        resultado.append(lineas[pos : pos + len(lote)])
        pos += len(lote)
    return resultado


def fonemizar_trozos(trozos: list[str], idioma: str) -> list[str]:
    """IPA for each text chunk of one sentence."""
    return fonemizar_lotes([trozos], idioma)[0]


def fonemizar(texto: str, idioma: str) -> str:
    """IPA for a single chunk. For whole sentences use ``tokens.fonemas_de_frase``."""
    return fonemizar_trozos([texto], idioma)[0]
