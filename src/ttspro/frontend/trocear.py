"""Split normalized text into text chunks and punctuation tokens.
Mirror of ``web/src/frontend/trocear.ts`` — keep rule-for-rule identical.

Why chunk at all: espeak-ng drops punctuation, and the model needs it (pauses,
intonation). So the text is split on punctuation, each text chunk is
phonemized on its own, and the marks survive as tokens between chunks.

What does NOT split: ``.`` ``,`` ``:`` between digits ("3,5", "3.5", "10:30")
— espeak reads them as decimal/time and that reading must survive —
and apostrophes inside words ("Don't"). A dash splits only when spaced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ASCII [0-9] on purpose: JS \d is ASCII-only, Python \d is not.
# `.` `,` `:` split unless BOTH neighbours are digits (either alternative below).
_SEPARADOR = re.compile(r'(?<![0-9])[.,:]|[.,:](?![0-9])|[;!?¡¿…"()]| - ')


@dataclass(frozen=True)
class Trozo:
    tipo: str  # "texto" | "puntuacion"
    valor: str


def trocear(texto: str) -> list[Trozo]:
    trozos: list[Trozo] = []
    pos = 0
    for m in _SEPARADOR.finditer(texto):
        _texto(trozos, texto[pos : m.start()])
        trozos.append(Trozo("puntuacion", m.group(0).strip()))
        pos = m.end()
    _texto(trozos, texto[pos:])
    return trozos


def _texto(trozos: list[Trozo], fragmento: str) -> None:
    fragmento = fragmento.strip()
    if fragmento:
        trozos.append(Trozo("texto", fragmento))
