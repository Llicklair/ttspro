"""Deterministic text normalization. Mirror of ``web/src/frontend/normalizar.ts``.

Keep the two files rule-for-rule identical, in the same order. The parity test
runs both over ``tests/fixtures/frontend/frases.txt``.
"""

from __future__ import annotations

import re
import unicodedata

_ESPACIOS = re.compile(r"\s+")
# Typographic quotes and dashes that espeak-ng does not read consistently.
_TIPOGRAFIA = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)


def normalizar(texto: str) -> str:
    """Rule order matters and must match the TypeScript mirror exactly.

    1. Unicode NFC, so composed and decomposed accents tokenize the same.
    2. Typographic punctuation -> ASCII.
    3. Collapse whitespace runs to one space; strip the ends.
    """
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.translate(_TIPOGRAFIA)
    texto = _ESPACIOS.sub(" ", texto).strip()
    return texto
