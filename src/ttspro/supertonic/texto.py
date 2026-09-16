"""Text -> token ids for the Supertonic engine, with no phonemizer at all.
Mirror of ``web/src/frontend/unicode.ts`` — keep rule-for-rule identical (rule 3).

Supertonic reads **characters**, not phonemes: every Unicode code point is mapped
through ``onnx/unicode_indexer.json`` (a flat table indexed by code point) and the
sentence is wrapped in a language tag, ``<es>…</es>``. That is the whole frontend.
espeak-ng, and the GPL obligation it dragged into every deployed build, is gone
(ADR 0012 supersedes ADR 0004).

One deliberate divergence from the upstream Python reference: this walks **UTF-16
code units**, not Python characters, because the browser's ``codePointAt`` does.
For a character outside the BMP the browser emits two ids — the full code point at
the lead surrogate and the bare trail surrogate after it — and the parity test
(rule 3) would fail against anything else. Upstream's Python and JS disagree here;
this side follows the runtime that actually ships.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

IDIOMAS = (
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
)

# Code point that is not in the table. ONNX Gather reads -1 as the last row, which
# is where the checkpoint keeps its unknown slot; do not "fix" this to 0.
DESCONOCIDO = -1

_EMOJI = re.compile(
    "[\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)

_SUSTITUCIONES = {
    "\u2013": "-",
    "\u2011": "-",
    "\u2014": "-",
    "_": " ",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u00b4": "'",
    "`": "'",
    "[": " ",
    "]": " ",
    "|": " ",
    "/": " ",
    "#": " ",
    "\u2192": " ",
    "\u2190": " ",
}
_SIMBOLOS = re.compile("[" + re.escape("\u2665\u2606\u2661\u00a9" + chr(92)) + "]")
_EXPRESIONES = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}
_ESPACIO_ANTES = re.compile(r" ([,.!?;:'])")
_FIN = re.compile(
    "[" + re.escape(".!?;:,'\"')]}\u2026\u3002\u300d\u300f\u3011\u3009\u300b\u203a\u00bb") + "]$"
)


def preprocesar(texto: str, idioma: str) -> str:
    """Normalize a sentence and wrap it in its language tag.

    Upstream calls this a placeholder ("Need advanced normalizer"). It stays
    rule-for-rule faithful anyway: the checkpoint was trained on text that went
    through exactly these steps, so improving them here would move the input off
    the distribution the weights know.
    """
    texto = unicodedata.normalize("NFKD", texto)
    texto = _EMOJI.sub("", texto)
    for k, v in _SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    texto = _SIMBOLOS.sub("", texto)
    for k, v in _EXPRESIONES.items():
        texto = texto.replace(k, v)
    texto = _ESPACIO_ANTES.sub(r"\1", texto)
    while '""' in texto:
        texto = texto.replace('""', '"')
    while "''" in texto:
        texto = texto.replace("''", "'")
    while "``" in texto:
        texto = texto.replace("``", "`")
    texto = re.sub(r"\s+", " ", texto).strip()
    if not _FIN.search(texto):
        texto += "."
    if idioma not in IDIOMAS:
        raise ValueError(
            f"idioma no soportado: {idioma!r}; hay {len(IDIOMAS)}: {', '.join(IDIOMAS)}"
        )
    return f"<{idioma}>{texto}</{idioma}>"


def puntos_de_codigo(texto: str) -> list[int]:
    """The sequence JS's ``codePointAt`` yields over every UTF-16 code unit."""
    unidades = np.frombuffer(texto.encode("utf-16-le"), dtype="<u2")
    salida: list[int] = []
    for i, u in enumerate(unidades):
        u = int(u)
        if 0xD800 <= u <= 0xDBFF and i + 1 < len(unidades):
            baja = int(unidades[i + 1])
            if 0xDC00 <= baja <= 0xDFFF:
                salida.append(0x10000 + ((u - 0xD800) << 10) + (baja - 0xDC00))
                continue
        salida.append(u)
    return salida


class Indexador:
    """``unicode_indexer.json``: a flat table from code point to token id."""

    def __init__(self, tabla: list[int]) -> None:
        self.tabla = tabla

    @classmethod
    def desde(cls, ruta: Path | str) -> Indexador:
        return cls(json.loads(Path(ruta).read_text(encoding="utf-8")))

    def ids(self, texto: str) -> list[int]:
        return [
            self.tabla[c] if c < len(self.tabla) else DESCONOCIDO for c in puntos_de_codigo(texto)
        ]

    def __call__(self, textos: list[str], idiomas: list[str]) -> tuple[np.ndarray, np.ndarray]:
        filas = [self.ids(preprocesar(t, i)) for t, i in zip(textos, idiomas, strict=True)]
        largos = np.array([len(f) for f in filas], dtype=np.int64)
        ids = np.zeros((len(filas), int(largos.max())), dtype=np.int64)
        for i, fila in enumerate(filas):
            ids[i, : len(fila)] = fila
        return ids, mascara(largos)


def mascara(largos: np.ndarray, maximo: int | None = None) -> np.ndarray:
    """``[B]`` lengths -> ``[B, 1, max]`` float mask, the shape both graphs want."""
    maximo = maximo or int(largos.max())
    m = np.arange(maximo) < np.expand_dims(largos, axis=1)
    return m.astype(np.float32).reshape(-1, 1, maximo)


_ABREVIATURAS = (
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)"
    r"(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)"
    r"(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+"
)
_FRASE = re.compile(_ABREVIATURAS)


def trocear(texto: str, maximo: int = 300) -> list[str]:
    """Split a long text into chunks the model synthesizes one at a time.

    Not the same job as ``ttspro.frontend.trocear``, which splits a sentence on
    punctuation for the phoneme frontend. This one splits a *document* into
    sentence groups under ``maximo`` characters, because attention over the whole
    thing degrades past that (upstream: 300, or 120 for ko and ja).
    """
    trozos: list[str] = []
    for parrafo in (p.strip() for p in re.split(r"\n\s*\n+", texto.strip())):
        if not parrafo:
            continue
        actual = ""
        for frase in _FRASE.split(parrafo):
            if len(actual) + len(frase) + 1 <= maximo:
                actual += (" " if actual else "") + frase
            else:
                if actual:
                    trozos.append(actual.strip())
                actual = frase
        if actual:
            trozos.append(actual.strip())
    return trozos
