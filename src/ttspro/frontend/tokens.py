"""Phoneme string <-> token ids, from the symbol table in ``models/contrato.json``.
Mirror of ``web/src/frontend/tokens.ts``.

The token sequence for a sentence is: the elements of ``trocear`` in order
(phonemized text chunks and punctuation marks), joined by a single space, then
one id per character. A character outside the table raises — never a silent
<unk>, because the model would then be fed something it never saw.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ttspro.frontend.fonemas import fonemizar_trozos
from ttspro.frontend.normalizar import normalizar
from ttspro.frontend.trocear import trocear

RAIZ = Path(__file__).resolve().parents[3]
CONTRATO = RAIZ / "models" / "contrato.json"


@lru_cache(maxsize=1)
def simbolos() -> dict:
    return json.loads(CONTRATO.read_text(encoding="utf-8"))["simbolos"]


@lru_cache(maxsize=1)
def tabla() -> dict[str, int]:
    """symbol -> id. Empty slots (Piper leaves gaps in its map) are skipped."""
    return {s: i for i, s in enumerate(simbolos()["tabla"]) if s != ""}


def fonemas_de_frase(texto: str, idioma: str) -> str:
    """Same assembly as ``fonemasDeFrase`` in tokens.ts: chunks phonemized as one
    stream, punctuation kept, elements joined by a single space."""
    trozos = trocear(normalizar(texto))
    fonemas = iter(fonemizar_trozos([t.valor for t in trozos if t.tipo == "texto"], idioma))
    elementos = []
    for trozo in trozos:
        if trozo.tipo == "texto":
            fon = next(fonemas)
            if fon:
                elementos.append(fon)
        else:
            elementos.append(trozo.valor)
    return " ".join(elementos)


def ids(fonemas: str) -> list[int]:
    """One id per character, wrapped as the model expects (mirrored in tokens.ts).

    The contract decides the shape of the sequence, because it is part of what the
    weights were trained on and getting it wrong makes audio that is almost right:

    - `blank_entre_tokens` (VITS add_blank): a pad next to every symbol;
    - `blank_al_inicio`: pad BEFORE each symbol (coqui) or AFTER it (Piper);
    - `bos`/`eos`: wrap the sentence (Piper's `^` and `$`).
    """
    t = tabla()
    cfg = simbolos()
    equivalencias = cfg.get("equivalencias") or {}
    if equivalencias:
        fonemas = "".join(equivalencias.get(c, c) for c in fonemas)
    desconocidos = sorted({c for c in fonemas if c not in t})
    if desconocidos:
        raise ValueError(
            f"símbolos fuera de models/contrato.json: {desconocidos!r} en {fonemas!r}. "
            "Añádelos a la tabla en los dos lados, no los ignores."
        )
    secuencia = [t[c] for c in fonemas]
    if not cfg.get("blank_entre_tokens"):
        return (
            ([cfg["bos"]] if cfg.get("bos") is not None else [])
            + secuencia
            + ([cfg["eos"]] if cfg.get("eos") is not None else [])
        )
    pad = cfg["pad"]
    al_inicio = cfg.get("blank_al_inicio", True)
    salida = [cfg["bos"]] if cfg.get("bos") is not None else []
    for x in secuencia:
        salida += [pad, x] if al_inicio else [x, pad]
    if al_inicio:
        salida.append(pad)
    if cfg.get("eos") is not None:
        salida.append(cfg["eos"])
    return salida


def tokenizar(texto: str, idioma: str) -> tuple[str, list[int]]:
    fonemas = fonemas_de_frase(texto, idioma)
    return fonemas, ids(fonemas)
