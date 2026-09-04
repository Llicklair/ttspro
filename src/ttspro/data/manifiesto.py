"""The manifest: one utterance per line, `ruta_wav<TAB>idioma<TAB>locutor<TAB>texto`.

Paths are relative to the manifest's own directory unless absolute. Languages
must be in models/contrato.json. Every corpus a manifest points at has a row
in data/README.md with its license (rule 10) — this module cannot check that,
it can only remind you.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
IDIOMAS = json.loads((RAIZ / "models" / "contrato.json").read_text(encoding="utf-8"))["idiomas"]


@dataclass(frozen=True)
class Frase:
    wav: Path
    idioma: str
    locutor: str
    texto: str


def leer(ruta: Path) -> list[Frase]:
    base = ruta.parent
    frases: list[Frase] = []
    errores: list[str] = []
    for n, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1):
        if not linea.strip() or linea.startswith("#"):
            continue
        partes = linea.split("\t")
        if len(partes) != 4:
            errores.append(
                f"línea {n}: esperaba 4 columnas separadas por tabulador, hay {len(partes)}"
            )
            continue
        wav, idioma, locutor, texto = partes
        if idioma not in IDIOMAS:
            errores.append(f"línea {n}: idioma {idioma!r} no está en el contrato {IDIOMAS}")
            continue
        ruta_wav = Path(wav)
        if not ruta_wav.is_absolute():
            ruta_wav = base / ruta_wav
        frases.append(Frase(ruta_wav, idioma, locutor.strip(), texto.strip()))
    if errores:
        raise ValueError(f"manifiesto {ruta} con errores:\n  " + "\n  ".join(errores[:20]))
    return frases


def escribir(ruta: Path, frases: list[Frase]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    lineas = [f"{f.wav}\t{f.idioma}\t{f.locutor}\t{f.texto}" for f in frases]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
