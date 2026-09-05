"""The manifest: one utterance per line, `ruta_wav<TAB>idioma<TAB>locutor<TAB>texto`.

Paths are relative to the manifest's own directory unless absolute. Languages
must be in models/contrato.json. Every corpus a manifest points at has a row
in data/README.md with its license (rule 10) — this module cannot check that,
it can only remind you.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ttspro.frontend.fonemas import VOCES

RAIZ = Path(__file__).resolve().parents[3]
# What the FRONTEND can phonemize, not what the model in models/ happens to speak:
# a corpus is data, and it stays valid when the base voice changes (ADR 0008).
IDIOMAS = sorted(VOCES)


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
    # TSV: a tab or newline inside the text would add columns (VCTK has one, 2026-09-04)
    tab, nl = chr(9), chr(10)
    limpio = lambda t: " ".join(t.replace(tab, " ").replace(nl, " ").split())  # noqa: E731
    lineas = [tab.join([str(f.wav), f.idioma, f.locutor, limpio(f.texto)]) for f in frases]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
