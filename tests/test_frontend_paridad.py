"""Rule 3: the Python frontend and the TypeScript frontend agree, stage by stage.

The TypeScript side runs as a process (rule 6 — no imports across the line):
``node web/src/frontend/cli.ts`` reads ``idioma<TAB>frase`` lines on stdin and
writes one JSON line per sentence with every stage (normalized text, chunks,
phonemes, ids), so a failure names the stage that diverged. Node >= 22.6 strips
types natively: no build step.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from ttspro.frontend import fonemas_de_frase, ids, normalizar, trocear

RAIZ = Path(__file__).resolve().parents[1]
FRASES = RAIZ / "tests" / "fixtures" / "frontend" / "frases.txt"
CLI_TS = RAIZ / "web" / "src" / "frontend" / "cli.ts"


def _frases() -> list[tuple[str, str]]:
    out = []
    for ln in FRASES.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            idioma, frase = ln.split("\t", 1)
            out.append((idioma, frase))
    return out


def _frontend_js(frases: list[tuple[str, str]]) -> list[dict]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("node no está en PATH: la paridad Python/JS no se puede comprobar")
    if not (RAIZ / "web" / "node_modules" / "espeak-ng").exists():
        pytest.fail("web/node_modules sin espeak-ng: corre `npm ci` en web/")
    proceso = subprocess.run(
        [node, "--no-warnings", str(CLI_TS)],
        input="".join(f"{i}\t{f}\n" for i, f in frases),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proceso.returncode != 0:
        pytest.fail(f"cli.ts falló:\n{proceso.stderr[-2000:]}")
    return [json.loads(ln) for ln in proceso.stdout.splitlines() if ln.strip()]


@pytest.fixture(scope="module")
def js() -> dict[tuple[str, str], dict]:
    frases = _frases()
    salidas = _frontend_js(frases)
    assert len(salidas) == len(frases)
    return {(s["idioma"], s["frase"]): s for s in salidas}


def test_fixtures_cubren_los_dos_idiomas() -> None:
    idiomas = {i for i, _ in _frases()}
    assert idiomas == {"es", "en"}
    assert len(_frases()) >= 20


@pytest.mark.parametrize(("idioma", "frase"), _frases())
def test_paridad_por_etapas(js, idioma: str, frase: str) -> None:
    salida = js[(idioma, frase)]
    normalizado = normalizar(frase)
    assert salida["normalizado"] == normalizado, "etapa: normalizar"
    assert salida["trozos"] == [asdict(t) for t in trocear(normalizado)], "etapa: trocear"
    fonemas = fonemas_de_frase(frase, idioma)
    assert salida["fonemas"] == fonemas, "etapa: fonemas"
    assert salida["ids"] == ids(fonemas), "etapa: ids"
