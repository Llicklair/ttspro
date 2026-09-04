"""Rule 3: the Python frontend and the TypeScript frontend agree, sentence by sentence.

The TypeScript side runs as a process (rule 6 — no imports across the line):
``node web/src/frontend/cli.ts`` reads sentences on stdin and writes one JSON
line per sentence. Node >= 22.6 strips types natively, so there is no build step.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ttspro.frontend import normalizar

RAIZ = Path(__file__).resolve().parents[1]
FRASES = RAIZ / "tests" / "fixtures" / "frontend" / "frases.txt"
CLI_TS = RAIZ / "web" / "src" / "frontend" / "cli.ts"


def _frases() -> list[str]:
    return [ln for ln in FRASES.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _frontend_js(frases: list[str]) -> list[dict]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("node no está en PATH: la paridad Python/JS no se puede comprobar")
    proceso = subprocess.run(
        [node, "--no-warnings", str(CLI_TS)],
        input="\n".join(frases) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return [json.loads(ln) for ln in proceso.stdout.splitlines() if ln.strip()]


def test_fixtures_no_vacios() -> None:
    assert len(_frases()) >= 20, "hacen falta frases de es y en para que la paridad diga algo"


def test_normalizar_coincide_en_python_y_js() -> None:
    frases = _frases()
    js = _frontend_js(frases)
    assert len(js) == len(frases)
    for frase, salida in zip(frases, js, strict=True):
        assert salida["normalizado"] == normalizar(frase), frase
