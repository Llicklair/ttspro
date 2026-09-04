"""The floor stays filled and the contract stays parseable.

These are the cheapest facts in the repo: a decision document that still has a
``gb:pendiente`` mark passes ``gb floor`` without saying anything, and a
``models/contrato.json`` that does not parse silently breaks both the export
test and the web runtime.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]

DOCUMENTOS_DEL_SUELO = [
    "AGENTS.md",
    "SCOPE.md",
    "ARCHITECTURE.md",
    "docs/evidencia.md",
    ".githooks/pre-commit",
]


@pytest.mark.parametrize("relativo", DOCUMENTOS_DEL_SUELO)
def test_documento_del_suelo_sin_marcas_pendientes(relativo: str) -> None:
    texto = (RAIZ / relativo).read_text(encoding="utf-8")
    assert "gb:pendiente" not in texto, f"{relativo} conserva marcas sin rellenar"


def test_scope_tiene_criterio_de_terminado_ejecutable() -> None:
    texto = (RAIZ / "SCOPE.md").read_text(encoding="utf-8")
    valla = re.search(r"^```gb:terminado\s*\n(.*?)^```", texto, re.MULTILINE | re.DOTALL)
    assert valla, "SCOPE.md no tiene la valla ```gb:terminado"
    comandos = [ln for ln in valla.group(1).splitlines() if ln.strip() and not ln.startswith("#")]
    assert comandos, "la valla gb:terminado no contiene ningún comando"


def test_contrato_parsea_y_declara_los_dos_grafos() -> None:
    contrato = json.loads((RAIZ / "models" / "contrato.json").read_text(encoding="utf-8"))
    assert set(contrato["grafos"]) == {"speaker_encoder", "tts"}
    for nombre, grafo in contrato["grafos"].items():
        assert grafo["fichero"].endswith(".onnx"), nombre
        assert grafo["opset"] >= 17, nombre
        assert grafo["entradas"] and grafo["salidas"], nombre
        for tensor in [*grafo["entradas"], *grafo["salidas"]]:
            assert {"nombre", "dtype", "forma"} <= set(tensor), tensor


def test_fronteras_declaradas_parsean() -> None:
    lineas = (RAIZ / ".gb-boundaries").read_text(encoding="utf-8").splitlines()
    reglas = [ln for ln in lineas if "-/->" in ln and not ln.lstrip().startswith("#")]
    assert reglas, ".gb-boundaries no declara ninguna frontera"
    for regla in reglas:
        izquierda, derecha = (lado.strip() for lado in regla.split("-/->"))
        assert izquierda and derecha, regla
