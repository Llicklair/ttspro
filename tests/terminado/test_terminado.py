"""The MVP done criterion from SCOPE.md, as a command: ``uv run pytest tests/terminado -q``.

Red on purpose until there is a model. Each test is one numbered point of the
criterion; a test that cannot run yet FAILS with the reason instead of skipping,
because a skipped criterion reads as green in a summary and that is the lie
this file exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
MODELOS = RAIZ / "models"
CONTRATO = json.loads((MODELOS / "contrato.json").read_text(encoding="utf-8"))


def _grafo(nombre: str) -> Path:
    ruta = MODELOS / CONTRATO["grafos"][nombre]["fichero"]
    if not ruta.exists():
        pytest.fail(f"criterio 1: no existe {ruta.relative_to(RAIZ)} (aún no hay modelo exportado)")
    return ruta


@pytest.mark.parametrize("nombre", ["speaker_encoder", "tts"])
def test_1_existe_carga_y_coincide_con_el_contrato(nombre: str) -> None:
    ruta = _grafo(nombre)
    ort = pytest.importorskip("onnxruntime", reason="criterio 1: instala el extra `export`")
    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    declarado = CONTRATO["grafos"][nombre]
    assert [e.name for e in sesion.get_inputs()] == [e["nombre"] for e in declarado["entradas"]]
    assert [s.name for s in sesion.get_outputs()] == [s["nombre"] for s in declarado["salidas"]]


def test_2_paridad_pytorch_ort_logmel_menor_que_0_1() -> None:
    _grafo("tts")
    pytest.fail("criterio 2: pendiente — comparar torch vs ORT sobre 20 frases (10 es, 10 en)")


def test_3_paridad_frontend_python_js_con_fonemas() -> None:
    # Es exactamente la suite de paridad por etapas (normalizar, trocear, fonemas,
    # ids) que corre en la suite rápida; aquí se invoca como proceso para que el
    # criterio no dependa de que alguien recuerde correr la otra suite.
    import subprocess
    import sys

    proceso = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_frontend_paridad.py", "-q"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proceso.returncode == 0, f"criterio 3: paridad rota:\n{proceso.stdout[-3000:]}"


def test_4_wer_menor_que_10_y_secs_mayor_que_0_70() -> None:
    _grafo("tts")
    pytest.fail("criterio 4: pendiente — WER (Whisper-small) y SECS sobre eval/")


def test_5_navegador_wasm_10_palabras_en_menos_de_3s_y_80mb() -> None:
    _grafo("tts")
    pytest.fail("criterio 5: pendiente — Playwright + Chromium headless, EP wasm")
