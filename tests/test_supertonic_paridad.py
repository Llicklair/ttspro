"""Rule 3 for the Supertonic frontend: Python and TypeScript agree, stage by stage.

Same shape as ``test_frontend_paridad.py`` and the same reason — a frontend that
exists twice will drift unless a test refuses to let it — but this one needs no
espeak-ng, because ADR 0012 removed the phonemizer. ``node web/src/frontend/
cli-unicode.ts`` reads ``idioma<TAB>frase`` and writes a JSON line per sentence
with the preprocessed text, the code points and the ids.

Skipped, not failed, when ``models/supertonic/`` is not downloaded: the graphs are
398 MB and never travel in git (``uv run python -m ttspro.supertonic.descargar``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ttspro.supertonic.texto import Indexador, preprocesar, puntos_de_codigo, trocear

RAIZ = Path(__file__).resolve().parents[1]
FRASES = RAIZ / "tests" / "fixtures" / "frontend" / "frases.txt"
INDEXADOR = RAIZ / "models" / "supertonic" / "onnx" / "unicode_indexer.json"
CLI_TS = RAIZ / "web" / "src" / "frontend" / "cli-unicode.ts"

# Texto largo: el troceador es el unico sitio donde las dos implementaciones
# tienen bucles distintos, asi que se comprueba con algo que de verdad trocee.
LARGO = (
    "Primera frase del parrafo. Segunda frase, un poco mas larga que la anterior. "
    "El Sr. Garcia llego tarde. Una tercera para pasar de los trescientos caracteres "
    "que es donde el troceador tiene que partir, porque si no parte nunca no hay nada "
    "que comparar y el test seria un adorno. Y una cuarta por si acaso.\n\n"
    "Segundo parrafo, que empieza aparte."
)

requiere_modelos = pytest.mark.skipif(
    not INDEXADOR.exists(),
    reason="falta models/supertonic; corre `uv run python -m ttspro.supertonic.descargar`",
)


def _frases() -> list[tuple[str, str]]:
    salida = []
    for ln in FRASES.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            idioma, frase = ln.split("\t", 1)
            salida.append((idioma, frase))
    return salida


def _js(entradas: list[tuple[str, str]]) -> list[dict]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("node no esta en PATH: la paridad Python/JS no se puede comprobar")
    proceso = subprocess.run(
        [node, "--no-warnings", str(CLI_TS)],
        input="".join(f"{i}\t{f}\n" for i, f in entradas),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proceso.returncode != 0:
        pytest.fail(f"cli-unicode.ts fallo:\n{proceso.stderr[-2000:]}")
    return [json.loads(ln) for ln in proceso.stdout.splitlines() if ln.strip()]


@pytest.fixture(scope="module")
def js() -> dict[tuple[str, str], dict]:
    entradas = _frases()
    salidas = _js(entradas)
    assert len(salidas) == len(entradas)
    return {(s["idioma"], s["frase"]): s for s in salidas}


@requiere_modelos
@pytest.mark.parametrize(("idioma", "frase"), _frases())
def test_paridad_por_etapas(js, idioma: str, frase: str) -> None:
    salida = js[(idioma, frase)]
    esperado = preprocesar(frase, idioma)
    assert salida["preprocesado"] == esperado, "etapa: preprocesar"
    assert salida["puntos"] == puntos_de_codigo(esperado), "etapa: puntos de codigo"
    indexador = Indexador.desde(INDEXADOR)
    assert salida["ids"] == indexador.ids(esperado), "etapa: ids"


@requiere_modelos
def test_paridad_troceador() -> None:
    (salida,) = _js([("trocear", LARGO.replace("\n", "\\n"))])
    # El CLI recibe una linea, asi que los saltos viajan escapados y se rehacen aqui.
    assert salida["trozos"] == trocear(salida["frase"]), "etapa: trocear"
    assert len(salida["trozos"]) >= 2, "el texto de prueba tiene que trocearse de verdad"


@requiere_modelos
def test_toda_letra_del_espanol_tiene_id() -> None:
    """Ninguna letra del espanol puede caer en el hueco de desconocido.

    La tabla no trae la n con tilde ni las vocales acentuadas precompuestas, pero
    la NFKD las parte en letra + marca combinante y ambas SI estan. Si un dia se
    tocara ese orden, esto lo caza antes que el oido.
    """
    indexador = Indexador.desde(INDEXADOR)
    for letra in "abcdefghijklmnopqrstuvwxyzñáéíóúüABCDEFGHIJKLMNOPQRSTUVWXYZÑÁÉÍÓÚÜ¿¡":
        ids = indexador.ids(preprocesar(letra, "es"))
        assert -1 not in ids, f"{letra!r} no tiene id en la tabla"
