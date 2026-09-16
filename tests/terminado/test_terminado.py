"""The MVP done criterion from SCOPE.md, as a command: ``uv run pytest tests/terminado -q``.

Each test is one numbered point of the criterion. **A test that cannot run FAILS
with the reason instead of skipping**, because a skipped criterion reads as green
in a summary and that is the lie this file exists to prevent. That rule is older
than the current engine and it survives it.

Rewritten on 2026-09-16 against the chain of
`ADR 0012 <../../docs/adr/0012-supertonic-como-motor.md>`_ and its amendment. What
it measured before was the Piper + OpenVoice chain: ``models/tts.onnx``, a torch
checkpoint to compare the export against, and a 110 MB download budget. None of
the three exists any more, so the file was red for the wrong reason — the worst
state for a criterion, because red stops meaning anything.

What changed, point by point, and what did not:

- **1** is the same question against four graphs instead of two.
- **2 was PyTorch ↔ ORT parity and is retired.** It guarded «what runs is not what
  was measured», a risk that came from exporting a trained model ourselves.
  Upstream publishes ONNX now: there is no torch side to disagree with. The same
  risk in this chain is the noise draw, so the point becomes **reproducibility**
  (rule 5): same text, same voice, same seed, same audio. Without it no number in
  docs/evidencia.md can be re-checked.
- **3** is the same rule 3, now over the Unicode frontend instead of the phonemes.
- **4 and 5** are measured **in the browser**, by ``web/e2e/criterio.spec.ts``, and
  scored here. Reimplementing synthesis in Python to grade it would grade the
  reimplementation: the thing that ships is a page, and the model that clones only
  exists as a Worker. So the e2e runs the page and leaves the wav files; this file
  puts a number on them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
MODELOS = RAIZ / "models" / "supertonic"
CONTRATO_RUTA = MODELOS / "contrato.json"
# No `web/test-results/`: Playwright lo borra al empezar cada tirada, asi que
# correr cualquier otro spec se llevaba por delante lo medido.
MEDIDO = RAIZ / "web" / "medido" / "criterio.json"

CORRE_LA_PAGINA = "cd web && npx playwright test e2e/criterio.spec.ts"


def _contrato() -> dict:
    if not CONTRATO_RUTA.exists():
        pytest.fail(
            f"criterio 1: no existe {CONTRATO_RUTA.relative_to(RAIZ)}."
            " Descarga el motor: uv run python -m ttspro.supertonic.descargar"
        )
    return json.loads(CONTRATO_RUTA.read_text(encoding="utf-8"))


def _grafo(nombre: str) -> Path:
    ruta = MODELOS / _contrato()["grafos"][nombre]["fichero"]
    if not ruta.exists():
        pytest.fail(
            f"criterio 1: no existe {ruta.relative_to(RAIZ)}."
            " Descarga el motor: uv run python -m ttspro.supertonic.descargar"
        )
    return ruta


def _medido() -> dict:
    """Lo que la página midió de verdad. Sin esto no hay criterio 4 ni 5."""
    if not MEDIDO.exists():
        pytest.fail(
            f"criterios 4 y 5: no existe {MEDIDO.relative_to(RAIZ)}."
            f" Lo escribe el navegador: {CORRE_LA_PAGINA}"
        )
    return json.loads(MEDIDO.read_text(encoding="utf-8"))


def _onda(ruta: Path) -> tuple:
    sf = pytest.importorskip("soundfile", reason="instala el extra `export`")
    onda, frecuencia = sf.read(str(ruta), dtype="float32", always_2d=True)
    return onda.mean(axis=1), frecuencia


# --------------------------------------------------------------------- 1


@pytest.mark.parametrize(
    "nombre", ["text_encoder", "duration_predictor", "vector_estimator", "vocoder"]
)
def test_1_existe_carga_y_coincide_con_el_contrato(nombre: str) -> None:
    """Los cuatro grafos abren en onnxruntime y su firma es la que dice el contrato.

    El contrato se GENERA leyendo los propios `.onnx` (ADR 0012, decisión 5), así
    que esto no comprueba que dos ficheros escritos a mano coincidan: comprueba
    que lo descargado sigue siendo lo que se leyó, que es lo que se rompe cuando
    alguien mueve una revisión o baja media descarga.
    """
    ruta = _grafo(nombre)
    ort = pytest.importorskip("onnxruntime", reason="criterio 1: instala el extra `export`")
    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    declarado = _contrato()["grafos"][nombre]
    assert [e.name for e in sesion.get_inputs()] == [e["nombre"] for e in declarado["entradas"]]
    assert [s.name for s in sesion.get_outputs()] == [s["nombre"] for s in declarado["salidas"]]


# --------------------------------------------------------------------- 2


def test_2_la_misma_semilla_da_la_misma_toma() -> None:
    """Regla 5: el ruido entra como tensor, así que una semilla reproduce una toma.

    Sustituye a la paridad PyTorch ↔ ORT, que dejó de tener sentido cuando el
    modelo pasó a venir ya exportado (ver el docstring de arriba). Es el mismo
    tipo de garantía: que lo que se mide hoy se pueda volver a medir mañana. Si
    esto falla, ningún número de docs/evidencia.md es comprobable y la palabra
    «medido» deja de significar nada en este repo.
    """
    for nombre in ["text_encoder", "duration_predictor", "vector_estimator", "vocoder"]:
        _grafo(nombre)
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime", reason="criterio 2: instala el extra `export`")

    from ttspro.supertonic.estilo import Estilo, catalogo
    from ttspro.supertonic.motor import Motor

    estilos = catalogo(MODELOS / "voice_styles")
    if "F1" not in estilos:
        pytest.fail("criterio 2: no hay models/supertonic/voice_styles/F1.json")
    motor = Motor(MODELOS)
    voz = Estilo.cargar(estilos["F1"])
    frase = "Una frase cualquiera para comprobar que la semilla manda."

    a = motor.sintetizar(frase, voz, "es", pasos=4, semilla=7)
    b = motor.sintetizar(frase, voz, "es", pasos=4, semilla=7)
    c = motor.sintetizar(frase, voz, "es", pasos=4, semilla=8)

    assert a.onda.shape == b.onda.shape, "criterio 2: la misma semilla cambia hasta la duración"
    assert np.array_equal(a.onda, b.onda), (
        "criterio 2: la misma semilla da dos ondas distintas; "
        f"diferencia máxima {float(np.abs(a.onda - b.onda).max()):.2e}"
    )
    # Y la vuelta: si dos semillas dan lo mismo, la semilla no se está usando y el
    # test de arriba pasaría por el motivo equivocado.
    assert not np.array_equal(a.onda, c.onda), "criterio 2: la semilla no cambia nada"


# --------------------------------------------------------------------- 3


def test_3_paridad_frontend_python_js() -> None:
    """Regla 3: el frontend Unicode dice lo mismo en Python y en el navegador.

    Es la suite de paridad que ya corre en la rápida; aquí se invoca como proceso
    para que el criterio no dependa de que alguien se acuerde de correr la otra.
    """
    proceso = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_supertonic_paridad.py", "-q"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proceso.returncode == 0, f"criterio 3: paridad rota:\n{proceso.stdout[-3000:]}"


# --------------------------------------------------------------------- 4


def test_4a_inteligibilidad_wer_menor_que_10() -> None:
    """Lo que dice la página se entiende: WER medio de Whisper-small ≤ 0,10.

    Sobre los wav que salieron del navegador, no de una síntesis en Python: lo que
    se juzga es el producto.

    Son **cuatro frases**, no una, y la razón está medida: con una sola de diez
    palabras cada palabra vale 0,1 de WER, así que el criterio lo decidía un único
    fallo de transcripción — y la primera tirada dio 0,100 exacto porque Whisper
    oyó «sarpó» donde decía «zarpó», que es seseo del ASR y no del sintetizador.
    """
    medido = _medido()
    whisper = pytest.importorskip(
        "whisper",
        # `uv sync` PODA Whisper si no pides el extra, y el fallo por defecto no
        # dice cual es el extra.
        reason="criterio 4: uv sync --extra dev --extra export --extra eval",
    )
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")

    from ttspro.medida import normalizar_para_wer, wer

    asr = whisper.load_model("small")
    medidas, detalle = [], []
    for f in medido["lectura"]:
        ruta = MEDIDO.parent / f["wav"]
        if not ruta.exists():
            pytest.fail(f"criterio 4: falta {ruta.name}; vuelve a correr {CORRE_LA_PAGINA}")
        onda, frecuencia = _onda(ruta)
        a16 = torchaudio.functional.resample(torch.from_numpy(onda), frecuencia, 16000).numpy()
        dicho = asr.transcribe(a16, language="es", fp16=False)["text"]
        medidas.append(wer(normalizar_para_wer(f["frase"]), normalizar_para_wer(dicho)))
        detalle.append(f"    {medidas[-1]:.3f}  oyó {dicho.strip()!r}")
    medio = sum(medidas) / len(medidas)
    assert medio <= 0.10, (
        f"criterio 4, inteligibilidad: WER medio {medio:.3f} sobre {len(medidas)} frases\n"
        + "\n".join(detalle)
    )


def test_4b_similitud_de_locutor_mayor_que_0_55() -> None:
    """Clonar se parece: coseno de WeSpeaker ≥ 0,55 contra la referencia.

    **Este punto está en rojo y se sabe por qué**: 0,413 de media sobre 14
    locutores (evidencia, 2026-09-16). No es un parámetro mal puesto — se barrió
    temperatura y pasos, no sigue a la calidad de la referencia (r = +0,247) y se
    estanca en 0,41–0,45 locutor a locutor. Lo único que lo movió fue darle más
    grabación, +13 puntos de 2 s a 20 s, y por eso la página graba 20.

    Aquí se mide con UNA referencia de 6 s, en otro idioma que el de la síntesis,
    así que este número es más bajo que aquel 0,413 y no lo sustituye: es el
    guardarraíl de que la cadena entera sigue clonando algo reconocible, no la
    medida del modelo.
    """
    medido = _medido()
    referencia = RAIZ / "web" / medido["referencia"]
    if not referencia.exists():
        pytest.fail(f"criterio 4: falta {referencia.name}")
    pytest.importorskip("onnxruntime", reason="criterio 4: instala el extra `export`")

    from ttspro.supertonic.constructor import Similitud

    sim = Similitud()
    objetivo = sim.embedding(*_onda(referencia))
    cosenos = []
    for f in medido["clonada"]:
        ruta = MEDIDO.parent / f["wav"]
        if not ruta.exists():
            pytest.fail(f"criterio 4: falta {ruta.name}; vuelve a correr {CORRE_LA_PAGINA}")
        cosenos.append(float(sim.embedding(*_onda(ruta)) @ objetivo))
    coseno = sum(cosenos) / len(cosenos)
    assert coseno >= 0.55, (
        f"criterio 4, similitud: {coseno:.3f} sobre 0,55."
        " El techo de este encoder publicado es 0,413 de media (evidencia 2026-09-16):"
        " subir de ahí necesita otro encoder, no otro ajuste."
    )


# --------------------------------------------------------------------- 5


def test_5_en_el_navegador_una_frase_en_menos_de_3s() -> None:
    """Una frase en < 3 s en el navegador, con el EP que tocara (wasm en headless).

    El presupuesto de descarga que este punto llevaba —110 MB— lo retiró el
    ADR 0012 por instrucción explícita de Marcos, así que los MB se anotan y no se
    juzgan. La latencia sigue juzgándose: es lo que decide si la página es usable.
    """
    medido = _medido()
    ms = medido.get("ms_lectura")
    assert isinstance(ms, (int, float)) and ms > 0, f"criterio 5: sin medida de latencia: {medido}"
    assert ms < 3000, (
        f"criterio 5: {ms:.0f} ms para «{medido['lectura'][0]['frase']}»"
        f" en {medido.get('proveedor')} ({medido.get('mb_descarga_total')} MB de descarga)"
    )
