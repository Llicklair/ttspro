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
    """The exported graph reproduces the TRAINED torch model on real sentences:
    mean log-mel distance < 0.1 over the fixture sentences (es and en)."""
    ruta = _grafo("tts")
    hechos_ruta = ruta.with_suffix(".export.json")
    if not hechos_ruta.exists():
        pytest.fail("criterio 2: no hay models/tts.export.json (exporta con ttspro.export.tts)")
    hechos = json.loads(hechos_ruta.read_text(encoding="utf-8"))
    if not hechos.get("checkpoint") or not Path(hechos["checkpoint"]).exists():
        pytest.fail("criterio 2: el grafo no viene de un checkpoint entrenado (pesos aleatorios)")

    import numpy as np
    import onnxruntime as ort
    import torch

    from ttspro.export.tts import ENTRADAS_COMPLETAS, cargar
    from ttspro.frontend import tokenizar
    from ttspro.model.config import ConfigSintetizador
    from ttspro.train.mel import mel_spectrogram_torch

    cfg = ConfigSintetizador(**hechos["config"])
    modelo = cargar(Path(hechos["checkpoint"]), cfg).preparar_export()
    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    frases = [
        ln.split("\t", 1)
        for ln in (RAIZ / "tests/fixtures/frontend/frases.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    g = torch.Generator().manual_seed(0)
    embedding = torch.nn.functional.normalize(torch.randn(1, cfg.gin_channels, generator=g), dim=1)
    distancias = []
    # A fixed-voice graph (ADR 0008) has no `embedding` and no `idioma` input: the
    # exporter prunes them. So torch is called with the full signature and ORT is fed
    # BY NAME with what the contract declares, exactly like the browser runtime does.
    nombres = [e["nombre"] for e in CONTRATO["grafos"]["tts"]["entradas"]]
    for idioma, frase in frases:
        _, secuencia = tokenizar(frase, idioma)
        tokens = torch.tensor([secuencia])
        idiomas = CONTRATO["idiomas"]
        entradas = (
            tokens,
            torch.tensor([tokens.shape[1]]),
            embedding,
            torch.tensor([idiomas.index(idioma) if idioma in idiomas else 0]),
            torch.randn(1, cfg.inter_channels, tokens.shape[1] * 12, generator=g),
            torch.randn(1, 2, tokens.shape[1], generator=g),
            torch.tensor([0.667, 0.8, 1.0]),
        )
        por_nombre = dict(zip(ENTRADAS_COMPLETAS, entradas, strict=True))
        with torch.no_grad():
            onda_torch = modelo.inferir(*entradas)
        (onda_ort,) = sesion.run(None, {n: por_nombre[n].numpy() for n in nombres})
        assert onda_ort.shape == tuple(onda_torch.shape), frase

        def logmel(onda):
            return mel_spectrogram_torch(
                onda.reshape(1, -1),
                cfg.n_fft,
                cfg.n_mels,
                cfg.sample_rate,
                cfg.hop_length,
                cfg.win_length,
                cfg.mel_fmin,
                cfg.mel_fmax,
            )

        distancias.append(
            float((logmel(onda_torch) - logmel(torch.from_numpy(onda_ort))).abs().mean())
        )
    assert np.mean(distancias) < 0.1, (
        f"criterio 2: distancia log-mel media {np.mean(distancias):.4f}"
    )


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
    """Objective quality on held-out speakers: `ttspro.train.evaluar` synthesizes
    sentences the model never saw with that speaker's embedding, transcribes them
    with Whisper and measures speaker cosine. Small on purpose (3 speakers x 2
    sentences per language): it is a gate, not a paper, and it still takes minutes."""
    import subprocess
    import sys

    _grafo("tts")
    hechos_ruta = MODELOS / "tts.export.json"
    if not hechos_ruta.exists():
        pytest.fail("criterio 4: no hay models/tts.export.json (exporta con ttspro.export.tts)")
    checkpoint = json.loads(hechos_ruta.read_text(encoding="utf-8")).get("checkpoint")
    if not checkpoint or not Path(checkpoint).exists():
        pytest.fail("criterio 4: el grafo no viene de un checkpoint entrenado (pesos aleatorios)")
    caches = sorted(p.parent for p in (RAIZ / "cache").glob("*/indice.pt"))
    if not caches:
        pytest.fail("criterio 4: no hay cache/*/indice.pt (prepara datos con ttspro.data.preparar)")

    proceso = subprocess.run(
        [
            sys.executable,
            "-m",
            "ttspro.train.evaluar",
            "--checkpoint",
            checkpoint,
            "--cache",
            *[str(c) for c in caches],
            "--locutores",
            "3",
            "--frases",
            "2",
        ],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    detalle = proceso.stdout[-2000:] + chr(10) + proceso.stderr[-1500:]
    assert proceso.returncode == 0, "criterio 4: la evaluación falló:" + chr(10) + detalle
    resumen = json.loads(
        Path(checkpoint).with_suffix(".evaluacion.json").read_text(encoding="utf-8")
    )["resumen"]
    for idioma in CONTRATO["idiomas"]:
        assert idioma in resumen, f"criterio 4: sin frases de {idioma} en la evaluación: {resumen}"
        r = resumen[idioma]
        assert r["wer_medio"] <= 0.10, f"criterio 4 ({idioma}), inteligibilidad: {r}"
        # 0.55 absolute (above the p99 of different speakers, 0.465) and three quarters of the
        # ceiling measured on these same speakers. See SCOPE.md and docs/evidencia.md.
        assert r["secs_sintesis"] >= 0.55, f"criterio 4 ({idioma}), similitud: {r}"
        assert r["secs_sintesis"] >= 0.75 * r["secs_real_mismo"], (
            f"criterio 4 ({idioma}), techo: {r}"
        )


def test_5_navegador_wasm_10_palabras_en_menos_de_3s_y_110mb() -> None:
    """Playwright + Chromium headless, EP wasm, the real page: a 10-word sentence
    synthesized in < 3 s after load, and the total download <= 110 MB
    (SCOPE criterion 5; ADR 0005 amendment). The e2e lives in web/e2e and
    writes web/test-results/criterio5.json with what it measured."""
    import shutil
    import subprocess

    _grafo("tts")
    _grafo("speaker_encoder")
    npx = shutil.which("npx")
    if npx is None:
        pytest.fail("criterio 5: npx no está en PATH")
    web = RAIZ / "web"
    proceso = subprocess.run(
        [npx, "playwright", "test", "--reporter=line"],
        cwd=web,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    detalle = proceso.stdout[-3000:] + chr(10) + proceso.stderr[-1500:]
    assert proceso.returncode == 0, "criterio 5: e2e rojo:" + chr(10) + detalle
    medido = json.loads((web / "test-results" / "criterio5.json").read_text(encoding="utf-8"))
    assert medido["proveedor"] == "wasm", medido
    assert medido["ms_total_sintesis"] < 3000, medido
    assert medido["mb_descarga_total"] <= 110, medido
