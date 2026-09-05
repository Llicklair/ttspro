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
from ttspro.frontend.chat import nombre_legible, normalizar_chat

RAIZ = Path(__file__).resolve().parents[1]
FRASES = RAIZ / "tests" / "fixtures" / "frontend" / "frases.txt"
CHAT = RAIZ / "tests" / "fixtures" / "frontend" / "chat.txt"
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


# ---------------------------------------------------------------- chat stage (ADR 0010)


def _mensajes() -> list[str]:
    return [
        ln.split("	", 1)[1]
        for ln in CHAT.read_text(encoding="utf-8").splitlines()
        if ln.strip() and ln.startswith("chat	")
    ]


@pytest.fixture(scope="module")
def js_chat() -> dict[str, str]:
    mensajes = _mensajes()
    salidas = _frontend_js([("chat", m) for m in mensajes])
    assert len(salidas) == len(mensajes)
    return {s["frase"]: s["chat"] for s in salidas}


@pytest.mark.parametrize("mensaje", _mensajes())
def test_paridad_chat(js_chat, mensaje: str) -> None:
    assert js_chat[mensaje] == normalizar_chat(mensaje), "etapa: chat"


def test_chat_deja_algo_legible_o_nada() -> None:
    """Emote-only messages come out EMPTY (the caller skips them); the rest keep words."""
    assert normalizar_chat("KEKW") == "jajaja"  # laughter, rule 4, wins over the emote list
    assert normalizar_chat("Kappa :) D: F o7") == ""
    assert normalizar_chat("😀") == ""
    assert normalizar_chat("holaaaa q tal todos!!!!") == "hola que tal todos!"
    assert normalizar_chat("JAJAJAJAJA no me lo creo") == "jajaja no me lo creo"
    assert normalizar_chat("@Dark_Lord99 tienes razón") == "Dark Lord99 tienes razón"
    assert normalizar_chat("mira https://x.y/z brutal") == "mira, enlace, brutal"
    # punctuation glued to a word does not hide it from the tables
    assert normalizar_chat("gracias por el stream, nice!") == "gracias por el estrim, nais!"
    assert normalizar_chat("¿q? (hype)") == "¿que? (jaip)"
    assert len(normalizar_chat("palabra " * 100)) <= 200


def _nombres() -> list[str]:
    return [
        ln.split("	", 1)[1]
        for ln in CHAT.read_text(encoding="utf-8").splitlines()
        if ln.startswith("nombre	")
    ]


@pytest.fixture(scope="module")
def js_nombres() -> dict[str, str]:
    nombres = _nombres()
    salidas = _frontend_js([("nombre", n) for n in nombres])
    assert len(salidas) == len(nombres)
    return {s["frase"]: s["chat"] for s in salidas}


@pytest.mark.parametrize("usuario", _nombres())
def test_paridad_nombre(js_nombres, usuario: str) -> None:
    assert js_nombres[usuario] == nombre_legible(usuario), "etapa: nombre"


def test_nombre_legible_dice_algo() -> None:
    assert nombre_legible("xXDark_Lord99Xx") == "Dark Lord"
    assert nombre_legible("@mod_ana") == "mod ana"
    assert nombre_legible("SuperStreamerTV") == "Super Streamer TV"
    assert nombre_legible("Raúl99") == "Raúl"
    # nothing sayable left -> the raw name, never an empty author
    assert nombre_legible("12345") == "12345"
    assert nombre_legible("xXx") == "xXx"
    assert nombre_legible("KEKW") == "KEKW"
    # a real trailing x is part of the name; only the "xX…Xx" decoration goes
    assert nombre_legible("streex_bot") == "streex bot"
    assert nombre_legible("Alex") == "Alex"
    assert nombre_legible("Max_99") == "Max"
    assert (
        nombre_legible("FelixXx") == "Felix"
    )  # no emote list for names: it is what they are called
