"""Python-side frontend facts that do not need node."""

from __future__ import annotations

import pytest

from ttspro.frontend import Trozo, fonemizar, ids, normalizar, tokenizar, trocear
from ttspro.frontend.fonemas import VERSION_ESPERADA, version


def test_trocear_no_parte_entre_digitos() -> None:
    assert trocear("a las 10:30 son 3,5 o 3.5.") == [
        Trozo("texto", "a las 10:30 son 3,5 o 3.5"),
        Trozo("puntuacion", "."),
    ]


def test_trocear_conserva_puntuacion_como_tokens() -> None:
    assert trocear(normalizar("Hola, ¿cómo estás?")) == [
        Trozo("texto", "Hola"),
        Trozo("puntuacion", ","),
        Trozo("puntuacion", "¿"),
        Trozo("texto", "cómo estás"),
        Trozo("puntuacion", "?"),
    ]


def test_espeak_es_la_version_pinneada() -> None:
    assert version().startswith(VERSION_ESPERADA), version()


def test_fonemizar_es_y_en() -> None:
    assert fonemizar("Hola", "es") == "ˈola"
    assert fonemizar("Hello", "en") == "həlˈoʊ"


def test_tokenizar_sigue_la_convencion_del_contrato() -> None:
    """The sequence shape is data (contract), not a constant: coqui wants a pad
    before each symbol, Piper a pad after plus ^ and $ around the sentence."""
    from ttspro.frontend.tokens import simbolos

    cfg = simbolos()
    fonemas, secuencia = tokenizar("Hola, ¿cómo estás?", "es")
    assert fonemas == "ˈola , ¿ kˈomo estˈas ?"
    utiles = len(fonemas) - sum(fonemas.count(c) for c in (cfg.get("equivalencias") or {}))
    esperado = 2 * utiles + (1 if cfg.get("blank_al_inicio", True) else 0)
    esperado += (cfg.get("bos") is not None) + (cfg.get("eos") is not None)
    assert len(secuencia) == esperado, (len(secuencia), esperado)
    if cfg.get("bos") is not None:
        assert secuencia[0] == cfg["bos"] and secuencia[-1] == cfg["eos"]
        assert all(i == cfg["pad"] for i in secuencia[2:-1:2])
    else:
        assert all(i == cfg["pad"] for i in secuencia[0::2])


def test_simbolo_desconocido_falla_en_voz_alta() -> None:
    with pytest.raises(ValueError, match="fuera de models/contrato.json"):
        ids("ok☃")
