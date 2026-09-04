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


def test_tokenizar_devuelve_ids_de_la_tabla() -> None:
    fonemas, secuencia = tokenizar("Hola, ¿cómo estás?", "es")
    assert fonemas == "ˈola , ¿ kˈomo estˈas ?"
    assert len(secuencia) == len(fonemas)
    assert all(i > 0 for i in secuencia)


def test_simbolo_desconocido_falla_en_voz_alta() -> None:
    with pytest.raises(ValueError, match="fuera de models/contrato.json"):
        ids("ok☃")
