"""Manifest round-trip and validation (no audio, no network)."""

from __future__ import annotations

import pytest

from ttspro.data.manifiesto import Frase, escribir, leer


def test_escribir_y_leer_conserva_todo(tmp_path) -> None:
    wav = tmp_path / "audio" / "a.wav"
    frases = [Frase(wav, "es", "loc1", "Hola, ¿cómo estás?"), Frase(wav, "en", "loc2", "Hello.")]
    ruta = tmp_path / "m.tsv"
    escribir(ruta, frases)
    assert leer(ruta) == frases


def test_rutas_relativas_al_manifiesto(tmp_path) -> None:
    ruta = tmp_path / "m.tsv"
    ruta.write_text("wavs/a.wav\tes\tloc\tTexto\n", encoding="utf-8")
    (frase,) = leer(ruta)
    assert frase.wav == tmp_path / "wavs" / "a.wav"


def test_idioma_fuera_del_contrato_falla_en_voz_alta(tmp_path) -> None:
    ruta = tmp_path / "m.tsv"
    ruta.write_text("a.wav\tfr\tloc\tBonjour\n", encoding="utf-8")
    with pytest.raises(ValueError, match="idioma 'fr'"):
        leer(ruta)
