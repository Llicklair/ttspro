"""La vara de medir del proyecto: WER e inteligibilidad, en un solo sitio.

Esto vivía dentro de ``ttspro.train.evaluar``, que es el evaluador de la cadena
que retiró el `ADR 0012 <../../docs/adr/0012-supertonic-como-motor.md>`_. Medir no
es entrenar: el criterio de terminado tiene que poder pedir un WER sin arrastrar
torch, el exportador y el modelo VITS entero sólo para normalizar una cadena.

Y hay una razón más fuerte que la comodidad: una vara duplicada deja de ser una
vara. Si el criterio normaliza el texto de una manera y el evaluador de otra, los
dos números dejan de ser comparables sin que nadie lo note, y este repo compara
números de hace semanas todo el rato.

La similitud de locutor NO está aquí: vive en ``ttspro.supertonic.constructor``
(``Similitud``), porque además de medir es la entrada del constructor de voces.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalizar_para_wer", "wer"]


def normalizar_para_wer(texto: str) -> str:
    """Lo que se compara al medir WER: minúsculas, sin tildes y sin puntuación.

    Un TTS no escribe, así que castigarlo por una coma o por una tilde mide la
    transcripción de Whisper, no la síntesis. Nota: la ``ñ`` cae a ``n`` aquí,
    porque la descomposición NFKD la parte y el filtro se queda con la letra
    base. Es consistente en los dos lados de la comparación, así que no infla ni
    hunde el número; lo que no puede es cambiarse en un lado solo.
    """
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", texto).split())


def wer(referencia: str, hipotesis: str) -> float:
    """WER entre dos textos, ya normalizados por `normalizar_para_wer`.

    `jiwer` vive en el extra ``eval`` y ``uv sync`` lo PODA si no lo pides, así
    que el import va aquí dentro y el fallo dice qué instalar.
    """
    try:
        from jiwer import wer as _wer
    except ModuleNotFoundError as e:  # pragma: no cover - depende del entorno
        raise ModuleNotFoundError(
            "para medir WER hace falta `jiwer`: uv sync --extra dev --extra export --extra eval"
        ) from e
    return float(_wer(referencia, hipotesis))
