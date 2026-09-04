"""Text frontend: normalization -> chunks -> phonemes -> token ids.

Every function here has a mirror in ``web/src/frontend`` and
``tests/test_frontend_paridad.py`` compares both over the shared fixtures
(ARCHITECTURE.md rule 3, ADR 0004). Add a rule on one side only and that test
goes red — which is the point.
"""

from ttspro.frontend.fonemas import EspeakNoDisponible, fonemizar
from ttspro.frontend.normalizar import normalizar
from ttspro.frontend.tokens import fonemas_de_frase, ids, tokenizar
from ttspro.frontend.trocear import Trozo, trocear

__all__ = [
    "EspeakNoDisponible",
    "Trozo",
    "fonemas_de_frase",
    "fonemizar",
    "ids",
    "normalizar",
    "tokenizar",
    "trocear",
]
