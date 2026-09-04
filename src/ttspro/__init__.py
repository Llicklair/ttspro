"""ttspro — multilingual non-autoregressive TTS with one-shot voice cloning.

Layout (ARCHITECTURE.md rule 7, enforced by .gb-boundaries):

- ``frontend``  text -> phonemes -> token ids. Mirrored 1:1 in ``web/src/frontend``.
- ``model``     the VITS-style synthesizer and the speaker encoder (inference only).
- ``export``    torch -> ONNX, against ``models/contrato.json``.
- ``train``     training loop, losses, monotonic alignment (never imported by the above).
- ``data``      datasets and loaders (never imported by the above).
"""

__version__ = "0.0.1"
