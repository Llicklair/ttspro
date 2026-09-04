"""The exportable speaker encoder: raw 16 kHz waveform -> L2-normalized embedding.

Wraps any backbone that maps (B, frames, n_mels) features to an embedding
(speechbrain's ECAPA-TDNN today; whatever wins in docs/evidencia.md tomorrow),
with the feature extraction inside the graph so the browser only resamples
(ADR 0002). The backbone is injected: this module never imports speechbrain
(rule 11 — experiments live behind the `exp` extra).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ttspro.model.fbank import FbankConv


class SpeakerEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, fbank: FbankConv | None = None) -> None:
        super().__init__()
        self.fbank = fbank or FbankConv()
        self.backbone = backbone

    def forward(self, onda: torch.Tensor) -> torch.Tensor:
        """(1, muestras) -> (1, dim), unit norm."""
        feats = self.fbank(onda)
        emb = self.backbone(feats)
        emb = emb.reshape(emb.shape[0], -1)
        return F.normalize(emb, dim=1)
