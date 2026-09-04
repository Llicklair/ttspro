"""Speaker-consistency loss (YourTTS §3.3): push the generated segment to have
the same speaker embedding as the real one, through the SAME encoder the
browser uses.

    L_scl = mean(1 - cos(phi(y_hat), phi(y)))

`phi` is WeSpeaker ResNet34 (frozen, eval) over our Kaldi fbank, so the gradient
reaches the generator. Two details that matter:

- the encoder eats 16 kHz and the model emits 22 050 Hz: `torchaudio`'s resample
  is a convolution, so it is differentiable and stays in the graph;
- it runs in fp32 even under autocast: the Kaldi fbank scales by 32768 and
  squares, which overflows fp16 (measured 2026-09-04 on the ONNX conversion).

This is the lever that turns "a voice" into "this voice": without it the model
learns to use the embedding as a weak style hint. Cost is measured in
docs/evidencia.md before it is switched on by default.
"""

from __future__ import annotations

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from ttspro.model.fbank import FbankKaldiConv
from ttspro.model.wespeaker import cargar


class ConsistenciaLocutor(nn.Module):
    def __init__(self, sr_modelo: int = 22050, sr_encoder: int = 16000) -> None:
        super().__init__()
        self.remuestreo = torchaudio.transforms.Resample(sr_modelo, sr_encoder)
        self.fbank = FbankKaldiConv()
        self.encoder = cargar()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def embedding(self, onda: torch.Tensor) -> torch.Tensor:
        """(B, muestras) at 22.05 kHz -> (B, 256) unit norm."""
        return F.normalize(self.encoder(self.fbank(self.remuestreo(onda))), dim=1)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emisor = y_hat.squeeze(1).float()
        real = y.squeeze(1).float()
        with torch.autocast(device_type=emisor.device.type, enabled=False):
            emb_hat = self.embedding(emisor)
            with torch.no_grad():
                emb_real = self.embedding(real)
            return (1.0 - (emb_hat * emb_real).sum(dim=1)).mean()
