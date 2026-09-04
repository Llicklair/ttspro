"""Posterior encoder: linear spectrogram -> latent z. Training only; never in the
exported graph (the flow runs in reverse from the prior at inference)."""

from __future__ import annotations

import torch
from torch import nn

from ttspro.model.comun import WN, sequence_mask


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor | None = None):
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
