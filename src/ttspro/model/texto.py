"""Text (phoneme) encoder with a language embedding added to the symbol
embedding, the multilingual ingredient of YourTTS in its simplest form."""

from __future__ import annotations

import math

import torch
from torch import nn

from ttspro.model.atencion import Encoder
from ttspro.model.comun import sequence_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_symbols: int,
        n_langs: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.emb = nn.Embedding(n_symbols, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.emb_lang = nn.Embedding(n_langs, hidden_channels)
        nn.init.normal_(self.emb_lang.weight, 0.0, hidden_channels**-0.5)
        self.encoder = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, window_size
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, lang: torch.Tensor):
        """tokens (b, t) int64, lengths (b,), lang (b,) -> x, m, logs, mask (b, c, t)."""
        x = self.emb(tokens) * math.sqrt(self.hidden_channels) + self.emb_lang(lang).unsqueeze(1)
        x = x.transpose(1, -1)  # (b, h, t)
        x_mask = sequence_mask(lengths, tokens.size(1)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask
