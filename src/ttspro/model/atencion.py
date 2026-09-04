"""Relative-position transformer encoder (VITS attentions.py, export-friendly).

The one deliberate change from the reference: `_get_relative_embeddings` pads
by `length` on both sides unconditionally instead of `max(length - w - 1, 0)`,
so no Python `max()` on a traced size bakes a constant into the graph and the
exported model works for any text length.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ttspro.model.comun import LayerNorm


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int | None = None,
        heads_share: bool = True,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.window_size = window_size
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x = self.attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores = scores + self._relative_position_to_absolute_position(rel_logits)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = self.drop(F.softmax(scores, dim=-1))
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        return output.transpose(2, 3).contiguous().view(b, d, t_t)

    @staticmethod
    def _matmul_with_relative_values(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.unsqueeze(0))

    @staticmethod
    def _matmul_with_relative_keys(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, relative_embeddings: torch.Tensor, length) -> torch.Tensor:
        # pad by `length` on both sides (always enough), then take the 2*length-1
        # window starting at window_size + 1. Equivalent to the reference for every
        # length and free of Python max() on traced sizes.
        padded = F.pad(relative_embeddings, [0, 0, length, length, 0, 0])
        start = self.window_size + 1
        return padded[:, start : start + 2 * length - 1]

    @staticmethod
    def _relative_position_to_absolute_position(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, [0, length - 1, 0, 0, 0, 0])
        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]

    @staticmethod
    def _absolute_position_to_relative_position(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length * length + length * (length - 1)])
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self._pad(x * x_mask))
        x = self.drop(torch.relu(x))
        x = self.conv_2(self._pad(x * x_mask))
        return x * x_mask

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        return F.pad(x, [pad_l, pad_r, 0, 0, 0, 0])


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2, strict=True
        ):
            y = self.drop(attn(x, x, attn_mask))
            x = norm1(x + y)
            y = self.drop(ffn(x, x_mask))
            x = norm2(x + y)
        return x * x_mask
