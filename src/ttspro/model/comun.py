"""Building blocks shared by the synthesizer, written for `torch.onnx.export`
on opset 17 and ORT Web's WebGPU op set (rule 4):

- `softplus`/`logsigmoid` are spelled out with Where/Log/Exp because ONNX
  `Softplus` is not on the WebGPU list;
- `flip_channels` is a Gather with constant indices, not `torch.flip`
  (Slice with negative steps);
- nothing here draws random numbers (rule 5).

Architecture after VITS (Kim, Kong, Son 2021; reference code MIT).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(torch.clamp(x, max=20.0))))


def logsigmoid(x: torch.Tensor) -> torch.Tensor:
    return -softplus(-x)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if "Conv" in m.__class__.__name__:
        m.weight.data.normal_(mean, std)


def remove_weight_norm(module: nn.Module) -> None:
    """Fold every weight_norm parametrization before export."""
    for m in module.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)


def sequence_mask(
    length: torch.Tensor, max_length: torch.Tensor | int | None = None
) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """duration (b, 1, t_x), mask (b, 1, t_y, t_x) -> hard alignment (b, 1, t_y, t_x)."""
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


def fused_add_tanh_sigmoid_multiply(
    a: torch.Tensor, b: torch.Tensor, n_channels: int
) -> torch.Tensor:
    in_act = a + b
    return torch.tanh(in_act[:, :n_channels]) * torch.sigmoid(in_act[:, n_channels:])


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = int(ids_str[i])
        ret[i] = x[i, :, idx_str : idx_str + segment_size]
    return ret


def rand_slice_segments(x: torch.Tensor, x_lengths: torch.Tensor, segment_size: int):
    b, _, t = x.size()
    ids_str_max = (x_lengths - segment_size + 1).clamp(min=1)
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    return slice_segments(x, ids_str, segment_size), ids_str


class LayerNorm(nn.Module):
    """LayerNorm over the channel axis of (b, c, t)."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class FlipChannels(nn.Module):
    """x[:, ::-1] on the channel axis as a Gather with constant indices."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("indices", torch.arange(channels - 1, -1, -1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, 1, self.indices)


class WN(nn.Module):
    """WaveNet-style residual stack with optional global conditioning."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        if gin_channels:
            self.cond_layer = weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            )
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                weight_norm(
                    nn.Conv1d(
                        hidden_channels,
                        2 * hidden_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=padding,
                    )
                )
            )
            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(
                weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1))
            )

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                offset = i * 2 * self.hidden_channels
                g_l = g[:, offset : offset + 2 * self.hidden_channels]
            else:
                g_l = torch.zeros_like(x_in)
            acts = self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels))
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :]
            else:
                output = output + res_skip_acts
        return output * x_mask


class ResBlock1(nn.Module):
    """HiFi-GAN residual block (type 1)."""

    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: tuple[int, ...] = (1, 3, 5)
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class ResBlock2(nn.Module):
    """HiFi-GAN residual block (type 2): two dilated convolutions, no second stack.

    Piper's `medium` voices use this one, and it is a third of the parameters of
    ResBlock1 for the same kernel sizes.
    """

    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: tuple[int, ...] = (1, 3)
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            x = c(xt) + x
        if x_mask is not None:
            x = x * x_mask
        return x


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q) between diagonal Gaussians, elementwise."""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    return kl


SQRT_2 = math.sqrt(2.0)
