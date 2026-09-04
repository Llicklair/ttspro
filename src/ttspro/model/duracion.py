"""Stochastic duration predictor (VITS) with export-safe spline flows.

Changes from the reference, all for rule 4/5 and none changing the maths:
- the rational-quadratic spline is evaluated for EVERY element (inputs clamped
  to the interval) and `torch.where` picks spline vs identity, instead of
  boolean-mask indexing, which exports to NonZero (not on WebGPU);
- in-place edits of the end bins become `torch.cat` with constants;
- softplus/logsigmoid come from `ttspro.model.comun`;
- at inference the noise is an INPUT (`noise`), never drawn here.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ttspro.model.comun import LayerNorm, logsigmoid, softplus
from ttspro.model.flow import Flip

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def _searchsorted(
    bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    bin_locations = torch.cat([bin_locations[..., :-1], bin_locations[..., -1:] + eps], dim=-1)
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def _cumulative(unnormalized: torch.Tensor, min_size: float, left: float, right: float):
    num_bins = unnormalized.shape[-1]
    sizes = F.softmax(unnormalized, dim=-1)
    sizes = min_size + (1 - min_size * num_bins) * sizes
    cum = torch.cumsum(sizes, dim=-1)
    cum = F.pad(cum, pad=(1, 0), mode="constant", value=0.0)
    cum = (right - left) * cum + left
    cum = torch.cat(
        [torch.full_like(cum[..., :1], left), cum[..., 1:-1], torch.full_like(cum[..., :1], right)],
        dim=-1,
    )
    sizes = cum[..., 1:] - cum[..., :-1]
    return cum, sizes


def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
):
    cumwidths, widths = _cumulative(unnormalized_widths, min_bin_width, left, right)
    derivatives = min_derivative + softplus(unnormalized_derivatives)
    cumheights, heights = _cumulative(unnormalized_heights, min_bin_height, bottom, top)

    bin_idx = _searchsorted(cumheights if inverse else cumwidths, inputs)[..., None]
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)
        discriminant = torch.clamp(b.pow(2) - 4 * a * c, min=0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)
    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    return outputs, logabsdet


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
):
    """Linear tails outside [-tail_bound, tail_bound]; spline inside."""
    inside = inputs.abs() <= tail_bound  # not `&`: ONNX And is off the WebGPU list
    constant = math.log(math.exp(1 - min_derivative) - 1)
    borde = torch.full_like(unnormalized_derivatives[..., :1], constant)
    unnormalized_derivatives = torch.cat([borde, unnormalized_derivatives, borde], dim=-1)
    clamped = torch.clamp(inputs, -tail_bound, tail_bound)
    outputs_in, logabsdet_in = rational_quadratic_spline(
        clamped,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    outputs = torch.where(inside, outputs_in, inputs)
    logabsdet = torch.where(inside, logabsdet_in, torch.zeros_like(inputs))
    return outputs, logabsdet


class DDSConv(nn.Module):
    """Dilated depth-separable convolution stack."""

    def __init__(
        self, channels: int, kernel_size: int, n_layers: int, p_dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = F.gelu(self.norms_1[i](y))
            y = self.convs_1x1[i](y)
            y = F.gelu(self.norms_2[i](y))
            x = x + self.drop(y)
        return x * x_mask


class ConvFlow(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: float = 5.0,
    ) -> None:
        super().__init__()
        self.filter_channels = filter_channels
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2
        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask
        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # (b, c, t, 3*num_bins - 1)
        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]
        x1, logabsdet = unconstrained_rational_quadratic_spline(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tail_bound=self.tail_bound,
        )
        x = torch.cat([x0, x1], 1) * x_mask
        if reverse:
            return x
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        return x, logdet


class Log(nn.Module):
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        return torch.exp(x) * x_mask


class ElementwiseAffine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False, **kwargs):
        if not reverse:
            y = (self.m + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        return (x - self.m) * torch.exp(-self.logs) * x_mask


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        filter_channels = in_channels  # as in the reference ("to be removed in a future version")
        self.log_flow = Log()
        self.flows = nn.ModuleList([ElementwiseAffine(2)])
        for _ in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip(2))
        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList([ElementwiseAffine(2)])
        for _ in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip(2))
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        reverse: bool = False,
        noise_scale=1.0,
        noise: torch.Tensor | None = None,
    ):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(torch.detach(g))
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            assert w is not None
            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2), device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q = logdet_tot_q + logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q = logdet_tot_q + torch.sum(
                (logsigmoid(z_u) + logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot = logdet_tot + logdet
            z = torch.cat([z0, z1], 1)
            for flow in self.flows:
                z, logdet = flow(z, x_mask, g=x, reverse=False)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # (b,)

        assert noise is not None, "at inference the noise is an input (rule 5)"
        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]  # drop the last Flip as in the reference
        z = noise * noise_scale
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=True)
        z0, _ = torch.split(z, [1, 1], 1)
        return z0  # logw
