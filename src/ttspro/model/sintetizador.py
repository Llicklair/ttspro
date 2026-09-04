"""The synthesizer (VITS, YourTTS flavour): `forward` for training with the
posterior encoder and monotonic alignment, `inferir` for the exported graph.

`inferir` IS the contract in models/contrato.json:
    tokens (1, L) int64 · longitud_tokens (1,) · embedding (1, 256) · idioma (1,)
    ruido_flow (1, inter, frames) · ruido_duracion (1, 2, L) · escala_ruido (3,)
    -> onda (1, 1, muestras)
`ruido_flow` may be shorter than the frames the durations ask for: it is tiled
(ONNX Tile) and cut, so any length is valid and the caller never has to guess.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ttspro.model.alineacion import maximum_path
from ttspro.model.comun import generate_path, rand_slice_segments, remove_weight_norm, sequence_mask
from ttspro.model.config import ConfigSintetizador
from ttspro.model.duracion import StochasticDurationPredictor
from ttspro.model.flow import ResidualCouplingBlock
from ttspro.model.generador import Generator
from ttspro.model.posterior import PosteriorEncoder
from ttspro.model.texto import TextEncoder


class Sintetizador(nn.Module):
    def __init__(self, cfg: ConfigSintetizador) -> None:
        super().__init__()
        self.cfg = cfg
        self.enc_p = TextEncoder(
            cfg.n_symbols,
            cfg.n_langs,
            cfg.inter_channels,
            cfg.hidden_channels,
            cfg.filter_channels,
            cfg.n_heads,
            cfg.n_layers,
            cfg.kernel_size,
            cfg.p_dropout,
            cfg.window_size,
        )
        self.dec = Generator(
            cfg.inter_channels,
            cfg.resblock_kernel_sizes,
            cfg.resblock_dilation_sizes,
            cfg.upsample_rates,
            cfg.upsample_initial_channel,
            cfg.upsample_kernel_sizes,
            gin_channels=cfg.gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            cfg.spec_channels,
            cfg.inter_channels,
            cfg.inter_channels,
            cfg.posterior_kernel,
            1,
            cfg.posterior_layers,
            gin_channels=cfg.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            cfg.inter_channels,
            cfg.inter_channels,
            cfg.flow_kernel,
            cfg.flow_dilation,
            cfg.flow_layers,
            n_flows=cfg.flow_n_flows,
            gin_channels=cfg.gin_channels,
        )
        self.dp = StochasticDurationPredictor(
            cfg.inter_channels,
            cfg.sdp_filter_channels,
            cfg.sdp_kernel,
            cfg.sdp_dropout,
            cfg.sdp_flows,
            gin_channels=cfg.gin_channels,
        )

    # ------------------------------------------------------------------ training
    def forward(
        self,
        tokens: torch.Tensor,
        tokens_len: torch.Tensor,
        spec: torch.Tensor,
        spec_len: torch.Tensor,
        embedding: torch.Tensor,
        idioma: torch.Tensor,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(tokens, tokens_len, idioma)
        g = embedding.unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(spec, spec_len, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)  # (b, d, t_s)
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # (b, 1, t_s)
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)  # (b, t_t, t_s)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4  # (b, t_t, t_s)
            attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)  # (b, 1, t_t, t_s)
            attn = maximum_path(
                neg_cent.transpose(1, 2), attn_mask.squeeze(1).transpose(1, 2)
            )  # (b, t_s, t_t)
            attn = attn.transpose(1, 2).unsqueeze(1)  # (b, 1, t_t, t_s)

        w = attn.sum(2)  # (b, 1, t_s)
        l_length = self.dp(x, x_mask, w, g=g) / torch.sum(x_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = rand_slice_segments(
            z, spec_len, self.cfg.segment_size // self.cfg.hop_length
        )
        o = self.dec(z_slice, g=g)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    # ------------------------------------------------------------------ export
    def inferir(
        self,
        tokens: torch.Tensor,
        longitud_tokens: torch.Tensor,
        embedding: torch.Tensor,
        idioma: torch.Tensor,
        ruido_flow: torch.Tensor,
        ruido_duracion: torch.Tensor,
        escala_ruido: torch.Tensor,
    ) -> torch.Tensor:
        noise_scale = escala_ruido[0]
        noise_scale_w = escala_ruido[1]
        length_scale = escala_ruido[2]
        x, m_p, logs_p, x_mask = self.enc_p(tokens, longitud_tokens, idioma)
        g = embedding.unsqueeze(-1)
        logw = self.dp(
            x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w, noise=ruido_duracion
        )
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        attn = generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        t_y = m_p.size(2)
        frames = ruido_flow.size(2)
        repeticiones = (t_y + frames - 1) // frames
        ruido = ruido_flow.repeat(1, 1, repeticiones)[:, :, :t_y]
        z_p = m_p + ruido * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        return self.dec(z * y_mask, g=g)

    def preparar_export(self) -> Sintetizador:
        self.eval()
        remove_weight_norm(self)
        return self


class SintetizadorExport(nn.Module):
    """`forward` = `inferir`, so torch.onnx.export sees the contract directly."""

    def __init__(self, modelo: Sintetizador) -> None:
        super().__init__()
        self.modelo = modelo

    def forward(
        self, tokens, longitud_tokens, embedding, idioma, ruido_flow, ruido_duracion, escala_ruido
    ):
        return self.modelo.inferir(
            tokens, longitud_tokens, embedding, idioma, ruido_flow, ruido_duracion, escala_ruido
        )
