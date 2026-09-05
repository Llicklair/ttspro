"""Tone-colour converter: base audio + target voice -> the same words in that voice.

Cloning as post-processing (ADR 0007): a base TTS says the sentence in whatever
voice it has, and this model repaints the timbre. It is the OpenVoice v2
converter (MIT, myshell-ai/OpenVoiceV2), reimplemented on the blocks this repo
already has — its `enc_q`, `flow` and `dec` are, tensor for tensor, our
PosteriorEncoder, ResidualCouplingBlock and Generator — plus a reference encoder
that turns a spectrogram into a 256-d voice vector.

The maths, once the weights are in place:

    z      = enc_q(spec)                 # what is said, plus the source timbre
    z_p    = flow(z,   g = voz_origen)   # remove the source timbre
    z_hat  = flow(z_p, g = voz_destino, reverse=True)   # paint the target one
    onda   = dec(z_hat)

`zero_g` (true in the published config) means the posterior encoder and the
decoder get NO conditioning: all the identity lives in the flow. Everything runs
at 22 050 Hz with n_fft 1024 and hop 256 — the same settings as our synthesizer,
so its output feeds straight in.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from ttspro.model.flow import ResidualCouplingBlock
from ttspro.model.generador import Generator
from ttspro.model.posterior import PosteriorEncoder

RAIZ = Path(__file__).resolve().parents[3]
REPO_OPENVOICE = "myshell-ai/OpenVoiceV2"


class ReferenceEncoder(nn.Module):
    """Linear spectrogram -> 256-d voice vector. Six strided 2-D convolutions, a
    GRU over time and a projection. It runs ONCE per voice, not per sentence."""

    def __init__(self, spec_channels: int = 513, gin_channels: int = 256) -> None:
        super().__init__()
        self.spec_channels = spec_channels
        filtros = [1, 32, 32, 64, 64, 128, 128]
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(filtros[i], filtros[i + 1], (3, 3), (2, 2), (1, 1)))
                for i in range(6)
            ]
        )
        canales = spec_channels
        for _ in range(6):
            canales = (canales - 3 + 2) // 2 + 1
        self.gru = nn.GRU(input_size=128 * canales, hidden_size=128, batch_first=True)
        self.proj = nn.Linear(128, gin_channels)
        self.layernorm = nn.LayerNorm(spec_channels)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """(B, 513, T) -> (B, 256, 1), ready to condition the flow."""
        x = self.layernorm(spec.transpose(1, 2)).unsqueeze(1)  # (B, 1, T, 513)
        for conv in self.convs:
            x = F.relu(conv(x))
        x = x.transpose(1, 2).contiguous()  # (B, T', 128, F')
        x = x.view(x.size(0), x.size(1), -1)
        _, estado = self.gru(x)
        return self.proj(estado.squeeze(0)).unsqueeze(-1)


class Conversor(nn.Module):
    def __init__(
        self,
        spec_channels: int = 513,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        gin_channels: int = 256,
        upsample_rates: tuple[int, ...] = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        zero_g: bool = True,
    ) -> None:
        super().__init__()
        self.zero_g = zero_g
        self.inter_channels = inter_channels
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, n_flows=4, gin_channels=gin_channels
        )
        self.dec = Generator(
            inter_channels,
            list(resblock_kernel_sizes),
            [list(d) for d in resblock_dilation_sizes],
            list(upsample_rates),
            upsample_initial_channel,
            list(upsample_kernel_sizes),
            gin_channels=gin_channels,
        )
        self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)

    def voz(self, spec: torch.Tensor) -> torch.Tensor:
        """Spectrogram of a reference recording -> (B, 256, 1)."""
        return self.ref_enc(spec)

    def forward(
        self,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        voz_origen: torch.Tensor,
        voz_destino: torch.Tensor,
        ruido: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        cero = torch.zeros_like(voz_origen)
        z, _, _, y_mask = self.enc_q(
            spec,
            spec_lengths,
            g=cero if self.zero_g else voz_origen,
            ruido=ruido,
            tau=tau,
        )
        z_p = self.flow(z, y_mask, g=voz_origen)
        z_hat = self.flow(z_p, y_mask, g=voz_destino, reverse=True)
        return self.dec(z_hat * y_mask, g=cero if self.zero_g else voz_destino)


def _traducir(clave: str) -> str:
    if clave.endswith(".weight_g"):
        return clave[: -len(".weight_g")] + ".parametrizations.weight.original0"
    if clave.endswith(".weight_v"):
        return clave[: -len(".weight_v")] + ".parametrizations.weight.original1"
    return clave


def cargar(carpeta: Path | None = None) -> tuple[Conversor, dict]:
    """Download (once) and load the published converter. Returns (model, report)."""
    if carpeta is None:
        from huggingface_hub import hf_hub_download

        cache = str(RAIZ / ".scratch" / "openvoice" / "hf")
        cfg_ruta = hf_hub_download(REPO_OPENVOICE, "converter/config.json", cache_dir=cache)
        ck_ruta = hf_hub_download(REPO_OPENVOICE, "converter/checkpoint.pth", cache_dir=cache)
    else:
        cfg_ruta, ck_ruta = carpeta / "config.json", carpeta / "checkpoint.pth"
    cfg = json.loads(Path(cfg_ruta).read_text(encoding="utf-8"))
    m = cfg["model"]
    modelo = Conversor(
        spec_channels=cfg["data"]["filter_length"] // 2 + 1,
        inter_channels=m["inter_channels"],
        hidden_channels=m["hidden_channels"],
        gin_channels=m["gin_channels"],
        upsample_rates=tuple(m["upsample_rates"]),
        upsample_initial_channel=m["upsample_initial_channel"],
        upsample_kernel_sizes=tuple(m["upsample_kernel_sizes"]),
        resblock_kernel_sizes=tuple(m["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(d) for d in m["resblock_dilation_sizes"]),
        zero_g=bool(m.get("zero_g", False)),
    )
    suyo = torch.load(ck_ruta, map_location="cpu", weights_only=False)
    suyo = suyo.get("model", suyo)
    mio = modelo.state_dict()
    traducido, sin_destino = {}, []
    for k, v in suyo.items():
        nueva = _traducir(k)
        if nueva in mio and tuple(mio[nueva].shape) == tuple(v.shape):
            traducido[nueva] = v
        else:
            sin_destino.append(f"{k} -> {nueva}")
    faltan = sorted(set(mio) - set(traducido))
    modelo.load_state_dict(traducido, strict=False)
    informe = {
        "licencia": "MIT (myshell-ai/OpenVoiceV2)",
        "tensores_suyos": len(suyo),
        "cargados": len(traducido),
        "mios_sin_cargar": faltan,
        "suyos_sin_destino": sin_destino,
        "data": cfg["data"],
        "zero_g": modelo.zero_g,
    }
    return modelo.eval(), informe
