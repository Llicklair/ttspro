"""Synthesizer hyperparameters. One dataclass, no YAML: the export script prints
the resulting size and that number, not a recipe, decides the values (ADR 0005:
~37 MB fp16 left for this graph)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConfigSintetizador:
    # text
    n_symbols: int = 180
    n_langs: int = 2
    # audio (VITS/LJSpeech defaults)
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float | None = None
    segment_size: int = 8192
    # speaker embedding coming from speaker_encoder.onnx (contract: 256)
    gin_channels: int = 256
    # Sizes below VITS base (enc 6 layers, WN 4 layers, decoder 512) on purpose:
    # measured 2026-09-04, base exports to 62 MB fp16 and the budget left for
    # this graph is ~37 MB (ADR 0005). This config: 15.97 M params, ~32 MB.
    # text encoder
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 4
    kernel_size: int = 3
    p_dropout: float = 0.1
    window_size: int = 4
    # latent / flow / posterior
    inter_channels: int = 192
    flow_n_flows: int = 4
    flow_kernel: int = 5
    flow_dilation: int = 1
    flow_layers: int = 3
    posterior_kernel: int = 5
    posterior_layers: int = 16
    # HiFi-GAN decoder
    upsample_initial_channel: int = 256
    upsample_rates: list[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 4, 4])
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    # stochastic duration predictor
    sdp_filter_channels: int = 192
    sdp_kernel: int = 3
    sdp_dropout: float = 0.5
    sdp_flows: int = 4

    @property
    def spec_channels(self) -> int:
        return self.n_fft // 2 + 1


def config_pequena() -> ConfigSintetizador:
    """For tests: every block present, every dimension tiny."""
    return ConfigSintetizador(
        n_symbols=32,
        hidden_channels=16,
        filter_channels=32,
        n_heads=2,
        n_layers=2,
        inter_channels=16,
        gin_channels=8,
        flow_layers=2,
        posterior_layers=2,
        upsample_initial_channel=32,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3]],
        sdp_filter_channels=16,
        sdp_flows=2,
        segment_size=2048,
    )
