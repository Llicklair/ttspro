"""Log-mel filterbank with the STFT expressed as a strided Conv1d (ADR 0002).

ORT Web's WebGPU provider has no `STFT` op, so the DFT is a fixed-kernel
convolution and the mel projection a MatMul: ops every provider runs (rule 4).
Parameters default to what speechbrain's ECAPA-TDNN was trained on
(16 kHz, n_fft 400, hop 160, hamming, 80 mels 0-8 kHz, power spectrum, dB with
top_db 80, per-utterance mean subtraction); the export script checks this
module against speechbrain's own pipeline numerically before exporting.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _hz_a_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_a_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def banco_mel_triangular(
    n_fft: int, n_mels: int, sample_rate: int, f_min: float, f_max: float
) -> torch.Tensor:
    """(n_fft//2 + 1, n_mels) triangular filters on the mel scale, speechbrain style."""
    freqs = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1)
    puntos_mel = torch.linspace(
        _hz_a_mel(torch.tensor(f_min)), _hz_a_mel(torch.tensor(f_max)), n_mels + 2
    )
    puntos_hz = _mel_a_hz(puntos_mel)
    banda = puntos_hz[1:] - puntos_hz[:-1]  # (n_mels + 1,)
    f_central = puntos_hz[1:-1]  # (n_mels,)
    pendiente = (freqs[:, None] - f_central[None, :]) / banda[:-1]  # (F, n_mels)
    izquierda = pendiente + 1.0
    derecha = -pendiente * (banda[:-1] / banda[1:]) + 1.0
    return torch.clamp(torch.min(izquierda, derecha), min=0.0)


class FbankConv(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        amin: float = 1e-10,
        top_db: float = 80.0,
        media_por_frase: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.amin = amin
        self.top_db = top_db
        self.media_por_frase = media_por_frase
        ventana = torch.hamming_window(win_length)  # periodic, as torch.stft/speechbrain
        if win_length < n_fft:
            ventana = F.pad(
                ventana, ((n_fft - win_length) // 2, n_fft - win_length - (n_fft - win_length) // 2)
            )
        n = torch.arange(n_fft, dtype=torch.float32)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
        angulo = 2.0 * math.pi * k[:, None] * n[None, :] / n_fft
        nucleo = torch.cat([torch.cos(angulo), -torch.sin(angulo)], dim=0) * ventana[None, :]
        self.register_buffer("nucleo", nucleo.unsqueeze(1))  # (2F, 1, n_fft)
        self.register_buffer(
            "banco", banco_mel_triangular(n_fft, n_mels, sample_rate, f_min, f_max).T
        )  # (n_mels, F)

    def forward(self, onda: torch.Tensor) -> torch.Tensor:
        """(B, muestras) in [-1, 1] -> (B, frames, n_mels) log-mel, mean-normalized."""
        x = F.pad(onda.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2))  # center=True, constant
        espectro = F.conv1d(x, self.nucleo, stride=self.hop_length)  # (B, 2F, frames)
        mitad = espectro.shape[1] // 2
        potencia = espectro[:, :mitad] ** 2 + espectro[:, mitad:] ** 2  # (B, F, frames)
        mel = torch.matmul(self.banco, potencia)  # (B, n_mels, frames)
        db = 10.0 * torch.log10(torch.clamp(mel, min=self.amin))
        techo = db.amax(dim=(1, 2), keepdim=True) - self.top_db
        # `torch.maximum` exports as ONNX `Max`, which WebGPU does not run; `Where` it does.
        db = torch.where(db < techo, techo, db)
        feats = db.transpose(1, 2)  # (B, frames, n_mels)
        if self.media_por_frase:
            feats = feats - feats.mean(dim=1, keepdim=True)
        return feats


def _banco_mel_kaldi(
    n_fft: int, n_mels: int, sample_rate: int, f_min: float, f_max: float
) -> torch.Tensor:
    """(n_mels, n_fft//2 + 1) Kaldi-style mel filters (natural-log mel scale, no
    Nyquist bin), as torchaudio.compliance.kaldi.get_mel_banks builds them."""
    mel = lambda hz: 1127.0 * math.log(1.0 + hz / 700.0)  # noqa: E731
    mel_lo, mel_hi = mel(f_min), mel(f_max)
    delta = (mel_hi - mel_lo) / (n_mels + 1)
    bins = torch.arange(n_mels, dtype=torch.float32)
    izq, centro, der = (
        mel_lo + bins * delta,
        mel_lo + (bins + 1) * delta,
        mel_lo + (bins + 2) * delta,
    )
    fft_hz = torch.arange(n_fft // 2, dtype=torch.float32) * (sample_rate / n_fft)
    fft_mel = 1127.0 * torch.log1p(fft_hz / 700.0)  # (F-1,)
    subida = (fft_mel[None, :] - izq[:, None]) / (centro - izq)[:, None]
    bajada = (der[:, None] - fft_mel[None, :]) / (der - centro)[:, None]
    banco = torch.clamp(torch.min(subida, bajada), min=0.0)  # (n_mels, F-1)
    return F.pad(banco, (0, 1))  # Nyquist column is zero


class FbankKaldiConv(nn.Module):
    """Kaldi fbank (what WeSpeaker/wenet models eat) as one strided Conv1d.

    Matches ``torchaudio.compliance.kaldi.fbank(onda * 32768, num_mel_bins=80,
    frame_length=25, frame_shift=10, dither=0, window_type="hamming",
    use_energy=False)`` followed by per-utterance mean subtraction. Per-frame
    DC removal, pre-emphasis (0.97, first sample against itself), the
    non-periodic Hamming window and the 512-point DFT are all linear in the
    400-sample frame, so they fold into a single (2·257, 400) kernel.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 400,
        frame_shift: int = 160,
        n_fft: int = 512,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        preemphasis: float = 0.97,
        escala: float = 32768.0,
    ) -> None:
        super().__init__()
        self.frame_shift = frame_shift
        self.escala = escala
        self.eps = torch.finfo(torch.float32).eps
        n = frame_length
        quita_dc = torch.eye(n) - torch.full((n, n), 1.0 / n)
        preenfasis = torch.eye(n)
        preenfasis[torch.arange(1, n), torch.arange(0, n - 1)] = -preemphasis
        preenfasis[0, 0] = 1.0 - preemphasis
        ventana = torch.diag(torch.hamming_window(n, periodic=False))
        t = torch.arange(n, dtype=torch.float32)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
        angulo = 2.0 * math.pi * k[:, None] * t[None, :] / n_fft
        dft = torch.cat(
            [torch.cos(angulo), -torch.sin(angulo)], dim=0
        )  # (2F, n) on the unpadded frame
        nucleo = dft @ ventana @ preenfasis @ quita_dc
        self.register_buffer("nucleo", nucleo.unsqueeze(1))  # (2F, 1, n)
        self.register_buffer(
            "banco", _banco_mel_kaldi(n_fft, n_mels, sample_rate, f_min, f_max)
        )  # (n_mels, F)

    def forward(self, onda: torch.Tensor) -> torch.Tensor:
        """(B, muestras) in [-1, 1] -> (B, frames, n_mels), snip_edges (no padding)."""
        x = (onda * self.escala).unsqueeze(1)
        espectro = F.conv1d(x, self.nucleo, stride=self.frame_shift)  # (B, 2F, frames)
        mitad = espectro.shape[1] // 2
        potencia = espectro[:, :mitad] ** 2 + espectro[:, mitad:] ** 2
        mel = torch.matmul(self.banco, potencia)  # (B, n_mels, frames)
        logmel = torch.log(torch.where(mel < self.eps, torch.full_like(mel, self.eps), mel))
        feats = logmel.transpose(1, 2)
        return feats - feats.mean(dim=1, keepdim=True)
