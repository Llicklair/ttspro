"""Linear and mel spectrograms for training (VITS mel_processing). Uses
torch.stft freely: this never enters the exported graph (rule 7)."""

from __future__ import annotations

import torch
import torchaudio

_ventanas: dict[tuple, torch.Tensor] = {}
_bancos: dict[tuple, torch.Tensor] = {}


def _ventana(win_length: int, device, dtype) -> torch.Tensor:
    clave = (win_length, str(device), dtype)
    if clave not in _ventanas:
        _ventanas[clave] = torch.hann_window(win_length, device=device, dtype=dtype)
    return _ventanas[clave]


def _banco_mel(
    n_fft: int, n_mels: int, sample_rate: int, fmin: float, fmax: float | None, device, dtype
) -> torch.Tensor:
    clave = (n_fft, n_mels, sample_rate, fmin, fmax, str(device), dtype)
    if clave not in _bancos:
        banco = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=fmin,
            f_max=fmax if fmax is not None else sample_rate / 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )  # (n_freqs, n_mels), same as librosa.filters.mel
        _bancos[clave] = banco.T.to(device=device, dtype=dtype)
    return _bancos[clave]


def spectrogram_torch(
    y: torch.Tensor, n_fft: int, hop_length: int, win_length: int
) -> torch.Tensor:
    """(b, muestras) -> (b, n_fft//2 + 1, frames) linear magnitude."""
    if y.dim() == 3:
        y = y.squeeze(1)
    pad = (n_fft - hop_length) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=_ventana(win_length, y.device, y.dtype),
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.sqrt(spec.real**2 + spec.imag**2 + 1e-6)


def spec_to_mel_torch(
    spec: torch.Tensor, n_fft: int, n_mels: int, sample_rate: int, fmin: float, fmax: float | None
) -> torch.Tensor:
    banco = _banco_mel(n_fft, n_mels, sample_rate, fmin, fmax, spec.device, spec.dtype)
    return torch.log(torch.clamp(torch.matmul(banco, spec), min=1e-5))


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    fmin: float,
    fmax: float | None,
) -> torch.Tensor:
    return spec_to_mel_torch(
        spectrogram_torch(y, n_fft, hop_length, win_length), n_fft, n_mels, sample_rate, fmin, fmax
    )
