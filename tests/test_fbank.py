"""The Conv1d STFT equals torch.stft (what the backbones were trained on)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="extra `train` no instalado")

from ttspro.model.fbank import FbankConv, banco_mel_triangular  # noqa: E402


def _referencia(onda: torch.Tensor, fb: FbankConv) -> torch.Tensor:
    stft = torch.stft(
        onda,
        n_fft=fb.n_fft,
        hop_length=fb.hop_length,
        win_length=fb.n_fft,
        window=torch.hamming_window(fb.n_fft),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    potencia = stft.real**2 + stft.imag**2
    mel = torch.matmul(fb.banco, potencia)
    db = 10.0 * torch.log10(torch.clamp(mel, min=fb.amin))
    db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - fb.top_db)
    feats = db.transpose(1, 2)
    return feats - feats.mean(dim=1, keepdim=True)


def test_fbank_conv_coincide_con_torch_stft() -> None:
    torch.manual_seed(0)
    fb = FbankConv().eval()
    onda = torch.randn(1, 16000) * 0.3
    with torch.no_grad():
        nuestro = fb(onda)
    ref = _referencia(onda, fb)
    assert nuestro.shape == ref.shape == (1, 101, 80)
    assert (nuestro - ref).abs().max() < 1e-3


def test_fbank_kaldi_conv_coincide_con_torchaudio() -> None:
    torchaudio = pytest.importorskip("torchaudio")
    from ttspro.model.fbank import FbankKaldiConv

    torch.manual_seed(0)
    onda = torch.randn(1, 16000 * 2) * 0.3
    ref = torchaudio.compliance.kaldi.fbank(
        onda * 32768,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
        window_type="hamming",
        use_energy=False,
    )
    ref = ref - ref.mean(dim=0, keepdim=True)
    with torch.no_grad():
        nuestro = FbankKaldiConv().eval()(onda)[0]
    assert nuestro.shape == ref.shape == (198, 80)
    assert (nuestro - ref).abs().max() < 1e-2


def test_banco_mel_cubre_todas_las_frecuencias() -> None:
    banco = banco_mel_triangular(400, 80, 16000, 0.0, 8000.0)
    assert banco.shape == (201, 80)
    assert (banco >= 0).all()
    # every filter has some support, and the interior bins are covered
    assert (banco.sum(dim=0) > 0).all()
    assert (banco[1:-1].sum(dim=1) > 0).all()
