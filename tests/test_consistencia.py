"""The speaker-consistency loss must be zero for identical audio, positive for
different speakers, and differentiable back to the generator."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="extra `train` no instalado")
pytest.importorskip("torchaudio")
pytest.importorskip("huggingface_hub", reason="extra `exp` no instalado")

from ttspro.train.consistencia import ConsistenciaLocutor  # noqa: E402


@pytest.fixture(scope="module")
def scl() -> ConsistenciaLocutor:
    return ConsistenciaLocutor()


def _voz(semilla: int, muestras: int = 8192) -> torch.Tensor:
    """A crude periodic signal: not speech, but two seeds give two timbres."""
    g = torch.Generator().manual_seed(semilla)
    t = torch.arange(muestras, dtype=torch.float32) / 22050
    f0 = 90 + 120 * torch.rand(1, generator=g).item()
    onda = sum(torch.sin(2 * torch.pi * f0 * k * t) / k for k in range(1, 8))
    ruido = torch.randn(muestras, generator=g) * 0.02
    return ((onda / onda.abs().max()) * 0.5 + ruido).reshape(1, 1, muestras)


def test_identico_da_cero(scl) -> None:
    y = _voz(0)
    with torch.no_grad():
        assert float(scl(y, y)) < 1e-4


def test_distinto_da_mas_que_identico(scl) -> None:
    y, otro = _voz(0), _voz(7)
    with torch.no_grad():
        assert float(scl(otro, y)) > float(scl(y, y)) + 1e-3


def test_el_gradiente_llega_a_la_onda_generada(scl) -> None:
    y = _voz(0)
    y_hat = _voz(7).clone().requires_grad_(True)
    perdida = scl(y_hat, y)
    perdida.backward()
    assert y_hat.grad is not None
    assert torch.isfinite(y_hat.grad).all()
    assert float(y_hat.grad.abs().max()) > 0


def test_el_encoder_esta_congelado(scl) -> None:
    assert all(not p.requires_grad for p in scl.encoder.parameters())
    assert not scl.encoder.training
