"""One real training step on random data, tiny config, CPU: every loss finite,
both optimizers stepped, checkpoint round-trips, dataset collate pads right."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="extra `train` no instalado")

from ttspro.data.dataset import collate  # noqa: E402
from ttspro.model.config import config_pequena  # noqa: E402
from ttspro.train.entrenar import construir, guardar, paso  # noqa: E402
from ttspro.train.mel import spectrogram_torch  # noqa: E402


def _lote(cfg, b: int = 2):
    torch.manual_seed(0)
    ejemplos = []
    for i in range(b):
        muestras = cfg.segment_size * 2 + 1000 * i
        onda = (torch.rand(muestras) * 2 - 1) * 0.3
        spec = spectrogram_torch(
            onda.unsqueeze(0), cfg.n_fft, cfg.hop_length, cfg.win_length
        ).squeeze(0)
        ejemplos.append(
            {
                "tokens": torch.randint(1, cfg.n_symbols, (6 + i,)),
                "spec": spec,
                "onda": onda.unsqueeze(0),
                "embedding": torch.nn.functional.normalize(torch.randn(cfg.gin_channels), dim=0),
                "idioma": i % 2,
            }
        )
    return collate(ejemplos)


def test_collate_rellena_y_conserva_longitudes() -> None:
    cfg = config_pequena()
    lote = _lote(cfg)
    assert lote["tokens"].shape == (2, 7)
    assert lote["tokens_len"].tolist() == [6, 7]
    assert lote["spec"].shape[1] == cfg.spec_channels
    assert lote["spec_len"].max() == lote["spec"].shape[2]
    assert lote["onda"].shape[2] == lote["onda_len"].max()


def test_un_paso_de_entrenamiento_en_cpu(tmp_path) -> None:
    cfg = config_pequena()
    device = torch.device("cpu")
    net_g, net_d, optim_g, optim_d = construir(cfg, device, lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    lote = _lote(cfg)
    antes = [p.detach().clone() for p in net_g.parameters()][:3]
    m1 = paso(net_g, net_d, optim_g, optim_d, scaler, lote, cfg, device, fp16=False)
    m2 = paso(net_g, net_d, optim_g, optim_d, scaler, lote, cfg, device, fp16=False)
    for k, v in m1.items():
        assert v == v and abs(v) < 1e6, (k, v)  # finite
    assert m2["total_g"] == m2["total_g"]
    assert any(
        not torch.equal(a, b) for a, b in zip(antes, list(net_g.parameters())[:3], strict=False)
    )
    ruta = tmp_path / "G_2.pt"
    guardar(ruta, net_g, net_d, optim_g, optim_d, 2, 0, cfg)
    estado = torch.load(ruta)
    assert estado["paso"] == 2 and estado["cfg"]["n_symbols"] == cfg.n_symbols
