"""The synthesizer trains, infers, exports and runs in onnxruntime — on a tiny
config, so the whole thing fits in a few seconds and runs in the fast suite."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="extra `train` no instalado")

import numpy as np  # noqa: E402

from ttspro.model.alineacion import maximum_path  # noqa: E402
from ttspro.model.config import config_pequena  # noqa: E402
from ttspro.model.sintetizador import Sintetizador, SintetizadorExport  # noqa: E402


def _entradas_inferencia(cfg, longitud: int = 7, frames_ruido: int = 3):
    torch.manual_seed(0)
    return (
        torch.randint(1, cfg.n_symbols, (1, longitud)),
        torch.tensor([longitud]),
        torch.nn.functional.normalize(torch.randn(1, cfg.gin_channels), dim=1),
        torch.tensor([1]),
        torch.randn(1, cfg.inter_channels, frames_ruido),
        torch.randn(1, 2, longitud),
        torch.tensor([0.667, 0.8, 1.0]),
    )


def _mas_lento(neg_cent: np.ndarray, t_x: int, t_y: int) -> np.ndarray:
    """The reference O(t_x * t_y) dynamic programme, cell by cell."""
    value = np.full((t_x, t_y), -np.inf)
    value[0, 0] = neg_cent[0, 0]
    for y in range(1, t_y):
        for x in range(0, min(t_x, y + 1)):
            stay = value[x, y - 1]
            advance = value[x - 1, y - 1] if x > 0 else -np.inf
            value[x, y] = max(stay, advance) + neg_cent[x, y]
    path = np.zeros((t_x, t_y), dtype=np.int64)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index -= 1
    return path


def test_alineacion_coincide_con_la_referencia_lenta() -> None:
    torch.manual_seed(1)
    b, t_x, t_y = 3, 6, 15
    neg_cent = torch.randn(b, t_x, t_y)
    t_xs, t_ys = [6, 4, 5], [15, 9, 12]
    mask = torch.zeros(b, t_x, t_y)
    for i in range(b):
        mask[i, : t_xs[i], : t_ys[i]] = 1
    path = maximum_path(neg_cent, mask)
    for i in range(b):
        esperado = _mas_lento(neg_cent[i].numpy(), t_xs[i], t_ys[i])
        assert np.array_equal(path[i, : t_xs[i], : t_ys[i]].numpy(), esperado), i
        assert path[i].sum() == t_ys[i]  # one text token per frame
        assert (path[i, : t_xs[i], : t_ys[i]].sum(1) >= 1).all()  # every token gets a frame


def test_forward_de_entrenamiento_devuelve_las_formas_esperadas() -> None:
    cfg = config_pequena()
    modelo = Sintetizador(cfg).train()
    b, t_x, t_y = 2, 9, 40
    tokens = torch.randint(1, cfg.n_symbols, (b, t_x))
    tokens_len = torch.tensor([9, 6])
    spec = torch.rand(b, cfg.spec_channels, t_y)
    spec_len = torch.tensor([40, 30])
    emb = torch.nn.functional.normalize(torch.randn(b, cfg.gin_channels), dim=1)
    idioma = torch.tensor([0, 1])
    o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = modelo(
        tokens, tokens_len, spec, spec_len, emb, idioma
    )
    assert o.shape == (b, 1, cfg.segment_size)
    assert l_length.shape == (b,)
    assert attn.shape == (b, 1, t_y, t_x)  # VITS convention: frames x tokens
    assert torch.equal(attn.sum(2).squeeze(1).sum(1), spec_len.float())  # every frame assigned
    assert z.shape == z_p.shape == m_p.shape == logs_p.shape == (b, cfg.inter_channels, t_y)
    assert torch.isfinite(o).all() and torch.isfinite(l_length).all()
    (o.sum() + l_length.sum()).backward()  # the graph is differentiable end to end


def test_inferir_produce_onda_y_respeta_la_escala_de_duracion() -> None:
    cfg = config_pequena()
    modelo = Sintetizador(cfg).eval()
    entradas = _entradas_inferencia(cfg)
    with torch.no_grad():
        onda = modelo.inferir(*entradas)
        lenta = modelo.inferir(*entradas[:-1], torch.tensor([0.667, 0.8, 2.0]))
    assert onda.shape[:2] == (1, 1)
    assert onda.shape[2] % cfg.hop_length == 0
    assert lenta.shape[2] > onda.shape[2]


def test_export_onnx_corre_en_ort_y_solo_usa_ops_de_webgpu(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from ttspro.export.ops_ort_web import ops_fuera_de_webgpu

    cfg = config_pequena()
    modelo = Sintetizador(cfg).preparar_export()
    entradas = _entradas_inferencia(cfg, longitud=7, frames_ruido=3)
    nombres = [
        "tokens",
        "longitud_tokens",
        "embedding",
        "idioma",
        "ruido_flow",
        "ruido_duracion",
        "escala_ruido",
    ]
    ruta = tmp_path / "tts.onnx"
    torch.onnx.export(
        SintetizadorExport(modelo).eval(),
        entradas,
        str(ruta),
        opset_version=17,
        input_names=nombres,
        output_names=["onda"],
        dynamic_axes={
            "tokens": {1: "longitud"},
            "ruido_flow": {2: "frames"},
            "ruido_duracion": {2: "longitud"},
            "onda": {2: "muestras"},
        },
        dynamo=False,
    )
    m = onnx.load(str(ruta))
    onnx.checker.check_model(m)
    assert ops_fuera_de_webgpu(m) == {}, ops_fuera_de_webgpu(m)

    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        esperado = modelo.inferir(*entradas).numpy()
    (salida,) = sesion.run(None, {n: e.numpy() for n, e in zip(nombres, entradas, strict=True)})
    assert salida.shape == esperado.shape
    assert (
        np.abs(salida - esperado).max() < 1e-2
    )  # fp32 both sides; ORT fuses differently and the graph is deep

    # a different, longer sentence with even shorter noise: the graph is truly dynamic
    otras = _entradas_inferencia(cfg, longitud=13, frames_ruido=2)
    with torch.no_grad():
        esperado2 = modelo.inferir(*otras).numpy()
    (salida2,) = sesion.run(None, {n: e.numpy() for n, e in zip(nombres, otras, strict=True)})
    assert salida2.shape == esperado2.shape
    assert (
        np.abs(salida2 - esperado2).max() < 1e-2
    )  # fp32 both sides; ORT fuses differently and the graph is deep
