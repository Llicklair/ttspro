"""The PyTorch WeSpeaker must give the same embedding as the exported ONNX:
otherwise the speaker-consistency loss would push the model toward a different
speaker space than the one the browser uses."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="extra `train` no instalado")
np = pytest.importorskip("numpy")

RAIZ = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (RAIZ / "models" / "speaker_encoder.onnx").exists(),
    reason="exporta primero: python -m ttspro.export.speaker_encoder",
)
def test_wespeaker_torch_coincide_con_el_onnx_exportado() -> None:
    ort = pytest.importorskip("onnxruntime", reason="extra `export` no instalado")
    pytest.importorskip("huggingface_hub", reason="extra `exp` no instalado")

    from ttspro.model.fbank import FbankKaldiConv
    from ttspro.model.wespeaker import cargar

    torch.manual_seed(0)
    onda = (torch.randn(1, 16000 * 4) * 0.3).clamp(-1, 1)
    with torch.no_grad():
        emb_torch = torch.nn.functional.normalize(cargar()(FbankKaldiConv().eval()(onda)), dim=1)

    sesion = ort.InferenceSession(
        str(RAIZ / "models" / "speaker_encoder.onnx"), providers=["CPUExecutionProvider"]
    )
    (emb_onnx,) = sesion.run(None, {sesion.get_inputs()[0].name: onda.numpy()})
    coseno = float(np.dot(emb_onnx[0], emb_torch[0].numpy()))
    assert coseno > 0.999, coseno
