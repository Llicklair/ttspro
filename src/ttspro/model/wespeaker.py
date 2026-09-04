"""WeSpeaker ResNet34-TSTP in PyTorch, so a loss can flow through the speaker
encoder (ADR 0002 exports the ONNX; this is the same network, differentiable).

Reimplemented from the published architecture and loaded from the official
`avg_model` checkpoint (`Wespeaker/wespeaker-voxceleb-resnet34-LM`, CC-BY-4.0),
integrated by reference (rule 11). `ttspro.export.speaker_encoder` still composes
the published ONNX for inference; this module exists for training, and
`tests/test_wespeaker.py` holds both to the same numbers.

Shapes: features (B, T, 80) -> permute to (B, 80, T) -> unsqueeze channel ->
ResNet34 with strides (1, 2, 2, 2) -> (B, 256, 10, T/8) -> mean and std over
time -> (B, 5120) -> linear -> (B, 256).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet34TSTP(nn.Module):
    def __init__(self, m_channels: int = 32, feat_dim: int = 80, embed_dim: int = 256) -> None:
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._capa(m_channels, 3, 1)
        self.layer2 = self._capa(m_channels * 2, 4, 2)
        self.layer3 = self._capa(m_channels * 4, 6, 2)
        self.layer4 = self._capa(m_channels * 8, 3, 2)
        stats_dim = (feat_dim // 8) * m_channels * 8
        self.seg_1 = nn.Linear(stats_dim * 2, embed_dim)

    def _capa(self, planes: int, bloques: int, stride: int) -> nn.Sequential:
        capas = []
        for s in [stride] + [1] * (bloques - 1):
            capas.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*capas)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, T, 80) log-mel (Kaldi fbank, mean-normalized) -> (B, 256)."""
        x = feats.permute(0, 2, 1).unsqueeze(1)  # (B, 1, 80, T)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        media = out.mean(dim=-1)
        desviacion = torch.sqrt(out.var(dim=-1) + 1e-7)
        stats = torch.cat([media.flatten(1), desviacion.flatten(1)], dim=1)
        return self.seg_1(stats)


def cargar(checkpoint: Path | str | None = None) -> ResNet34TSTP:
    """Load the official checkpoint (downloads it once through huggingface_hub)."""
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        raiz = Path(__file__).resolve().parents[3]
        checkpoint = hf_hub_download(
            "Wespeaker/wespeaker-voxceleb-resnet34-LM",
            "avg_model",
            cache_dir=str(raiz / ".scratch" / "wespeaker-resnet34" / "hf"),
        )
    estado = torch.load(checkpoint, map_location="cpu", weights_only=False)
    estado = estado.get("state_dict", estado)
    estado = {k: v for k, v in estado.items() if not k.startswith("projection.")}
    modelo = ResNet34TSTP()
    modelo.load_state_dict(estado, strict=True)
    return modelo.eval()
