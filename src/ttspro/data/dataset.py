"""Dataset over the index written by `ttspro.data.preparar`, with length-bucketed batches."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler

from ttspro.model.config import ConfigSintetizador
from ttspro.train.mel import spectrogram_torch


class DatasetTTS(Dataset):
    def __init__(self, indice: list[dict] | Path, cfg: ConfigSintetizador) -> None:
        self.indice = torch.load(indice) if isinstance(indice, (str, Path)) else indice
        self.cfg = cfg
        self.idiomas = {"es": 0, "en": 1}

    def __len__(self) -> int:
        return len(self.indice)

    def frames(self, i: int) -> int:
        return int(self.indice[i]["segundos"] * self.cfg.sample_rate / self.cfg.hop_length)

    def __getitem__(self, i: int):
        e = self.indice[i]
        onda, sr = sf.read(e["wav"], dtype="float32", always_2d=True)
        onda = torch.from_numpy(onda.mean(axis=1))
        if sr != self.cfg.sample_rate:
            onda = torchaudio.functional.resample(onda, sr, self.cfg.sample_rate)
        spec = spectrogram_torch(
            onda.unsqueeze(0), self.cfg.n_fft, self.cfg.hop_length, self.cfg.win_length
        ).squeeze(0)
        return {
            "tokens": e["tokens"],
            "spec": spec,
            "onda": onda.unsqueeze(0),
            "embedding": e["embedding"],
            "idioma": self.idiomas[e["idioma"]],
        }


def collate(lote: list[dict]) -> dict:
    b = len(lote)
    max_tok = max(x["tokens"].numel() for x in lote)
    max_spec = max(x["spec"].shape[1] for x in lote)
    max_onda = max(x["onda"].shape[1] for x in lote)
    n_freqs = lote[0]["spec"].shape[0]
    tokens = torch.zeros(b, max_tok, dtype=torch.int64)
    spec = torch.zeros(b, n_freqs, max_spec)
    onda = torch.zeros(b, 1, max_onda)
    tokens_len = torch.zeros(b, dtype=torch.int64)
    spec_len = torch.zeros(b, dtype=torch.int64)
    onda_len = torch.zeros(b, dtype=torch.int64)
    for i, x in enumerate(lote):
        tokens[i, : x["tokens"].numel()] = x["tokens"]
        tokens_len[i] = x["tokens"].numel()
        spec[i, :, : x["spec"].shape[1]] = x["spec"]
        spec_len[i] = x["spec"].shape[1]
        onda[i, :, : x["onda"].shape[1]] = x["onda"]
        onda_len[i] = x["onda"].shape[1]
    return {
        "tokens": tokens,
        "tokens_len": tokens_len,
        "spec": spec,
        "spec_len": spec_len,
        "onda": onda,
        "onda_len": onda_len,
        "embedding": torch.stack([x["embedding"] for x in lote]),
        "idioma": torch.tensor([x["idioma"] for x in lote], dtype=torch.int64),
    }


class LotesPorLongitud(Sampler[list[int]]):
    """Batches of similar length (less padding), shuffled every epoch."""

    def __init__(self, dataset: DatasetTTS, tam_lote: int, semilla: int = 0) -> None:
        self.frames = np.array([dataset.frames(i) for i in range(len(dataset))])
        self.tam_lote = tam_lote
        self.rng = random.Random(semilla)
        self.epoca = 0

    def __iter__(self):
        ruido = np.array([self.rng.random() for _ in self.frames]) * 0.1 * self.frames
        orden = np.argsort(self.frames + ruido)
        lotes = [orden[i : i + self.tam_lote].tolist() for i in range(0, len(orden), self.tam_lote)]
        lotes = [lote for lote in lotes if len(lote) == self.tam_lote]
        self.rng.shuffle(lotes)
        self.epoca += 1
        return iter(lotes)

    def __len__(self) -> int:
        return len(self.frames) // self.tam_lote
