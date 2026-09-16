"""A voice, as two tensors.

This is the whole reason for ADR 0012. In the old chain the identity of a voice
lived in the weights of a converter that ran after the synthesizer and cost 2.9 s
and 63 MB; here it is an **input**:

    style_ttl  [1, 50, 256]   50 style tokens, read by the text encoder and by
                              every step of the vector estimator
    style_dp   [1, 8, 16]     the same identity for the duration predictor,
                              which is what makes a voice fast or slow

Nothing else distinguishes one speaker from another. That is why a voice weighs
292 KB of JSON instead of a checkpoint, why building one is a search problem
rather than a training problem (``ttspro.supertonic.constructor``), and why two
voices can be averaged into a third that has never existed.

The format is upstream's, so a style built here loads in any of the Supertonic
runtimes and a style from anywhere else loads here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


@dataclass
class Estilo:
    """One or more voices, batched on the first axis."""

    ttl: np.ndarray  # [B, 50, 256] float32
    dp: np.ndarray  # [B, 8, 16] float32
    nombres: list[str] = field(default_factory=list)
    metadatos: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ttl = np.ascontiguousarray(self.ttl, dtype=np.float32)
        self.dp = np.ascontiguousarray(self.dp, dtype=np.float32)
        if self.ttl.ndim != 3 or self.dp.ndim != 3:
            raise ValueError(
                "se esperaban dos tensores [B,50,256] y [B,8,16]; "
                f"llegaron {self.ttl.shape} y {self.dp.shape}"
            )
        if self.ttl.shape[0] != self.dp.shape[0]:
            raise ValueError(
                f"lotes distintos: {self.ttl.shape[0]} en ttl y {self.dp.shape[0]} en dp"
            )
        if not self.nombres:
            self.nombres = [f"voz{i}" for i in range(self.ttl.shape[0])]

    def __len__(self) -> int:
        return int(self.ttl.shape[0])

    def __getitem__(self, i: int) -> Estilo:
        return Estilo(
            self.ttl[i : i + 1], self.dp[i : i + 1], [self.nombres[i]], dict(self.metadatos)
        )

    @property
    def vector(self) -> np.ndarray:
        """The voice as one flat vector per item — what the builder searches over."""
        b = len(self)
        return np.concatenate([self.ttl.reshape(b, -1), self.dp.reshape(b, -1)], axis=1)

    @classmethod
    def desde_vector(cls, v: np.ndarray, nombres: list[str] | None = None) -> Estilo:
        v = np.atleast_2d(np.asarray(v, dtype=np.float32))
        b, corte = v.shape[0], 50 * 256
        if v.shape[1] != corte + 8 * 16:
            raise ValueError(f"un vector de voz mide {corte + 8 * 16}; llegaron {v.shape[1]}")
        return cls(v[:, :corte].reshape(b, 50, 256), v[:, corte:].reshape(b, 8, 16), nombres or [])

    @classmethod
    def cargar(cls, *rutas: Path | str) -> Estilo:
        """Load one or more ``*.json`` voices into a single batched style."""
        crudos = [json.loads(Path(r).read_text(encoding="utf-8")) for r in rutas]

        def apilar(clave: str) -> np.ndarray:
            return np.stack(
                [
                    np.asarray(c[clave]["data"], dtype=np.float32).reshape(c[clave]["dims"][1:])
                    for c in crudos
                ]
            )

        return cls(
            apilar("style_ttl"),
            apilar("style_dp"),
            [Path(r).stem for r in rutas],
            crudos[0].get("metadata", {}) if len(crudos) == 1 else {},
        )

    def guardar(self, ruta: Path | str, **metadatos) -> Path:
        if len(self) != 1:
            raise ValueError("se guarda una voz cada vez; usa estilo[i]")
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        cuerpo = {
            "style_ttl": {
                "data": self.ttl.tolist(),
                "dims": list(self.ttl.shape),
                "type": "float32",
            },
            "style_dp": {"data": self.dp.tolist(), "dims": list(self.dp.shape), "type": "float32"},
            "metadata": {
                **self.metadatos,
                **metadatos,
                "extracted_at": datetime.now(UTC).isoformat(),
            },
        }
        ruta.write_text(json.dumps(cuerpo), encoding="utf-8")
        return ruta

    def mezclar(self, pesos: np.ndarray) -> Estilo:
        """Combine the voices in this batch into one, token by token.

        ``pesos`` is either ``[B]`` (one weight per voice) or ``[50, B]`` (a
        different mix for each of the 50 style tokens, which is what gives the
        builder enough freedom to leave the straight line between two speakers).
        The duration style always uses the column mean, because there is no
        token axis to vary along.
        """
        pesos = np.asarray(pesos, dtype=np.float32)
        if pesos.ndim == 1:
            pesos = np.tile(pesos.reshape(1, -1), (self.ttl.shape[1], 1))
        if pesos.shape != (self.ttl.shape[1], len(self)):
            raise ValueError(
                f"se esperaban pesos [{self.ttl.shape[1]},{len(self)}]; llegaron {pesos.shape}"
            )
        ttl = np.einsum("tb,btd->td", pesos, self.ttl)[None]
        dp = np.einsum("b,bkd->kd", pesos.mean(axis=0), self.dp)[None]
        return Estilo(ttl, dp, ["mezcla"], {"mezclada_de": list(self.nombres)})


def catalogo(carpeta: Path | str) -> dict[str, Path]:
    """Every ``*.json`` voice in a folder, by name. ``indice.json`` is not a voice."""
    return {r.stem: r for r in sorted(Path(carpeta).glob("*.json")) if r.stem != "indice"}


def publicar(ruta: Path | str, raiz: Path | str | None = None) -> Path | None:
    """Copy a freshly built voice where the page will find it, and reindex.

    Otherwise building a voice takes a command **and** remembering to run
    ``npm run preparar``, and a voice nobody can hear is not a voice. If the page
    has never been prepared there is nothing to copy into, and that is fine: this
    returns ``None`` and says nothing.
    """
    import json as _json

    ruta = Path(ruta)
    destino = (
        Path(raiz or Path(__file__).resolve().parents[3])
        / "web"
        / "public"
        / "models"
        / "supertonic"
        / "voice_styles"
    )
    if not destino.is_dir():
        return None
    (destino / ruta.name).write_bytes(ruta.read_bytes())
    nombres = sorted(p.stem for p in destino.glob("*.json") if p.stem != "indice")
    (destino / "indice.json").write_text(_json.dumps(nombres, indent=1), encoding="utf-8")
    return destino / ruta.name
