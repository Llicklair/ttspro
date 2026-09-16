"""The Supertonic engine: four ONNX graphs, no phonemizer, no converter.
Mirror of ``web/src/runtime/supertonic.ts`` — same order, same tensors (rule 2).

    texto --indexador--> ids ---+--> duration_predictor --> segundos
                                |
                                +--> text_encoder --------> text_emb
                                                              |
       ruido ---> [ vector_estimator x N pasos (flow matching) ] --> latente
                                                                       |
                                                              vocoder --> onda 44,1 kHz

The identity of the speaker enters three of those four graphs as ``style_ttl`` /
``style_dp`` (see ``estilo.py``), so there is nothing to run afterwards: the voice
you asked for is the voice that comes out.

Two rules of ARCHITECTURE.md meet their exception here, both recorded in ADR 0012.
Rule 1 (one forward per sentence) becomes *N* forwards, because flow matching is
an ODE solved in steps — ``pasos`` is that N, and it is the only quality-latency
dial the model has. Rule 5 survives intact: the noise is sampled out here and
handed in as a tensor, so a seed reproduces a take exactly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .estilo import Estilo
from .texto import Indexador, trocear

GRAFOS = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")

# Upstream defaults. 8 steps is the quality-latency knee; 1.05 is slightly faster
# than life on purpose, because the model reads a touch slowly at 1.0.
PASOS = 8
VELOCIDAD = 1.05
SILENCIO = 0.3


@dataclass
class Resultado:
    """One synthesis, plus everything docs/evidencia.md ever wants to know."""

    onda: np.ndarray  # float32 [T], -1..1
    frecuencia: int
    texto: str
    idioma: str
    voz: str
    pasos: int
    ms: float
    ms_por_grafo: dict[str, float] = field(default_factory=dict)
    # El latente que entro al vocoder, `[144, T]`, solo si se pidio. Lo usa el
    # voice builder: es la misma representacion a la que `inversor.py` lleva una
    # grabacion real, asi que ahi se pueden comparar una voz sintetica y una real.
    latente: np.ndarray | None = None

    @property
    def segundos(self) -> float:
        return len(self.onda) / self.frecuencia

    @property
    def rtf(self) -> float:
        """Real-time factor: under 1 means it synthesizes faster than it speaks."""
        return (self.ms / 1000) / max(self.segundos, 1e-9)

    def guardar(self, ruta: Path | str) -> Path:
        import soundfile as sf

        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(ruta), self.onda, self.frecuencia)
        return ruta


class Motor:
    """Loads the four graphs once and synthesizes many times."""

    def __init__(self, carpeta: Path | str, proveedores: list[str] | None = None) -> None:
        import onnxruntime as ort

        carpeta = Path(carpeta)
        self.carpeta = carpeta
        onnx = carpeta / "onnx"
        self.cfgs = json.loads((onnx / "tts.json").read_text(encoding="utf-8"))
        self.indexador = Indexador.desde(onnx / "unicode_indexer.json")

        opciones = ort.SessionOptions()
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        disponibles = ort.get_available_providers()
        preferidos = ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
        self.proveedores = proveedores or [p for p in preferidos if p in disponibles]
        self.sesiones = {}
        for nombre in GRAFOS:
            ruta = onnx / f"{nombre}.onnx"
            if not ruta.exists():
                raise FileNotFoundError(
                    f"falta {ruta}; ejecuta `uv run python -m ttspro.supertonic.descargar`"
                )
            self.sesiones[nombre] = ort.InferenceSession(
                str(ruta), sess_options=opciones, providers=self.proveedores
            )

        self.frecuencia = int(self.cfgs["ae"]["sample_rate"])
        self._trozo = int(self.cfgs["ae"]["base_chunk_size"]) * int(
            self.cfgs["ttl"]["chunk_compress_factor"]
        )
        self._dim_latente = int(self.cfgs["ttl"]["latent_dim"]) * int(
            self.cfgs["ttl"]["chunk_compress_factor"]
        )

    # -- el camino corto: una frase, una voz ---------------------------------

    def sintetizar(
        self,
        texto: str,
        estilo: Estilo,
        idioma: str = "es",
        pasos: int = PASOS,
        velocidad: float = VELOCIDAD,
        silencio: float = SILENCIO,
        semilla: int | None = None,
    ) -> Resultado:
        """Speak ``texto`` in one voice, splitting a long text into chunks."""
        if len(estilo) != 1:
            raise ValueError(
                f"una voz cada vez; este estilo trae {len(estilo)}. Usa estilo[i] o lote()"
            )
        maximo = 120 if idioma in ("ko", "ja") else 300
        trozos = trocear(texto, maximo) or [texto]
        ondas: list[np.ndarray] = []
        ms, por_grafo = 0.0, {}
        for i, trozo in enumerate(trozos):
            semilla_i = None if semilla is None else semilla + i
            ondas_i, dur, t, g, _ = self._inferir(
                [trozo], [idioma], estilo, pasos, velocidad, semilla_i
            )
            ms += t
            for k, v in g.items():
                por_grafo[k] = por_grafo.get(k, 0.0) + v
            onda = ondas_i[0, : int(self.frecuencia * float(dur[0]))]
            if ondas:
                ondas.append(np.zeros(int(silencio * self.frecuencia), dtype=np.float32))
            ondas.append(onda)
        return Resultado(
            np.concatenate(ondas),
            self.frecuencia,
            texto,
            idioma,
            estilo.nombres[0],
            pasos,
            ms,
            por_grafo,
        )

    def lote(
        self,
        textos: list[str],
        estilo: Estilo,
        idiomas: list[str] | str = "es",
        pasos: int = PASOS,
        velocidad: float = VELOCIDAD,
        semilla: int | None = None,
        con_latente: bool = False,
    ) -> list[Resultado]:
        """One sentence per voice, all in one forward. The builder lives on this."""
        if isinstance(idiomas, str):
            idiomas = [idiomas] * len(textos)
        if len(estilo) != len(textos):
            raise ValueError(
                f"{len(textos)} textos y {len(estilo)} voces: el lote exige una voz por texto"
            )
        ondas, dur, ms, por_grafo, latente = self._inferir(
            textos, idiomas, estilo, pasos, velocidad, semilla
        )
        return [
            Resultado(
                ondas[b, : int(self.frecuencia * float(dur[b]))].copy(),
                self.frecuencia,
                textos[b],
                idiomas[b],
                estilo.nombres[b],
                pasos,
                ms / len(textos),
                por_grafo,
                latente[b].copy() if con_latente else None,
            )
            for b in range(len(textos))
        ]

    # -- el motor ------------------------------------------------------------

    def _inferir(self, textos, idiomas, estilo, pasos, velocidad, semilla):
        marca = time.perf_counter()
        por_grafo: dict[str, float] = {}

        def correr(nombre: str, entradas: dict):
            t = time.perf_counter()
            salida = self.sesiones[nombre].run(None, entradas)
            por_grafo[nombre] = por_grafo.get(nombre, 0.0) + (time.perf_counter() - t) * 1000
            return salida[0]

        ids, mascara_texto = self.indexador(textos, idiomas)
        duracion = correr(
            "duration_predictor",
            {"text_ids": ids, "style_dp": estilo.dp, "text_mask": mascara_texto},
        )
        duracion = duracion / velocidad
        emb = correr(
            "text_encoder",
            {"text_ids": ids, "style_ttl": estilo.ttl, "text_mask": mascara_texto},
        )

        xt, mascara_latente = self._ruido(duracion, semilla)
        total = np.full(len(textos), pasos, dtype=np.float32)
        for paso in range(pasos):
            xt = correr(
                "vector_estimator",
                {
                    "noisy_latent": xt,
                    "text_emb": emb,
                    "style_ttl": estilo.ttl,
                    "text_mask": mascara_texto,
                    "latent_mask": mascara_latente,
                    "current_step": np.full(len(textos), paso, dtype=np.float32),
                    "total_step": total,
                },
            )
        onda = correr("vocoder", {"latent": xt})
        return onda, duracion, (time.perf_counter() - marca) * 1000, por_grafo, xt

    def _ruido(self, duracion: np.ndarray, semilla: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian noise of the right length, and the mask that trims it.

        Rule 5 lives here: sampled outside the graph, handed in as a tensor. A
        seed makes a take reproducible, which is what the page semilla field is for.
        """
        rng = np.random.default_rng(semilla)
        muestras = (duracion * self.frecuencia).astype(np.int64)
        largo = int((duracion.max() * self.frecuencia + self._trozo - 1) // self._trozo)
        mascara = _mascara((muestras + self._trozo - 1) // self._trozo, largo)
        # Con semilla, el lote entero comparte la misma toma: el constructor compara
        # candidatos y una diferencia de ruido entre ellos seria ruido en la medida.
        forma = (1 if semilla is not None else len(duracion), self._dim_latente, largo)
        ruido = rng.standard_normal(forma, dtype=np.float32)
        return np.broadcast_to(ruido, (len(duracion), *forma[1:])) * mascara, mascara


def _mascara(largos: np.ndarray, maximo: int) -> np.ndarray:
    return (np.arange(maximo) < np.expand_dims(largos, 1)).astype(np.float32).reshape(-1, 1, maximo)
