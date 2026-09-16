"""Build a voice from a recording, without the encoder that built the built-in ones.

    uv run python -m ttspro.supertonic.constructor --referencia mi_voz.wav --nombre marcos

Supertone shipped the four inference graphs and ten voices, and kept the tool that
turns a recording into a voice: its Voice Builder was a hosted service and it went
dark on 2026-08-31, so there is no ``audio -> style`` encoder to download, here or
anywhere (checked in all four published model repos — ADR 0012 records the search).

What they did ship is enough, because of one fact: the voice is an **input**
(``estilo.py``). So building a voice stops being "train an encoder" and becomes
"find 12 928 numbers that sound like this person", and that is a search with a
cheap, honest score — the cosine between the speaker embedding of the recording
and of the synthesis, measured with the same WeSpeaker graph
(``models/speaker_encoder.onnx``) that ``tests/terminado`` uses to grade the
project. The search cannot cheat the number that judges it, so it is validated
against a second recording of the same speaker it never optimized on.

The search runs over **mixtures of the voices you already have**, one mixture per
style token, so the result is not stuck on the straight line between two speakers.
It runs in two stages, each measured:

    etapa 1   10 numbers   one global mix          seconds
    etapa 2   50 x 10      a mix per style token   minutes

Every voice you build joins the basis for the next one, so the reachable space
grows with use. What it cannot do is leave that space altogether: this is a
projection onto the voices the checkpoint already knows, not zero-shot cloning.
Whether that is close enough is a measurement, and it is in docs/evidencia.md.

Only build a voice whose owner agreed to it (rule 10, MODEL_CARD.md).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .estilo import Estilo, catalogo, publicar
from .motor import Motor

RAIZ = Path(__file__).resolve().parents[3]
SONDAS = (
    "Hola, esta es mi voz leyendo una frase corta para la prueba.",
    "El tiempo pasa despacio cuando uno espera algo importante.",
)


class Similitud:
    """WeSpeaker cosine — the project's own yardstick, not a new one.

    Contract in ``models/contrato.json``: mono 16 kHz in ``[-1, 1]``, out comes a
    256-d L2-normalised embedding, so a dot product is the cosine. The batch axis
    is pinned to 1 upstream, hence the loop.
    """

    def __init__(self, ruta: Path | str | None = None) -> None:
        import onnxruntime as ort

        ruta = Path(ruta or RAIZ / "models" / "speaker_encoder.onnx")
        if not ruta.exists():
            raise FileNotFoundError(
                f"falta {ruta}; ejecuta `uv run python -m ttspro.export.speaker_encoder`"
            )
        self.sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])

    def embedding(self, onda: np.ndarray, frecuencia: int) -> np.ndarray:
        onda = a_16k(onda, frecuencia)
        if len(onda) < 16000:  # menos de un segundo: el encoder no tiene con que trabajar
            onda = np.pad(onda, (0, 16000 - len(onda)))
        salida = self.sesion.run(None, {"onda": onda[None].astype(np.float32)})[0]
        return salida[0].astype(np.float32)

    def lote(self, ondas: list[np.ndarray], frecuencia: int) -> np.ndarray:
        return np.stack([self.embedding(o, frecuencia) for o in ondas])


def a_16k(onda: np.ndarray, frecuencia: int) -> np.ndarray:
    """Down to the 16 kHz mono the speaker encoder expects."""
    from math import gcd

    from scipy.signal import resample_poly

    onda = np.asarray(onda, dtype=np.float32)
    if onda.ndim > 1:
        onda = onda.mean(axis=tuple(range(onda.ndim - 1)))
    if frecuencia == 16000:
        return onda
    d = gcd(16000, frecuencia)
    return resample_poly(onda, 16000 // d, frecuencia // d).astype(np.float32)


def cargar_referencia(ruta: Path | str, segundos: float = 12.0) -> tuple[np.ndarray, int]:
    import soundfile as sf

    onda, frecuencia = sf.read(str(ruta), dtype="float32", always_2d=True)
    onda = onda.mean(axis=1)
    return onda[: int(segundos * frecuencia)], frecuencia


@dataclass
class Traza:
    """One generation of the search, so the run can be plotted and doubted."""

    etapa: int
    generacion: int
    mejor: float
    media: float
    segundos: float


@dataclass
class Construccion:
    voz: Estilo
    similitud: float
    similitud_inicial: float
    traza: list[Traza] = field(default_factory=list)
    evaluaciones: int = 0
    segundos: float = 0.0


class Constructor:
    """Searches the space of mixtures for the voice that scores highest."""

    def __init__(
        self,
        motor: Motor,
        base: Estilo,
        similitud: Similitud | None = None,
        sondas: tuple[str, ...] = SONDAS,
        idioma: str = "es",
        pasos: int = 4,
    ) -> None:
        self.motor = motor
        self.base = base
        self.similitud = similitud or Similitud()
        self.sondas = sondas
        self.idioma = idioma
        self.pasos = pasos  # menos pasos durante la busqueda: se comparan candidatos, no se publica
        self.evaluaciones = 0

    # -- puntuar ------------------------------------------------------------

    def _estilos(self, pesos: np.ndarray) -> Estilo:
        """``[P, 50, K]`` weights -> a batch of P mixed voices."""
        ttl = np.einsum("ptk,ktd->ptd", pesos, self.base.ttl)
        dp = np.einsum("pk,kjd->pjd", pesos.mean(axis=1), self.base.dp)
        return Estilo(ttl, dp, [f"c{i}" for i in range(len(pesos))])

    def puntuar(self, pesos: np.ndarray, objetivo: np.ndarray, semilla: int = 0) -> np.ndarray:
        """Mean cosine to ``objetivo`` over the probe sentences, one score per candidate."""
        estilos = self._estilos(pesos)
        total = np.zeros(len(pesos), dtype=np.float32)
        for s, sonda in enumerate(self.sondas):
            resultados = self.motor.lote(
                [sonda] * len(pesos), estilos, self.idioma, pasos=self.pasos, semilla=semilla + s
            )
            embs = self.similitud.lote([r.onda for r in resultados], self.motor.frecuencia)
            total += embs @ objetivo
        self.evaluaciones += len(pesos) * len(self.sondas)
        return total / len(self.sondas)

    # -- buscar -------------------------------------------------------------

    def construir(
        self,
        referencia: np.ndarray,
        frecuencia: int,
        generaciones: tuple[int, int] = (12, 25),
        poblacion: int = 16,
        semilla: int = 0,
        registro=print,
    ) -> Construccion:
        """Two stages of a cross-entropy search, from a global mix to a per-token one."""
        arranque = time.perf_counter()
        objetivo = self.similitud.embedding(referencia, frecuencia)
        k, t = len(self.base), self.base.ttl.shape[1]
        rng = np.random.default_rng(semilla)
        traza: list[Traza] = []

        # Punto de partida: cada voz de la base, sola. Tambien da el "antes" honesto.
        solas = self.puntuar(np.eye(k)[:, None, :].repeat(t, axis=1), objetivo, semilla)
        orden = np.argsort(-solas)
        registro(f"  mejor voz suelta: {self.base.nombres[orden[0]]} con {solas[orden[0]]:.3f}")
        inicial = float(solas[orden[0]])

        # Etapa 1: un peso por voz, igual para los 50 tokens.
        media = np.zeros(k, dtype=np.float32)
        media[orden[0]] = 1.0
        sigma = np.full(k, 0.35, dtype=np.float32)
        for g in range(generaciones[0]):
            t0 = time.perf_counter()
            cand = np.clip(
                media + sigma * rng.standard_normal((poblacion, k)).astype(np.float32), -0.5, 1.5
            )
            cand = np.concatenate([media[None], cand])
            puntos = self.puntuar(_extender(cand, t), objetivo, semilla + g)
            elite = cand[np.argsort(-puntos)[: max(2, poblacion // 4)]]
            media, sigma = elite.mean(axis=0), np.maximum(elite.std(axis=0), 0.02)
            traza.append(
                Traza(1, g, float(puntos.max()), float(puntos.mean()), time.perf_counter() - t0)
            )
            registro(
                f"  etapa 1 gen {g + 1}/{generaciones[0]}: "
                f"mejor {puntos.max():.3f}  media {puntos.mean():.3f}  "
                f"({traza[-1].segundos:.0f} s)"
            )

        # Etapa 2: un peso por voz y por token de estilo.
        media_t = np.tile(media, (t, 1))
        sigma_t = np.full((t, k), 0.12, dtype=np.float32)
        mejor_pesos, mejor_punto = media_t.copy(), -1.0
        for g in range(generaciones[1]):
            t0 = time.perf_counter()
            cand = np.clip(
                media_t + sigma_t * rng.standard_normal((poblacion, t, k)).astype(np.float32),
                -0.5,
                1.5,
            )
            cand = np.concatenate([media_t[None], cand])
            puntos = self.puntuar(cand, objetivo, semilla + 100 + g)
            idx = np.argsort(-puntos)
            if puntos[idx[0]] > mejor_punto:
                mejor_punto, mejor_pesos = float(puntos[idx[0]]), cand[idx[0]].copy()
            elite = cand[idx[: max(2, poblacion // 4)]]
            media_t, sigma_t = elite.mean(axis=0), np.maximum(elite.std(axis=0), 0.01)
            traza.append(
                Traza(2, g, float(puntos.max()), float(puntos.mean()), time.perf_counter() - t0)
            )
            registro(
                f"  etapa 2 gen {g + 1}/{generaciones[1]}: "
                f"mejor {puntos.max():.3f}  media {puntos.mean():.3f}  "
                f"({traza[-1].segundos:.0f} s)"
            )

        voz = self._estilos(mejor_pesos[None])
        voz.metadatos = {
            "construida_por": "ttspro.supertonic.constructor",
            "base": list(self.base.nombres),
            "similitud_busqueda": mejor_punto,
            "similitud_mejor_voz_suelta": inicial,
            "pasos_busqueda": self.pasos,
        }
        return Construccion(
            voz, mejor_punto, inicial, traza, self.evaluaciones, time.perf_counter() - arranque
        )

    def verificar(
        self,
        voz: Estilo,
        referencia: np.ndarray,
        frecuencia: int,
        pasos: int = 8,
        frases: tuple[str, ...] | None = None,
    ) -> float:
        """Score at publication quality, on sentences the search never saw."""
        objetivo = self.similitud.embedding(referencia, frecuencia)
        frases = frases or (
            "Buenas tardes, gracias por acompanarme un rato esta noche.",
            "No sabia que se pudiera construir una voz de esta manera.",
            "Manana seguimos con lo que dejamos a medias hoy por la tarde.",
        )
        cos = []
        for i, frase in enumerate(frases):
            r = self.motor.sintetizar(frase, voz, self.idioma, pasos=pasos, semilla=9000 + i)
            cos.append(float(self.similitud.embedding(r.onda, r.frecuencia) @ objetivo))
        return float(np.mean(cos))


def _extender(pesos: np.ndarray, tokens: int) -> np.ndarray:
    """``[P, K]`` -> ``[P, T, K]``: the same mix for every style token."""
    return np.repeat(pesos[:, None, :], tokens, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--referencia", type=Path, required=True, help="grabacion de la voz a construir (3 s o mas)"
    )
    ap.add_argument("--nombre", required=True, help="como se llamara la voz")
    ap.add_argument(
        "--validacion", type=Path, help="otra grabacion del mismo locutor, para puntuar sin trampa"
    )
    ap.add_argument("--modelos", type=Path, default=RAIZ / "models" / "supertonic")
    ap.add_argument("--salida", type=Path, default=RAIZ / "models" / "supertonic" / "voice_styles")
    ap.add_argument("--generaciones", type=int, nargs=2, default=(12, 25))
    ap.add_argument("--poblacion", type=int, default=16)
    ap.add_argument("--pasos", type=int, default=4, help="pasos de flow durante la busqueda")
    ap.add_argument("--idioma", default="es")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    motor = Motor(args.modelos)
    base = Estilo.cargar(*catalogo(args.modelos / "voice_styles").values())
    print(f"base de {len(base)} voces: {', '.join(base.nombres)}")
    referencia, frecuencia = cargar_referencia(args.referencia)
    print(
        f"referencia {args.referencia.name}: {len(referencia) / frecuencia:.1f} s a {frecuencia} Hz"
    )

    constructor = Constructor(motor, base, idioma=args.idioma, pasos=args.pasos)
    c = constructor.construir(
        referencia, frecuencia, tuple(args.generaciones), args.poblacion, args.semilla
    )

    verificada = constructor.verificar(c.voz, referencia, frecuencia)
    linea = {
        "voz": args.nombre,
        "similitud_mejor_voz_suelta": round(c.similitud_inicial, 4),
        "similitud_busqueda": round(c.similitud, 4),
        "similitud_verificada": round(verificada, 4),
        "evaluaciones": c.evaluaciones,
        "segundos": round(c.segundos, 1),
    }
    if args.validacion:
        otra, f2 = cargar_referencia(args.validacion)
        linea["similitud_validacion"] = round(constructor.verificar(c.voz, otra, f2), 4)
        linea["techo_dos_reales"] = round(
            float(
                constructor.similitud.embedding(referencia, frecuencia)
                @ constructor.similitud.embedding(otra, f2)
            ),
            4,
        )

    ruta = c.voz.guardar(
        args.salida / f"{args.nombre}.json", nombre=args.nombre, source_file=args.referencia.name
    )
    print(json.dumps(linea, indent=1))
    print(f"guardada en {ruta}")
    publicada = publicar(ruta)
    if publicada:
        print(f"y publicada en {publicada}: recarga la pagina y ahi esta")
    else:
        print("corre `npm run preparar` en web/ para que la pagina la vea")


if __name__ == "__main__":
    main()
