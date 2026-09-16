"""Acuñar un pack de voces nuevas, porque no hay ninguno que descargar.

    uv run python -m ttspro.supertonic.pack --n 24 --nombre chat

Buscado el 2026-09-16: **no existe ni un solo pack de voces de Supertonic** fuera
de las diez de fábrica. Ni en Hugging Face (buscar «voice_styles» da cero), ni en
los mirrors de la comunidad, que llevan todos los mismos M1–M5 y F1–F5. La razón
es la de siempre: la única herramienta que sabía fabricar una voz era el Voice
Builder de Supertone, era un servicio alojado, y cerró el 2026-08-31.

Así que en vez de descargarlas, se acuñan. Y aquí hay una distinción que decide
si esto sirve o no:

    clonar a una persona concreta   ->  medido, 0,227 sobre un objetivo de 0,55: NO llega
    inventar voces distintas entre sí ->  otra cosa, y esta sí se puede

Para «una voz por usuario» en un chat no hace falta que una voz se parezca a
nadie: hace falta que **se distingan entre ellas** y que hablen claro. Eso es
exactamente lo que esto optimiza. Se muestrean candidatas mezclando las voces que
ya hay, se tiran las que no hablan, y de las que quedan se eligen las N más
separadas entre sí — muestreo del punto más lejano, que es lo que maximiza la
distancia mínima del conjunto en vez de la media.

El listón es el del propio proyecto: el percentil 99 del coseno entre locutores
DISTINTOS con este encoder es 0,465 (docs/evidencia.md). Un pack cuyas voces se
queden por debajo de eso entre sí es un pack de voces que nadie confundiría.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .constructor import Similitud
from .estilo import Estilo, catalogo, publicar
from .motor import Motor

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "models" / "supertonic"
# El p99 del coseno entre locutores distintos con este encoder (docs/evidencia.md).
DISTINTAS = 0.465


def elegir_lejanas(embeddings: np.ndarray, n: int, semilla: int = 0) -> list[int]:
    """Las ``n`` más separadas entre sí, por muestreo del punto más lejano.

    Elegir al azar da un pack donde dos voces se parecen por mala suerte. Esto
    empieza por una y va añadiendo, cada vez, la candidata cuya **distancia a la
    más cercana ya elegida** es mayor: el resultado maximiza la separación
    mínima, que es la que se nota, no la media, que la esconde.
    """
    rng = np.random.default_rng(semilla)
    elegidas = [int(rng.integers(len(embeddings)))]
    cercania = embeddings @ embeddings[elegidas[0]]
    while len(elegidas) < min(n, len(embeddings)):
        siguiente = int(np.argmin(cercania))
        elegidas.append(siguiente)
        cercania = np.maximum(cercania, embeddings @ embeddings[siguiente])
    return elegidas


def acunar(
    motor: Motor,
    base: Estilo,
    similitud: Similitud,
    n: int,
    candidatas: int,
    lote: int = 16,
    pasos: int = 4,
    idioma: str = "es",
    semilla: int = 0,
    registro=print,
) -> tuple[Estilo, np.ndarray]:
    """Muestrea candidatas, tira las que no hablan y devuelve las ``n`` más distintas."""
    from .puente import SONDAS, _usable

    rng = np.random.default_rng(semilla)
    k, t = base.ttl.shape[0], base.ttl.shape[1]
    vectores: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    descartadas = 0

    while len(vectores) < candidatas:
        pesos = rng.dirichlet(np.full(k, 0.5), size=lote).astype(np.float32)
        tanda = np.empty((lote, base.vector.shape[1]), dtype=np.float32)
        for i in range(lote):
            # Mezcla con ruido por ficha de estilo: sin el ruido, todas las
            # candidatas caen en el casco convexo y se parecen demasiado.
            w = np.clip(pesos[i] + 0.3 * rng.standard_normal((t, k)), -0.4, 1.4).astype(np.float32)
            ttl = np.einsum("tk,ktd->td", w, base.ttl)
            dp = np.einsum("k,kjd->jd", w.mean(axis=0), base.dp)
            tanda[i] = np.concatenate([ttl.reshape(-1), dp.reshape(-1)])
        sonda = SONDAS[len(vectores) // lote % len(SONDAS)]
        for i, r in enumerate(
            motor.lote(
                [sonda] * lote,
                Estilo.desde_vector(tanda),
                idioma,
                pasos=pasos,
                semilla=semilla + len(vectores),
            )
        ):
            if not _usable(r.onda, r.segundos):
                descartadas += 1
                continue
            vectores.append(tanda[i])
            embeddings.append(similitud.embedding(r.onda, r.frecuencia))
        registro(
            f"  {len(vectores)}/{candidatas} candidatas ({descartadas} descartadas por no hablar)"
        )

    embs = np.stack(embeddings[:candidatas])
    indices = elegir_lejanas(embs, n, semilla)
    elegidas = np.stack([vectores[i] for i in indices])
    return Estilo.desde_vector(elegidas), embs[indices]


def separacion(embeddings: np.ndarray) -> dict:
    """Cómo de distintas son entre sí: lo que decide si el pack vale."""
    import itertools

    pares = [
        float(embeddings[i] @ embeddings[j])
        for i, j in itertools.combinations(range(len(embeddings)), 2)
    ]
    return {
        "voces": int(len(embeddings)),
        "coseno_medio": round(float(np.mean(pares)), 4),
        "coseno_maximo": round(float(np.max(pares)), 4),
        "pares_por_encima_de_0.465": int(sum(p > DISTINTAS for p in pares)),
        "pares": len(pares),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=16, help="cuantas voces tendra el pack")
    ap.add_argument("--nombre", default="voz", help="prefijo de los ficheros: voz01.json, voz02...")
    ap.add_argument(
        "--candidatas", type=int, default=0, help="cuantas muestrear antes de elegir (0: 8 por voz)"
    )
    ap.add_argument("--pasos", type=int, default=4)
    ap.add_argument("--idioma", default="es")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--modelos", type=Path, default=MODELOS)
    ap.add_argument("--salida", type=Path, default=MODELOS / "voice_styles")
    args = ap.parse_args()

    candidatas = args.candidatas or args.n * 8
    motor = Motor(args.modelos)
    base = Estilo.cargar(*catalogo(args.modelos / "voice_styles").values())
    print(f"acunando {args.n} voces a partir de {len(base)}, sobre {candidatas} candidatas")

    pack, embs = acunar(
        motor,
        base,
        Similitud(),
        args.n,
        candidatas,
        pasos=args.pasos,
        idioma=args.idioma,
        semilla=args.semilla,
    )
    medidas = separacion(embs)
    print("\n" + json.dumps(medidas, indent=1))
    if medidas["pares_por_encima_de_0.465"]:
        print(
            f"  aviso: {medidas['pares_por_encima_de_0.465']} pares se parecen mas de la cuenta;"
            f" sube --candidatas o baja --n"
        )

    # Solo se publica lo que va a la carpeta de verdad: mandar la salida a otro
    # sitio (una prueba, un experimento) y que aparezca igual en la pagina es
    # justo lo que nadie espera, y ensucia hasta los tests.
    oficial = args.salida.resolve() == (args.modelos / "voice_styles").resolve()
    for i in range(len(pack)):
        nombre = f"{args.nombre}{i + 1:02d}"
        voz = pack[i]
        voz.nombres = [nombre]
        ruta = voz.guardar(
            args.salida / f"{nombre}.json", nombre=nombre, acunada_por="ttspro.supertonic.pack"
        )
        if oficial:
            publicar(ruta)
    print(f"\n{len(pack)} voces en {args.salida}")
    print(
        "publicadas para la pagina: recarga y ahi estan"
        if oficial
        else f"NO publicadas: --salida apunta fuera de {args.modelos / 'voice_styles'}"
    )


if __name__ == "__main__":
    main()
