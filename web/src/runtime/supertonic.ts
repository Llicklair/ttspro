/**
 * The Supertonic engine in the browser: four ONNX graphs, no phonemizer, no converter.
 * Mirror of `src/ttspro/supertonic/motor.py` — same order, same tensors (rule 2).
 *
 *   texto --indexador--> ids ---+--> duration_predictor --> segundos
 *                               |
 *                               +--> text_encoder --------> text_emb
 *                                                             |
 *      ruido ---> [ vector_estimator x N pasos (flow matching) ] --> latente
 *                                                                      |
 *                                                             vocoder --> onda 44,1 kHz
 *
 * The speaker's identity enters three of those four graphs as `style_ttl` /
 * `style_dp`, so there is nothing to run afterwards: the voice you asked for is
 * the voice that comes out. That is the whole of ADR 0012.
 *
 * Rule 1 (one forward per sentence) is the one casualty: flow matching is an ODE
 * solved in `pasos` steps, so the vector estimator runs N times. Rule 5 survives —
 * the noise is generated here and handed in as a tensor, so a seed repeats a take.
 */

import { Indexador, trocearLargo } from "../frontend/unicode.ts";
import { ruidoNormal } from "./audio.ts";
import { type AlDescargar, type Proveedor, type Sesion, crearSesion, ort } from "./ort.ts";

export const GRAFOS = [
  "duration_predictor",
  "text_encoder",
  "vector_estimator",
  "vocoder",
] as const;
export type Grafo = (typeof GRAFOS)[number];

/** Upstream defaults. 8 steps is the quality-latency knee; 1.05 is slightly
 * faster than life on purpose, because the model reads a touch slowly at 1.0. */
export const PASOS = 8;
export const VELOCIDAD = 1.05;
export const SILENCIO = 0.3;

export interface Estilo {
  nombre: string;
  ttl: Float32Array; // [50, 256]
  dp: Float32Array; // [8, 16]
  metadatos?: Record<string, unknown>;
}

export interface Opciones {
  idioma?: string;
  pasos?: number;
  velocidad?: number;
  silencio?: number;
  semilla?: number;
}

export interface Resultado {
  onda: Float32Array;
  sampleRate: number;
  texto: string;
  idioma: string;
  voz: string;
  pasos: number;
  ms: number;
  msPorGrafo: Record<string, number>;
  rtf: number;
}

/** A voice is 292 KB of JSON, which is why a page can hold a hundred of them. */
export async function descargarEstilo(url: string, nombre?: string): Promise<Estilo> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`no se pudo descargar la voz ${url}: ${r.status}`);
  return leerEstilo(
    (await r.json()) as EstiloJSON,
    nombre ??
      url
        .split("/")
        .pop()
        ?.replace(/\.json$/, "") ??
      "voz",
  );
}

export function leerEstilo(crudo: EstiloJSON, nombre: string): Estilo {
  const plano = (t: TensorJSON, esperado: number[]): Float32Array => {
    const dims: number[] = t?.dims ?? [];
    if (dims.length !== 3 || dims[1] !== esperado[0] || dims[2] !== esperado[1]) {
      throw new Error(
        `la voz ${nombre} trae ${JSON.stringify(dims)} y se esperaba [1,${esperado}]`,
      );
    }
    return Float32Array.from(t.data.flat(Number.POSITIVE_INFINITY) as number[]);
  };
  return {
    nombre,
    ttl: plano(crudo.style_ttl, [50, 256]),
    dp: plano(crudo.style_dp, [8, 16]),
    metadatos: crudo.metadata,
  };
}

/** `tts.json`: lo que el motor necesita saber del checkpoint. */
export interface Cfgs {
  ae: { sample_rate: number; base_chunk_size: number };
  ttl: { latent_dim: number; chunk_compress_factor: number };
}

/** Un tensor tal y como viaja en los `voice_styles/*.json` de upstream. */
interface TensorJSON {
  data: unknown[];
  dims: number[];
  type: string;
}

export interface EstiloJSON {
  style_ttl: TensorJSON;
  style_dp: TensorJSON;
  metadata?: Record<string, unknown>;
}

export interface Cargando {
  grafo: Grafo;
  indice: number;
  total: number;
  recibidos: number;
  bytes: number;
}

export class Supertonic {
  private constructor(
    readonly sesiones: Record<Grafo, Sesion>,
    readonly indexador: Indexador,
    readonly cfgs: Cfgs,
    readonly proveedor: Proveedor,
    readonly bytes: number,
  ) {
    this.sampleRate = cfgs.ae.sample_rate;
    this.trozo = cfgs.ae.base_chunk_size * cfgs.ttl.chunk_compress_factor;
    this.dimLatente = cfgs.ttl.latent_dim * cfgs.ttl.chunk_compress_factor;
  }

  readonly sampleRate: number;
  private readonly trozo: number;
  private readonly dimLatente: number;

  static async cargar(
    base: string,
    preferidos: Proveedor[] = ["webgpu", "wasm"],
    alCargar?: (c: Cargando) => void,
  ): Promise<Supertonic> {
    const raiz = base.endsWith("/") ? base : `${base}/`;
    const [cfgs, indexador] = await Promise.all([
      fetch(`${raiz}tts.json`).then((r) => r.json() as Promise<Cfgs>),
      Indexador.descargar(`${raiz}unicode_indexer.json`),
    ]);
    const sesiones = {} as Record<Grafo, Sesion>;
    let bytes = 0;
    for (const [indice, grafo] of GRAFOS.entries()) {
      const aviso: AlDescargar | undefined = alCargar
        ? (recibidos, total) =>
            alCargar({ grafo, indice, total: GRAFOS.length, recibidos, bytes: total })
        : undefined;
      sesiones[grafo] = await crearSesion(`${raiz}${grafo}.onnx`, preferidos, aviso);
      bytes += sesiones[grafo].bytes;
    }
    // Todos los grafos en el mismo proveedor o las medidas no significan nada.
    const proveedor = sesiones.vector_estimator.proveedor;
    return new Supertonic(sesiones, indexador, cfgs, proveedor, bytes);
  }

  /** Speak `texto` in one voice, splitting a long text into chunks. */
  async sintetizar(texto: string, voz: Estilo, opciones: Opciones = {}): Promise<Resultado> {
    const idioma = opciones.idioma ?? "es";
    const pasos = opciones.pasos ?? PASOS;
    const silencio = opciones.silencio ?? SILENCIO;
    const maximo = idioma === "ko" || idioma === "ja" ? 120 : 300;
    const trozos = trocearLargo(texto, maximo);
    const partes: Float32Array[] = [];
    let ms = 0;
    const msPorGrafo: Record<string, number> = {};
    for (const [i, trozo] of (trozos.length ? trozos : [texto]).entries()) {
      const semilla = opciones.semilla === undefined ? undefined : opciones.semilla + i;
      const paso = await this.inferir(
        trozo,
        idioma,
        voz,
        pasos,
        opciones.velocidad ?? VELOCIDAD,
        semilla,
      );
      ms += paso.ms;
      for (const [k, v] of Object.entries(paso.msPorGrafo))
        msPorGrafo[k] = (msPorGrafo[k] ?? 0) + v;
      if (partes.length) partes.push(new Float32Array(Math.round(silencio * this.sampleRate)));
      partes.push(paso.onda);
    }
    const onda = unir(partes);
    return {
      onda,
      sampleRate: this.sampleRate,
      texto,
      idioma,
      voz: voz.nombre,
      pasos,
      ms,
      msPorGrafo,
      rtf: ms / 1000 / (onda.length / this.sampleRate),
    };
  }

  private async inferir(
    texto: string,
    idioma: string,
    voz: Estilo,
    pasos: number,
    velocidad: number,
    semilla?: number,
  ) {
    const t0 = performance.now();
    const msPorGrafo: Record<string, number> = {};
    const correr = async (grafo: Grafo, feeds: Record<string, InstanceType<typeof ort.Tensor>>) => {
      const t = performance.now();
      const salida = await this.sesiones[grafo].sesion.run(feeds);
      msPorGrafo[grafo] = (msPorGrafo[grafo] ?? 0) + (performance.now() - t);
      return salida;
    };

    const { ids, mascara, largo } = this.indexador.tokenizar([texto], [idioma]);
    const textIds = new ort.Tensor("int64", ids, [1, largo]);
    const textMask = new ort.Tensor("float32", mascara, [1, 1, largo]);
    const styleTtl = new ort.Tensor("float32", voz.ttl, [1, 50, 256]);
    const styleDp = new ort.Tensor("float32", voz.dp, [1, 8, 16]);

    const dp = await correr("duration_predictor", {
      text_ids: textIds,
      style_dp: styleDp,
      text_mask: textMask,
    });
    const duracion =
      (dp[this.sesiones.duration_predictor.sesion.outputNames[0]].data as Float32Array)[0] /
      velocidad;

    const te = await correr("text_encoder", {
      text_ids: textIds,
      style_ttl: styleTtl,
      text_mask: textMask,
    });
    const textEmb = te[this.sesiones.text_encoder.sesion.outputNames[0]];

    const muestras = Math.round(duracion * this.sampleRate);
    const largoLatente = Math.max(1, Math.ceil(muestras / this.trozo));
    const ruido = ruidoNormal(
      this.dimLatente * largoLatente,
      semilla ?? Math.floor(Math.random() * 2 ** 31),
    );
    const mascaraLatente = new Float32Array(largoLatente).fill(1);
    // `ort.Tensor` a secas, no `TypedTensor<"float32">`: lo que devuelve `run()`
    // es el tipo ancho y este tensor se reasigna en cada paso del flow.
    let xt: InstanceType<typeof ort.Tensor> = new ort.Tensor("float32", ruido, [
      1,
      this.dimLatente,
      largoLatente,
    ]);
    const latentMask = new ort.Tensor("float32", mascaraLatente, [1, 1, largoLatente]);
    const total = new ort.Tensor("float32", Float32Array.from([pasos]), [1]);
    const salidaVE = this.sesiones.vector_estimator.sesion.outputNames[0];
    for (let paso = 0; paso < pasos; paso++) {
      const r = await correr("vector_estimator", {
        noisy_latent: xt,
        text_emb: textEmb,
        style_ttl: styleTtl,
        text_mask: textMask,
        latent_mask: latentMask,
        current_step: new ort.Tensor("float32", Float32Array.from([paso]), [1]),
        total_step: total,
      });
      xt = r[salidaVE];
    }
    const voc = await correr("vocoder", { latent: xt });
    const cruda = voc[this.sesiones.vocoder.sesion.outputNames[0]].data as Float32Array;
    return {
      onda: cruda.slice(0, Math.min(muestras, cruda.length)),
      ms: performance.now() - t0,
      msPorGrafo,
    };
  }
}

function unir(partes: Float32Array[]): Float32Array {
  const onda = new Float32Array(partes.reduce((n, p) => n + p.length, 0));
  let pos = 0;
  for (const p of partes) {
    onda.set(p, pos);
    pos += p.length;
  }
  return onda;
}
