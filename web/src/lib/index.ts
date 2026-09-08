/**
 * ttspro as a library: one object, imported from any HTML page.
 *
 *   import { TTS } from "./ttspro.js";
 *   const tts = await TTS.cargar({ modelos: "/models/", espeak: "/espeak/espeak-ng.wasm" });
 *   await tts.clonar(fileInput.files[0]);            // a recording -> the voice to use
 *   const r = await tts.predict("hola a todos");     // -> { onda, sampleRate, ms, wav() }
 *   tts.reproducir(r);
 *   for await (const r of tts.predict(streamDeMensajes, { reproducir: true })) { ... }
 *
 * `predict` takes a string and returns one result, or takes a stream (an
 * AsyncIterable, a ReadableStream, an array, or a plain Iterable of strings or
 * `{ usuario, texto }`) and returns an AsyncIterable of results in order,
 * synthesized ahead of playback through the same queue the demo uses. Chat
 * lines go through the chat normalizer when `chat: true`.
 *
 * Everything runs in the page: no server. The page must be served over HTTP
 * (fetch does not work from file://) and, for multithreaded wasm, with the
 * COOP/COEP headers — `scripts/servir.mjs` does exactly that.
 */

import { nombreLegible, normalizarChat } from "../frontend/chat.ts";
import { configurarEspeak } from "../frontend/fonemas.ts";
import {
  aWav,
  decodificar,
  ponerVolumen,
  reproducir as reproducirOnda,
  volumen,
} from "../runtime/audio.ts";
import { Cola, type Estadisticas, type Mensaje } from "../runtime/cola.ts";
import type { Contrato } from "../runtime/contrato.ts";
import { convertir, vectorVoz } from "../runtime/conversor.ts";
import {
  type Hardware,
  type Medida,
  type Recomendacion,
  calibrar as calibrarGrafo,
  detectar,
  recomendar,
} from "../runtime/hardware.ts";
import {
  type AlDescargar,
  type Proveedor,
  type Sesion,
  crearSesion,
  soportaF16,
} from "../runtime/ort.ts";
import {
  type MetaPaquete,
  URL_VOCES,
  type VozBase,
  cargarIndice,
  cargarPaquete,
} from "../runtime/paquetes.ts";
import { type Opciones as OpcionesSintesis, sintetizar } from "../runtime/sintetizador.ts";

export interface OpcionesCarga {
  /** Folder with contrato.json, voces.json and the .onnx graphs. Default "/models/". */
  modelos?: string;
  /** URL of espeak-ng.wasm. Default "/espeak/espeak-ng.wasm". */
  espeak?: string;
  /** "auto" tries WebGPU then wasm. Default "auto". */
  proveedor?: Proveedor | "auto";
  /** "" for fp32, ".fp16" for fp16, "auto" picks fp16 only on WebGPU with shader-f16. */
  precision?: "" | ".fp16" | "auto";
  /** Download progress over all files. */
  alDescargar?: AlDescargar;
  /**
   * Where the voice packs live (indice.json + <clave>.json + <clave>.tts*.onnx).
   * Default: the project's GitHub release. `null` disables packs.
   */
  vocesBase?: string | null;
}

export interface Voz {
  id: string;
  nombre: string;
  idioma: string;
  voz: number[];
}

export interface OpcionesPredict extends OpcionesSintesis {
  /** Voice vector or preset id for this call; default the voice chosen with `elegirVoz`/`clonar`. */
  voz?: Float32Array | string | null;
  /** How much of the base audio the converter keeps. Default 0.3. */
  tau?: number;
  /** Pass the text through the chat normalizer (links, emotes, "jajaja", …). Default false. */
  chat?: boolean;
  /** Streams only: play each result as it is ready, back to back. Default false. */
  reproducir?: boolean;
  /** Streams only: prefix "Nombre: " when the item carries a user. Default true. */
  leerNombre?: boolean;
  /** Streams only: drop items that waited longer than this. Default 30 000. */
  maxEdadMs?: number;
  /** Which base voice speaks: a pack key from `vocesBase`, or "local". Default the current one. */
  vozBase?: string;
}

export interface Resultado {
  texto: string;
  usuario?: string;
  onda: Float32Array;
  sampleRate: number;
  /** ms of synthesis, converter included. */
  ms: number;
  fonemas: string;
  /** The same audio as a WAV blob, for download or for an <audio> element. */
  wav(): Blob;
}

export type Entrada = string | { usuario?: string; texto: string };
export type Fuente =
  | Iterable<Entrada>
  | AsyncIterable<Entrada>
  | ReadableStream<Entrada | Uint8Array>;

interface Voces {
  base: { locutor: string; idioma: string; embedding_tts?: number[]; voz: number[] };
  voces: Voz[];
}

export class TTS {
  /** Preset voices shipped in voces.json. */
  readonly voces: Voz[];
  /** The voice `predict` uses when none is given: a vector, or null for the base voice. */
  vozActual: Float32Array | null = null;
  /** Provider each graph actually ran on. */
  readonly proveedores: { tts: Proveedor; conversor: Proveedor; voz: Proveedor };
  /** Base voices available for download (packs), by key. Empty when offline or disabled. */
  readonly vocesBase: Record<string, MetaPaquete> = {};
  /** Key of the base voice that speaks by default: "local" or a loaded pack. */
  vozBaseActual = "local";
  /**
   * Defaults for every `predict` from now on: `tts.ajustes.tau = 0.2`,
   * `tts.ajustes.length_scale = 1.1`… A per-call option still wins. Same knobs
   * as the page's sliders: noise_scale (timbre variation, 0.667), noise_scale_w
   * (duration variation, 0.5), length_scale (speed, >1 slower), tau (how much of
   * the base audio the converter keeps, 0.3), semilla.
   */
  ajustes: OpcionesSintesis & { tau?: number } = {};
  /** What `detectar()` saw on this machine, and the rule's recommendation. */
  readonly hardware: Hardware;
  readonly recomendacion: Recomendacion;

  private readonly bases = new Map<string, VozBase>();
  private readonly cargando = new Map<string, Promise<VozBase>>();

  private constructor(
    private readonly contrato: Contrato,
    private readonly presets: Voces,
    private readonly sesionTts: Sesion,
    private readonly sesionConversor: Sesion,
    private readonly sesionVoz: Sesion,
    private readonly rutas: {
      base: string;
      precision: "" | ".fp16";
      proveedores: Proveedor[];
      vocesBase: string | null;
    },
    hardware: Hardware,
  ) {
    this.voces = presets.voces;
    this.proveedores = {
      tts: sesionTts.proveedor,
      conversor: sesionConversor.proveedor,
      voz: sesionVoz.proveedor,
    };
    this.hardware = hardware;
    this.recomendacion = recomendar(hardware);
    this.bases.set("local", {
      meta: {
        clave: "local",
        nombre: presets.base.locutor,
        idioma: presets.base.idioma,
        region: "",
        calidad: "",
        licencia: "",
        parametros_M: 0,
        MB: { fp32: 0, fp16: 0 },
        ms_frase_cpu: 0,
        ficheros: { fp32: "tts.onnx", fp16: "tts.fp16.onnx" },
      },
      sesion: sesionTts,
      contrato,
      vozBase: Float32Array.from(presets.base.voz),
    });
  }

  /** What this machine can run, without loading anything. */
  static hardware(): Promise<Hardware> {
    return detectar();
  }

  /** The rule's recommendation for this machine, without loading anything. */
  static async recomendar(): Promise<Recomendacion> {
    return recomendar(await detectar());
  }

  static async cargar(opciones: OpcionesCarga = {}): Promise<TTS> {
    const base = (opciones.modelos ?? "/models/").replace(/\/?$/, "/");
    const espeak = opciones.espeak ?? "/espeak/espeak-ng.wasm";
    configurarEspeak({ locateFile: (f: string) => (f.endsWith(".wasm") ? espeak : f) });
    const hardware = await detectar();
    const regla = recomendar(hardware);
    const preferido = opciones.proveedor ?? "auto";
    const proveedores: Proveedor[] =
      preferido === "auto"
        ? regla.proveedor === "webgpu"
          ? ["webgpu", "wasm"]
          : ["wasm"]
        : [preferido];
    let precision = opciones.precision ?? "auto";
    if (precision === "auto") {
      precision = proveedores[0] === "webgpu" && (await soportaF16()) ? ".fp16" : "";
    }
    const bytes: Record<string, [number, number]> = {};
    const progreso = (clave: string) => (r: number, t: number) => {
      bytes[clave] = [r, t];
      const todos = Object.values(bytes);
      opciones.alDescargar?.(
        todos.reduce((a, [x]) => a + x, 0),
        todos.reduce((a, [, y]) => a + y, 0),
      );
    };
    const [contrato, voces] = await Promise.all([
      fetch(`${base}contrato.json`).then((r) => r.json() as Promise<Contrato>),
      fetch(`${base}voces.json`).then((r) => r.json() as Promise<Voces>),
    ]);
    const [tts, conversor, voz] = await Promise.all([
      crearSesion(`${base}tts${precision}.onnx`, proveedores, progreso("tts")),
      crearSesion(`${base}conversor${precision}.onnx`, proveedores, progreso("conversor")),
      // the voice extractor carries a GRU, which WebGPU has no kernel for
      crearSesion(`${base}voz${precision}.onnx`, ["wasm"], progreso("voz")),
    ]);
    const vocesBase = opciones.vocesBase === undefined ? URL_VOCES : opciones.vocesBase;
    const salida = new TTS(
      contrato,
      voces,
      tts,
      conversor,
      voz,
      { base, precision, proveedores, vocesBase },
      hardware,
    );
    if (vocesBase) {
      try {
        Object.assign(salida.vocesBase, await cargarIndice(vocesBase));
      } catch {
        // offline, or no packs published: the local base voice still works
      }
    }
    return salida;
  }

  /** Download a base voice pack (once) and open its graph. Returns its meta. */
  async cargarVozBase(clave: string, alDescargar?: AlDescargar): Promise<MetaPaquete> {
    const hecha = this.bases.get(clave);
    if (hecha) return hecha.meta;
    if (!this.rutas.vocesBase) throw new Error("los paquetes de voz están desactivados");
    let pendiente = this.cargando.get(clave);
    if (!pendiente) {
      pendiente = cargarPaquete(
        this.rutas.vocesBase,
        clave,
        this.contrato,
        this.rutas.precision,
        this.rutas.proveedores,
        alDescargar,
      );
      this.cargando.set(clave, pendiente);
    }
    const voz = await pendiente;
    this.bases.set(clave, voz);
    this.cargando.delete(clave);
    return voz.meta;
  }

  /** Make a (loaded) base voice the default speaker. */
  elegirVozBase(clave: string): void {
    if (!this.bases.has(clave))
      throw new Error(`voz base ${clave} no cargada: cargarVozBase antes`);
    this.vozBaseActual = clave;
  }

  /** Keys of the base voices loaded and ready. */
  get vocesBaseCargadas(): string[] {
    return [...this.bases.keys()];
  }

  /**
   * Measure, on this machine, how long ONE sentence takes on each candidate
   * provider/precision, for the TTS and (optionally) the converter, and return
   * the fastest. Slow (it opens sessions) and honest (it runs them).
   */
  async calibrar(
    opciones: { conversor?: boolean; alProgresar?: (grafo: string, m: Medida) => void } = {},
  ): Promise<{ tts: Medida[]; conversor: Medida[]; mejor: Medida | null }> {
    const candidatos: Array<[Proveedor, "" | ".fp16"]> = this.hardware.webgpu
      ? this.hardware.f16
        ? [
            ["webgpu", ".fp16"],
            ["webgpu", ""],
            ["wasm", ""],
          ]
        : [
            ["webgpu", ""],
            ["wasm", ""],
          ]
      : [["wasm", ""]];
    const frase = "Hola a todos, bienvenidos al directo de hoy.";
    const idioma = this.contrato.idiomas[0];
    const tts = await calibrarGrafo(
      (p) => `${this.rutas.base}tts${p}.onnx`,
      async (s) => (await sintetizar(s, this.contrato, frase, idioma, null, {})).onda,
      candidatos,
      (m) => opciones.alProgresar?.("tts", m),
    );
    let conversor = { medidas: [] as Medida[], mejor: null as Medida | null };
    if (opciones.conversor ?? true) {
      const base = await sintetizar(this.sesionTts, this.contrato, frase, idioma, null, {});
      const origen = Float32Array.from(this.presets.base.voz);
      const destino = Float32Array.from(this.presets.voces[0]?.voz ?? this.presets.base.voz);
      conversor = await calibrarGrafo(
        (p) => `${this.rutas.base}conversor${p}.onnx`,
        async (s) => (await convertir(s, this.contrato, base.onda, origen, destino, {})).onda,
        candidatos,
        (m) => opciones.alProgresar?.("conversor", m),
      );
    }
    // the converter dominates when it runs; otherwise the TTS decides
    const mejor = conversor.mejor ?? tts.mejor;
    return { tts: tts.medidas, conversor: conversor.medidas, mejor };
  }

  /** Languages the base voice speaks. */
  get idiomas(): string[] {
    return this.contrato.idiomas;
  }

  /** Master volume for `reproducir`/`leer`, 0..1. Applies to what plays from now on. */
  get volumen(): number {
    return volumen();
  }
  set volumen(v: number) {
    ponerVolumen(v);
  }

  /** Sample rate of every result. */
  get sampleRate(): number {
    return this.contrato.grafos.voz.frecuencia_entrada_hz ?? 22050;
  }

  /** Pick a preset by id, a raw vector, or null for the base voice. */
  elegirVoz(voz: string | Float32Array | null): Float32Array | null {
    this.vozActual = this.resolverVoz(voz);
    return this.vozActual;
  }

  /**
   * A recording -> the voice to use. Accepts a File/Blob (any format the browser
   * decodes), an ArrayBuffer of an audio file, or raw samples with their rate.
   * Three seconds or more. Returns the 256-d vector and makes it the current voice.
   */
  async clonar(
    audio: Blob | ArrayBuffer | Float32Array,
    sampleRate?: number,
  ): Promise<Float32Array> {
    let onda: Float32Array;
    if (audio instanceof Float32Array) {
      if (!sampleRate) throw new Error("clonar: raw samples need their sampleRate");
      onda =
        sampleRate === this.sampleRate
          ? audio
          : await remuestrear(audio, sampleRate, this.sampleRate);
    } else {
      const datos = audio instanceof Blob ? await audio.arrayBuffer() : audio;
      onda = await decodificar(datos, this.sampleRate);
    }
    if (onda.length < this.sampleRate * 2) {
      throw new Error(
        `clonar: ${(onda.length / this.sampleRate).toFixed(1)} s of audio; 3 s or more needed`,
      );
    }
    this.vozActual = await vectorVoz(this.sesionVoz, this.contrato, onda);
    return this.vozActual;
  }

  /** One text -> one result. A stream of texts -> results in order. */
  predict(texto: string, opciones?: OpcionesPredict): Promise<Resultado>;
  predict(fuente: Fuente, opciones?: OpcionesPredict): AsyncIterable<Resultado>;
  predict(
    entrada: string | Fuente,
    opciones: OpcionesPredict = {},
  ): Promise<Resultado> | AsyncIterable<Resultado> {
    if (typeof entrada === "string") return this.uno(entrada, undefined, opciones);
    return this.varios(entrada, opciones);
  }

  /** Synthesize and play one text. */
  async leer(texto: string, opciones: OpcionesPredict = {}): Promise<Resultado> {
    const r = await this.uno(texto, undefined, opciones);
    await this.reproducir(r);
    return r;
  }

  /** Play a result; resolves when it ends. */
  reproducir(r: { onda: Float32Array; sampleRate: number }): Promise<void> {
    return new Promise((resolve) => {
      const nodo = reproducirOnda(r.onda, r.sampleRate);
      nodo.onended = () => resolve();
    });
  }

  // ---------------------------------------------------------------- internals

  private resolverVoz(voz: string | Float32Array | null | undefined): Float32Array | null {
    if (voz === undefined) return this.vozActual;
    if (voz === null) return null;
    if (voz instanceof Float32Array) return voz;
    const preset = this.presets.voces.find((v) => v.id === voz);
    if (!preset) throw new Error(`no hay ninguna voz preset con id ${voz}`);
    return Float32Array.from(preset.voz);
  }

  private async uno(
    texto: string,
    usuario: string | undefined,
    porLlamada: OpcionesPredict,
  ): Promise<Resultado> {
    const opciones: OpcionesPredict = { ...this.ajustes, ...porLlamada };
    let dicho = opciones.chat ? normalizarChat(texto) : texto;
    if (!dicho.trim()) throw new Error("predict: nothing readable in the text");
    if (usuario && (opciones.leerNombre ?? true)) dicho = `${nombreLegible(usuario)}: ${dicho}`;
    const t0 = performance.now();
    const clave = opciones.vozBase ?? this.vozBaseActual;
    if (!this.bases.has(clave)) await this.cargarVozBase(clave);
    const hablante = this.bases.get(clave) as VozBase;
    const base = await sintetizar(
      hablante.sesion,
      hablante.contrato,
      dicho,
      hablante.contrato.idiomas[0],
      clave === "local" && this.presets.base.embedding_tts
        ? Float32Array.from(this.presets.base.embedding_tts)
        : null,
      opciones,
    );
    let onda = base.onda;
    let sampleRate = base.sampleRate;
    const voz = this.resolverVoz(opciones.voz);
    if (voz) {
      const c = await convertir(
        this.sesionConversor,
        this.contrato,
        base.onda,
        hablante.vozBase,
        voz,
        { tau: opciones.tau, semilla: opciones.semilla },
      );
      onda = c.onda;
      sampleRate = c.sampleRate;
    }
    return {
      texto: dicho,
      usuario,
      onda,
      sampleRate,
      ms: performance.now() - t0,
      fonemas: base.fonemas,
      wav: () => aWav(onda, sampleRate),
    };
  }

  private async *varios(fuente: Fuente, opciones: OpcionesPredict): AsyncIterable<Resultado> {
    // Items go through the queue (synthesize ahead, drop the stale, count it all)
    // and come out in order as they reach the playback slot. With `reproducir`
    // off that slot is instantaneous.
    const resultados = new WeakMap<Mensaje, Resultado>();
    const listos: Resultado[] = [];
    let aceptados = 0;
    let cerrados = 0;
    let fin = false;
    let ociosa = true;
    let despertar: (() => void) | null = null;
    const avisar = () => {
      const d = despertar;
      despertar = null;
      d?.();
    };
    const cola = new Cola(
      async (m: Mensaje) => {
        const r = await this.uno(m.texto, m.usuario, { ...opciones, leerNombre: false });
        resultados.set(m, r);
        return { onda: r.onda, sampleRate: r.sampleRate };
      },
      async (onda) => {
        if (opciones.reproducir) await this.reproducir(onda);
      },
      (_s: Estadisticas, evento: string, m?: Mensaje) => {
        if (evento === "encolado") ociosa = false;
        if (evento === "vacía") ociosa = true;
        if (evento === "sonando" && m) {
          const r = resultados.get(m);
          if (r) listos.push(r);
          cerrados++;
        } else if ((evento === "viejo" || evento.startsWith("error")) && m) cerrados++;
        avisar();
      },
      { maxEdadMs: opciones.maxEdadMs },
    );
    (async () => {
      for await (const item of iterar(fuente)) {
        const texto = typeof item === "string" ? item : item.texto;
        const usuario = typeof item === "string" ? undefined : item.usuario;
        const dicho = opciones.chat ? normalizarChat(texto) : texto;
        if (!dicho.trim()) continue;
        const prefijo =
          usuario && (opciones.leerNombre ?? true) ? `${nombreLegible(usuario)}: ` : "";
        if (cola.encolar({ texto: prefijo + dicho, usuario })) aceptados++;
      }
      fin = true;
      avisar();
    })();
    while (true) {
      while (listos.length > 0) yield listos.shift() as Resultado;
      const hecho = fin && cerrados === aceptados && (!opciones.reproducir || ociosa);
      if (hecho) break;
      await new Promise<void>((r) => {
        despertar = r;
      });
    }
  }
}

/** Any of the accepted stream shapes -> one async iteration of items. */
async function* iterar(fuente: Fuente): AsyncIterable<Entrada> {
  if (fuente instanceof ReadableStream) {
    const lector = fuente.getReader();
    const decodificador = new TextDecoder();
    let resto = "";
    while (true) {
      const { done, value } = await lector.read();
      if (done) break;
      if (value instanceof Uint8Array) {
        // bytes: one message per line
        resto += decodificador.decode(value, { stream: true });
        const lineas = resto.split("\n");
        resto = lineas.pop() ?? "";
        for (const l of lineas) if (l.trim()) yield l;
      } else yield value;
    }
    if (resto.trim()) yield resto;
    return;
  }
  if (Symbol.asyncIterator in fuente) {
    for await (const x of fuente as AsyncIterable<Entrada>) yield x;
    return;
  }
  for (const x of fuente as Iterable<Entrada>) yield x;
}

async function remuestrear(onda: Float32Array, de: number, a: number): Promise<Float32Array> {
  const ctx = new OfflineAudioContext(1, Math.ceil((onda.length * a) / de), a);
  const buffer = ctx.createBuffer(1, onda.length, de);
  // a fresh copy: the samples may sit on a SharedArrayBuffer, which copyToChannel refuses
  buffer.copyToChannel(new Float32Array(onda), 0);
  const nodo = ctx.createBufferSource();
  nodo.buffer = buffer;
  nodo.connect(ctx.destination);
  nodo.start();
  return (await ctx.startRendering()).getChannelData(0).slice();
}
