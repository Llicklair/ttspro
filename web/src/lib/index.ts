/**
 * ttspro as a library: one object, imported from any HTML page.
 *
 *   import { TTS } from "./ttspro.js";
 *   const tts = await TTS.cargar({ modelos: "/models/supertonic/" });
 *   tts.voces;                                       // ["F1", …, "M5"]
 *   await tts.elegirVoz("M3");
 *   const r = await tts.predict("hola a todos");     // -> { onda, sampleRate, ms, wav() }
 *   await tts.reproducir(r);
 *   for await (const r of tts.predict(stream, { chat: true, reproducir: true })) { … }
 *
 * `predict` takes a string and returns one result, or takes a stream (an
 * AsyncIterable, a ReadableStream, an array, or a plain Iterable of strings or
 * `{ usuario, texto }`) and returns an AsyncIterable of results in order,
 * synthesized ahead of playback through the same queue the page uses. Chat lines
 * go through the chat normalizer when `chat: true`.
 *
 * **Lo que cambió con ADR 0012, si venías de la versión anterior.** El motor es
 * Supertonic 3 y la voz es una *entrada* del modelo, así que:
 *
 *   - se fue `espeak`: no hay fonemizador, el frontend es Unicode;
 *   - se fue `precision`: no hay variantes fp16;
 *   - se fueron `vocesBase`, `cargarVozBase` y `elegirVozBase`: ya no hay voz base
 *     contra voz destino, hay **una** lista de voces;
 *   - se fueron `tau`, `noise_scale` y `length_scale`, y entran `pasos`,
 *     `velocidad` y `semilla`;
 *   - `clonar()` sigue existiendo y ahora **funciona**, pero no con Supertonic:
 *     su encoder de voz no se publicó nunca y reconstruirlo no llega (seis
 *     medidas en docs/evidencia.md). Lo hace un segundo motor, Pocket TTS, que
 *     se baja solo la primera vez que se llama.
 *
 * Todo corre en la página: sin servidor. La página tiene que venir por HTTP
 * (fetch no funciona desde file://) y, para wasm multihilo, con las cabeceras
 * COOP/COEP — `scripts/servir.mjs` hace exactamente eso.
 */

import { nombreLegible, normalizarChat } from "../frontend/chat.ts";
import { normalizar } from "../frontend/normalizar.ts";
import { IDIOMAS } from "../frontend/unicode.ts";
import { aWav, ponerVolumen, reproducir as reproducirOnda, volumen } from "../runtime/audio.ts";
import { Cola, type Estadisticas, type Mensaje } from "../runtime/cola.ts";
import { type Hardware, type Recomendacion, detectar, recomendar } from "../runtime/hardware.ts";
import type { Proveedor } from "../runtime/ort.ts";
import { Pocket } from "../runtime/pocket.ts";
import {
  type Cargando,
  type Estilo,
  type EstiloJSON,
  Supertonic,
  descargarEstilo,
  leerEstilo,
} from "../runtime/supertonic.ts";
import { cambiarVelocidad } from "../runtime/velocidad.ts";

export interface OpcionesCarga {
  /** Dónde viven los grafos y las voces. Por defecto `/models/supertonic/`. */
  modelos?: string;
  /** `auto` prueba webgpu y cae a wasm. */
  proveedor?: Proveedor | "auto";
  /** Progreso de la descarga, grafo a grafo. */
  alCargar?: (c: Cargando) => void;
  /** Idioma por defecto de `predict`. Uno de los 31 del motor. */
  idioma?: string;
  /** Qué voz queda elegida al cargar. Por defecto, la primera del origen. */
  voz?: string;
}

export interface Ajustes {
  /** Pasos de flow matching: más es mejor y más lento. 8 es la rodilla. */
  pasos: number;
  /** Velocidad del habla; por encima de 1 habla más rápido. */
  velocidad: number;
  /** Fija el ruido para repetir una toma. `undefined` es aleatorio; **0 es una semilla**. */
  semilla?: number;
  /** Idioma por defecto. */
  idioma: string;
}

export interface OpcionesPredict extends Partial<Ajustes> {
  /** Qué voz dice esto: un nombre ya cargado, o la elegida si se omite. */
  voz?: string;
  /** Pasa el texto por el normalizador de chat (risas, emotes, enlaces). */
  chat?: boolean;
  /** En un stream, reproduce cada resultado en orden según sale. */
  reproducir?: boolean;
  /** Antepone el nombre de quien habla. Por defecto sí. */
  leerNombre?: boolean;
  /** Descarta lo que lleve más de esto esperando en la cola. */
  maxEdadMs?: number;
}

export interface Resultado {
  texto: string;
  usuario?: string;
  onda: Float32Array;
  sampleRate: number;
  ms: number;
  voz: string;
  wav(): Blob;
}

export type Entrada = string | { usuario?: string; texto: string };
export type Fuente =
  | AsyncIterable<Entrada>
  | Iterable<Entrada>
  | ReadableStream<Entrada | Uint8Array>;

export class TTS {
  private constructor(
    private readonly motor: Supertonic,
    private readonly raiz: string,
    readonly voces: string[],
    readonly proveedor: Proveedor,
    readonly hardware: Hardware,
    readonly recomendacion: Recomendacion,
    readonly ajustes: Ajustes,
  ) {}

  private readonly estilos = new Map<string, Estilo>();
  /** Las voces que habla el otro motor: clonadas, sin estilo que guardar. */
  private readonly clonadas = new Set<string>();
  private pocket: Pocket | null = null;
  private vozActual: string | null = null;

  static hardware(): Promise<Hardware> {
    return detectar();
  }

  static async recomendar(): Promise<Recomendacion> {
    return recomendar(await detectar());
  }

  static async cargar(opciones: OpcionesCarga = {}): Promise<TTS> {
    const raiz = (opciones.modelos ?? "/models/supertonic/").replace(/\/?$/, "/");
    const hw = await detectar();
    const rec = recomendar(hw);
    const preferidos: Proveedor[] =
      !opciones.proveedor || opciones.proveedor === "auto"
        ? ["webgpu", "wasm"]
        : [opciones.proveedor];

    const voces = await listaDeVoces(raiz);
    const motor = await Supertonic.cargar(`${raiz}onnx/`, preferidos, opciones.alCargar);
    const tts = new TTS(motor, raiz, voces, motor.proveedor, hw, rec, {
      pasos: 8,
      velocidad: 1.05,
      idioma: opciones.idioma ?? "es",
    });
    await tts.elegirVoz(opciones.voz ?? voces[0]);
    return tts;
  }

  get idiomas(): string[] {
    return IDIOMAS.filter((i) => i !== "na");
  }

  get sampleRate(): number {
    return this.motor.sampleRate;
  }

  get volumen(): number {
    return volumen();
  }

  set volumen(v: number) {
    ponerVolumen(v);
  }

  /** Qué voz habla ahora. */
  get voz(): string | null {
    return this.vozActual;
  }

  /** Elige una voz, descargándola la primera vez. 292 KB cada una. */
  async elegirVoz(nombre: string): Promise<string> {
    if (!this.estilos.has(nombre)) {
      this.estilos.set(
        nombre,
        await descargarEstilo(`${this.raiz}voice_styles/${nombre}.json`, nombre),
      );
      if (!this.voces.includes(nombre)) this.voces.push(nombre);
    }
    this.vozActual = nombre;
    return nombre;
  }

  /**
   * Añade una voz desde un `.json` de estilo: una de fábrica de cualquier
   * runtime de Supertonic, o una construida con `ttspro.supertonic.constructor`.
   */
  async importar(origen: File | Blob | EstiloJSON, nombre?: string): Promise<string> {
    const crudo =
      origen instanceof Blob ? ((await origen.text().then(JSON.parse)) as EstiloJSON) : origen;
    const id =
      nombre ??
      (origen instanceof File
        ? origen.name.replace(/\.json$/i, "")
        : `voz${this.voces.length + 1}`);
    this.estilos.set(id, leerEstilo(crudo, id));
    if (!this.voces.includes(id)) this.voces.push(id);
    this.vozActual = id;
    return id;
  }

  /**
   * Clonar desde una grabación. Devuelve el nombre de la voz, ya elegida.
   *
   * No lo hace Supertonic: Supertone nunca publicó el encoder que convierte un
   * audio en `style_ttl`, y reconstruirlo no llega — seis intentos medidos en
   * docs/evidencia.md, el mejor 0,227 sobre un objetivo de 0,55. Lo hace Pocket
   * TTS, que sí publica el suyo, y que se baja en esta llamada (216 MB, una vez)
   * porque quien solo lee texto no tiene por qué cargar con él.
   *
   * Dale lo más largo que puedas: el encoder lee hasta 20 s y se nota mucho, de
   * 0,40 de similitud con 2 s a 0,53 con 20 (docs/evidencia.md). Y clona solo
   * voces cuyo dueño te haya dado permiso.
   */
  async clonar(fichero: File | Blob, nombre?: string): Promise<string> {
    const id =
      nombre ??
      (fichero instanceof File
        ? fichero.name.replace(/\.[^.]+$/, "")
        : `voz${this.voces.length + 1}`);
    this.pocket ??= await Pocket.cargar({ idioma: "spanish" });
    await this.pocket.clonarFichero(fichero as File, id);
    this.clonadas.add(id);
    if (!this.voces.includes(id)) this.voces.push(id);
    this.vozActual = id;
    return id;
  }

  /** Mide una frase en cada proveedor disponible y devuelve los ms. */
  async calibrar(frase = "Una frase corta para medir."): Promise<Record<string, number>> {
    const medidas: Record<string, number> = {};
    for (const proveedor of ["webgpu", "wasm"] as Proveedor[]) {
      try {
        const otro = await Supertonic.cargar(`${this.raiz}onnx/`, [proveedor]);
        const estilo = this.estiloActual();
        const t0 = performance.now();
        await otro.sintetizar(frase, estilo, { ...this.ajustes });
        medidas[proveedor] = performance.now() - t0;
      } catch {
        // Ese proveedor no está en esta máquina: no es un error, es un dato.
      }
    }
    return medidas;
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

  private estiloActual(nombre?: string): Estilo {
    const id = nombre ?? this.vozActual;
    const estilo = id ? this.estilos.get(id) : undefined;
    if (!estilo) {
      throw new Error(
        `no hay ninguna voz cargada con el nombre ${id}; usa await tts.elegirVoz(nombre)`,
      );
    }
    return estilo;
  }

  private async uno(
    texto: string,
    usuario: string | undefined,
    porLlamada: OpcionesPredict,
  ): Promise<Resultado> {
    const o = { ...this.ajustes, ...porLlamada };
    const cual = o.voz ?? this.vozActual ?? "";
    const clonada = this.clonadas.has(cual);
    if (o.voz && !clonada && !this.estilos.has(o.voz)) await this.elegirVoz(o.voz);
    const estilo = clonada ? null : this.estiloActual(o.voz);

    let dicho = porLlamada.chat ? normalizarChat(texto) : texto;
    if (!dicho.trim()) throw new Error("predict: nothing readable in the text");
    if (usuario && (porLlamada.leerNombre ?? true)) dicho = `${nombreLegible(usuario)}: ${dicho}`;
    // El normalizador español va delante del frontend Unicode: el motor lee
    // "3.522" como dígitos, y esta pieza ya existía y ya está probada.
    const listo = o.idioma === "es" ? normalizar(dicho) : dicho;

    // Una voz clonada la dice el otro motor, que no tiene velocidad dentro: se
    // estira el tiempo después, sin tocar el tono, que es la mitad de lo que
    // hace reconocible a una voz clonada.
    const r =
      estilo === null
        ? await (this.pocket as Pocket)
            // Ese motor no pasa de 4 pasos: pedirle los 8 de Supertonic no es
            // "mejor", es un valor que no admite.
            .sintetizar(listo, cual, { pasos: Math.min(4, o.pasos), semilla: o.semilla })
            .then((x) => ({
              ...x,
              onda: cambiarVelocidad(x.onda, o.velocidad, x.sampleRate),
            }))
        : await this.motor.sintetizar(listo, estilo, {
            idioma: o.idioma,
            pasos: o.pasos,
            velocidad: o.velocidad,
            semilla: o.semilla,
          });
    return {
      texto: dicho,
      usuario,
      onda: r.onda,
      sampleRate: r.sampleRate,
      ms: r.ms,
      voz: estilo?.nombre ?? cual,
      wav: () => aWav(r.onda, r.sampleRate),
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

/**
 * Qué voces hay en este origen. `voice_styles/indice.json` lo escribe
 * `npm run preparar` y lo reescribe el voice builder al publicar una voz, así que
 * una voz nueva aparece sola. Si no existe, valen las del contrato; y si tampoco,
 * las diez de fábrica.
 *
 * Se comprueba la forma, no el código de estado: el servidor de desarrollo de
 * Vite responde 200 con el index.html a lo que no encuentra.
 */
async function listaDeVoces(raiz: string): Promise<string[]> {
  const lista = async (url: string, campo?: string): Promise<string[] | null> => {
    try {
      const r = await fetch(url);
      if (!r.ok) return null;
      const d = (await r.json()) as unknown;
      const v = campo ? (d as Record<string, unknown>)[campo] : d;
      return Array.isArray(v) && v.length && v.every((x) => typeof x === "string")
        ? (v as string[])
        : null;
    } catch {
      return null;
    }
  };
  return (
    (await lista(`${raiz}voice_styles/indice.json`)) ??
    (await lista(`${raiz}contrato.json`, "voces")) ?? [
      "F1",
      "F2",
      "F3",
      "F4",
      "F5",
      "M1",
      "M2",
      "M3",
      "M4",
      "M5",
    ]
  );
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
