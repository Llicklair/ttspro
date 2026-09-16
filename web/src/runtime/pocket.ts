/**
 * Pocket TTS: el motor que **sí** sabe clonar desde una grabación.
 *
 * Supertonic lee mejor (WER 0,008 contra 0,027) y sale a 44,1 kHz en 31 idiomas,
 * pero no puede clonar: el encoder que convierte un audio en voz no se publicó
 * nunca, y reconstruirlo no llega — seis intentos medidos en docs/evidencia.md,
 * el mejor 0,227 sobre un objetivo de 0,55, y el definitivo falla incluso
 * reconstruyendo una voz de fábrica desde su propio audio.
 *
 * Este clona en medio segundo y saca **0,413** de similitud, que es el mejor
 * número que ha tenido el proyecto. Lo paga en frecuencia (24 kHz) y en idiomas
 * (7), así que **solo clona**: leer lo hace Supertonic, y las dos voces de
 * fábrica de este (`javert`, `lola`) no se ofrecen en ningún sitio. El usuario
 * no elige motor; elige una voz, y la voz sabe quién la habla.
 *
 * El trabajo pesado corre en un **Worker**: onnxruntime en wasm decodifica de
 * forma síncrona y en el hilo principal congelaría la página y mataría el reloj
 * de audio. Lo que llega aquí son marcos de 80 ms ya hechos.
 *
 * Pesos: 177 MB el modelo, 1,6 MB el tokenizador y las voces, y 39 MB más el
 * encoder. Nada de eso se descarga al abrir la página: esta clase se carga la
 * primera vez que alguien suelta una grabación, y quien solo lee texto no paga
 * un byte. Se quedan en la Cache API, así que la segunda vez es inmediata.
 */

import {
  Engine,
  decodeAudioFile,
  gained,
  measure,
  normalGain,
  resample,
} from "pocket-tts-onnx/browser";
import PocketWorker from "pocket-tts-onnx/worker?worker";

/** Los siete que publica upstream. El español está y es lo que importa aquí. */
export const IDIOMAS_POCKET = [
  "spanish",
  "english",
  "french",
  "german",
  "italian",
  "portuguese",
  "hebrew",
] as const;

export interface OpcionesPocket {
  idioma?: string;
  /** Progreso de la descarga: etapa y bytes. */
  alCargar?: (etapa: string, recibidos: number, total: number) => void;
  /** Dónde está el wasm de onnxruntime, si lo sirves tú. */
  ortWasm?: string;
}

export interface ResultadoPocket {
  onda: Float32Array;
  sampleRate: number;
  ms: number;
  voz: string;
}

export class Pocket {
  private constructor(
    private readonly motor: Engine,
    readonly idioma: string,
  ) {}

  /**
   * Las muestras de cada voz clonada, por nombre.
   *
   * El worker de upstream guarda **una sola**: `let cloned = null`, y cualquier
   * `Float32Array` que le pases se resuelve a esa ultima. Asi que guardar aqui un
   * marcador vacio hacia que clonar A, clonar B y volver a A hablara con B. Se
   * guardan las muestras y se vuelve a clonar al cambiar, que cuesta medio
   * segundo y es lo unico que esa API permite.
   */
  private readonly clonadas = new Map<string, Float32Array>();
  /** Cual tiene el worker ahora mismo, para no reclonar sin motivo. */
  private enElWorker: string | null = null;

  static async cargar(opciones: OpcionesPocket = {}): Promise<Pocket> {
    const idioma = opciones.idioma ?? "spanish";
    const motor = await Engine.load({
      language: idioma,
      // Vite no sabe seguir `new URL("./worker.js", import.meta.url)` dentro de
      // una dependencia: lo inlinea como data URL y pierde sus imports. Hay que
      // dárselo hecho.
      worker: () => new PocketWorker(),
      ...(opciones.ortWasm ? { ortWasmUrl: opciones.ortWasm } : {}),
      onProgress: (etapa, p) => opciones.alCargar?.(etapa, p.loaded, p.total),
    });
    return new Pocket(motor, idioma);
  }

  get sampleRate(): number {
    return this.motor.sampleRate;
  }

  /** Las de fábrica más las que hayas clonado en esta sesión. */
  get voces(): string[] {
    return [...this.motor.voices, ...this.clonadas.keys()];
  }

  get vozPorDefecto(): string {
    return this.motor.defaultVoice;
  }

  /** En que idioma se grabó cada voz, cuando el export lo dice. Sexo no hay. */
  idiomaDeVoz(nombre: string): string | undefined {
    return this.motor.manifest.voiceLanguages?.[nombre];
  }

  /** Lo que el modelo usa si no le dices nada. El deslizador arranca aquí. */
  get pasosPorDefecto(): number {
    return this.motor.defaults.decodeSteps;
  }

  get temperaturaPorDefecto(): number {
    return this.motor.defaults.temperature;
  }

  /** Una grabación -> una voz, en medio segundo. Esto es todo el voice builder. */
  async clonar(onda: Float32Array, nombre: string): Promise<number> {
    // El encoder lee 20 s como mucho; pasarle más es tiempo tirado, y pasarle
    // menos se paga caro: medido el 2026-09-16, la similitud sube de 0,396 con
    // 2 s a 0,527 con 20. Es la palanca de calidad mas grande que hay aqui.
    const maximo = 20 * this.sampleRate;
    const recortada = onda.length > maximo ? onda.slice(0, maximo) : onda;
    // La referencia va **tal cual**, sin nivelar. Nivelarla parecia buena idea
    // —una grabacion de movil llega floja y otra saturando— y resulto ser lo que
    // metia el siseo: medido el 2026-09-16 con una referencia floja y con ruido,
    // el +4x del nivelado cuesta **11,7 dB de senal/ruido** en la salida y sube
    // su suelo 24 dB. El encoder es sensible al nivel absoluto y con la
    // referencia amplificada se queda el ruido como parte de la voz.
    await this.motor.clone(recortada);
    this.clonadas.set(nombre, recortada);
    this.enElWorker = nombre;
    return recortada.length / this.sampleRate;
  }

  /** Un fichero de audio de cualquier formato que el navegador sepa abrir. */
  async clonarFichero(fichero: File | ArrayBuffer, nombre: string): Promise<number> {
    const { samples, sampleRate } = await decodeAudioFile(fichero);
    // El encoder quiere la frecuencia del modelo; `resample` es el del paquete,
    // sinc con ventana de Kaiser, que para un prompt de voz es de sobra.
    const onda =
      sampleRate === this.sampleRate ? samples : resample(samples, sampleRate, this.sampleRate);
    return this.clonar(onda, nombre);
  }

  /**
   * Texto -> onda. Los marcos llegan de 80 ms; aquí se juntan porque la página
   * dibuja y descarga la frase entera. Para reproducir según sale, usa
   * `marcos()` con el `FramePlayer` del paquete.
   */
  async sintetizar(
    texto: string,
    voz: string,
    opciones: { pasos?: number; temperatura?: number; semilla?: number } = {},
  ): Promise<ResultadoPocket> {
    const t0 = performance.now();
    const trozos: Float32Array[] = [];
    for await (const marco of this.marcos(texto, voz, opciones)) trozos.push(marco);
    const cruda = new Float32Array(trozos.reduce((n, t) => n + t.length, 0));
    let pos = 0;
    for (const t of trozos) {
      cruda.set(t, pos);
      pos += t.length;
    }
    // La salida si se nivela: es una ganancia constante, asi que **no puede**
    // cambiar la relacion senal/ruido, solo iguala el volumen entre voces, que
    // en el chat se nota mas que la calidad.
    return { onda: nivelar(cruda), sampleRate: this.sampleRate, ms: performance.now() - t0, voz };
  }

  /** Los marcos según los va decodificando el worker. */
  async *marcos(
    texto: string,
    voz: string,
    opciones: { pasos?: number; temperatura?: number; semilla?: number } = {},
  ): AsyncGenerator<Float32Array> {
    const muestras = this.clonadas.get(voz);
    if (muestras && this.enElWorker !== voz) {
      // Otra voz clonada ocupa el worker: hay que devolver esta a su sitio.
      await this.motor.clone(muestras);
      this.enElWorker = voz;
    }
    const cual = muestras ?? voz;
    // Sin temperatura, la del modelo. El bundle español no declara ninguna (el
    // inglés usa 0,2), asi que el valor efectivo lo pone el checkpoint: forzar
    // aqui un numero medido con OTRO export seria adivinar.
    yield* this.motor.speak(texto, cual, {
      decodeSteps: opciones.pasos,
      ...(opciones.temperatura === undefined ? {} : { temperature: opciones.temperatura }),
      seed: opciones.semilla,
    });
  }

  /** Suelta el worker y el modelo, para dejar sitio a otro motor. */
  cerrar(): void {
    this.motor.dispose();
  }
}

/** A −18 dB RMS con techo en −1 dB, que son los objetivos del propio paquete. */
function nivelar(onda: Float32Array): Float32Array {
  if (onda.length === 0) return onda;
  const { rms, peak } = measure([onda]);
  if (!(rms > 0)) return onda;
  return gained(onda, normalGain(rms, peak));
}
