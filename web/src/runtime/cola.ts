/**
 * A speech queue for a stream of messages (ADR 0010: a Twitch chat reader).
 *
 * Messages arrive faster than they can be spoken, and in bursts. Three rules
 * keep the reader useful instead of two minutes behind:
 *
 *   1. Synthesize AHEAD: the next message is rendered while the current one
 *      plays, so playback is back-to-back and the model never idles.
 *   2. Drop the stale: a message older than `maxEdadMs` when its turn comes is
 *      skipped (nobody wants "hola" read a minute after it scrolled by), and a
 *      queue longer than `maxPendientes` refuses new ones.
 *   3. Drop the duplicate: the same text already waiting is not queued twice.
 *
 * Everything dropped is counted, and the counters are the product's honesty:
 * they say how far behind the reader really is.
 */

export interface Mensaje {
  texto: string;
  /** Who said it; only used by the caller (to pick a voice, to log). */
  usuario?: string;
  /** Voice vector for the converter, or null for the base voice. */
  voz?: Float32Array | null;
  /** performance.now() when it arrived. Set by `encolar` if absent. */
  llegada?: number;
}

export interface Onda {
  onda: Float32Array;
  sampleRate: number;
}

export interface Estadisticas {
  pendientes: number;
  reproducidos: number;
  descartadosViejos: number;
  descartadosLlenos: number;
  descartadosRepetidos: number;
  /** Mean ms from arrival to the start of playback, over what was played. */
  esperaMediaMs: number;
  /** ms the synthesizer spent per message, mean. */
  sintesisMediaMs: number;
  /** Seconds of audio produced per second of synthesis (> 1 means it keeps up). */
  velocidad: number;
}

export interface OpcionesCola {
  maxPendientes?: number;
  maxEdadMs?: number;
}

export type Sintetizador = (m: Mensaje) => Promise<Onda | null>;
export type Reproductor = (onda: Onda) => Promise<void>;

export class Cola {
  private pendientes: Mensaje[] = [];
  /** Texts being rendered or played right now (rule 3 counts them as waiting). */
  private activos = new Set<string>();
  private corriendo = false;
  private detenida = false;
  private readonly maxPendientes: number;
  private readonly maxEdadMs: number;
  private stats: Estadisticas = {
    pendientes: 0,
    reproducidos: 0,
    descartadosViejos: 0,
    descartadosLlenos: 0,
    descartadosRepetidos: 0,
    esperaMediaMs: 0,
    sintesisMediaMs: 0,
    velocidad: 0,
  };
  private sumaEspera = 0;
  private sumaSintesis = 0;
  private sumaAudio = 0;

  constructor(
    private readonly sintetizar: Sintetizador,
    private readonly reproducir: Reproductor,
    private readonly alCambiar: (stats: Estadisticas, evento: string, m?: Mensaje) => void,
    opciones: OpcionesCola = {},
  ) {
    this.maxPendientes = opciones.maxPendientes ?? 20;
    this.maxEdadMs = opciones.maxEdadMs ?? 30_000;
  }

  /** Queue a message. Returns false when it was refused (full or duplicate). */
  encolar(m: Mensaje): boolean {
    m.llegada ??= performance.now();
    if (this.pendientes.length >= this.maxPendientes) {
      this.stats.descartadosLlenos++;
      this.emitir("lleno", m);
      return false;
    }
    // "Waiting" includes the one being rendered or played: the loop shifts a
    // message out of `pendientes` the moment it starts on it.
    if (this.activos.has(m.texto) || this.pendientes.some((p) => p.texto === m.texto)) {
      this.stats.descartadosRepetidos++;
      this.emitir("repetido", m);
      return false;
    }
    this.pendientes.push(m);
    this.emitir("encolado", m);
    if (!this.corriendo) void this.bucle();
    return true;
  }

  vaciar(): void {
    this.pendientes = [];
    this.emitir("vaciada");
  }

  /** Stops after the message currently playing. */
  parar(): void {
    this.detenida = true;
    this.pendientes = [];
  }

  get estadisticas(): Estadisticas {
    return { ...this.stats, pendientes: this.pendientes.length };
  }

  private emitir(evento: string, m?: Mensaje): void {
    this.alCambiar(this.estadisticas, evento, m);
  }

  /** Next message worth saying, skipping the stale ones. */
  private siguiente(): Mensaje | null {
    while (this.pendientes.length > 0) {
      const m = this.pendientes.shift() as Mensaje;
      if (performance.now() - (m.llegada ?? 0) > this.maxEdadMs) {
        this.stats.descartadosViejos++;
        this.emitir("viejo", m);
        continue;
      }
      return m;
    }
    return null;
  }

  private async render(m: Mensaje): Promise<{ m: Mensaje; onda: Onda | null; ms: number }> {
    const t0 = performance.now();
    this.activos.add(m.texto);
    let onda: Onda | null = null;
    try {
      onda = await this.sintetizar(m);
    } catch (err) {
      this.emitir(`error: ${String(err)}`, m);
    }
    if (!onda) this.activos.delete(m.texto);
    return { m, onda, ms: performance.now() - t0 };
  }

  private async bucle(): Promise<void> {
    this.corriendo = true;
    this.detenida = false;
    let proximo = this.siguiente();
    let enCurso = proximo ? this.render(proximo) : null;
    while (enCurso && !this.detenida) {
      const listo = await enCurso;
      // Rule 1: start the next render BEFORE playing this one.
      proximo = this.siguiente();
      enCurso = proximo ? this.render(proximo) : null;
      if (listo.onda) {
        this.sumaSintesis += listo.ms;
        this.sumaAudio += listo.onda.onda.length / listo.onda.sampleRate;
        this.sumaEspera += performance.now() - (listo.m.llegada ?? performance.now());
        this.stats.reproducidos++;
        this.stats.esperaMediaMs = this.sumaEspera / this.stats.reproducidos;
        this.stats.sintesisMediaMs = this.sumaSintesis / this.stats.reproducidos;
        this.stats.velocidad = this.sumaAudio / (this.sumaSintesis / 1000);
        this.emitir("sonando", listo.m);
        await this.reproducir(listo.onda);
        this.activos.delete(listo.m.texto);
      }
      // A message that arrived while the last one played.
      if (!enCurso) {
        proximo = this.siguiente();
        enCurso = proximo ? this.render(proximo) : null;
      }
    }
    this.corriendo = false;
    this.emitir("vacía");
  }
}
