/**
 * The waveform panel: draws the audio, animates a cursor while it plays, and
 * shows a pulse while the models are working.
 *
 * It draws an ENVELOPE (min and max per pixel column), not samples: at 22 050 Hz
 * a three-second clip is 66 000 points for ~800 columns, and plotting them one by
 * one draws the same picture far more slowly.
 */

export type EstadoVisor = "vacio" | "trabajando" | "listo";

export class Visor {
  private ctx: CanvasRenderingContext2D;
  private onda: Float32Array | null = null;
  private sampleRate = 22050;
  private estado: EstadoVisor = "vacio";
  private inicio = 0;
  private animando = false;
  private sobre: number | null = null;

  constructor(private lienzo: HTMLCanvasElement) {
    this.ctx = lienzo.getContext("2d") as CanvasRenderingContext2D;
    new ResizeObserver(() => this.pintar()).observe(lienzo);
    lienzo.addEventListener("pointermove", (e) => {
      const caja = lienzo.getBoundingClientRect();
      this.sobre = (e.clientX - caja.left) / caja.width;
      if (!this.animando) this.pintar();
    });
    lienzo.addEventListener("pointerleave", () => {
      this.sobre = null;
      if (!this.animando) this.pintar();
    });
  }

  private color(nombre: string, alterna: string): string {
    const v = getComputedStyle(this.lienzo).getPropertyValue(nombre).trim();
    return v || alterna;
  }

  private tamano(): { ancho: number; alto: number } {
    const dpr = window.devicePixelRatio || 1;
    const ancho = Math.max(1, Math.floor(this.lienzo.clientWidth * dpr));
    const alto = Math.max(1, Math.floor(this.lienzo.clientHeight * dpr));
    if (this.lienzo.width !== ancho || this.lienzo.height !== alto) {
      this.lienzo.width = ancho;
      this.lienzo.height = alto;
    }
    return { ancho, alto };
  }

  /** min/max per column, computed once per draw. */
  private envolvente(columnas: number): Array<[number, number]> {
    const onda = this.onda as Float32Array;
    const porColumna = onda.length / columnas;
    const salida: Array<[number, number]> = new Array(columnas);
    for (let c = 0; c < columnas; c++) {
      const desde = Math.floor(c * porColumna);
      const hasta = Math.min(onda.length, Math.floor((c + 1) * porColumna));
      let min = 0;
      let max = 0;
      for (let i = desde; i < hasta; i++) {
        const x = onda[i];
        if (x < min) min = x;
        if (x > max) max = x;
      }
      salida[c] = [min, max];
    }
    return salida;
  }

  mostrar(onda: Float32Array, sampleRate: number): void {
    this.onda = onda;
    this.sampleRate = sampleRate;
    this.estado = "listo";
    this.pintar();
  }

  poner(estado: EstadoVisor): void {
    this.estado = estado;
    if (estado !== "listo") this.onda = null;
    if (estado === "trabajando") this.pulso();
    else this.pintar();
  }

  private pulso(): void {
    const paso = () => {
      if (this.estado !== "trabajando") return;
      this.pintar();
      requestAnimationFrame(paso);
    };
    requestAnimationFrame(paso);
  }

  /** Plays and runs the cursor across; resolves when the sound ends. */
  async reproducir(reproductor: (onda: Float32Array, sr: number) => void): Promise<void> {
    if (!this.onda) return;
    reproductor(this.onda, this.sampleRate);
    this.inicio = performance.now();
    this.animando = true;
    const duracion = (this.onda.length / this.sampleRate) * 1000;
    await new Promise<void>((resolve) => {
      const paso = () => {
        this.pintar();
        if (performance.now() - this.inicio < duracion) requestAnimationFrame(paso);
        else {
          this.animando = false;
          this.pintar();
          resolve();
        }
      };
      requestAnimationFrame(paso);
    });
  }

  pintar(): void {
    const { ancho, alto } = this.tamano();
    const ctx = this.ctx;
    const acento = this.color("--acento", "#a78bfa");
    const acento2 = this.color("--acento-2", "#22d3ee");
    const tenue = this.color("--onda-tenue", "#ffffff26");
    ctx.clearRect(0, 0, ancho, alto);

    if (this.estado !== "listo" || !this.onda) {
      const t = performance.now() / 700;
      const filas = 44;
      for (let i = 0; i < filas; i++) {
        const x = (i + 0.5) * (ancho / filas);
        const base = this.estado === "trabajando" ? Math.abs(Math.sin(t + i * 0.35)) : 0.08;
        const h = alto * 0.06 + base * alto * 0.34;
        ctx.fillStyle = this.estado === "trabajando" ? acento : tenue;
        ctx.globalAlpha = this.estado === "trabajando" ? 0.35 + 0.45 * base : 1;
        ctx.fillRect(x - ancho / filas / 5, alto / 2 - h / 2, (ancho / filas) * 0.4, h);
      }
      ctx.globalAlpha = 1;
      return;
    }

    const columnas = Math.max(1, Math.floor(ancho / (2 * (window.devicePixelRatio || 1))));
    const env = this.envolvente(columnas);
    const avance = this.animando
      ? Math.min(
          1,
          (performance.now() - this.inicio) / ((this.onda.length / this.sampleRate) * 1000),
        )
      : 1;
    const degradado = ctx.createLinearGradient(0, 0, ancho, 0);
    degradado.addColorStop(0, acento);
    degradado.addColorStop(1, acento2);
    const anchoCol = ancho / columnas;
    for (let c = 0; c < columnas; c++) {
      const [min, max] = env[c];
      const y1 = alto / 2 - max * alto * 0.46;
      const y2 = alto / 2 - min * alto * 0.46;
      const tocado = c / columnas <= avance;
      ctx.fillStyle = tocado ? degradado : tenue;
      ctx.globalAlpha = tocado ? 1 : 0.85;
      ctx.fillRect(c * anchoCol, y1, Math.max(1, anchoCol * 0.62), Math.max(1, y2 - y1));
    }
    ctx.globalAlpha = 1;

    if (this.animando) {
      ctx.fillStyle = acento2;
      ctx.fillRect(avance * ancho - 1, 0, 2, alto);
    }
    if (this.sobre !== null) {
      const segundos = this.sobre * (this.onda.length / this.sampleRate);
      ctx.fillStyle = tenue;
      ctx.fillRect(this.sobre * ancho - 0.5, 0, 1, alto);
      ctx.fillStyle = this.color("--texto-tenue", "#94a3b8");
      ctx.font = `${12 * (window.devicePixelRatio || 1)}px ui-monospace, monospace`;
      ctx.fillText(
        `${segundos.toFixed(2)} s`,
        Math.min(this.sobre * ancho + 6, ancho - 60),
        16 * (window.devicePixelRatio || 1),
      );
    }
  }
}

/** Small horizontal meter, used live while recording. */
export function pintarNivel(lienzo: HTMLCanvasElement, nivel: number): void {
  const ctx = lienzo.getContext("2d") as CanvasRenderingContext2D;
  const dpr = window.devicePixelRatio || 1;
  const ancho = Math.max(1, Math.floor(lienzo.clientWidth * dpr));
  const alto = Math.max(1, Math.floor(lienzo.clientHeight * dpr));
  if (lienzo.width !== ancho || lienzo.height !== alto) {
    lienzo.width = ancho;
    lienzo.height = alto;
  }
  ctx.clearRect(0, 0, ancho, alto);
  const barras = 32;
  // sqrt so quiet speech still moves the meter: RMS alone barely leaves the floor
  const activo = Math.min(1, Math.sqrt(nivel) * 2.6) * barras;
  const estilo = getComputedStyle(lienzo);
  for (let i = 0; i < barras; i++) {
    const x = (i + 0.2) * (ancho / barras);
    const h = alto * (0.28 + 0.72 * (i / barras));
    ctx.fillStyle =
      i < activo
        ? estilo.getPropertyValue("--acento-2").trim() || "#22d3ee"
        : estilo.getPropertyValue("--linea").trim() || "#ffffff22";
    ctx.fillRect(x, alto - h, (ancho / barras) * 0.55, h);
  }
}
