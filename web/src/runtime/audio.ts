/** Browser audio helpers: decode + resample to mono Float32, record, play, wav. */

export async function decodificar(datos: ArrayBuffer, sampleRate: number): Promise<Float32Array> {
  const ctx = new AudioContext();
  const buffer = await ctx.decodeAudioData(datos.slice(0));
  await ctx.close();
  const longitud = Math.ceil(buffer.duration * sampleRate);
  const offline = new OfflineAudioContext(1, longitud, sampleRate);
  const fuente = offline.createBufferSource();
  fuente.buffer = buffer;
  fuente.connect(offline.destination);
  fuente.start();
  const salida = await offline.startRendering();
  return salida.getChannelData(0);
}

export async function grabar(segundos: number, alProgresar?: (s: number) => void): Promise<Blob> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const grabadora = new MediaRecorder(stream);
  const trozos: BlobPart[] = [];
  grabadora.ondataavailable = (e) => trozos.push(e.data);
  const fin = new Promise<Blob>((resolve) => {
    grabadora.onstop = () => resolve(new Blob(trozos, { type: grabadora.mimeType }));
  });
  grabadora.start();
  for (let s = 1; s <= segundos; s++) {
    await new Promise((r) => setTimeout(r, 1000));
    alProgresar?.(s);
  }
  grabadora.stop();
  for (const pista of stream.getTracks()) pista.stop();
  return fin;
}

/**
 * Raw PCM from the microphone at `sampleRate`, no container, no codec:
 * `decodeAudioData` refuses some MediaRecorder webm blobs (Chrome, 2026-09-04:
 * "Unable to decode audio data"), so the samples are taken straight from the
 * graph. ScriptProcessorNode is deprecated but universal; 5 s is nothing.
 */
export async function grabarPCM(
  segundos: number,
  sampleRate: number,
  alProgresar?: (s: number) => void,
): Promise<Float32Array> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const ctx = new AudioContext({ sampleRate });
  const fuente = ctx.createMediaStreamSource(stream);
  const procesador = ctx.createScriptProcessor(4096, 1, 1);
  const trozos: Float32Array[] = [];
  procesador.onaudioprocess = (e) => trozos.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  fuente.connect(procesador);
  procesador.connect(ctx.destination);
  for (let s = 1; s <= segundos; s++) {
    await new Promise((r) => setTimeout(r, 1000));
    alProgresar?.(s);
  }
  procesador.disconnect();
  fuente.disconnect();
  for (const pista of stream.getTracks()) pista.stop();
  await ctx.close();
  const total = trozos.reduce((n, t) => n + t.length, 0);
  const onda = new Float32Array(total);
  let pos = 0;
  for (const t of trozos) {
    onda.set(t, pos);
    pos += t.length;
  }
  return onda;
}

let contexto: AudioContext | null = null;
export function reproducir(onda: Float32Array, sampleRate: number): AudioBufferSourceNode {
  contexto ??= new AudioContext();
  const buffer = contexto.createBuffer(1, onda.length, sampleRate);
  buffer.copyToChannel(new Float32Array(onda), 0); // copy: SharedArrayBuffer-backed views are rejected
  const fuente = contexto.createBufferSource();
  fuente.buffer = buffer;
  fuente.connect(contexto.destination);
  fuente.start();
  return fuente;
}

export function aWav(onda: Float32Array, sampleRate: number): Blob {
  const n = onda.length;
  const buffer = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buffer);
  const escribir = (pos: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(pos + i, s.charCodeAt(i));
  };
  escribir(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  escribir(8, "WAVE");
  escribir(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  escribir(36, "data");
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const x = Math.max(-1, Math.min(1, onda[i]));
    v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/** N(0, 1) samples (Box-Muller). The graphs take their noise as inputs (rule 5). */
export function ruidoNormal(n: number, semilla = 0): Float32Array {
  const salida = new Float32Array(n);
  let estado = semilla >>> 0 || 0x9e3779b9;
  const uniforme = () => {
    // xorshift32
    estado ^= estado << 13;
    estado ^= estado >>> 17;
    estado ^= estado << 5;
    return ((estado >>> 0) + 1) / 4294967297;
  };
  for (let i = 0; i < n; i += 2) {
    const u1 = uniforme();
    const u2 = uniforme();
    const r = Math.sqrt(-2 * Math.log(u1));
    salida[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) salida[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return salida;
}
