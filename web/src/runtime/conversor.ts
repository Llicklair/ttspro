/**
 * Tone-colour conversion in the browser (ADR 0007): the TTS says the words in a
 * fixed base voice and this repaints the timbre.
 *
 * Two graphs, on purpose:
 * - `voz.onnx` turns a reference recording into a 256-d voice vector. It carries
 *   a GRU, which WebGPU has no kernel for, so it is created on `wasm` — it runs
 *   once per voice, not per sentence, and nobody notices.
 * - `conversor.onnx` runs per sentence and stays inside the WebGPU op set.
 */

import { ruidoNormal } from "./audio.ts";
import type { Contrato } from "./contrato.ts";
import { comprobarFirma } from "./contrato.ts";
import { type Sesion, ort } from "./ort.ts";

export interface Conversion {
  onda: Float32Array;
  sampleRate: number;
  ms: number;
}

/** Reference wave at 22 050 Hz -> voice vector (1, 256, 1). */
export async function vectorVoz(
  sesion: Sesion,
  contrato: Contrato,
  onda22k: Float32Array,
): Promise<Float32Array> {
  const grafo = contrato.grafos.voz;
  comprobarFirma(grafo, [...sesion.sesion.inputNames], [...sesion.sesion.outputNames]);
  const entrada = new ort.Tensor("float32", onda22k, [1, onda22k.length]);
  const salida = await sesion.sesion.run({ onda: entrada });
  return new Float32Array(salida.salida.data as Float32Array);
}

export async function convertir(
  sesion: Sesion,
  contrato: Contrato,
  onda: Float32Array,
  vozOrigen: Float32Array,
  vozDestino: Float32Array,
  opciones: { tau?: number; semilla?: number } = {},
): Promise<Conversion> {
  const grafo = contrato.grafos.conversor;
  comprobarFirma(grafo, [...sesion.sesion.inputNames], [...sesion.sesion.outputNames]);
  const canales = grafo.entradas.find((e) => e.nombre === "ruido")?.forma[1] as number;
  // Any length works: the graph tiles the noise to cover the audio.
  const frames = Math.max(1, Math.ceil(onda.length / 256));
  const t0 = performance.now();
  const salida = await sesion.sesion.run({
    onda: new ort.Tensor("float32", onda, [1, onda.length]),
    voz_origen: new ort.Tensor("float32", vozOrigen, [1, vozOrigen.length, 1]),
    voz_destino: new ort.Tensor("float32", vozDestino, [1, vozDestino.length, 1]),
    ruido: new ort.Tensor("float32", ruidoNormal(canales * frames, (opciones.semilla ?? 0) + 3), [
      1,
      canales,
      frames,
    ]),
    tau: new ort.Tensor("float32", Float32Array.from([opciones.tau ?? 0.3]), []),
  });
  const datos = salida.salida.data as Float32Array;
  return {
    onda: new Float32Array(datos),
    sampleRate: grafo.frecuencia_hz ?? 22050,
    ms: performance.now() - t0,
  };
}
