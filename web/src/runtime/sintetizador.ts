/**
 * Text + speaker embedding -> waveform through tts.onnx, feeding exactly the
 * contract (models/contrato.json): tokens from the shared frontend, noise
 * generated here (rule 5), scales as a 3-vector.
 */

import { fonemasDeFrase, ids } from "../frontend/tokens.ts";
import { ruidoNormal } from "./audio.ts";
import type { Contrato } from "./contrato.ts";
import { comprobarFirma } from "./contrato.ts";
import { type Sesion, ort } from "./ort.ts";

export interface Opciones {
  noise_scale?: number; // 0.667
  noise_scale_w?: number; // 0.8
  length_scale?: number; // 1.0
  semilla?: number;
}

export interface Resultado {
  onda: Float32Array;
  sampleRate: number;
  fonemas: string;
  tokens: number;
  ms_frontend: number;
  ms_modelo: number;
  rtf: number;
}

export async function sintetizar(
  tts: Sesion,
  contrato: Contrato,
  texto: string,
  idioma: string,
  embedding: Float32Array,
  opciones: Opciones = {},
): Promise<Resultado> {
  const grafo = contrato.grafos.tts;
  comprobarFirma(grafo, [...tts.sesion.inputNames], [...tts.sesion.outputNames]);
  const indiceIdioma = contrato.idiomas.indexOf(idioma);
  if (indiceIdioma < 0) throw new Error(`idioma ${idioma} no está en el contrato`);

  const t0 = performance.now();
  const fonemas = await fonemasDeFrase(texto, idioma);
  const secuencia = ids(fonemas, contrato.simbolos);
  const L = secuencia.length;
  const ms_frontend = performance.now() - t0;

  const inter = grafo.entradas.find((e) => e.nombre === "ruido_flow")?.forma[1] as number;
  const frames = L * 12; // any length works (the graph tiles), this one avoids tiling
  const semilla = opciones.semilla ?? 0;
  const feeds: Record<string, ort.Tensor> = {
    tokens: new ort.Tensor("int64", BigInt64Array.from(secuencia.map((x) => BigInt(x))), [1, L]),
    longitud_tokens: new ort.Tensor("int64", BigInt64Array.from([BigInt(L)]), [1]),
    embedding: new ort.Tensor("float32", embedding, [1, embedding.length]),
    idioma: new ort.Tensor("int64", BigInt64Array.from([BigInt(indiceIdioma)]), [1]),
    ruido_flow: new ort.Tensor("float32", ruidoNormal(inter * frames, semilla + 1), [
      1,
      inter,
      frames,
    ]),
    ruido_duracion: new ort.Tensor("float32", ruidoNormal(2 * L, semilla + 2), [1, 2, L]),
    escala_ruido: new ort.Tensor(
      "float32",
      Float32Array.from([
        opciones.noise_scale ?? 0.667,
        opciones.noise_scale_w ?? 0.8,
        opciones.length_scale ?? 1.0,
      ]),
      [3],
    ),
  };
  const t1 = performance.now();
  const salida = await tts.sesion.run(feeds);
  const ms_modelo = performance.now() - t1;
  const onda = new Float32Array(salida.onda.data as Float32Array);
  const sampleRate = grafo.frecuencia_salida_hz ?? 22050;
  return {
    onda,
    sampleRate,
    fonemas,
    tokens: L,
    ms_frontend,
    ms_modelo,
    rtf: ms_modelo / 1000 / (onda.length / sampleRate),
  };
}
