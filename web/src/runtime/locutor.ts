/** Reference audio -> speaker embedding through speaker_encoder.onnx (ADR 0002). */

import type { Contrato } from "./contrato.ts";
import { comprobarFirma } from "./contrato.ts";
import { type Sesion, ort } from "./ort.ts";

export interface Locutor {
  embedding: Float32Array;
  segundos: number;
  ms: number;
}

export async function calcularEmbedding(
  encoder: Sesion,
  contrato: Contrato,
  onda16k: Float32Array,
): Promise<Locutor> {
  const grafo = contrato.grafos.speaker_encoder;
  comprobarFirma(grafo, [...encoder.sesion.inputNames], [...encoder.sesion.outputNames]);
  const t0 = performance.now();
  const entrada = new ort.Tensor("float32", onda16k, [1, onda16k.length]);
  const salida = await encoder.sesion.run({ [grafo.entradas[0].nombre]: entrada });
  const embedding = salida[grafo.salidas[0].nombre].data as Float32Array;
  return {
    embedding: new Float32Array(embedding),
    segundos: onda16k.length / 16000,
    ms: performance.now() - t0,
  };
}
