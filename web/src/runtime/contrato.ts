/**
 * Typed view of `models/contrato.json` (ARCHITECTURE rule 2, ADR 0002).
 * The runtime loads the JSON and checks every session's inputs/outputs against
 * it before the first `run()`: a graph that drifted from the contract fails
 * loudly here instead of producing noise.
 */

export type Dtype = "float32" | "int64";

export interface Tensor {
  nombre: string;
  dtype: Dtype;
  forma: (number | string)[];
  nota?: string;
}

export interface Grafo {
  fichero: string;
  opset: number;
  frecuencia_entrada_hz?: number;
  frecuencia_salida_hz?: number;
  frecuencia_hz?: number;
  entradas: Tensor[];
  salidas: Tensor[];
}

export interface Contrato {
  version: number;
  embedding_locutor: { dim: number; encoder: string };
  /** tts + speaker_encoder (ADR 0002); voz + conversor (ADR 0007). */
  grafos: { speaker_encoder: Grafo; tts: Grafo; voz: Grafo; conversor: Grafo };
  idiomas: string[];
  simbolos: { pad: number; tabla: string[]; blank_entre_tokens?: boolean };
}

export function comprobarFirma(grafo: Grafo, entradas: string[], salidas: string[]): void {
  const esperadasIn = grafo.entradas.map((t) => t.nombre);
  const esperadasOut = grafo.salidas.map((t) => t.nombre);
  if (!iguales(esperadasIn, entradas) || !iguales(esperadasOut, salidas)) {
    throw new Error(
      `${grafo.fichero} no coincide con models/contrato.json: ` +
        `entradas ${JSON.stringify(entradas)} vs ${JSON.stringify(esperadasIn)}, ` +
        `salidas ${JSON.stringify(salidas)} vs ${JSON.stringify(esperadasOut)}`,
    );
  }
}

function iguales(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}
