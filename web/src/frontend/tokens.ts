/**
 * Phoneme string <-> token ids from the symbol table in `models/contrato.json`.
 * Mirror of `src/ttspro/frontend/tokens.py`.
 *
 * Sequence = elements of `trocear` in order (phonemized chunks and punctuation
 * marks) joined by one space, then one id per character. An unknown character
 * throws — never a silent <unk>.
 */

import { fonemizarTrozos } from "./fonemas.ts";
import { normalizar } from "./normalizar.ts";
import { trocear } from "./trocear.ts";

export interface Simbolos {
  pad: number;
  tabla: string[];
  blank_entre_tokens?: boolean;
  blank_al_inicio?: boolean;
  bos?: number | null;
  eos?: number | null;
  /** symbol -> replacement ("" drops it); declared, never silent. */
  equivalencias?: Record<string, string>;
}

export async function fonemasDeFrase(texto: string, idioma: string): Promise<string> {
  const trozos = trocear(normalizar(texto));
  const textos = trozos.filter((t) => t.tipo === "texto").map((t) => t.valor);
  const fonemas = await fonemizarTrozos(textos, idioma);
  const elementos: string[] = [];
  let i = 0;
  for (const trozo of trozos) {
    if (trozo.tipo === "texto") {
      const fon = fonemas[i++];
      if (fon) elementos.push(fon);
    } else {
      elementos.push(trozo.valor);
    }
  }
  return elementos.join(" ");
}

/**
 * One id per character, wrapped as the model expects (mirrored in tokens.py).
 * The contract decides the shape: `blank_entre_tokens` puts a pad next to every
 * symbol, `blank_al_inicio` says before (coqui) or after (Piper), and `bos`/`eos`
 * wrap the sentence. Getting this wrong makes audio that is almost right.
 */
export function ids(fonemas: string, simbolos: Simbolos): number[] {
  const tabla = new Map(simbolos.tabla.flatMap((s, i) => (s === "" ? [] : [[s, i] as const])));
  const equivalencias = simbolos.equivalencias ?? {};
  const texto = Object.keys(equivalencias).length
    ? [...fonemas].map((c) => equivalencias[c] ?? c).join("")
    : fonemas;
  const desconocidos = [...new Set([...texto].filter((c) => !tabla.has(c)))].sort();
  if (desconocidos.length > 0) {
    throw new Error(
      `símbolos fuera de models/contrato.json: ${JSON.stringify(desconocidos)} en ${JSON.stringify(texto)}. Añádelos a la tabla en los dos lados, no los ignores.`,
    );
  }
  const secuencia = [...texto].map((c) => tabla.get(c) as number);
  const bos = simbolos.bos ?? null;
  const eos = simbolos.eos ?? null;
  if (!simbolos.blank_entre_tokens) {
    return [...(bos !== null ? [bos] : []), ...secuencia, ...(eos !== null ? [eos] : [])];
  }
  const alInicio = simbolos.blank_al_inicio ?? true;
  const salida: number[] = bos !== null ? [bos] : [];
  for (const x of secuencia) {
    if (alInicio) salida.push(simbolos.pad, x);
    else salida.push(x, simbolos.pad);
  }
  if (alInicio) salida.push(simbolos.pad);
  if (eos !== null) salida.push(eos);
  return salida;
}

export async function tokenizar(
  texto: string,
  idioma: string,
  simbolos: Simbolos,
): Promise<{ fonemas: string; ids: number[] }> {
  const fonemas = await fonemasDeFrase(texto, idioma);
  return { fonemas, ids: ids(fonemas, simbolos) };
}
