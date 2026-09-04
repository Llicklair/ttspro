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

export function ids(fonemas: string, simbolos: Simbolos): number[] {
  const tabla = new Map(simbolos.tabla.map((s, i) => [s, i] as const));
  const desconocidos = [...new Set([...fonemas].filter((c) => !tabla.has(c)))].sort();
  if (desconocidos.length > 0) {
    throw new Error(
      `símbolos fuera de models/contrato.json: ${JSON.stringify(desconocidos)} en ${JSON.stringify(fonemas)}. Añádelos a la tabla en los dos lados, no los ignores.`,
    );
  }
  const secuencia = [...fonemas].map((c) => tabla.get(c) as number);
  if (!simbolos.blank_entre_tokens) return secuencia;
  // VITS add_blank: pad id 0 before, between and after every symbol.
  const conBlank: number[] = new Array(2 * secuencia.length + 1).fill(0);
  secuencia.forEach((id, i) => {
    conBlank[2 * i + 1] = id;
  });
  return conBlank;
}

export async function tokenizar(
  texto: string,
  idioma: string,
  simbolos: Simbolos,
): Promise<{ fonemas: string; ids: number[] }> {
  const fonemas = await fonemasDeFrase(texto, idioma);
  return { fonemas, ids: ids(fonemas, simbolos) };
}
