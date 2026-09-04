/**
 * Split normalized text into text chunks and punctuation tokens.
 * Mirror of `src/ttspro/frontend/trocear.py` — keep rule-for-rule identical.
 *
 * `.` `,` `:` between digits do not split ("3,5", "3.5", "10:30"); apostrophes
 * inside words do not split; a dash splits only when spaced.
 */

// ASCII [0-9] on purpose: JS \d is ASCII-only, Python \d is not.
// `.` `,` `:` split unless BOTH neighbours are digits (either alternative below).
const SEPARADOR = /(?<![0-9])[.,:]|[.,:](?![0-9])|[;!?¡¿…"()]| - /g;

export interface Trozo {
  tipo: "texto" | "puntuacion";
  valor: string;
}

export function trocear(texto: string): Trozo[] {
  const trozos: Trozo[] = [];
  let pos = 0;
  for (const m of texto.matchAll(SEPARADOR)) {
    pushTexto(trozos, texto.slice(pos, m.index));
    trozos.push({ tipo: "puntuacion", valor: m[0].trim() });
    pos = m.index + m[0].length;
  }
  pushTexto(trozos, texto.slice(pos));
  return trozos;
}

function pushTexto(trozos: Trozo[], fragmento: string): void {
  const limpio = fragmento.trim();
  if (limpio) trozos.push({ tipo: "texto", valor: limpio });
}
