/**
 * Deterministic text normalization. Mirror of `src/ttspro/frontend/normalizar.py`.
 *
 * Keep the two files rule-for-rule identical, in the same order. The Python
 * parity test runs both over `tests/fixtures/frontend/frases.txt`.
 */

const ESPACIOS = /\s+/g;

// Typographic quotes and dashes that espeak-ng does not read consistently.
const TIPOGRAFIA: Record<string, string> = {
  "‘": "'",
  "’": "'",
  "“": '"',
  "”": '"',
  "–": "-",
  "—": "-",
  " ": " ",
};
const TIPOGRAFIA_RE = new RegExp(`[${Object.keys(TIPOGRAFIA).join("")}]`, "g");

/**
 * Rule order matters and must match the Python mirror exactly.
 *
 * 1. Unicode NFC, so composed and decomposed accents tokenize the same.
 * 2. Typographic punctuation -> ASCII.
 * 3. Collapse whitespace runs to one space; strip the ends.
 */
export function normalizar(texto: string): string {
  let salida = texto.normalize("NFC");
  salida = salida.replace(TIPOGRAFIA_RE, (c) => TIPOGRAFIA[c] ?? c);
  salida = salida.replace(ESPACIOS, " ").trim();
  return salida;
}
