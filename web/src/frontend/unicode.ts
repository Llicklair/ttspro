/**
 * Text -> token ids for the Supertonic engine, with no phonemizer at all.
 * Mirror of `src/ttspro/supertonic/texto.py` — keep rule-for-rule identical (rule 3),
 * and `npm test` compares the two over `tests/fixtures/frontend/`.
 *
 * Supertonic reads characters, not phonemes: every code point goes through
 * `unicode_indexer.json` (a flat table indexed by code point) and the sentence is
 * wrapped in a language tag, `<es>…</es>`. That is the whole frontend — espeak-ng
 * and its 17.6 MB of wasm and its GPL obligation are gone (ADR 0012).
 */

export const IDIOMAS = [
  "en",
  "ko",
  "ja",
  "ar",
  "bg",
  "cs",
  "da",
  "de",
  "el",
  "es",
  "et",
  "fi",
  "fr",
  "hi",
  "hr",
  "hu",
  "id",
  "it",
  "lt",
  "lv",
  "nl",
  "pl",
  "pt",
  "ro",
  "ru",
  "sk",
  "sl",
  "sv",
  "tr",
  "uk",
  "vi",
  "na",
] as const;

export type Idioma = (typeof IDIOMAS)[number];

/** Code point outside the table. ONNX Gather reads -1 as the last row, which is
 * where the checkpoint keeps its unknown slot; do not "fix" this to 0. */
export const DESCONOCIDO = -1;

const EMOJI =
  /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F1E6}-\u{1F1FF}]+/gu;

const SUSTITUCIONES: Record<string, string> = {
  "–": "-",
  "‑": "-",
  "—": "-",
  _: " ",
  "“": '"',
  "”": '"',
  "‘": "'",
  "’": "'",
  "´": "'",
  "`": "'",
  "[": " ",
  "]": " ",
  "|": " ",
  "/": " ",
  "#": " ",
  "→": " ",
  "←": " ",
};
const SIMBOLOS = /[♥☆♡©\\]/g;
const EXPRESIONES: Record<string, string> = {
  "@": " at ",
  "e.g.,": "for example, ",
  "i.e.,": "that is, ",
};
const ESPACIO_ANTES = / ([,.!?;:'])/g;
const FIN = /[.!?;:,'"')\]}…。」』】〉》›»]$/;

/**
 * Normalize a sentence and wrap it in its language tag.
 *
 * Upstream calls this a placeholder ("Need advanced normalizer"). It stays
 * rule-for-rule faithful anyway: the checkpoint was trained on text that went
 * through exactly these steps, so improving them here would move the input off
 * the distribution the weights know. What the page *does* put in front of it is
 * `normalizar.ts`, which is a different job — Spanish numbers, chat, abbreviations.
 */
export function preprocesar(texto: string, idioma: string): string {
  if (!(IDIOMAS as readonly string[]).includes(idioma)) {
    throw new Error(`idioma no soportado: ${idioma}; hay ${IDIOMAS.length}: ${IDIOMAS.join(", ")}`);
  }
  let t = texto.normalize("NFKD");
  t = t.replace(EMOJI, "");
  for (const [k, v] of Object.entries(SUSTITUCIONES)) t = t.replaceAll(k, v);
  t = t.replace(SIMBOLOS, "");
  for (const [k, v] of Object.entries(EXPRESIONES)) t = t.replaceAll(k, v);
  t = t.replace(ESPACIO_ANTES, "$1");
  while (t.includes('""')) t = t.replace('""', '"');
  while (t.includes("''")) t = t.replace("''", "'");
  while (t.includes("``")) t = t.replace("``", "`");
  t = t.replace(/\s+/g, " ").trim();
  if (!FIN.test(t)) t += ".";
  return `<${idioma}>${t}</${idioma}>`;
}

/** The code points `codePointAt` yields over every UTF-16 code unit. The Python
 * side walks UTF-16 too, on purpose, so an astral character tokenizes the same
 * on both. */
export function puntosDeCodigo(texto: string): number[] {
  const salida: number[] = [];
  for (let i = 0; i < texto.length; i++) salida.push(texto.codePointAt(i) as number);
  return salida;
}

export class Indexador {
  // Sin propiedad de parámetro: `node --experimental-strip-types` (el arnés de
  // paridad, rule 3) no la soporta, y este fichero tiene que correr ahí sin build.
  readonly tabla: number[];

  constructor(tabla: number[]) {
    this.tabla = tabla;
  }

  static async descargar(url: string): Promise<Indexador> {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`no se pudo descargar ${url}: ${r.status}`);
    return new Indexador(await r.json());
  }

  ids(texto: string): number[] {
    return puntosDeCodigo(texto).map((c) => (c < this.tabla.length ? this.tabla[c] : DESCONOCIDO));
  }

  /** `[B, L]` ids padded with 0 and the `[B, 1, L]` float mask both graphs want. */
  tokenizar(
    textos: string[],
    idiomas: string[],
  ): { ids: BigInt64Array; mascara: Float32Array; largo: number } {
    const filas = textos.map((t, i) => this.ids(preprocesar(t, idiomas[i])));
    const largo = Math.max(...filas.map((f) => f.length));
    const ids = new BigInt64Array(filas.length * largo);
    const mascara = new Float32Array(filas.length * largo);
    filas.forEach((fila, b) => {
      for (let i = 0; i < fila.length; i++) {
        ids[b * largo + i] = BigInt(fila[i]);
        mascara[b * largo + i] = 1;
      }
    });
    return { ids, mascara, largo };
  }
}

const ABREVIATURAS =
  /(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+/;

/**
 * Split a long text into chunks the model synthesizes one at a time.
 *
 * Not the same job as `trocear.ts`, which splits a sentence on punctuation for
 * the phoneme frontend. This one splits a *document* into sentence groups under
 * `maximo` characters, because attention over the whole thing degrades past that.
 */
export function trocearLargo(texto: string, maximo = 300): string[] {
  const trozos: string[] = [];
  for (const parrafo of texto.trim().split(/\n\s*\n+/)) {
    const p = parrafo.trim();
    if (!p) continue;
    let actual = "";
    for (const frase of p.split(ABREVIATURAS)) {
      if (actual.length + frase.length + 1 <= maximo) {
        actual += (actual ? " " : "") + frase;
      } else {
        if (actual) trozos.push(actual.trim());
        actual = frase;
      }
    }
    if (actual) trozos.push(actual.trim());
  }
  return trozos;
}
