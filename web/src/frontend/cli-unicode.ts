/**
 * Parity harness for the Supertonic frontend (rule 3 / 6): the Python test runs
 * this as a process. Input: one `idioma<TAB>frase` per stdin line. Output: one
 * JSON object per line with every stage, so a failure names WHICH stage diverged.
 *
 * Separate from `cli.ts` on purpose: that one needs espeak-ng from node_modules,
 * and this frontend has no dependencies at all — which is the whole point of
 * ADR 0012. Node >= 22.6 strips types natively: no build step.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { Indexador, preprocesar, puntosDeCodigo, trocearLargo } from "./unicode.ts";

const RAIZ = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const tabla = JSON.parse(
  readFileSync(resolve(RAIZ, "models/supertonic/onnx/unicode_indexer.json"), "utf8"),
);
const indexador = new Indexador(tabla);

const lineas = createInterface({ input: process.stdin, crlfDelay: Number.POSITIVE_INFINITY });
for await (const linea of lineas) {
  if (linea.trim() === "") continue;
  const tab = linea.indexOf("\t");
  const idioma = linea.slice(0, tab);
  const frase = linea.slice(tab + 1);
  // `trocear<TAB>texto`: only the long-text splitter, which takes no language.
  if (idioma === "trocear") {
    process.stdout.write(`${JSON.stringify({ idioma, frase, trozos: trocearLargo(frase) })}\n`);
    continue;
  }
  const preprocesado = preprocesar(frase, idioma);
  process.stdout.write(
    `${JSON.stringify({
      idioma,
      frase,
      preprocesado,
      puntos: puntosDeCodigo(preprocesado),
      ids: indexador.ids(preprocesado),
    })}\n`,
  );
}
