/**
 * Parity harness (ARCHITECTURE rule 3 / 6): the Python test runs this as a
 * process. Input: one `idioma<TAB>frase` per stdin line. Output: one JSON
 * object per line with every stage, so the test can say WHICH stage diverged.
 * Node >= 22.6 strips types natively: no build step.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { normalizar } from "./normalizar.ts";
import { fonemasDeFrase, ids } from "./tokens.ts";
import { trocear } from "./trocear.ts";

const RAIZ = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const contrato = JSON.parse(readFileSync(resolve(RAIZ, "models/contrato.json"), "utf8"));

const lineas = createInterface({ input: process.stdin, crlfDelay: Number.POSITIVE_INFINITY });
for await (const linea of lineas) {
  if (linea.trim() === "") continue;
  const tab = linea.indexOf("\t");
  const idioma = linea.slice(0, tab);
  const frase = linea.slice(tab + 1);
  const normalizado = normalizar(frase);
  const trozos = trocear(normalizado);
  const fonemas = await fonemasDeFrase(frase, idioma);
  const salida = {
    idioma,
    frase,
    normalizado,
    trozos,
    fonemas,
    ids: ids(fonemas, contrato.simbolos),
  };
  process.stdout.write(`${JSON.stringify(salida)}\n`);
}
