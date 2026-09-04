/**
 * Parity harness (ARCHITECTURE rule 3 / 6): the Python test runs this as a
 * process. One sentence per stdin line, one JSON object per stdout line.
 * Node >= 22.6 strips types natively: no build step.
 */

import { createInterface } from "node:readline";
import { normalizar } from "./normalizar.ts";

const lineas = createInterface({ input: process.stdin, crlfDelay: Number.POSITIVE_INFINITY });
for await (const linea of lineas) {
  if (linea.trim() === "") continue;
  process.stdout.write(`${JSON.stringify({ normalizado: normalizar(linea) })}\n`);
}
