/**
 * IPA phonemes from the espeak-ng WASM build (npm `espeak-ng`, ADR 0004).
 *
 * The package is a CLI build: one Emscripten instantiation runs `main` once
 * (~125 ms, dominated by mounting the data filesystem — measured 2026-09-04,
 * docs/evidencia.md). So all chunks of a sentence go in ONE call, one per
 * line, each with " ." appended: espeak then closes exactly one clause per
 * line and the output lines align with the input chunks.
 *
 * The text is passed as an ARGUMENT, not through `-f`: espeak-ng 1.52 appends
 * a garbage clause when reading a file (nondeterministic, 14 of 20 runs on
 * Windows). Python feeds the same text through stdin; inside espeak both are
 * the same memory-buffer path.
 */

import ESpeakNg from "espeak-ng";

// Language id used by the model -> espeak-ng voice. Same table in fonemas.py.
const VOCES: Record<string, string> = { es: "es", en: "en-us" };

export async function fonemizarTrozos(trozos: string[], idioma: string): Promise<string[]> {
  if (trozos.length === 0) return [];
  const voz = VOCES[idioma] ?? idioma;
  const flujo = `${trozos.map((t) => `${t} .`).join("\n")}\n`;
  const modulo = await ESpeakNg({
    arguments: ["-q", "--ipa", "-v", voz, "--phonout", "salida.txt", flujo],
  });
  const salida = modulo.FS.readFile("salida.txt", { encoding: "utf8" });
  const lineas = salida
    .replace(/\r/g, "")
    .split("\n")
    .filter((l) => l.length > 0);
  if (lineas.length !== trozos.length) {
    throw new Error(
      `espeak-ng devolvió ${lineas.length} líneas para ${trozos.length} trozos: ${JSON.stringify(trozos)} -> ${JSON.stringify(lineas)}`,
    );
  }
  return lineas.map((l) => l.trim());
}

export async function versionEspeak(): Promise<string> {
  const lineas: string[] = [];
  await ESpeakNg({ arguments: ["--version"], print: (s: string) => lineas.push(s) });
  return lineas.join(" ").trim();
}
