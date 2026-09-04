// Copy what the browser must fetch same-origin into public/: ORT Web's wasm
// runtime, espeak-ng's wasm, and the exported models + contract. Same-origin
// matters because the page runs cross-origin isolated (COOP/COEP) for
// multithreaded wasm (ADR 0005), which blocks third-party resources without CORP.
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const raiz = resolve(web, "..");
const publico = join(web, "public");

function copiar(origen, destino) {
  mkdirSync(dirname(destino), { recursive: true });
  // Size alone is not enough: a re-exported model weighs the same and would be
  // skipped (2026-09-05: the demo kept serving the previous weights).
  if (existsSync(destino)) {
    const a = statSync(origen);
    const b = statSync(destino);
    if (a.size === b.size && a.mtimeMs <= b.mtimeMs) return false;
  }
  copyFileSync(origen, destino);
  return true;
}

let copiados = 0;
// ORT Web's wasm/mjs are NOT copied: the runtime lets ORT find them next to its
// own module (Vite handles that in dev and in build).
copiados += copiar(
  join(web, "node_modules", "espeak-ng", "dist", "espeak-ng.wasm"),
  join(publico, "espeak", "espeak-ng.wasm"),
)
  ? 1
  : 0;
const modelos = join(raiz, "models");
const faltan = [];
for (const f of [
  "contrato.json",
  "voces.json",
  "tts.export.json",
  "speaker_encoder.onnx",
  "speaker_encoder.fp16.onnx",
  "tts.onnx",
  "tts.fp16.onnx",
]) {
  const origen = join(modelos, f);
  if (!existsSync(origen)) {
    faltan.push(f);
    continue;
  }
  copiados += copiar(origen, join(publico, "models", f)) ? 1 : 0;
}
console.log(`preparar-estaticos: ${copiados} fichero(s) copiado(s) a web/public/`);
if (faltan.length) console.warn(`  faltan en models/ (exporta primero): ${faltan.join(", ")}`);
