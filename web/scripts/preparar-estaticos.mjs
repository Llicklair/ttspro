// Copy what the browser must fetch same-origin into public/: ORT Web's wasm
// runtime and the Supertonic graphs, voices and contract. Same-origin matters
// because the page runs cross-origin isolated (COOP/COEP) for multithreaded wasm
// (ADR 0005), which blocks third-party resources without CORP.
//
// espeak-ng is no longer copied: ADR 0012 removed the phonemizer, and with it
// 17.6 MB of wasm and a GPL-3.0 obligation on every deployed build.
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from "node:fs";
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
const faltan = [];
const supertonic = join(raiz, "models", "supertonic");

// Los cuatro grafos y el contrato. 398 MB: la copia tarda, pero solo la primera vez.
for (const f of [
  "contrato.json",
  join("onnx", "tts.json"),
  join("onnx", "unicode_indexer.json"),
  join("onnx", "duration_predictor.onnx"),
  join("onnx", "text_encoder.onnx"),
  join("onnx", "vector_estimator.onnx"),
  join("onnx", "vocoder.onnx"),
  "LICENSE", // OpenRAIL-M: viaja con los pesos, siempre
]) {
  const origen = join(supertonic, f);
  if (!existsSync(origen)) {
    faltan.push(f);
    continue;
  }
  copiados += copiar(origen, join(publico, "models", "supertonic", f)) ? 1 : 0;
}

// Las voces: las diez de fábrica y las que haya construido el voice builder.
// Se escribe además un indice.json con lo que hay, para que la página no dependa
// del contrato (que solo conoce las diez de fábrica) ni de una lista a mano: una
// voz nueva aparece sola en cuanto pasa por aquí.
const voces = join(supertonic, "voice_styles");
if (existsSync(voces)) {
  const nombres = [];
  for (const f of readdirSync(voces).filter((n) => n.endsWith(".json") && n !== "indice.json")) {
    copiados += copiar(join(voces, f), join(publico, "models", "supertonic", "voice_styles", f))
      ? 1
      : 0;
    nombres.push(f.replace(/\.json$/, ""));
  }
  const indice = join(publico, "models", "supertonic", "voice_styles", "indice.json");
  mkdirSync(dirname(indice), { recursive: true });
  writeFileSync(indice, JSON.stringify(nombres.sort(), null, 1));
} else {
  faltan.push("voice_styles/");
}

// El puente y su encoder, si ya se generaron: sin ellos la página no ofrece clonar.
for (const f of ["puente.json", "puente.bin", "speaker_encoder.onnx"]) {
  const origen = join(supertonic, f);
  if (existsSync(origen)) {
    copiados += copiar(origen, join(publico, "models", "supertonic", f)) ? 1 : 0;
  }
}

console.log(`preparar-estaticos: ${copiados} fichero(s) copiado(s) a web/public/`);
if (faltan.length) {
  console.warn(
    `  faltan en models/supertonic/ (descargalos con \`uv run python -m ttspro.supertonic.descargar\`, o usa el boton de Hugging Face en la pagina): ${faltan.join(", ")}`,
  );
}
