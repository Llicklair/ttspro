// `vitest/config` and not `vite`: the same file carries the unit-test
// settings, so `vitest run` does not try to collect the Playwright specs.
import { defineConfig } from "vitest/config";
import { espeakFueraDelBuild } from "./scripts/espeak-fuera-del-build.mjs";

// COOP/COEP: cross-origin isolation, required for multithreaded wasm in ORT Web
// (ADR 0005). Everything the page fetches is served same-origin from public/.
const aislamiento = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [espeakFueraDelBuild()],
  server: { headers: aislamiento, port: 5173 },
  preview: { headers: aislamiento },
  build: { target: "esnext" },
  // El worker de pocket-tts-onnx importa onnxruntime, asi que su bundle sale en
  // varios trozos y el formato `iife` que Vite usa por defecto para workers no
  // admite mas de uno. Con `es` se empaqueta como modulo y se acabo.
  // El Worker es otro build de rollup y no hereda `plugins`: hay que darselo.
  worker: { format: "es", plugins: () => [espeakFueraDelBuild()] },
  optimizeDeps: { exclude: ["onnxruntime-web", "espeak-ng"] },
  test: { include: ["src/**/*.test.ts"] },
});
