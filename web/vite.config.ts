// `vitest/config` and not `vite`: the same file carries the unit-test
// settings, so `vitest run` does not try to collect the Playwright specs.
import { defineConfig } from "vitest/config";

// COOP/COEP: cross-origin isolation, required for multithreaded wasm in ORT Web
// (ADR 0005). Everything the page fetches is served same-origin from public/.
const aislamiento = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  server: { headers: aislamiento, port: 5173 },
  preview: { headers: aislamiento },
  build: { target: "esnext" },
  optimizeDeps: { exclude: ["onnxruntime-web", "espeak-ng"] },
  test: { include: ["src/**/*.test.ts"] },
});
