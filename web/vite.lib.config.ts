// Library build: `npm run build:lib` -> dist/lib/ttspro.js, one ES module that
// any HTML page imports, plus ORT's wasm/mjs files next to it (ORT finds them
// through import.meta.url). Models and espeak's wasm are NOT in the bundle: they
// are served from public/ (`npm run preparar` puts them there) and their URLs are
// options of TTS.cargar.
//
// Not Vite's `build.lib`: library mode inlines every asset as base64, and with
// two ORT wasm variants inside that was a 99 MB ttspro.js. A plain build with a
// TS entry emits the wasm as files (assetsInlineLimit 0) and keeps the module
// at a few megabytes.
import { defineConfig } from "vite";

export default defineConfig({
  publicDir: false,
  build: {
    target: "esnext",
    outDir: "dist/lib",
    emptyOutDir: true,
    assetsInlineLimit: 0,
    modulePreload: false,
    rollupOptions: {
      input: "src/lib/index.ts",
      preserveEntrySignatures: "strict",
      output: {
        format: "es",
        entryFileNames: "ttspro.js",
        chunkFileNames: "[name].js",
        assetFileNames: "[name][extname]",
      },
    },
  },
  optimizeDeps: { exclude: ["onnxruntime-web", "espeak-ng"] },
});
