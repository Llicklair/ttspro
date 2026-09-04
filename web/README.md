# @ttspro/web

Browser runtime for ttspro. Consumes the two ONNX graphs and `models/contrato.json`; never
imports anything from `src/ttspro` (ARCHITECTURE rule 6).

```bash
npm ci
npm test          # vitest
npm run lint      # biome
npm run typecheck # tsc --noEmit
npm run dev       # copies models/ + espeak wasm into public/, serves the test page with COOP/COEP
npx playwright test   # the page end to end in Chromium headless (wasm floor), writes test-results/criterio5.json
```

The test page (`index.html` + `src/demo/main.ts`): load the two graphs (fp16 or fp32, WebGPU
with wasm fallback), clone a voice from a file or a 5 s mic recording, type text in es/en,
listen, download the wav. Every number on the page is measured in that browser.

## Serving the demo

ONNX Runtime Web uses multithreaded WASM only under **cross-origin isolation**. Serve with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them the runtime silently falls back to a single thread — the number one cause of
"fast locally, slow in production" (ADR 0005).
