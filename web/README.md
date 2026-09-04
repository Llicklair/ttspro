# @ttspro/web

Browser runtime for ttspro. Consumes the two ONNX graphs and `models/contrato.json`; never
imports anything from `src/ttspro` (ARCHITECTURE rule 6).

```bash
npm ci
npm test          # vitest
npm run lint      # biome
npm run typecheck # tsc --noEmit
```

## Serving the demo

ONNX Runtime Web uses multithreaded WASM only under **cross-origin isolation**. Serve with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them the runtime silently falls back to a single thread — the number one cause of
"fast locally, slow in production" (ADR 0005).
