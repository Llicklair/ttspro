# ttspro

Multilingual, non-autoregressive text-to-speech with one-shot voice cloning. Trained in PyTorch,
exported to ONNX, run **in the browser** with ONNX Runtime Web — no server, no network.

Think of what [coqui TTS](https://github.com/coqui-ai/tts) did with YourTTS (VITS + speaker
encoder), made small, exportable, and with the browser as a first-class runtime rather than a demo.

**Status:** day zero. The floor is laid, the design law is written, the done criterion is a
command and it is red. No model yet.

## Layout

| Path | What |
|---|---|
| `src/ttspro/` | Python: text frontend, model, export, training, data |
| `web/` | TypeScript: the same text frontend, the ONNX Runtime Web runtime |
| `models/` | The two `.onnx` graphs (not in git) and `contrato.json`, the signed contract between both worlds |
| `tests/` | Fast suite; `tests/terminado/` is the MVP done criterion |
| `docs/adr/` | Why things are the way they are |
| `docs/evidencia.md` | Every number ever measured, including the bad ones |

## Setup

```bash
uv sync --extra dev --extra export
# training: install torch for your CUDA first, then
uv sync --extra train
cd web && npm ci
```

Decision documents (`SCOPE.md`, `ARCHITECTURE.md`, `AGENTS.md`, ADRs, evidence) are in Spanish
by design; everything publishable is in English.
