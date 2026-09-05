# ttspro

Spanish text-to-speech with one-shot voice cloning, running **entirely in your browser**. No
server, no network after load, no account. Four ONNX graphs and ONNX Runtime Web.

Think of what [coqui TTS](https://github.com/coqui-ai/tts) did with YourTTS, made small enough to
download and run on a laptop, with the browser as the runtime rather than a demo.

**Status:** it works end to end and **nothing was trained for it**. The base voice is a
[Piper](https://github.com/OHF-Voice/piper1-gpl) voice ported onto this repo's own modules, and
cloning is post-processing with [OpenVoice v2](https://github.com/myshell-ai/OpenVoice)
([ADR 0007](docs/adr/0007-clonacion-como-postproceso.md),
[ADR 0008](docs/adr/0008-la-voz-base-es-una-voz-de-piper.md)). Two of the project's own acceptance
thresholds are still red, and [the numbers say exactly how red](#where-it-stands).

## How it works

```
text ──espeak-ng──► phonemes ──tts.onnx──► audio in the base voice
                                                │
recording ──voz.onnx──► voice vector ───────────┼──conversor.onnx──► audio in that voice
preset (voces.json) ────────────────────────────┘
```

The base voice is fixed and monolingual; identity comes from the converter, not from the
synthesizer. That is why cloning needs no training: the voice vector is an input, not a weight.

`models/contrato.json` is the contract between the Python and the TypeScript side — symbol table,
languages, graph inputs, licences. Both text frontends read it, and the export writes it, so the
browser can never disagree with the checkpoint about what a token means.

## Run it

```bash
uv sync --extra dev --extra export          # torch comes from the cu126 index pinned in pyproject
uv run python -m ttspro.export.modelos      # downloads public weights, writes models/*.onnx
cd web && npm ci && npm run dev             # http://localhost:5173
```

The second command is the whole model pipeline: it fetches the Piper voice and the OpenVoice
converter from Hugging Face, ports both onto this repo's modules and exports the ONNX graphs. It
needs no corpus, no GPU and no training — a few minutes on a laptop CPU. The preset voices
(`models/voces.json`) travel in git, because measuring them needs ~12 GB of speech corpora.

For the Python side of the text frontend you also need espeak-ng 1.52 installed natively
(`winget install eSpeak-NG.eSpeak-NG`, or `apt install espeak-ng`); the browser uses the WASM build
from npm. A test holds the two to identical output.

## Where it stands

Measured on 2026-09-05, full detail in [docs/evidencia.md](docs/evidencia.md).

| | Result | Target |
|---|---|---|
| Intelligibility (WER, Whisper-small, ordinary sentences) | 0.030 base, 0.000 median converted | ≤ 0.10 ✅ |
| Speaker similarity (cosine, six target speakers) | 0.079 → **0.338** | ≥ 0.55 ❌ |
| Same encoder's ceiling (two real recordings of the target) | 0.758 | — |
| Browser, wasm, fp32, 13-word sentence | **3.46 s** end to end (560 ms TTS + 2 861 ms converter) | < 3 s ❌ |
| Download (models + ORT wasm + espeak-ng) | 249.4 MB fp32, 148.8 MB fp16 | ≤ 110 MB ❌ |

So: the Spanish is clean and the voice is recognisable, but it is a resemblance rather than a copy,
it weighs more than the budget allows and it is just over the latency line. The converter dominates
all three — it is 66.5 MB of the download and 2.9 of the 3.5 seconds.

`uv run pytest tests/terminado -q` is that table as a command: criteria 1, 2 and 3 (the graphs load
and match the contract, PyTorch and ONNX Runtime agree, both text frontends agree) pass; 4 and 5 fail.

## Layout

| Path | What |
|---|---|
| `src/ttspro/frontend/` | Text → phonemes → tokens, mirrored exactly in TypeScript |
| `src/ttspro/model/` | VITS-family blocks: encoder, flow, duration predictor, HiFi-GAN, converter |
| `src/ttspro/export/` | Weight ports and ONNX export, constrained to ORT Web's WebGPU op set |
| `src/ttspro/train/` | Optional: training, fine-tuning and evaluation. Nothing shipped depends on it |
| `web/` | The browser runtime and the demo page |
| `models/` | `contrato.json` and `voces.json` in git, the `.onnx` graphs built locally |
| `docs/adr/` | Why things are the way they are |
| `docs/evidencia.md` | Every number ever measured, including the bad ones |

`ttspro.train` exists for the road not taken: training a voice-cloning synthesizer from scratch was
measured at 1.5–3 days on a GTX 1070 for a worse result than porting. The code stayed, the
dependency did not — a boundary check enforces that inference never imports training.

## Cloning, honestly

This clones a voice from about six seconds of audio. **Only clone a voice whose owner agreed to
it.** The demo page says so, [MODEL_CARD.md](MODEL_CARD.md) says what the model must not be used
for, and the output carries no watermark today — which is stated plainly rather than left to be
discovered.

## Licence

The weights come from MIT-licensed projects and the preset vectors from CC BY / CC BY-SA corpora;
the phonemizer, espeak-ng, is GPL-3.0-or-later and the browser app loads it.
[THIRD_PARTY.md](THIRD_PARTY.md) lists every component with its terms, and
[ADR 0009](docs/adr/0009-publicacion-en-github.md) is where the repo's own licence is decided.

Decision documents (`SCOPE.md`, `ARCHITECTURE.md`, `AGENTS.md`, ADRs, evidence) are in Spanish by
design; everything publishable is in English.
