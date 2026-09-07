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

One script does the whole setup and then offers to start the demo (add `--sin-demo` to only
install):

- Windows: double-click **`instalar.bat`**. Missing tools (uv, Node.js, eSpeak NG) come from winget.
  Afterwards, **`arrancar.bat`** starts the page (and `arrancar.bat --lib` the library's example page).
- Linux and macOS: **`./instalar.sh`**. uv and Node.js 22 install under `~/.local` without sudo;
  espeak-ng comes from your package manager and is only needed for the Python tests.

To update an existing install later, `./actualizar.sh` pulls, syncs the Python environment,
rebuilds the models only when the commits that came in touched the model pipeline (`--modelos`
forces it) and refreshes the browser side.

Both installers do exactly this:

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

## Base voices and hardware

The base voice is not fixed any more. Eight Spanish Piper voices (MIT) are ported and published as
**voice packs** ([ADR 0011](docs/adr/0011-paquetes-de-voz-y-hardware.md)): the page lists them,
downloads one on demand, and speaks with it. Each pack carries its own contract (symbol tables
differ between voices) and its own base-voice vector for the converter. In chat mode, *una voz base
por usuario* spreads the chat over the downloaded voices with no converter at all, which is the
fast and clear way to give people different voices.

| Pack | Quality | fp16 | ms / sentence, CPU |
|---|---|---|---|
| es_ES-carlfm-x_low | very low | 11.0 MB | 33 |
| es_ES-davefx-medium | medium | 32.3 MB | 43 |
| es_MX-claude-high (the local default: it survives the converter) | high | 32.3 MB | 60 |
| es_MX-ald-medium · es_ES-mls_9972-low · es_ES-mls_10246-low · es_MX-ald-x_low | | 11–32 MB | 62–129 |
| es_AR-daniela-high | high | 57.6 MB | 210 |

Packs are served from Hugging Face (`https://huggingface.co/Llicklair/ttspro-voces/resolve/main/`),
with GitHub Pages (`https://llicklair.github.io/ttspro/`) as a mirror; a release cannot host them
because its assets send no CORS header. The page's *origen de voces* field, `?voces=<url>`
and `TTS.cargar({ vocesBase })` point at any other host, and building your own is
`uv run python -m ttspro.export.paquete --todas es`.

**Hugging Face.** Any user can download the packs from a Hugging Face repo, which serves them
with the right CORS headers: publish yours with

```bash
uv run hf auth login                                        # once, token from huggingface.co/settings/tokens
uv run python -m ttspro.export.publicar --repo <usuario>/ttspro-voces
```

and use `https://huggingface.co/<usuario>/ttspro-voces/resolve/main/` as the origin, in the page
field or in `TTS.cargar({ vocesBase })`. The same flat files work on both hosts.

**Hardware.** The page detects WebGPU, the adapter, `shader-f16`, threads and isolation, states
the rule's recommendation (WebGPU + f16 → fp16 on the GPU; WebGPU without f16, such as a GTX 1070
→ fp32 on the GPU; no WebGPU → wasm fp32) and offers **calibrar**, which measures one sentence
through the TTS and the converter on every candidate and keeps the fastest. Library:
`TTS.hardware()`, `TTS.recomendar()`, `tts.calibrar()`.

## Chat mode

The page's fourth panel reads a Twitch chat aloud ([ADR 0010](docs/adr/0010-modo-chat-para-twitch.md)).
Type a channel name and connect: it joins anonymously over IRC-on-WebSocket, no account needed.
Every line goes through a chat normalizer (links, @mentions, emotes, emoji, "jajajaja", "holaaaa",
"q tal", repeated spam) mirrored in Python and TypeScript, then into a queue that renders the next
message while the current one plays and drops what has gone stale. Pick the voice policy per
message: the base voice keeps up on wasm; the converter costs ~3 s a message there and wants WebGPU.

### As a library, from any HTML page

```bash
cd web && npm run build:lib     # dist/lib/ttspro.js plus the wasm files next to it
npm run servir                  # http://127.0.0.1:8080 — ejemplo/, dist/lib/ and public/ with COOP/COEP
```

Open the page, then from the browser console (the example page already did `import { TTS }` and
left `tts` on `window`):

```js
import { TTS } from "./ttspro.js";
const tts = await TTS.cargar({ modelos: "/models/", espeak: "/espeak/espeak-ng.wasm" });

await tts.clonar(audioInput.files[0]);            // a recording (3 s+) -> the voice to use
tts.elegirVoz("cof_02484");                       // or a preset; tts.voces lists them
tts.elegirVoz(null);                              // or the base voice, no converter, fastest

const r = await tts.predict("hola a todos");      // { onda, sampleRate, ms, fonemas, wav() }
tts.volumen = 0.5;                                // master volume, 0..1
await tts.reproducir(r);

// a stream: any iterable, async iterable or ReadableStream of strings or { usuario, texto }
for await (const r of tts.predict(mensajes, { chat: true, reproducir: true })) console.log(r.texto);

await tts.cargarVozBase("es_MX-claude-high");     // a pack, downloaded once; tts.vocesBase lists them
tts.elegirVozBase("es_MX-claude-high");            // or per call: predict(texto, { vozBase })
const c = await tts.calibrar();                    // measured ms per provider; c.mejor is the pick
```

The stream form runs through the same queue as the demo: it synthesizes ahead of playback and
drops what has waited too long. `chat: true` applies the chat normalizer. The page has to come
over HTTP (fetch does not work from `file://`), and multithreaded wasm needs the COOP/COEP headers,
which `npm run servir` sends.

Embedding it in another site (a Rails app with importmap, say): copy the whole `dist/lib` folder
**as is** to a public path that is not fingerprinted, `public/tts_pro/` for instance. ORT finds its
wasm next to `ttspro.js` through `import.meta.url`, so an asset pipeline that renames the file
breaks that lookup. Then `pin "ttspro", to: "/tts_pro/ttspro.js"` and
`TTS.cargar({ modelos: "/tts_pro/models/", espeak: "/tts_pro/espeak-ng.wasm" })`. Without COOP/COEP
on that site ORT runs wasm on one thread (slower, still correct); adding those two headers to a site
that embeds third-party players or images breaks them, so measure before choosing.

### Feeding it from your own app (Rails, streex, anything)

The reader does not have to be the one talking to Twitch. If another application already reads
the chat, it hands lines to the page through one of three doors, chosen in the panel's *fuente*
selector. Every door delivers `{ usuario, texto }`; field names are forgiving (`user`, `username`,
`display_name`, `text`, `message`, `body` all work).

**ActionCable (Rails).** Point the page at your cable endpoint and channel:

```ruby
# app/channels/chat_channel.rb
class ChatChannel < ApplicationCable::Channel
  def subscribed = stream_from("chat")
end

# wherever a chat line arrives
ActionCable.server.broadcast("chat", { usuario: "Pepe", texto: "hola a todos" })

# config/environments/development.rb — Rails refuses other origins by default
config.action_cable.allowed_request_origins = ["http://localhost:5173"]
```

Page: fuente *WebSocket / ActionCable*, URL `ws://localhost:3000/cable`, canal `ChatChannel`.
Leave the channel empty for a plain WebSocket that sends one JSON object per frame.

**Server-Sent Events.** Any endpoint that streams `data: {"usuario":"Pepe","texto":"hola"}

`.
It must answer with `Access-Control-Allow-Origin` for the page's origin: the page is cross-origin
isolated and an `EventSource` is a CORS request.

**postMessage.** Embed the page in an `<iframe>` and post from the parent:

```js
iframe.contentWindow.postMessage({ tipo: "ttspro", usuario: "Pepe", texto: "hola" }, "*");
```

Fuente *ventana padre*; the *origen permitido* field restricts which parent is listened to.

For a script of your own, the page also exposes `window.__ttspro_chat.recibir(usuario, texto)`.

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

The code is **MIT** ([LICENSE](LICENSE)). The weights come from MIT-licensed projects and the
preset vectors from CC BY / CC BY-SA corpora; the phonemizer, espeak-ng, is GPL-3.0-or-later and
the browser app loads it, so a deployed build of `web/` carries that obligation.
[THIRD_PARTY.md](THIRD_PARTY.md) lists every component with its terms, and
[ADR 0009](docs/adr/0009-publicacion-en-github.md) records why it is set up this way.

Decision documents (`SCOPE.md`, `ARCHITECTURE.md`, `AGENTS.md`, ADRs, evidence) are in Spanish by
design; everything publishable is in English.
