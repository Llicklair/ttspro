# ttspro

Spanish text-to-speech with one-shot voice cloning, running **entirely in your browser**. No
server, no network after load, no account. ONNX graphs and ONNX Runtime Web.

Think of what [coqui TTS](https://github.com/coqui-ai/tts) did with YourTTS, made small enough to
download and run on a laptop, with the browser as the runtime rather than a demo.

**Status:** it works end to end and **nothing was trained for it**. Two published models do the
work: [Supertonic 3](https://github.com/supertone-oss-archive/supertonic-py) reads, and
[Pocket TTS](https://huggingface.co/kyutai/pocket-tts) clones
([ADR 0012](docs/adr/0012-supertonic-como-motor.md)). The Spanish is clean — WER 0.008 — and
cloning works from a recording in under a second, but the cloned voice is a resemblance rather than
a copy (0.413 cosine against a 0.55 target) and that is the model's ceiling, not a setting.
[The numbers say exactly how far it gets](#where-it-stands).

## How it works

```
texto ──indexador Unicode──► ids ──┬──► duration_predictor ──► segundos
                                   └──► text_encoder ───────► text_emb
                                                                 │
   ruido ──► [ vector_estimator × N pasos, flow matching ] ──► latente ──► vocoder ──► 44,1 kHz
                            ▲
                      style_ttl / style_dp  ← la voz, como ENTRADA
```

The speaker's identity enters three of those four graphs as two input tensors, so nothing runs
after the synthesizer: the voice you asked for is the voice that comes out. A voice is 292 KB of
JSON, not a checkpoint. And there is no phonemizer — the model reads Unicode characters, which is
why the reading path carries no espeak-ng
([ADR 0012](docs/adr/0012-supertonic-como-motor.md)).

What Supertonic does **not** ship is the encoder that turns a recording into those two tensors.
Supertone never published it, and six measured attempts to rebuild it — mixture search, a ridge
bridge, gradient fitting, and three variations on recovering the latent — all failed, the last one
with the answer within reach
([docs/evidencia.md](docs/evidencia.md), 2026-09-16). So cloning is done by a second model that
does publish its encoder, and that is the only thing it is here for:

```
grabación (5–20 s) ──► mimi_encoder ──► voz  ──►  Pocket TTS ──► 24 kHz ──► WSOLA ──► velocidad
```

**You never pick an engine.** You pick a voice, and the page knows which model speaks it. The
reading model downloads when the page opens; the cloning model downloads the first time you drop a
recording, and not before.

## Run it

**The fastest way: don't install anything.** Open the page and press **«usar Hugging Face»** — the
browser pulls the four graphs (398 MB) straight from the archive and speaks. Hugging Face serves
them with the CORS headers that make this work from a browser, which is also why the voice packs
live there ([ADR 0011](docs/adr/0011-paquetes-de-voz-y-hardware.md)). Keeping a local copy
(`uv run python -m ttspro.supertonic.descargar`) only makes every later reload faster.

One script does the whole setup and then offers to start the demo (add `--sin-demo` to only
install):

- Windows: double-click **`instalar.bat`**. Missing tools (uv, Node.js) come from winget.
  Afterwards, **`arrancar.bat`** starts the page (and `arrancar.bat --lib` the library's example page).
- Linux and macOS: **`./instalar.sh`**. uv and Node.js 22 install under `~/.local` without sudo.

To update an existing install later, `./actualizar.sh` pulls, syncs the Python environment,
re-downloads the graphs only when they are missing (`--modelos` forces it) and refreshes the
browser side.

Both installers do exactly this:

```bash
uv sync --extra dev --extra voz                # torch comes from the cu126 index pinned in pyproject
uv run python -m ttspro.supertonic.descargar   # 398 MB into models/supertonic/, revision pinned
cd web && npm ci && npm run dev                # http://localhost:5173
```

That middle command is a **download**, not a build: upstream publishes ONNX directly, so there is
no port, no export and no opset surgery between the checkpoint and the browser. Skip it entirely if
you are using the Hugging Face button.

The cloning model is never downloaded by a script. It arrives in the browser, from Hugging Face,
the first time somebody clones — 216 MB, once, then it lives in the Cache API.

## Voices

**Ten come with the model.** `F1`–`F5` and `M1`–`M5`, 292 KB of JSON each, downloaded on demand.
That convention is Supertone's own and is the only place the page states a voice's gender; where a
model does not publish it, the page says nothing rather than guessing from the name.

**Cloning one of your own** is dropping a recording on the page, or pressing *grabar*. The first
clone waits for the 216 MB download; after that it is about half a second. What matters most is not
a setting — it is **how much audio you give it**:

| Reference | Cosine similarity |
|---|---|
| 2 s | 0.396 |
| 5 s | 0.409 |
| 10 s | 0.441 |
| 20 s (what the encoder reads) | **0.527** |

so the recorder captures 20 s, counts them out loud, and says so when you hand it less than 15
(measured 2026-09-16, [docs/evidencia.md](docs/evidencia.md)). Reference *quality* barely moves the
number (r = +0.247); duration moves it 13 points.

**Building one without a recording.** `ttspro.supertonic.constructor` searches for a mixture of the
ten factory voices that lands nearest a target, one mixture per each of the 50 style slots. It is
slow (minutes per voice) and it cannot leave the convex hull of the ten, but it needs no encoder,
and it writes the same 292 KB `.json` the page imports. `ttspro.supertonic.pack` mints N mutually
distinct voices in one go.

**Three sliders, and they apply to whichever model is speaking.** *calidad* is decoder steps (the
knee is at 8; cloned voices cap at 4 and are clamped there rather than being sent a value they do
not accept). *variación* is what other projects call noise or temperature — it only affects cloned
voices, and 0 means "whatever the checkpoint chose". *velocidad* is predicted inside the model for
factory voices; for cloned ones the waveform is time-stretched afterwards with WSOLA, so the pitch
does not move — in a cloned voice, pitch is half of what makes it recognisable.

**Languages.** 31, from the reading model. Catalan is not one of them in either model, and it was
measured rather than assumed: Supertonic reads Catalan text at WER 0.237, which is rough but
followable; the cloning model returns 1.409, which is noise. So Catalan is not offered, and if you
paste it anyway, the factory voices are the ones that will make sense of it.

**Hardware.** The page detects WebGPU, the adapter, `shader-f16`, threads and isolation, states the
rule's recommendation and offers **calibrar**, which measures one sentence on every candidate
provider and keeps the fastest. Library: `TTS.hardware()`, `TTS.recomendar()`, `tts.calibrar()`.

## Chat mode

The page's second mode reads a Twitch chat aloud ([ADR 0010](docs/adr/0010-modo-chat-para-twitch.md)).
Type a channel name and connect: it joins anonymously over IRC-on-WebSocket, no account needed.
Every line goes through a chat normalizer (links, @mentions, emotes, emoji, "jajajaja", "holaaaa",
"q tal", repeated spam) mirrored in Python and TypeScript, then into a queue that renders the next
message while the current one plays and drops what has gone stale. Pick the voice policy per
message: *una voz por usuario* spreads the chat over the voices you have. A filter reads only the
lines viewers highlighted with channel points ("Highlight My Message"), or those plus custom
rewards that carry text, so a busy chat becomes something people pay points to have read.

### As a library, from any HTML page

```bash
cd web && npm run build:lib     # dist/lib/ttspro.js plus the wasm files next to it
npm run servir                  # http://127.0.0.1:8080 — ejemplo/, dist/lib/ and public/ with COOP/COEP
```

Open the page, then from the browser console (the example page already did `import { TTS }` and
left `tts` on `window`):

```js
import { TTS } from "./ttspro.js";
const tts = await TTS.cargar({ modelos: "/models/supertonic/" });

tts.voces;                                        // ["F1", …, "M5"]
await tts.elegirVoz("M3");                        // downloaded once, 292 KB
await tts.importar(jsonFile);                     // a style built with ttspro.supertonic.constructor
await tts.clonar(audioInput.files[0]);            // a recording (up to 20 s) -> the voice to use

const r = await tts.predict("hola a todos");      // { onda, sampleRate, ms, voz, wav() }
tts.volumen = 0.5;                                // master volume, 0..1
tts.ajustes.pasos = 8;                            // defaults for every predict: pasos, velocidad,
tts.ajustes.velocidad = 1.05;                     // semilla, idioma
await tts.reproducir(r);

// a stream: any iterable, async iterable or ReadableStream of strings or { usuario, texto }
for await (const r of tts.predict(mensajes, { chat: true, reproducir: true })) console.log(r.texto);

const c = await tts.calibrar();                   // measured ms per provider; c.mejor is the pick
```

`clonar()` downloads the cloning model on first call and returns the voice's name, already
selected; every voice after that, cloned or not, is just a name you pass to `predict`.

**If you used the previous version**, the API changed with
[ADR 0012](docs/adr/0012-supertonic-como-motor.md): `espeak`, `precision`, `vocesBase`,
`cargarVozBase` and `elegirVozBase` are gone (there is one list of voices now, not base against
target), and `tau`, `noise_scale` and `length_scale` became `pasos`, `velocidad` and `semilla`.

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

Measured on 2026-09-16, same harness as always — Whisper-small for WER, WeSpeaker cosine for
similarity. Full detail in [docs/evidencia.md](docs/evidencia.md).

| | Result | Target |
|---|---|---|
| Intelligibility (WER, ordinary sentences, reading model) | **0.008**, median 0.000 | ≤ 0.10 ✅ |
| Intelligibility, cloning model | 0.027 | ≤ 0.10 ✅ |
| Speaker similarity (cosine, 14 speakers, 42 sentences) | **0.413** | ≥ 0.55 ❌ |
| Same encoder's ceiling (two real recordings of the target) | 0.760 | — |
| Time to clone | **0.44 s** on CPU; 7.4 s the first time in the browser, encoder download included | — |
| RTF, CPU, 8 steps | 0.46 | < 1 ✅ |
| Download | 398 MB reading, +216 MB the first time you clone | (budget retired, ADR 0012) |

So: the Spanish is clean, it is fast enough on a laptop CPU, and cloning happens in under a second
— but the cloned voice lands at 0.413 against a 0.55 target and **that is the model's ceiling**.
The number was chased: it does not follow the reference's quality (r = +0.247), it plateaus at
0.41–0.45 across 14 speakers, and the only lever that moved it was giving the encoder more audio
(+13 points from 2 s to 20 s). A better number needs a better published encoder, not a better
setting here.

`uv run pytest tests/terminado -q` is the acceptance suite; it still describes the previous chain
and is being rewritten against this one.

## Layout

| Path | What |
|---|---|
| `src/ttspro/supertonic/` | The reading model: Unicode frontend, styles, engine, voice building |
| `src/ttspro/frontend/` | Text normalization, mirrored exactly in TypeScript |
| `web/src/runtime/` | `supertonic.ts` reads, `pocket.ts` clones, `velocidad.ts` time-stretches |
| `web/src/lib/` | The embeddable library: one `TTS` object |
| `web/` | The browser runtime and the demo page |
| `models/` | `contrato.json`, generated by reading the `.onnx` files that are downloaded locally |
| `docs/adr/` | Why things are the way they are |
| `docs/evidencia.md` | Every number ever measured, including the bad ones |

`src/ttspro/model/`, `src/ttspro/export/` and `src/ttspro/train/` are the previous chain — a Piper
voice ported onto this repo's modules plus an OpenVoice converter, and the training code for the
road not taken. Nothing shipped imports them any more; they stay because the measurements that
retired them are in the evidence log and the code behind those measurements should still be
readable.

## Cloning, honestly

This clones a voice from a recording of a few seconds. **Only clone a voice whose owner agreed to
it.** The demo page says so, [MODEL_CARD.md](MODEL_CARD.md) says what the models must not be used
for, and the output carries no watermark today — which is stated plainly rather than left to be
discovered.

## Licence

The code is **MIT** ([LICENSE](LICENSE)). The two models are not the same: Supertonic's weights are
**OpenRAIL-M**, which carries use restrictions and is not permissive, and Pocket TTS is
**CC BY 4.0**. The cloning runtime pulls in espeak-ng, which is **GPL-3.0-or-later**, so it is
deliberately **not bundled** — a published build loads it from a CDN on the one path that needs it,
which Spanish never takes, and therefore distributes nothing under the GPL.
[THIRD_PARTY.md](THIRD_PARTY.md) lists every component with its terms, and
[ADR 0009](docs/adr/0009-publicacion-en-github.md) records why it is set up this way.

Decision documents (`SCOPE.md`, `ARCHITECTURE.md`, `AGENTS.md`, ADRs, evidence) are in Spanish by
design; everything publishable is in English.
