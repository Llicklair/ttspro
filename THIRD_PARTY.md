# Third-party components

Everything this project ships, downloads or links, and under what terms. No weight in here was
trained here.

**Read this first.** Since [ADR 0012](docs/adr/0012-supertonic-como-motor.md) the engine's weights
are **OpenRAIL-M**, not MIT. OpenRAIL-M is an open licence with **use-based restrictions** — an
appendix of things the model may not be used for — so "the code is MIT" no longer describes what
this project distributes as a whole. The code still is; the weights are not.

## Weights

| Artifact | Source | Licence | Ships to the browser |
|---|---|---|---|
| `models/supertonic/onnx/*.onnx` — **the engine**, four graphs, 398 MB | [`supertone-oss-archive/supertonic-3`](https://huggingface.co/supertone-oss-archive/supertonic-3), revision `aafc6e32` | **OpenRAIL-M** (use restrictions) | yes |
| `models/supertonic/voice_styles/*.json` — 10 voices | same repo | **OpenRAIL-M** | yes |
| `models/supertonic/puente.bin` — the voice builder's matrix | fitted here from the engine's own output | derived from OpenRAIL-M weights; treat as OpenRAIL-M | yes |
| `models/tts.onnx` — base voice (**cadena anterior**) | [`es_MX-claude-high`](https://huggingface.co/rhasspy/piper-voices) (Piper) | MIT; its dataset is Apache-2.0 | no longer |
| voice packs on [`Llicklair/ttspro-voces`](https://huggingface.co/Llicklair/ttspro-voces) — 8 base voices | Piper voices ported | MIT; datasets CC0 / Apache-2.0 per voice, see each MODEL_CARD in `rhasspy/piper-voices` | on demand |
| `models/conversor.onnx`, `models/voz.onnx` — tone colour (**cadena anterior**) | [`myshell-ai/OpenVoiceV2`](https://huggingface.co/myshell-ai/OpenVoiceV2) | MIT | no longer |
| `models/speaker_encoder.onnx` — similarity metric **and** the voice builder's input | [`Wespeaker/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) | CC BY 4.0 | yes, since ADR 0012 it also builds voices |
| `models/voces.json` — 24 preset voices | 256-d vectors measured over VCTK and OpenSLR es | see below | yes |

`voces.json` holds no audio: each preset is a 256-float vector produced by the OpenVoice reference
encoder from recordings in [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) (CC BY 4.0) and
[OpenSLR 61/71/72/73/74/75](https://www.openslr.org/61) (CC BY-SA 4.0, Google's Spanish corpora for
Argentina, Chile, Colombia, Peru, Puerto Rico and Venezuela). Both licences are credited here; the
CC BY-SA share-alike condition is why this file is documented separately rather than folded into
the code's licence.

## Code and runtime

| Component | Licence | Note |
|---|---|---|
| [pocket-tts-onnx](https://github.com/thewh1teagle/pocket-tts-onnx) | CC BY 4.0 | the browser runtime for the engine that **can** clone |
| [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) weights (Kyutai) | CC BY 4.0 | 177 MB, fetched at runtime and cached in the browser |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) | **GPL-3.0-or-later** | a dependency of `pocket-tts-onnx`, **not bundled**: a published build fetches it from jsDelivr, so nothing GPL is distributed — see below |
| [supertonic-py](https://github.com/supertone-oss-archive/supertonic-py) | MIT | the reference implementation the engine is ported from (archived 2026-09-09) |
| [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch) | Apache-2.0 | voice builder only, never in the synthesis path (rule 7) |
| [ONNX Runtime Web](https://onnxruntime.ai/) | MIT | the browser runtime |
| PyTorch, ONNX, numpy, soundfile | BSD/MIT/Apache-2.0 | export and measurement only |

**The GPL problem came back by a side door, and is closed again.** ADR 0012 removed espeak-ng as
the phonemizer and with it the GPL obligation. Then the Pocket TTS engine arrived, and its browser
runtime depends on espeak-ng — only for Latin words inside Hebrew, a branch the Spanish model does
not even carry the adapter for — but Vite emitted the 18.5 MB wasm and 68 KB of Emscripten glue
anyway, and **a deployed build distributes them**. The obligation attaches to distribution, not to
execution, so shipping a binary nobody downloads would still have carried it: the GPLv3 notice, an
offer of the corresponding source, and the unsettled question of whether the bundle counts as a
single combined work with the application.

So the package is **not bundled**. In production builds it is aliased to
`web/src/runtime/espeak-remoto.ts`, which loads it from jsDelivr if a line ever mixes Hebrew with
Latin script. Nothing changes functionally — `pocket-tts-onnx` already fetched espeak's wasm bytes
from that same CDN and passed them in as `wasmBinary`, so the emitted copy was never requested —
and the party distributing espeak-ng is the CDN, which already was.
`web/e2e/libreria.spec.ts` asserts that no `espeak-ng` file appears in the build, because this is
the kind of thing that comes back silently: one removed alias, or one new dependency that pulls in
espeak, and the file is there again.

**The rest of the old GPL story, for the record.** Until 2026-09-15 a build of `web/`
distributed the espeak-ng WASM module, and anyone redistributing the built site took on GPLv3
obligations for it — the single fact that decided what licence the repo could carry
([ADR 0009](docs/adr/0009-publicacion-en-github.md)). The engine has no phonemizer, so that is
over.

What is not over: a deployed build now distributes **OpenRAIL-M weights**. That licence is not
copyleft and does not reach the code, but it does carry use restrictions that follow the weights
to whoever downloads them, and the `LICENSE` file is copied next to the graphs so it cannot be
separated from them. The repo itself stays MIT and still ships no weights
([ADR 0009](docs/adr/0009-publicacion-en-github.md) is unchanged on that point); what changed is
what a *hosted* build hands to a visitor.

## Reference audio in this repo

`web/e2e/referencia.wav` is speaker p225 of the CSTR VCTK Corpus (University of Edinburgh,
CC BY 4.0), utterance `p225_003`, resampled to 22 050 Hz mono and trimmed to 6 s. It exists so the
end-to-end test can exercise cloning from a real recording.

## What is deliberately absent

No XTTS-v2 weights or outputs, and none of its training data: its licence (CPML) is non-commercial
and this repo does not depend on it ([ADR 0006](docs/adr/0006-uso-no-comercial-y-xtts-como-maestro.md)).
No corpus audio is redistributed — `ttspro.data.descargar` fetches corpora from their original
hosts, and `data/README.md` lists the licence of each.
