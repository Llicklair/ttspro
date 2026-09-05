# Third-party components

Everything this project ships, downloads or links, and under what terms. Weights are **ported**,
not trained here: the tensors come from the projects below and are re-expressed on this repo's own
modules (`ttspro.export.piper`, `ttspro.model.conversor`), so their licences travel with them.

## Weights

| Artifact | Source | Licence | Ships to the browser |
|---|---|---|---|
| `models/tts.onnx` — base voice | [`es_ES-davefx-medium`](https://huggingface.co/rhasspy/piper-voices) (Piper) | MIT; its `davefx` dataset is CC0 | yes |
| `models/conversor.onnx`, `models/voz.onnx` — tone colour | [`myshell-ai/OpenVoiceV2`](https://huggingface.co/myshell-ai/OpenVoiceV2) | MIT | yes |
| `models/speaker_encoder.onnx` — similarity metric | [`Wespeaker/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) | CC BY 4.0 | no, it only measures |
| `models/voces.json` — 20 preset voices | 256-d vectors measured over VCTK and OpenSLR es | see below | yes |

`voces.json` holds no audio: each preset is a 256-float vector produced by the OpenVoice reference
encoder from recordings in [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) (CC BY 4.0) and
[OpenSLR 61/71/72/73/74/75](https://www.openslr.org/61) (CC BY-SA 4.0, Google's Spanish corpora for
Argentina, Chile, Colombia, Peru, Puerto Rico and Venezuela). Both licences are credited here; the
CC BY-SA share-alike condition is why this file is documented separately rather than folded into
the code's licence.

## Code and runtime

| Component | Licence | Note |
|---|---|---|
| [espeak-ng](https://github.com/espeak-ng/espeak-ng), via the [`espeak-ng`](https://www.npmjs.com/package/espeak-ng) npm WASM build | **GPL-3.0-or-later** | the phonemizer, on both sides of the contract |
| [ONNX Runtime Web](https://onnxruntime.ai/) | MIT | the browser runtime |
| PyTorch, ONNX, numpy, soundfile | BSD/MIT/Apache-2.0 | export and measurement only |

**espeak-ng is GPL-3.0-or-later and the browser app loads it.** A build of `web/` distributes that
WASM module, so anyone redistributing the built site takes on GPLv3 obligations for it. This is the
single fact that decides what licence the repo itself can carry, and it is the open question in
[ADR 0009](docs/adr/0009-publicacion-en-github.md).

## Reference audio in this repo

`web/e2e/referencia.wav` is speaker p225 of the CSTR VCTK Corpus (University of Edinburgh,
CC BY 4.0), utterance `p225_003`, resampled to 22 050 Hz mono and trimmed to 6 s. It exists so the
end-to-end test can exercise cloning from a real recording.

## What is deliberately absent

No XTTS-v2 weights or outputs, and none of its training data: its licence (CPML) is non-commercial
and this repo does not depend on it ([ADR 0006](docs/adr/0006-uso-no-comercial-y-xtts-como-maestro.md)).
No corpus audio is redistributed — `ttspro.data.descargar` fetches corpora from their original
hosts, and `data/README.md` lists the licence of each.
