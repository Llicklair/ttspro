# Model card — ttspro

Spanish text-to-speech with voice cloning, four ONNX graphs that run **in the browser**. Written
before publication because the scope asks for it: a repo that hands out a cloning model states what
it does, what it does badly, and what it must not be used for.

**Version:** 2026-09-05 · **Contact:** the repository issues · **Licence:** see
[THIRD_PARTY.md](THIRD_PARTY.md) and [ADR 0009](docs/adr/0009-publicacion-en-github.md)

## What it is

Text becomes phonemes with espeak-ng, phonemes become audio in one fixed base voice, and a
tone-colour converter turns that audio into the voice you picked — a preset or one cloned from a
recording. Nothing was trained for this project: the base voice is a ported Piper voice and the
converter is ported OpenVoice v2 ([ADR 0007](docs/adr/0007-clonacion-como-postproceso.md),
[ADR 0008](docs/adr/0008-la-voz-base-es-una-voz-de-piper.md)).

| Graph | Parameters | fp32 | fp16 | Runs in the browser |
|---|---|---|---|---|
| `tts.onnx` | 16.4 M | 63.3 MB | 32.3 MB | yes, WebGPU or wasm |
| `conversor.onnx` | — | 132.3 MB | 66.5 MB | yes |
| `voz.onnx` (recording → voice vector) | — | 7.5 MB | 3.7 MB | wasm only, it contains a GRU |
| `speaker_encoder.onnx` | — | 27.4 MB | 14.2 MB | no, it only measures similarity |

## Intended use

Personal, research and educational use of speech synthesis that runs entirely on the visitor's
machine, with no server and no network after load. Cloning your own voice, or a voice whose owner
gave you permission.

## Out of scope

- **Cloning anyone who has not agreed to it.** The demo says so on the page, and it is the one
  limit this project asks you to respect.
- Impersonation, fraud, evading voice-based identity checks, or any output presented as a genuine
  recording of a real person.
- Commercial use is outside the project's own scope ([ADR 0006](docs/adr/0006-uso-no-comercial-y-xtts-como-maestro.md));
  the upstream weights are MIT, so the constraint is this project's, not theirs.
- Languages other than Spanish. The base voice is monolingual and asking it for English produces
  noise, so the demo does not offer it.

## Measured behaviour

Every number is dated in [docs/evidencia.md](docs/evidencia.md), including the ones that fail.

| | Value | Note |
|---|---|---|
| Intelligibility (WER, Whisper-small) | 0.030 base, median 0.000 after conversion | ordinary sentences |
| | 0.153 base, 0.417 after conversion | corpus sentences: proper nouns, digits |
| Speaker similarity (cosine) | base 0.079 → **0.338** | six target speakers, OpenSLR es |
| Ceiling (two real recordings of the target) | 0.758 | the same encoder's own upper bound |
| Browser, wasm, fp32 | 3.46 s end to end, RTF 1.21 | Chromium headless, 13-word sentence |
| Download | 249.4 MB fp32, 148.8 MB fp16 | models + ORT wasm + espeak-ng |

## Known limitations

- **Similarity is 0.338 against a target of 0.55.** The project's own done-criterion 4 fails. What
  you get is a recognisable resemblance to the reference, not a copy of it. Treat that as the
  honest ceiling of a zero-training approach, not as a bug about to be fixed.
- **The download is 148.8 MB in fp16 against a 110 MB budget**, and a 13-word sentence takes 3.46 s
  against a 3 s limit. The converter dominates both: 66.5 MB and 2.9 of those seconds.
- **fp16 needs `shader-f16`.** Pascal GPUs (GTX 10xx) lack it; the page detects this and falls back.
- Sentence-level synthesis only: a non-autoregressive model emits the whole sentence, so long texts
  are split by sentence in the frontend.
- Prosody is the base voice's. There is no emotion, style or SSML control.

## Provenance and consent

The base voice is `es_MX-claude-high` (Piper, MIT); its dataset is Apache-2.0. `es_ES-davefx-medium`
(dataset CC0) stays available as a pack. The 24 preset voices are
vectors derived from VCTK (CC BY 4.0) and OpenSLR es (CC BY-SA 4.0) — corpora recorded with the
speakers' consent for research use. No audio from those corpora is redistributed here.

## Watermarking

**There is none today.** Output carries no audible or inaudible mark, so it cannot be told from
other synthetic audio by inspection. The scope deliberately deferred this until publication was on
the table, and [ADR 0009](docs/adr/0009-publicacion-en-github.md) reopens it as a decision to make
before weights are handed out.
