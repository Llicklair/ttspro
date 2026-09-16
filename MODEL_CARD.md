# Model card — ttspro

Spanish text-to-speech with voice cloning, running **in the browser**. Written before publication
because the scope asks for it: a repo that hands out a cloning model states what it does, what it
does badly, and what it must not be used for.

**Version:** 2026-09-16 · **Contact:** the repository issues · **Licence:** the code is MIT and
**two sets of weights with two different licences** ship here — Supertonic's are **OpenRAIL-M**, an
open licence *with use restrictions* that are part of this card and not a footnote, and Pocket
TTS's are CC BY 4.0. See [THIRD_PARTY.md](THIRD_PARTY.md),
[ADR 0009](docs/adr/0009-publicacion-en-github.md) and
[ADR 0012](docs/adr/0012-supertonic-como-motor.md).

## What it is

Since [ADR 0012](docs/adr/0012-supertonic-como-motor.md) the engine is **Supertonic 3**, a 99 M
parameter flow-matching model published by Supertone and now archived. Text is read as **Unicode
characters** (no phonemizer), and the speaker's identity is an **input tensor**, not a weight — so
there is no converter after the synthesizer. Nothing was trained for this project.

| Graph | fp32 | Runs in the browser |
|---|---|---|
| `text_encoder.onnx` | 36.4 MB | yes, WebGPU or wasm |
| `duration_predictor.onnx` | 3.7 MB | yes |
| `vector_estimator.onnx` (N steps of flow matching) | 256.5 MB | yes — it is the slow one |
| `vocoder.onnx` | 101.4 MB | yes |
| `speaker_encoder.onnx` (WeSpeaker) | 27.4 MB | it measures similarity **and** builds voices |

A voice is `style_ttl` `[1,50,256]` plus `style_dp` `[1,8,16]` — 292 KB of JSON, not a checkpoint.

**Cloning is a second model, and only cloning.** Supertone never published the encoder that turns a
recording into those two tensors, and six measured attempts to rebuild it failed (evidence,
2026-09-16). So a recording goes through **Pocket TTS** (Kyutai, 100 M parameters, CC BY 4.0,
24 kHz), which publishes its audio encoder. It is downloaded in the browser the first time somebody
clones — 216 MB — and never otherwise. Its own two factory voices are not offered; reading is
Supertonic's job in every case. Nothing was trained for this project, by either model.

## Intended use

Personal, research and educational use of speech synthesis that runs entirely on the visitor's
machine, with no server and no network after load. Cloning your own voice, or a voice whose owner
gave you permission.

## Out of scope

- **Cloning anyone who has not agreed to it.** The demo says so on the page, and it is the one
  limit this project asks you to respect.
- Impersonation, fraud, evading voice-based identity checks, or any output presented as a genuine
  recording of a real person.
- **Whatever OpenRAIL-M's use restrictions forbid.** That licence travels with the weights and its
  restrictions apply to you as much as to us; `LICENSE` is downloaded next to the graphs so it
  cannot get separated from them. Read it before deploying anything.
- Commercial use is outside the project's own scope ([ADR 0006](docs/adr/0006-uso-no-comercial-y-xtts-como-maestro.md)).
  Note this is now **two** constraints, not one: the project's, and the weights'.
- The engine handles 31 languages, but only Spanish is measured here. The rest are offered
  untested, and "untested" is the honest word for them.

## Measured behaviour

Every number is dated in [docs/evidencia.md](docs/evidencia.md), including the ones that fail.

| | Value | Note |
|---|---|---|
| Intelligibility (WER, Whisper-small), reading | **0.008**, median 0.000 | ordinary sentences |
| | 0.178 mean, median 0.000 | corpus sentences: proper nouns, digits |
| Intelligibility, cloned voices | 0.027 | same sentences, same harness |
| Speaker similarity (cosine) | **0.413** | 14 speakers, 42 sentences, OpenSLR es |
| | 0.396 at 2 s of reference → 0.527 at 20 s | duration is the only lever that moved it |
| Ceiling (two real recordings of the target) | 0.760 | the same encoder's own upper bound |
| Time to clone | 0.44 s on CPU; 7.4 s in the browser the first time | the first includes a 39 MB download |
| RTF, CPU, 8 steps | 0.46 | reading |
| Output rate | 44 100 Hz reading, 24 000 Hz cloned | each result carries its own |
| Download | 398 MB reading, +216 MB on first clone | the budget was retired by ADR 0012 |
| Catalan (not supported, measured anyway) | WER 0.237 reading, 1.409 cloned | see limitations |

## Known limitations

- **Similarity is 0.413 against a target of 0.55.** The project's own done-criterion 4 fails. What
  you get is a recognisable resemblance to the reference, not a copy of it. This was chased rather
  than assumed: it does not follow the reference's audio quality (r = +0.247), it plateaus at
  0.41–0.45 across 14 speakers, and sweeping temperature and decoder steps does not move it. The
  one thing that did was giving the encoder more audio (+13 points from 2 s to 20 s), which is why
  the page records 20 s and warns below 15. Treat 0.413 as the honest ceiling of a zero-training
  approach built on published encoders, not as a bug about to be fixed.
- **The download is 398 MB, and 614 MB if you clone.** [ADR 0012](docs/adr/0012-supertonic-como-motor.md)
  retired the 110 MB budget on the owner's explicit instruction; that does not make the number
  small, and on a slow connection the first load is long. The page says so before downloading.
- **A deployed build carries a GPL-3.0 obligation.** The cloning runtime depends on espeak-ng and
  the build distributes its 18.5 MB wasm, even though Spanish never invokes it. See
  [THIRD_PARTY.md](THIRD_PARTY.md); the resolution is still open.
- **Only Spanish is measured.** The engine handles 31 languages and the rest are offered untested.
  Catalan is in neither model's list and was measured to say what happens anyway: readable but
  rough with the factory voices (WER 0.237), noise with a cloned one (1.409).
- Sentence-level synthesis only: a non-autoregressive model emits the whole sentence, so long texts
  are split by sentence in the frontend.
- There is no emotion, style or SSML control. Speed is a real control for factory voices (the model
  predicts duration) and a post-hoc time-stretch for cloned ones.

## Provenance and consent

The ten factory voices are Supertone's own, published with the model. Voices built with
`ttspro.supertonic.constructor` are mixtures of those ten and contain nothing else. Every
measurement here uses VCTK (CC BY 4.0) and OpenSLR es (CC BY-SA 4.0) — corpora recorded with the
speakers' consent for research use — and no audio from them is redistributed here.

A cloned voice is not stored anywhere: it lives inside the browser worker for as long as the tab is
open and is gone when it closes. It cannot be exported to a `.json` the way a built voice can, and
nothing is uploaded.

## Watermarking

**There is none today.** Output carries no audible or inaudible mark, so it cannot be told from
other synthetic audio by inspection. The scope deliberately deferred this until publication was on
the table, and [ADR 0009](docs/adr/0009-publicacion-en-github.md) reopens it as a decision to make
before weights are handed out.
