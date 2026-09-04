# Datos — licencia escrita o no entra (regla 10)

Cada corpus que toca el entrenamiento o la evaluación tiene una fila aquí. Un fichero de audio
sin fila es un bug. Los datos **no** se versionan en git: esta tabla y los scripts de descarga sí.

| Corpus | Idiomas | Locutores | Horas | Licencia | Uso | URL |
|---|---|---|---|---|---|---|
| Multilingual LibriSpeech (MLS) | es, en (+ fr, de, it, pt, pl, nl) | miles | ~918 es / ~44 000 en | CC BY 4.0 | entrenamiento multi-locutor | https://www.openslr.org/94/ |
| VCTK | en | 110 | 44 | CC BY 4.0 | entrenamiento; baseline de YourTTS | https://datashare.ed.ac.uk/handle/10283/3443 |
| LibriTTS-R | en | 2 456 | 585 | CC BY 4.0 | entrenamiento (audio restaurado) | https://www.openslr.org/141/ |
| CSS10 (es) | es | 1 | 24 | Apache 2.0 | fine-tuning; voz limpia de un locutor | https://github.com/Kyubyong/css10 |
| Common Voice | es, en | miles | variable | CC0 | **solo** evaluación de robustez: calidad de micrófono muy variable | https://commonvoice.mozilla.org/ |

## Voces de referencia en `eval/`

Cinco segundos por locutor, no vistos en entrenamiento, con licencia CC o consentimiento escrito
del locutor guardado junto al audio (`eval/<locutor>/CONSENTIMIENTO.md`). Sin eso, no entran.

## Pendiente

- Decidir el subconjunto de MLS es/en que cabe en el tiempo de entrenamiento de una GTX 1070
  (ver [docs/evidencia.md](../docs/evidencia.md), "Batch en 8 GB").
- Script de descarga y verificación de checksums en `src/ttspro/data/`.
