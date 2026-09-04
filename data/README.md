# Datos — licencia escrita o no entra (regla 10)

Cada corpus que toca el entrenamiento o la evaluación tiene una fila aquí. Un fichero de audio
sin fila es un bug. Los datos **no** se versionan en git: esta tabla y los scripts de descarga sí.

| Corpus | Idiomas | Locutores | Horas | Licencia | Uso | URL |
|---|---|---|---|---|---|---|
| Multilingual LibriSpeech (MLS) | es, en (+ fr, de, it, pt, pl, nl) | miles | ~918 es / ~44 000 en | CC BY 4.0 | entrenamiento multi-locutor | https://www.openslr.org/94/ |
| VCTK | en | 110 | 44 | CC BY 4.0 | entrenamiento; baseline de YourTTS | https://datashare.ed.ac.uk/handle/10283/3443 |
| LibriTTS-R `train-clean-100` | en | **247** | ~53 | CC BY 4.0 | entrenamiento (audio restaurado, 24 kHz); receta `libritts_r` | https://www.openslr.org/141/ |
| OpenSLR 61/71/72/73/74/75 (Google, español de Argentina, Chile, Colombia, Perú, Puerto Rico, Venezuela) | es | ~170 | ~38 | CC BY-SA 4.0 | **entrenamiento multi-locutor es** (receta `openslr_es`) | https://www.openslr.org/61 … /75 |
| CSS10 (es) | es | 1 | 24 | Apache 2.0 | fine-tuning; voz limpia de un locutor | https://github.com/Kyubyong/css10 |
| LJSpeech | en | 1 | 24 | dominio público | referencia; receta `ljspeech` | https://keithito.com/LJ-Speech-Dataset/ |
| M-AILABS es_ES | es | 3 | ~108 | libre (licencia M-AILABS) | pendiente: caito.de no respondía el 2026-09-04 | https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/ |
| Common Voice | es, en | miles | variable | CC0 | **solo** evaluación de robustez: calidad de micrófono muy variable | https://commonvoice.mozilla.org/ |

## Voces de referencia en `eval/`

Cinco segundos por locutor, no vistos en entrenamiento, con licencia CC o consentimiento escrito
del locutor guardado junto al audio (`eval/<locutor>/CONSENTIMIENTO.md`). Sin eso, no entran.

## Lo que hay preparado hoy (2026-09-05)

| Corpus | Frases | Horas | Locutores | Idioma |
|---|---|---|---|---|
| OpenSLR 61/71–75 | 24 392 | 37,6 | 174 | es |
| VCTK 0.92 | 44 237 | 41,3 | 109 | en |
| LibriTTS-R train-clean-100 | 33 232 | — | 247 | en |
| **Total** | ~101 900 | ~79+ | **530** | es + en |

El número que importa para clonar voces no vistas es el de **locutores**, no el de horas.

## Manifiesto

Un fichero TSV por corpus en `data/manifests/`, una frase por línea:

```
ruta/al/audio.wav<TAB>es<TAB>id_locutor<TAB>Texto tal cual se leyó.
```

Rutas relativas al manifiesto o absolutas; idioma de `models/contrato.json`. Lo convierte en
índice de entrenamiento `ttspro.data.preparar`: fonemas con el mismo frontend que el navegador,
embedding de locutor con el mismo `speaker_encoder.onnx`, y un `resumen.json` que cuenta lo que
descartó y por qué (símbolos fuera de tabla, duración fuera de 1–12 s, wav ausente).

## Pendiente

- Decidir el subconjunto de MLS es/en que cabe en el tiempo de entrenamiento de una GTX 1070
  (ver [docs/evidencia.md](../docs/evidencia.md), "Batch en 8 GB").
- Script de descarga y verificación de checksums en `src/ttspro/data/`.
