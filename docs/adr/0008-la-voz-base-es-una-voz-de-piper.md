# 8. La voz base es una voz de Piper portada, no un modelo propio a medio entrenar

**Estado:** aceptada · **Fecha:** 2026-09-05 · **Decide:** Marcos · **Sigue a:** [0007](0007-clonacion-como-postproceso.md)

## Contexto

Con la clonación resuelta como postproceso ([ADR 0007](0007-clonacion-como-postproceso.md)), el
cuello de botella pasó a ser el TTS base. Medido: nuestro modelo de 2 000 pasos daba **WER 0,32–0,42**
en español y pesaba 61 MB en fp16, y llevarlo a calidad decente son días de GPU
([evidencia](../evidencia.md)).

[Piper](https://github.com/OHF-Voice/piper1-gpl) publica voces **MIT** que son VITS con fonemas de
espeak-ng, o sea nuestra misma familia y nuestro mismo frontend. Nueve en español.

## Decisión

La voz base es **`es_ES-davefx-medium`** de Piper, portada a `ttspro.model.Sintetizador` desde su
ONNX (`ttspro.train.piper`). No se sirve el ONNX de Piper tal cual: el suyo tiene
`RandomNormalLike`, `NonZero`, `And` y `Softplus`, que WebGPU no ejecuta y que además meten el
ruido dentro del grafo, en contra de la regla 5.

El port lee la arquitectura de las **formas de los tensores**, no de suposiciones, y descubrió dos
cosas que no eran como las nuestras: el decoder usa **ResBlock2** (dos convoluciones por bloque,
tres upsamples de 256 canales) y las tablas de posición relativa son compartidas entre cabezas, lo
que hacía deducir 1 cabeza donde hay 2. **349 tensores cargados**, y lo que no carga es solo la
parte de entrenamiento (`enc_q`, la rama `post_*` del predictor de duración y el `ConvFlow` que
VITS descarta en inferencia).

Tres cosas que el port arrastra y que ahora son **datos del contrato**, no reglas escondidas:

1. **Tabla de símbolos propia** de Piper, con su convención: `^` al principio, cada símbolo
   seguido de un `pad`, y `$` al final. La nuestra ponía el `pad` antes. Los dos frontends leen
   `blank_al_inicio`, `bos` y `eos` del contrato.
2. **Equivalencias declaradas**: Piper nunca vio `¿` ni `¡`, así que el contrato dice
   `{"¿": "", "¡": ""}` y se descartan **a la vista**, no en silencio.
3. **Firma del grafo menor**: una voz fija monolingüe no lee `embedding` ni `idioma`, y
   `torch.onnx.export` los poda. El contrato declara las entradas reales y el navegador construye
   su feed a partir de él.

## Consecuencias

- **Español de verdad**: WER mediana **0,091** frente a 0,32. Los dos peores casos de la medición
  son Whisper escribiendo cifras ("6 entradas" por "seis entradas"), no fallos de síntesis.
- **Más pequeño y mucho más rápido**: 16,4 M parámetros, 63,3 MB fp32 / **32,3 MB fp16** (eran
  121/61,2) y en el navegador **401 ms** por frase en wasm, frente a 3 825 ms. Es ResBlock2 con
  tres upsamples en vez de ResBlock1 con cuatro.
- **El demo es solo español.** Es lo honesto: la voz base es monolingüe, y ofrecer inglés
  produciría ruido. Añadir inglés es portar una segunda voz (`en_US-libritts_r-medium`, 904
  locutores) y decidir cómo conviven dos voces base.
- Lo que **no** se tira: `ttspro.train.entrenar`, la consistencia de locutor y el modelo propio
  siguen ahí. Cambia de qué se parte, no a dónde se va.
- El presupuesto sigue sin cumplirse: 148,8 MB fp16 previstos contra 110. Ahora el peso lo domina
  el conversor (66,5 MB), no el TTS.
