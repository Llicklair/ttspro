# 2. Dos grafos ONNX y un contrato escrito

**Estado:** aceptada · **Fecha:** 2026-09-04 · **Regla:** [ARCHITECTURE 2](../../ARCHITECTURE.md)

## Contexto

La clonación necesita convertir el audio de referencia en un vector de locutor, y el sintetizador
necesita ese vector en cada frase. Hay dos formas de empaquetarlo y la elección se nota en el
navegador, no en Python.

| Opción | Coste en el navegador |
|---|---|
| Un grafo: audio de referencia + texto → onda | Recalcula el embedding en cada frase (un encoder sobre 5 s de audio, cada vez); y mete la extracción de mel del audio dentro del mismo grafo que la síntesis, así que un cambio en uno reexporta el otro |
| **Dos grafos**: `speaker_encoder.onnx` y `tts.onnx` | El embedding se calcula **una vez** por voz y se cachea (y se puede guardar: una voz son 256 floats); cada grafo se exporta, prueba y versiona solo |

Queda una pregunta dentro de la segunda: **¿quién calcula el espectrograma mel** del audio de
referencia? Si lo hace JS, hay que reimplementar STFT + banco mel en TypeScript y mantener paridad
bit a bit con `torchaudio`, que es exactamente la clase de bug que no se ve hasta que la clonación
"suena rara". Si lo hace el grafo, hace falta que la STFT sea una op que ORT Web ejecute — la op
`STFT` de ONNX no está en el EP WebGPU — o expresarla como convolución 1-D con núcleos DFT fijos,
que sí está en todos los EP.

## Decisión

**Dos grafos.** `speaker_encoder.onnx` recibe **onda cruda a 16 kHz** y lleva dentro la STFT como
`Conv1d` con núcleos fijos y el banco mel como `MatMul`; JS solo remuestrea. `tts.onnx` recibe
tokens, embedding de locutor, id de idioma y los tensores de ruido, y devuelve onda a la frecuencia
del vocoder.

Las firmas viven en [models/contrato.json](../../models/contrato.json): nombre, dtype, forma y
ejes dinámicos de cada entrada y salida, más frecuencias de muestreo y dimensión del embedding.
Lo leen el test de paridad de `tests/terminado` y el runtime de `web/`. **Un `.onnx` cuya firma no
coincide con el contrato no pasa el test**, y eso es lo que impide que Python y JS se separen sin
que nadie se entere.

## Consecuencias

- Una voz es un fichero de 256 floats: se puede guardar, compartir y cargar sin el audio original.
- La STFT por convolución cuesta más memoria que la op nativa (núcleos de `n_fft × n_fft/2`);
  con `n_fft = 400` a 16 kHz es despreciable. Si un día ORT Web ejecuta `STFT` en WebGPU, se
  mide y se cambia con ADR.
- Dos exports, dos tests de paridad, dos presupuestos de tamaño. El contrato es el precio de que
  no haya un tercer sitio donde las firmas estén escritas.
- El speaker encoder puede cambiar (ECAPA-TDNN, ResNet H/ASP, WavLM-base destilado) sin tocar el
  sintetizador **siempre que** se reentrene el sintetizador con los embeddings nuevos: el vector no
  es intercambiable entre encoders, y el contrato lleva el nombre y la versión del encoder por eso.
