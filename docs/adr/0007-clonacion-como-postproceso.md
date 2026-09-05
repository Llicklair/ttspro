# 7. La clonación es un postproceso, no algo que el sintetizador aprenda

**Estado:** aceptada · **Fecha:** 2026-09-05 · **Decide:** Marcos · **Matiza a:** [0001](0001-sintetizador-no-autorregresivo.md), [0002](0002-dos-grafos-onnx-y-un-contrato.md)

## Contexto

El plan original era el de YourTTS: un sintetizador que lee un vector de locutor y habla con esa
voz. Medido ([evidencia](../evidencia.md)), eso cuesta **días de GPU** en la máquina que hay, y no
hay atajo: el puente lineal entre el espacio de locutor de coqui y el de WeSpeaker da coseno 0,03
fuera de muestra, peor que predecir la media.

Con el proyecto ya declarado **no comercial y de navegador** ([ADR 0006](0006-uso-no-comercial-y-xtts-como-maestro.md)),
la alternativa que no cuesta entrenamiento es separar las dos cosas que hasta ahora iban juntas:
**qué se dice** lo pone un TTS, y **quién lo dice** lo pone un conversor de timbre.

## Decisión

Tres grafos en vez de dos:

1. `tts.onnx` — texto → onda en una **voz base fija**. Ya no necesita entender embeddings de
   locutor, lo que lo hace un problema mucho más fácil (y permite partir de modelos ya
   entrenados: Piper español, MIT).
2. `voz.onnx` — espectrograma de una grabación de referencia → vector de voz de 256.
3. `conversor.onnx` — onda base + vector de origen + vector de destino → onda en la voz destino.

Los dos últimos son el **conversor de OpenVoice v2** (MIT, `myshell-ai/OpenVoiceV2`), integrado
por referencia (regla 11) y reimplementado sobre los bloques que este repo ya tenía: sus `enc_q`,
`flow` y `dec` son, tensor a tensor, nuestro `PosteriorEncoder`, `ResidualCouplingBlock` y
`Generator`; **486 de 486 tensores cargan** sin sobras ni faltas. Solo hubo que escribir el
codificador de referencia. Trabaja a 22 050 Hz con n_fft 1024 y hop 256, exactamente nuestros
ajustes, así que la salida del TTS entra sin adaptación.

## Consecuencias

- **La clonación deja de depender del entrenamiento.** Medido el mismo día: la cadena completa
  lleva la similitud con el locutor destino de 0,165 (solo TTS, prácticamente el suelo de 0,120)
  a **0,384**, y sobre audio real de un locutor a otro llega a 0,42–0,53. El techo del encoder es
  0,78: es un parecido claro, no una copia.
- **El cuello de botella pasa a ser el TTS base**, que es un problema resuelto: hay voces
  españolas de Piper (MIT, VITS con fonemas de espeak-ng, nuestra misma familia).
- El presupuesto de descarga sube: el conversor son 32,8 M parámetros (~66 MB fp16) más el TTS
  base. Hay que medirlo y decidir; puede obligar a otra enmienda del [ADR 0005](0005-onnx-runtime-web-webgpu-con-fallback-wasm.md).
- La GRU del codificador de referencia no está en el conjunto de ops de WebGPU. Como corre **una
  vez por voz** y no por frase, su grafo puede ir en `wasm` sin que se note.
- Lo que **no** se tira: el sintetizador propio, el afinado y la consistencia de locutor siguen
  siendo el camino a una clonación de verdad end-to-end. Quedan pausados, no descartados, y el
  criterio de terminado no cambia.
- La política de la regla 10 (clonar solo con consentimiento) pesa **más** ahora: clonar acaba de
  volverse fácil. El model card lo dice antes de publicar nada.
