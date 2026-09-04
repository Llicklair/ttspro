# 1. El sintetizador es no autorregresivo (VITS multilingüe con embedding de locutor)

**Estado:** aceptada · **Fecha:** 2026-09-04 · **Regla:** [ARCHITECTURE 1](../../ARCHITECTURE.md)

## Contexto

El objetivo fija tres cosas a la vez: clonación con un audio, varios idiomas y ejecución en el
navegador con ONNX Runtime Web. La familia de modelo decide si lo tercero es posible o no, y no
se arregla después optimizando.

Las familias sobre la mesa en 2026, con lo que cuesta cada una **en el navegador**:

| Familia | Ejemplos | Clona con 1 audio | Export ONNX | En ORT Web |
|---|---|---|---|---|
| Autorregresiva (GPT sobre tokens de audio + decoder) | XTTS-v2, Tortoise, VALL-E, Bark | sí, muy bien | por partes, con caché KV | un paso por token: cientos de forwards por frase; XTTS-v2 además es CPML (no comercial) |
| Flow-matching / difusión iterativa | F5-TTS, E2-TTS, StyleTTS2 (parte difusión) | sí | sí | 16–32 pasos del DiT por frase; posible, fuera de presupuesto hoy |
| No autorregresiva, un forward | VITS, YourTTS, Kokoro (StyleTTS2 sin difusión) | YourTTS sí; **Kokoro no** (voces fijas) | un grafo | Kokoro corre hoy en navegador con ~82 M parámetros vía `kokoro-js`: prueba de que el tramo navegador de la tesis es real |

## Decisión

**VITS multilingüe con embedding de locutor externo**, la receta de YourTTS: encoder de texto
sobre fonemas con embedding de idioma, flow, predictor estocástico de duración y generador HiFi-GAN
en **un solo grafo**; la identidad del locutor entra como un vector calculado por otro grafo
([ADR 0002](0002-dos-grafos-onnx-y-un-contrato.md)). Un forward por frase, regla 1.

## Alternativas descartadas

- **XTTS-v2**: la mejor prosodia de la tabla y el referente de coqui, pero autorregresivo y con
  licencia CPML. Ni cabe en la regla 8 ni permite publicar pesos con uso comercial.
- **F5-TTS**: no autorregresivo en el sentido de "sin caché KV", pero iterativo; 300 M parámetros y
  32 pasos por frase. Se reabre si ORT Web con WebGPU baja el coste por paso lo bastante — es la
  candidata natural a "v2" y la entrada de evidencia que lo mediría está pendiente.
- **Kokoro / StyleTTS2 sin difusión**: demuestra el navegador, pero no clona. Su arquitectura
  (predicción de estilo desde texto) es lo contrario de lo que se pide.

## Consecuencias

- Prosodia más plana que un autorregresivo. Se acepta a cambio de un forward y un grafo.
- La calidad de la clonación queda acotada por el speaker encoder, no por el sintetizador: elegirlo
  y medirlo es el experimento con más palanca del proyecto.
- El predictor estocástico de duración y el flow usan ruido: para exportar, el ruido entra como
  tensor (regla 5). Monotonic Alignment Search solo existe en entrenamiento y no toca el grafo.
- Requisito para reabrir esto: una medición en evidencia con RTF < 1 en WebGPU de una alternativa
  iterativa, con la misma calidad objetiva que la del criterio de terminado.
