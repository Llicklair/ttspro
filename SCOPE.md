# TTS pro — alcance

## En una frase

Un sintetizador de voz **multilingüe y no autorregresivo** que clona una voz a partir de **un solo
audio de referencia de 3–10 s**, se entrena en PyTorch, se exporta a **ONNX** y sintetiza **en el
navegador con ONNX Runtime Web**, sin servidor y sin red.

Referente: lo que hacía [coqui TTS](https://github.com/coqui-ai/tts) con YourTTS (VITS +
speaker encoder), hecho pequeño, exportable y con el navegador como runtime de primera clase, no
como demo.

## Lo que NO entra

| Fuera | Motivo |
|---|---|
| **Modelos autorregresivos** (XTTS, Tortoise, VALL-E, Bark) | Mejor prosodia, pero decodifican token a token con caché KV: ni caben en el presupuesto de latencia (regla 8) ni se exportan a un grafo ONNX que ORT Web ejecute bien. [ADR 0001](docs/adr/0001-sintetizador-no-autorregresivo.md). |
| **Servidor de inferencia** (API HTTP/gRPC, cola, cuentas) | El producto es `tts.onnx` + el runtime JS. Quien quiera servidor lo tiene con `onnxruntime` en Python en cinco líneas; aquí no se mantiene. |
| **Fork de coqui TTS o dependencia de él** | Archivado en 2024 y con export hostil (MAS en el grafo, config gigante, phonemizer acoplado). Se reimplementa lo mínimo. [ADR 0003](docs/adr/0003-implementacion-propia-no-fork.md). |
| **Entrenar o afinar en el navegador** | Entrenar es Python. El navegador solo ejecuta. |
| **Conversión de voz** (audio → audio) | Otro problema con otro modelo. |
| **Streaming dentro de una frase** | Un modelo no autorregresivo emite la frase entera. Lo que SÍ entra es trocear por oraciones en el frontend JS y reproducir la primera mientras se sintetiza la segunda. |
| **Emoción, estilo, SSML** | Después del criterio de terminado, y con ADR. |
| **Idiomas fuera del conjunto entrenado** | El MVP entrena **español e inglés**. Añadir un idioma es datos + fonemizador + reentrenar, no una rama de código. |
| **Watermarking de audio y anti-spoofing** | No porque no importe: porque hoy no se publica ningún peso. Se reabre en el ADR de publicación, **antes** de publicar. |
| **Un LLM en el camino de síntesis** (normalizar o "mejorar" texto con modelo) | Normalización determinista con paridad Python/JS (regla 3). El navegador sin red sintetiza (regla 9). |
| **Apps nativas** (móvil, Electron, React Native) | El runtime es el navegador. Si ORT Web corre ahí, corre. |
| **Clonar sin consentimiento** | Las voces de referencia del repo llevan licencia o consentimiento escrito (regla 10). La política de uso del modelo se escribe en el model card antes de publicar. |

## Criterio de terminado

El MVP está terminado cuando **este comando pasa**:

```gb:terminado
uv run pytest tests/terminado -q
```

Lo que comprueba, en orden, y por qué cada cosa:

1. **Existe y carga**: `models/tts.onnx` y `models/speaker_encoder.onnx` abren en `onnxruntime`,
   opset ≥ 17, y sus firmas coinciden con [models/contrato.json](models/contrato.json).
2. **Paridad PyTorch ↔ ORT**: sobre 20 frases fijadas (10 es, 10 en), la distancia log-mel media
   entre la salida de PyTorch y la de ORT es < 0,1. Si el export cambia el audio, el export está
   roto.
3. **Paridad Python ↔ JS del frontend**: las frases de `tests/fixtures/frontend/` producen la misma
   salida en `ttspro.frontend` y en `web/src/frontend` (regla 3).
4. **Calidad objetiva** sobre `eval/` (voces de referencia con licencia, 5 s cada una, no vistas en
   entrenamiento):
   - inteligibilidad: WER con Whisper-small ≤ 10 % en es y en en;
   - similitud de locutor: coseno entre el embedding de la referencia y el de la síntesis ≥ 0,70 de
     media.
   Umbrales **provisionales**: se recalibran contra el baseline (YourTTS reportó SECS ≈ 0,75–0,82
   en VCTK) en cuanto haya una primera medición en [docs/evidencia.md](docs/evidencia.md).
5. **En el navegador**: Playwright + Chromium headless con el EP `wasm`, una frase de 10 palabras
   en < 3 s desde que el modelo está cargado, y descarga total (los dos `.onnx` + wasm de ORT +
   espeak) ≤ 110 MB (80 hasta el 2026-09-04; ADR 0005, enmienda).

Que el criterio sea BUENO sigue sin poder juzgarlo ninguna herramienta (`exit 0` también pasa): que
el audio suene bien lo juzga una persona, y esa escucha se anota en evidencia con fecha, no en el
test.

Hoy (2026-09-04) el comando **falla**, y debe fallar: no hay modelo. Es el termómetro.
