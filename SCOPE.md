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
| **Uso comercial** (decidido el 2026-09-05, [ADR 0006](docs/adr/0006-uso-no-comercial-y-xtts-como-maestro.md)) | El proyecto es no comercial. Eso abre XTTS-v2 (CPML) como maestro y listón, nunca como runtime. Si cambia, se reentrena sin datos de XTTS. Matiz del [ADR 0009](docs/adr/0009-publicacion-en-github.md): el **código** se publica MIT, y hoy no hay ni un peso ni un dato de XTTS en la cadena, así que la restricción es de este proyecto, no de lo que reparte. |
| **Fork de coqui TTS o dependencia de él** | Archivado en 2024 y con export hostil (MAS en el grafo, config gigante, phonemizer acoplado). Se reimplementa lo mínimo. [ADR 0003](docs/adr/0003-implementacion-propia-no-fork.md). |
| **Entrenar o afinar en el navegador** | Entrenar es Python. El navegador solo ejecuta. |
| **Conversión de voz** (audio → audio) | Otro problema con otro modelo. |
| **Streaming dentro de una frase** | Un modelo no autorregresivo emite la frase entera. Lo que SÍ entra es trocear por oraciones en el frontend JS y reproducir la primera mientras se sintetiza la segunda. |
| **Emoción, estilo, SSML** | Después del criterio de terminado, y con ADR. |
| **Idiomas fuera del conjunto entrenado** | El MVP entrena **español e inglés**. Añadir un idioma es datos + fonemizador + reentrenar, no una rama de código. |
| **Watermarking de audio y anti-spoofing** | No porque no importe: porque hoy no se publica ningún peso. Se reabre en el ADR de publicación, **antes** de publicar. Reabierto y contestado el 2026-09-05 en [ADR 0009](docs/adr/0009-publicacion-en-github.md): se publica el repo **sin pesos**, así que la puerta sigue cerrada; se vuelve a abrir el día que se publique un peso o se sirva la demo online. |
| **Un LLM en el camino de síntesis** (normalizar o "mejorar" texto con modelo) | Normalización determinista con paridad Python/JS (regla 3). El navegador sin red sintetiza (regla 9). |
| **Apps nativas** (móvil, Electron, React Native) | El runtime es el navegador. Si ORT Web corre ahí, corre. |
| **Clonar sin consentimiento** | Las voces de referencia del repo llevan licencia o consentimiento escrito (regla 10). La política de uso del modelo se escribe en el model card antes de publicar: escrita el 2026-09-05 en [MODEL_CARD.md](MODEL_CARD.md). |

## Criterio de terminado

El MVP está terminado cuando **este comando pasa**:

```gb:terminado
uv run pytest tests/terminado -q
```

Lo que comprueba, en orden, y por qué cada cosa. **Reescrito el 2026-09-16** contra la cadena del
[ADR 0012](docs/adr/0012-supertonic-como-motor.md): lo de antes medía `models/tts.onnx`, un
checkpoint de PyTorch contra el que comparar el export y un presupuesto de 110 MB, y ninguna de las
tres cosas existe ya. El criterio estaba **rojo por el motivo equivocado**, que es el peor estado
posible: cuando el rojo no significa nada, deja de avisar.

1. **Existe y carga**: los cuatro grafos de `models/supertonic/` abren en `onnxruntime` y sus firmas
   coinciden con [contrato.json](models/supertonic/contrato.json). El contrato se **genera leyendo
   los propios `.onnx`**, así que esto no compara dos ficheros escritos a mano: comprueba que lo
   descargado sigue siendo lo que se leyó.
2. **Reproducibilidad (regla 5)**: el mismo texto, la misma voz y la misma semilla dan la **misma
   onda**, y dos semillas distintas dan ondas distintas. Sustituye a la paridad PyTorch ↔ ORT, que
   guardaba «lo que corre no es lo que se midió» cuando el export lo hacíamos aquí; hoy el modelo
   viene ya en ONNX y no hay lado de torch con el que discrepar. El riesgo equivalente es el sorteo
   del ruido: si esto falla, ningún número de [evidencia](docs/evidencia.md) se puede volver a
   comprobar.
3. **Paridad Python ↔ JS del frontend** (regla 3): `tests/test_supertonic_paridad.py`, sobre el
   frontend Unicode. Ya no hay fonemas que comparar.
4. **Calidad objetiva**, sobre el audio que sale del **navegador**, no de una síntesis en Python:
   - inteligibilidad: WER con Whisper-small ≤ 0,10 leyendo con una voz predeterminada;
   - similitud de locutor: coseno de WeSpeaker **≥ 0,55** entre la referencia y lo que dice la voz
     clonada de ella.
5. **En el navegador**: una frase de 10 palabras en < 3 s desde que el modelo está cargado. El
   presupuesto de descarga que este punto llevaba lo retiró el ADR 0012 por instrucción explícita;
   los MB se siguen anotando, pero no se juzgan.

Los puntos 4 y 5 los mide `web/e2e/criterio.spec.ts` ejecutando la página de verdad, que deja los
wav y un `criterio.json`; el test de Python les pone el número. Reimplementar la síntesis en Python
para puntuarla puntuaría la reimplementación: lo que se publica es una página, y el modelo que clona
sólo existe como Worker. Por eso el comando completo son dos:

```bash
cd web && npx playwright test e2e/criterio.spec.ts   # la página, de verdad; deja web/medido/
uv run pytest tests/terminado -q                     # le pone el número a lo que dejó
```

Que el criterio sea BUENO sigue sin poder juzgarlo ninguna herramienta (`exit 0` también pasa): que
el audio suene bien lo juzga una persona, y esa escucha se anota en evidencia con fecha, no en el
test.

Hoy (2026-09-16) el comando **falla en dos de los cinco puntos**, y los dos rojos están medidos y
explicados, que es distinto de estar rotos:

| | Estado | Medido |
|---|---|---|
| 1 existe y carga | ✅ | los cuatro grafos |
| 2 reproducibilidad | ✅ | misma semilla, onda idéntica |
| 3 paridad de frontends | ✅ | 26 casos |
| 4a inteligibilidad | ✅ | WER medio **0,048** sobre cuatro frases (los dos fallos son del ASR: «sarpó», «9») |
| 4b similitud | ❌ | **0,349** con esta referencia de 6 s en otro idioma; 0,413 de media sobre 14 locutores |
| 5 latencia | ❌ | **~4 100 ms** en `wasm` headless (4 078–4 088 en tiradas distintas), RTF ~0,95 |

El 4b es el techo del encoder publicado, no un parámetro mal puesto: barrer temperatura y pasos no
lo mueve, y lo único que lo movió fue darle más grabación (+13 puntos de 2 s a 20 s). El 5 es el
suelo de `wasm` sin GPU; queda por medir en WebGPU, que es donde el `vector_estimator` tendría que
notarse ([evidencia](docs/evidencia.md), mediciones pendientes).
