# TTS pro — la ley de diseño

Reglas **numeradas**, y lo de numeradas no es cosmético: una regla con número se cita en una
revisión ("esto viola la 4") y una cita decide. Un principio en prosa no se cita, y lo que no se
cita no ata. Una regla entra aquí solo si alguna vez se va a poder decir que algo la incumple.

1. **Un forward por frase.** El sintetizador es no autorregresivo: tokens → onda en **un** grafo
   ONNX, sin bucle fuera del grafo. Un cambio que introduzca un paso por token o por frame viola
   esta regla, por buena que sea la prosodia que traiga.
   [ADR 0001](docs/adr/0001-sintetizador-no-autorregresivo.md).

2. **Dos grafos, un contrato.** `speaker_encoder.onnx` (onda 16 kHz → embedding) y `tts.onnx`
   (tokens + embedding + idioma + ruido → onda). Sus entradas, salidas, dtypes y ejes dinámicos
   viven en [models/contrato.json](models/contrato.json); ese fichero lo leen el test de paridad y
   el runtime JS. Cambiar una firma sin cambiar el contrato es violación.
   [ADR 0002](docs/adr/0002-dos-grafos-onnx-y-un-contrato.md).

3. **Paridad del frontend por test, no por confianza.** Cada transformación de texto
   (normalización, fonemización, tokenización) existe en Python (`ttspro.frontend`) y en
   TypeScript (`web/src/frontend`), y un test las compara sobre `tests/fixtures/frontend/`.
   Añadir una regla a un lado sin el otro rompe el test — así debe ser.
   [ADR 0004](docs/adr/0004-fonemas-con-espeak-ng-en-los-dos-lados.md).

4. **Solo ops que ORT Web ejecuta.** Opset 17 y ops soportadas por los EP `webgpu` y `wasm` de
   ONNX Runtime Web. El test de export lo comprueba sobre el grafo exportado; una op que solo corra
   en CUDA no entra ni "temporalmente".
   [ADR 0005](docs/adr/0005-onnx-runtime-web-webgpu-con-fallback-wasm.md).

5. **Todo lo aleatorio es una entrada.** El ruido del flow y del predictor estocástico de duración
   entran como tensores. Dado su input, el grafo es determinista: se puede comparar, reproducir y
   exportar sin `RandomNormal` dentro.

6. **Python entrena, JS ejecuta, ninguno importa al otro.** `web/` no contiene Python ni lo llama;
   `src/ttspro` no contiene JS ni lo llama, salvo el test de paridad, que lanza `node` como
   proceso y compara salidas. Lo único que cruza es `models/` (dos `.onnx` + `contrato.json`).

7. **La inferencia no depende del entrenamiento.** `ttspro.model`, `ttspro.export` y
   `ttspro.frontend` no importan `ttspro.train` ni `ttspro.data`. Declarado en
   [.gb-boundaries](.gb-boundaries) y gateado en el pre-commit.

8. **Presupuestos, no deseos.** Descarga total ≤ 110 MB (era 80 hasta el 2026-09-04: ADR 0005,
   enmienda, para partir del VITS de VCTK de coqui); una frase de 10 palabras en < 3 s con el
   EP `wasm` en Chromium headless; RTF < 1 con `webgpu` en una GPU integrada. Se miden en
   `tests/terminado` y se anotan en [docs/evidencia.md](docs/evidencia.md). Superarlos es un bug de
   arquitectura, no un problema de rendimiento a optimizar luego.

9. **Cero modelos externos en el camino de síntesis.** Ni LLM ni servicio: un navegador sin red
   sintetiza. La IA, si entra, entra antes (entrenar) o después (evaluar), nunca entre el texto y la
   onda.

10. **Datos con licencia escrita.** Cada corpus figura en [data/README.md](data/README.md) con
    licencia y URL; audio sin licencia declarada no entra al entrenamiento. Las voces de referencia
    del repo son CC o llevan consentimiento escrito. Un fichero de audio sin fila en esa tabla es un
    bug.

11. **Lo externo, por referencia.** espeak-ng, ONNX Runtime y Whisper (evaluación) se detectan e
    instalan por su canal oficial, no se copian al repo. El WASM de espeak-ng en `web/` se pinnea
    como dependencia npm a la **misma versión** que el binario de Python — la paridad de la regla 3
    lo vigila.

## Cómo se cambia esto

Una regla se cambia o se retira con **un ADR que la cite por número** y **una entrada en
[docs/evidencia.md](docs/evidencia.md) con la medición** que lo motiva. Sin medición, no: "otros lo
hacen" no es una razón. Puede proponerlo cualquiera, persona o agente; lo firma Marcos. El ADR
que retira una regla deja la regla tachada aquí con el enlace, no la borra: la próxima persona tiene
que poder ver que existió y por qué dejó de aplicarse.
