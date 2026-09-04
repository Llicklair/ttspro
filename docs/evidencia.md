# Evidencia — la libreta

Cada medición real: qué se probó, qué salió, qué cambió por ello.

**Los resultados negativos se escriben con el mismo detalle que los positivos, o más.** Un proyecto
que solo registra lo que funcionó no tiene evidencia: tiene publicidad. Y el dato que no está en el
repo, no existe — la memoria de nadie cuenta.

## Formato

`## AAAA-MM-DD · qué se probó — VEREDICTO`, y debajo: montaje, resultado, consecuencia.

---

## 2026-09-04 · El suelo del proyecto con `gb floor` — SUELO PUESTO, CRITERIO EN ROJO

**Montaje.** Carpeta vacía, `git init`, `gb floor` (galaxy-brain 0.7.0) antes y después de
`--init`, y después de rellenar los documentos.

**Resultado.**

| Momento | Niveles sin cubrir (de 8) | Lo que faltaba |
|---|---|---|
| Carpeta vacía | 5 | tests, gates, mapa, invariantes, AGENTS.md; ADR y criterio de terminado en gris |
| Tras `--init` | 4 | los mismos, con 7 documentos puestos pero marcados sin rellenar |
| Tras rellenar | **0** | criterio de terminado en gris (nunca se marca verde, por diseño) |

Con el suelo relleno: `gb graph . --gate` ve 14 módulos (Python y TypeScript en un mismo mapa),
4 aristas, 0 ciclos, 8 reglas de frontera sin cruces. Suite rápida: 10 tests en verde en 0,2 s,
incluida la paridad Python ↔ JS de la normalización lanzando `node` como proceso. Criterio de
terminado: 6 tests en rojo, el primero por "no existe models/tts.onnx". `ruff check` y
`ruff format --check` limpios.

**Consecuencia.** El criterio de terminado queda como comando (`uv run pytest tests/terminado -q`)
y **falla hoy**, que es lo correcto: no hay modelo. Los cinco ADR fijan las decisiones que el
navegador impone antes de la primera línea de modelo; cada uno nombra la medición que lo reabriría.

## 2026-09-04 · Inventario de la máquina de entrenamiento — DATO, SIN VEREDICTO

**Montaje.** `nvidia-smi`, versiones de la cadena de herramientas.

**Resultado.** GPU NVIDIA GeForce GTX 1070, 8 GB. Python 3.11.9, uv 0.9.26, ruff 0.16.4,
pytest 9.1.1, node 24.13.0, npm 11.6.2, ast-grep 0.45.2. Sin torch, sin onnxruntime, sin
espeak-ng instalados aún.

**Consecuencia.** 8 GB acotan el batch de VITS a ~16 segmentos de 8 192 muestras en fp16, sin
cambios de arquitectura. YourTTS entrenó con más; la primera entrada de entrenamiento tiene que
medir si a ese batch converge en un tiempo aceptable o si toca alquilar GPU. Se anota aquí para que
la decisión de alquilar se tome con un número, no con impaciencia.

---

## Mediciones pendientes que deciden algo

No son tareas: son las preguntas cuyo número cambia una decisión escrita. Cuando se midan, cada una
sube arriba como entrada con fecha.

- **Speaker encoder** (ADR 0002): ECAPA-TDNN vs ResNet H/ASP vs WavLM destilado — tamaño en fp16,
  latencia en `wasm` sobre 5 s de audio, y SECS sobre `eval/`. Es el experimento con más palanca.
- **Ops en WebGPU** (ADR 0005): con `onnxruntime-web` pinneado, qué ops del grafo exportado caen a
  CPU. Cero es el objetivo; cualquier otra cifra es un bug de la regla 4.
- **Batch en 8 GB** (esta libreta): pasos/segundo y VRAM con batch 16 y 32 en fp16.
- **Umbrales del criterio de terminado** (SCOPE): WER y SECS del primer modelo que hable, para
  recalibrar el 10 % y el 0,70 provisionales contra el baseline de YourTTS.
- **Descarga real** (regla 8): bytes de los dos `.onnx` + wasm de ORT + espeak-ng, medidos en el
  demo servido, no sumados a mano.
- **F5-TTS en WebGPU** (ADR 0001): RTF de una alternativa iterativa. Si baja de 1 con la misma
  calidad, la regla 1 se reabre.
