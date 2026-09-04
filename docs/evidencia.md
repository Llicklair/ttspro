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
medir si a ese batch converge en un tiempo aceptable. **Decisión de Marcos (2026-09-04): no se
alquila GPU; si la 1070 no da, se tira de CPU.** Más lento, pero es el techo y está escrito.

Añadido el mismo día: `torch 2.14.0+cu126` ve la GPU (`cuda=True`, capability 6.1) desde el índice
`pytorch-cu126` fijado en `pyproject.toml`. Los wheels cu128+ ya no compilan para Pascal; si se
sube de CUDA, la 1070 se queda fuera y hay que volver a esta línea.

## 2026-09-04 · Fonemizador en el navegador: qué paquete WASM y si coincide con el binario — DECIDE EL ADR 0004

**Montaje.** espeak-ng 1.52.0 instalado por winget. Tres paquetes npm probados sobre las 24 frases
de `tests/fixtures/frontend/frases.txt` (12 es, 12 en), comparando con el binario carácter a
carácter.

| Paquete | Tamaño | Licencia | Idiomas | ¿Coincide con el binario? |
|---|---|---|---|---|
| `phonemizer` 1.2.1 (xenova, el de kokoro-js) | 2,6 MB | Apache-2.0 | **solo inglés** (`en`, `en-gb`, `en-us` y variantes) | no aplica: rechaza `es` |
| `espeak-ng` 1.0.2 (ianmarmour) | 18,5 MB wasm + 0,2 MB js | GPL-3.0 | todos | **24/24 idénticas**, espeak 1.52-dev |
| `@echogarden/espeak-ng-emscripten` 0.3.5 | 24 MB data + 0,8 MB js | GPL-3.0 | todos | no medido: sin API exportada por nombre y más pesado que el anterior sin ventaja |

**Resultado.** Se queda `espeak-ng` 1.0.2. Es un build CLI: cada instanciación ejecuta `main` una
vez, y `callMain` no está exportado, así que no se reutiliza. Coste medido en node: **124–144 ms
por instanciación**; precompilar el wasm una vez no ayuda (5 ms de compilación: el coste es montar
el sistema de ficheros con los datos). Solución: todos los trozos de una frase en una llamada, uno
por línea con " ." añadido, que cierra exactamente una cláusula por línea (probado hasta 480
caracteres sin puntuación: una línea).

**Consecuencia.** El presupuesto de descarga del ADR 0005 contaba ~2 MB para espeak; son 18,5.
Sigue cabiendo en 80 MB, pero come 16 MB del margen del sintetizador. Un build propio con solo
`es` y `en` bajaría a ~3–4 MB; es trabajo de vendorizar (regla 11) y se pospone hasta que el
presupuesto apriete de verdad.

## 2026-09-04 · La API C de espeak no da los mismos acentos que la CLI — NEGATIVO, CAMBIÓ EL DISEÑO

**Montaje.** Python fonemizando por `ctypes` → `espeak_TextToPhonemes` (0,2 ms por frase, sin
proceso) contra el WASM, que solo puede usar la CLI. Mismo flujo de texto en los dos lados.

**Resultado.** 3 de 24 frases difieren, siempre en marcas de acento de palabras función aisladas
por puntuación: `y` → `i` (API) frente a `ˈi` (CLI); `and` → `ænd` frente a `ˈænd`; `what` →
`wˌʌt` frente a `wˈʌt`. La CLI pasa por la fase de entonación de la síntesis, que reasigna acentos;
`espeak_TextToPhonemes` no.

**Consecuencia.** Python usa **el ejecutable**, no la librería. Se pierde la velocidad de la API,
y se recupera por lotes: `fonemizar_lotes` mete las frases que haga falta en un proceso. Anotado
en `models/contrato.json` para que nadie "optimice" de vuelta a la API sin leer esto.

## 2026-09-04 · `espeak-ng -f fichero` no es determinista en Windows — NEGATIVO

**Montaje.** El mismo texto de dos líneas, 20 ejecuciones por variante de entrada, contando cuántas
veces la salida no es exactamente las dos líneas esperadas. Python 3.14.2 (el del venv).

| Entrada | Salidas malas / 20 | Ejemplo de basura |
|---|---|---|
| `-f fichero.txt` | **14** | línea extra `dʒˈiː`, `sˈiː`, `ˈoʊ` (cambia cada vez) |
| `-f` sin salto de línea final | 5 | `hˌaʊ ɑːɹ juː dˈɑːt` |
| `--stdin` | 0 | — |
| texto como argumento | 0 | — |

Con el Python 3.11 del sistema la misma llamada `-f` salió limpia 3 de 3 veces: depende del
entorno del proceso, lo que huele a lectura de memoria sin inicializar en el lector de ficheros de
espeak-ng 1.52.0.

**Consecuencia.** Python alimenta por `--stdin`; el WASM, como argumento. Ambos son búferes en
memoria por el mismo camino interno. `-f` queda prohibido en los dos lados, con el motivo en el
docstring.

## 2026-09-04 · Paridad Python ↔ JS del frontend completo — VERDE

**Montaje.** `tests/test_frontend_paridad.py`: 24 frases, cuatro etapas comparadas por separado
(normalizar, trocear, fonemas, ids), el lado JS ejecutado como proceso `node` sin build.

**Resultado.** 24/24 en las cuatro etapas. Suite rápida: 39 tests en 6 s (los 5 s son node
instanciando el wasm). Web: 7 tests, biome y tsc limpios.

**Consecuencia.** El criterio 3 del terminado deja de ser un stub y pasa. Lo que la paridad **no**
cubre: que los fonemas sean *buenos* (eso lo dirá el WER), ni caracteres fuera de la tabla de
`contrato.json` (fallan en voz alta, que es lo buscado, pero solo cuando aparecen).

---

## Mediciones pendientes que deciden algo

No son tareas: son las preguntas cuyo número cambia una decisión escrita. Cuando se midan, cada una
sube arriba como entrada con fecha.

- **Speaker encoder** (ADR 0002): ECAPA-TDNN vs ResNet H/ASP vs WavLM destilado — tamaño en fp16,
  latencia en `wasm` sobre 5 s de audio, y SECS sobre `eval/`. Es el experimento con más palanca.
- **Ops en WebGPU** (ADR 0005): con `onnxruntime-web` pinneado, qué ops del grafo exportado caen a
  CPU. Cero es el objetivo; cualquier otra cifra es un bug de la regla 4.
- **Batch en 8 GB** (esta libreta): pasos/segundo y VRAM con batch 16 y 32 en fp16, y el mismo
  paso en CPU, porque esa es la alternativa decidida si no cabe.
- **espeak-ng WASM en navegador real** (ADR 0004): los 124 ms por instanciación se midieron en
  node; en Chrome con el wasm cacheado puede ser distinto. Y si el troceado por frase basta para
  que no se note, o hace falta el build recortado a es+en.
- **Umbrales del criterio de terminado** (SCOPE): WER y SECS del primer modelo que hable, para
  recalibrar el 10 % y el 0,70 provisionales contra el baseline de YourTTS.
- **Descarga real** (regla 8): bytes de los dos `.onnx` + wasm de ORT + espeak-ng, medidos en el
  demo servido, no sumados a mano.
- **F5-TTS en WebGPU** (ADR 0001): RTF de una alternativa iterativa. Si baja de 1 con la misma
  calidad, la regla 1 se reabre.
