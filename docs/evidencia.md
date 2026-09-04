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

## 2026-09-04 · Speaker encoder: ECAPA-TDNN (speechbrain) contra ResNet34-LM (WeSpeaker) — DECIDE EL ADR 0002

**Montaje.** `uv run python -m ttspro.export.speaker_encoder --backbone <x>`: features como Conv1d
+ MatMul dentro del grafo (`ttspro.model.fbank`), backbone por referencia, normalización L2,
compuesto con `onnx.compose`, opset 17. Paridad medida sobre 5 s de ruido gaussiano (semilla 0)
contra el pipeline propio de cada backbone; latencia en `onnxruntime` CPU, mediana de 10.

| | ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb` | ResNet34-LM `Wespeaker/wespeaker-voxceleb-resnet34-LM` |
|---|---|---|
| Licencia | Apache-2.0 | CC-BY-4.0 |
| Cómo llega | torch → export | **ONNX ya publicado**, usado tal cual |
| Parámetros | 20,77 M | **6,63 M** |
| Dimensión | 192 | **256** (la del contrato) |
| Features | fbank speechbrain (hamming periódica, 10·log10, top_db 80) | fbank Kaldi (×32768, quitar DC, preénfasis, hamming no periódica, DFT 512, ln) |
| Diferencia máx. de mis features vs las suyas | **2,4 dB** en 62 de 80 bandas (coseno del embedding aun así 0,99996; no diagnosticado del todo) | **0,0026** |
| fp32 | 84,1 MB · coseno 1,0 · 126 ms | **27,4 MB · coseno 1,0 · 61 ms** |
| fp16 | 42,3 MB · coseno 0,9997 · 99 ms | **14,2 MB · coseno 0,99999** · 82 ms |
| Ops fuera de WebGPU | 0 (tras cambiar `torch.maximum` → `Where`) | **0** |

**Resultado.** WeSpeaker ResNet34-LM: 3× menos parámetros, 3× menos descarga, 2× más rápido, la
dimensión del contrato, y sin exportar nada de torch salvo mis dos grafos auxiliares. Las
métricas publicadas por WeSpeaker para este modelo (EER 0,72 % en Vox1-O) son la misma clase que
ECAPA (0,80 %); no las he medido yo.

**Consecuencia.** `models/contrato.json` fija el encoder y describe las features. El criterio 1 del
terminado pasa para `speaker_encoder.onnx`. **Lo que esta medición no dice:** que el coseno sea 1,0
con ruido no garantiza que lo sea con voz; la paridad hay que repetirla con audio real cuando haya
`eval/`, y la SECS de verdad sale de ahí.

## 2026-09-04 · El fbank no sobrevive a fp16 y el conversor tiene trampas — NEGATIVO, CAMBIÓ EL DISEÑO

**Montaje.** `onnxruntime.transformers.float16.convert_float_to_float16` sobre el grafo compuesto.

**Resultado.** Embedding NaN. Diagnóstico con todas las salidas intermedias expuestas
(`.scratch/diag_fp16.py`): el `Conv` del fbank Kaldi produce valores de 1,2 millones (onda ×
32768) y su cuadrado 4,3 × 10⁹; fp16 llega a 65 504. Tres intentos, tres lecciones:

1. `node_block_list` con los nodos del fbank: sigue NaN. El conversor deja el nodo en fp32 pero
   guarda en fp16 los tensores **entre** nodos bloqueados.
2. Bloquear generó dos `Cast` con el mismo nombre (ORT los rechaza) y después dos con el mismo
   tensor de salida. Parche de deduplicación: funcionaba, pero no arreglaba el punto 1.
3. **Convertir solo el grafo del backbone antes de componer**, con `keep_io_types`: coseno 0,99999.
   El conversor además deja el `Cast` de entrada al final de la lista de nodos y el checker exige
   orden topológico: `_ordenar` en `ttspro.export.speaker_encoder`.

**Consecuencia.** La precisión se decide **por grafo, antes de componer**; el fbank y la
normalización son fp32 siempre. Escrito en el contrato (`precision`). En CPU el fp16 es más lento
que el fp32 (82 frente a 61 ms): la ganancia de fp16 es descarga, y velocidad solo en WebGPU.

## 2026-09-04 · El sintetizador exporta entero a un grafo WebGPU, y el tamaño decide la configuración — DECIDE ADR 0005

**Montaje.** VITS (YourTTS: embedding de locutor externo de 256, embedding de idioma sumado al
de símbolo) reimplementado en `ttspro.model` con el export como restricción: sin `Softplus`
(Where/Log/Exp), sin `torch.flip` (Gather), splines del predictor de duración sin indexado
booleano (evaluar todo + Where en vez de NonZero), `&` → `abs(x) <= b` (And no está en
WebGPU), ruido como entrada y `Tile` para que valga cualquier longitud. Export con pesos
aleatorios: mide el grafo, no el audio. `uv run python -m ttspro.export.tts`.

| Configuración | Parámetros exportados | fp16 | Cabe (≤ ~37 MB) |
|---|---|---|---|
| VITS base (enc 6, WN 4, dec 512) | 30,86 M | 61 MB | no |
| dec 256 | 20,22 M | 39,7 MB | no |
| dec 256 + WN 3 en el flow | 18,05 M | ~36 MB | justo |
| **dec 256 + enc 4 capas + WN 3** | **15,97 M** (enc_p 4,25 · flow 6,52 · dec 3,83 · dp 1,37) | **31,0 MB** | sí, con 5 MB de margen |
| dec 384 | 24,69 M | ~49 MB | no |

Para la configuración elegida: `tts.onnx` 61,0 MB fp32 / 31,0 MB fp16, **0 ops fuera de
WebGPU**, paridad torch ↔ ORT en la onda 0,0012 (fp32 y fp16), 1,42 s de audio por frase de 60
tokens en 116 ms (RTF CPU **0,08** fp32; fp16 0,17 en CPU, que no es su sitio). El encoder
posterior (8,82 M) no se exporta.

**Consecuencia.** Descarga total prevista: 31 + 14,2 (encoder) + 18,5 (espeak) + ~10 (ORT) ≈
**74 MB**, dentro de los 80. Si el modelo pequeño no llega a la calidad del criterio, la palanca
barata es recortar espeak a es+en (devuelve 15 MB al sintetizador), no subir el presupuesto.
Un paso de entrenamiento completo (MAS en torch vectorizado, pérdidas de VITS, discriminadores)
corre en CPU con la configuración pequeña en la suite rápida (`tests/test_entrenar.py`).
**Lo que esto no dice:** nada sobre la calidad — el sintetizador no ha visto un dato. El siguiente
número que importa es pasos/segundo en la 1070 con batch 16 fp16.

## 2026-09-04 · Un paso de entrenamiento real en la GTX 1070 — DECIDE CÓMO SE ENTRENA

**Montaje.** `.scratch/bench_gpu.py`: configuración real (16 M), lote sintético de frases de 6 s
y 80 tokens, `paso()` de `ttspro.train.entrenar` completo (generador + discriminadores + MAS),
mediana de 10 pasos tras 3 de calentamiento, VRAM pico de `torch.cuda`.

| Batch | Precisión | s/paso | frases/s | VRAM pico |
|---|---|---|---|---|
| 16 | fp16 (autocast + GradScaler) | 2,32 | **6,9** | 3,35 GB |
| 32 | fp16 | 4,01 | 8,0 | 5,73 GB |
| 8 | fp16 | 1,62 | 4,9 | 2,46 GB |
| 8 | fp32 | 1,41 | 5,7 | 3,35 GB |
| 16 | fp32 | **46,1** | 0,3 | "14 GB": se sale de los 8 y Windows lo pagina a RAM |

La alineación monótona (`ttspro.model.alineacion`, torch en CPU, bucle por frame) cuesta
**0,36 s** de los 2,32 con (16, 80, 520): el 15 %. El resto es la GPU: Pascal no tiene tensor
cores y fp16 no acelera el cómputo, solo ahorra memoria.

**Consecuencia.** Se entrena con **fp16 y batch 16** (o 32, si el corpus tiene frases largas y
la VRAM lo permite); fp32 solo cabe hasta batch 8 y rinde parecido. A 0,43 pasos/s, 100 k pasos
son ~65 h y 300 k (lo habitual en VITS) ~8 días de máquina encendida. Es el techo decidido
(no se alquila GPU); la alternativa CPU sería un orden de magnitud peor y no se ha medido.
Si hace falta acelerar, la palanca es la alineación (0,36 s → numba o cython opcional, regla 11
lo permite por referencia), no la precisión.

---

## Mediciones pendientes que deciden algo

No son tareas: son las preguntas cuyo número cambia una decisión escrita. Cuando se midan, cada una
sube arriba como entrada con fecha.

- **Speaker encoder con voz real** (ADR 0002): la paridad del grafo compuesto se midió con ruido.
  Repetir con `eval/` y medir SECS entre locutores distintos y el mismo locutor, que es lo que
  el modelo va a usar. También la latencia en `wasm` real, no en ORT CPU.
- **Ops en WebGPU** (ADR 0005): la lista `ttspro.export.ops_ort_web` es de la documentación, no
  del runtime: cargar `speaker_encoder.fp16.onnx` en Chrome con el EP `webgpu` y ver en el
  perfilador de ORT qué nodos cayeron a CPU. Cero es el objetivo.
- **Calidad con datos reales** (SCOPE, criterio 4): el sintetizador no ha visto un dato. Primer
  corpus (data/README.md), primeras muestras en `runs/<x>/muestras/`, primer WER y SECS. Ese
  número recalibra los umbrales provisionales del criterio de terminado.
- **espeak-ng WASM en navegador real** (ADR 0004): los 124 ms por instanciación se midieron en
  node; en Chrome con el wasm cacheado puede ser distinto. Y si el troceado por frase basta para
  que no se note, o hace falta el build recortado a es+en.
- **Umbrales del criterio de terminado** (SCOPE): WER y SECS del primer modelo que hable, para
  recalibrar el 10 % y el 0,70 provisionales contra el baseline de YourTTS.
- **Descarga real** (regla 8): bytes de los dos `.onnx` + wasm de ORT + espeak-ng, medidos en el
  demo servido, no sumados a mano.
- **F5-TTS en WebGPU** (ADR 0001): RTF de una alternativa iterativa. Si baja de 1 con la misma
  calidad, la regla 1 se reabre.
