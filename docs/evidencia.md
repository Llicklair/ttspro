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

## 2026-09-04 · El protocolo "una línea por trozo" se rompe una vez cada ~200 trozos — NEGATIVO, ARREGLADO

**Montaje.** `preparar` sobre el manifiesto de OpenSLR es (24 437 frases) en lotes de 200 frases.

**Resultado.** Primer lote: 223 líneas para 224 trozos. Los 216 trozos de las primeras 200
frases, fonemizados **uno a uno**, dan todos exactamente una línea: el fallo no es de un trozo,
es de la interacción entre dos consecutivos en el mismo flujo (espeak fusiona o suprime una
cláusula según lo que haya alrededor). No se ha aislado el par exacto.

**Consecuencia.** `fonemizar_lotes` biseca el lote cuando el recuento no cuadra hasta dejar el
culpable solo y toma su salida real. El lado JS fonemiza frase a frase (pocos trozos por llamada) y
lanza error si no cuadra: ahí no se ha visto, pero la paridad no cubre este caso y hay que darle
el mismo trato cuando se vea. Lo que sí queda claro: el protocolo es una heurística sobre la
CLI, no una garantía; la garantía sería una API que devuelva los terminadores, que el WASM no tiene.

## 2026-09-04 · Destruí la descarga de VCTK al extraerla — NEGATIVO, ERROR PROPIO

**Qué pasó.** El enlace de datashare (`download/DS_10283_3443.zip`, 11,7 GB) es un zip
**envoltorio** con `README.txt` y `VCTK-Corpus-0.92.zip` dentro. Lo guardé con el nombre del
zip interior; al extraer, el miembro homónimo sobreescribió el archivo que se estaba leyendo:
`EOFError` en la primera pasada, fichero de 0 bytes en la segunda (`BadZipFile`). bsdtar listaba
0 entradas por lo mismo. Diagnóstico tardío: leí "zip64 raro" donde había un nombre repetido.

**Consecuencia.** Receta corregida (envoltorio → interior → corpus, nombres reales) y 11,7 GB
otra vez. Regla para la libreta: cuando un archivo "se corrompe" durante una extracción, mirar
primero qué se escribe y dónde, no el formato.

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

## 2026-09-04 · ¿Se puede partir de un modelo ya entrenado? — SÍ: EL VITS DE VCTK DE COQUI ENTRA TENSOR A TENSOR

**Montaje.** Pregunta de Marcos: descargar "la versión entrenada" en vez de entrenar. Revisado el
zoo de coqui (`.models.json`) y los checkpoints de Piper.

| Modelo | Licencia | Encaja con el contrato |
|---|---|---|
| YourTTS (coqui, en/fr/pt) | **CC BY-NC-ND 4.0** | no: sin derivados ni uso comercial, no vale ni para afinar; y sin español |
| XTTS-v2 (coqui, 17 idiomas) | CPML | no: autorregresivo (ADR 0001) y no comercial |
| Piper (checkpoints de entrenamiento) | MIT | repo de HF tras login; las voces ONNX públicas no clonan |
| `es/css10/vits` (coqui) | BSD-3 | a medias: grafemas, un locutor, decoder distinto |
| **`en/vctk/vits` (coqui)** | **Apache 2.0** | **sí**: misma red que mi configuración base, 858 tensores, 22 050 Hz, fonemas espeak con el mismo inventario IPA, condicionamiento de locutor de 256 canales |

`ttspro.train.inicializar` traduce nombres (bloques, weight_norm antigua → parametrizaciones,
acoplamientos k → 2k porque coqui hace el flip en línea, ConvFlow j → 2j−1 en el predictor,
`translation/log_scale` → `m/logs`, `cond_layer` → `cond`) y copia el embedding de símbolos
carácter a carácter: **857 de 858 cargados**, 172 símbolos copiados, 7 nuevos (`- ( )` y cuatro
diacríticos), solo mi embedding de idioma queda en su inicialización aleatoria y solo su tabla de
109 locutores sobra. Coqui entrenó con `add_blank`; el frontend lo replica ahora en los dos lados
(`simbolos.blank_entre_tokens`) y la paridad Python/JS sigue en verde.

**Resultado.** Sin entrenar ni un paso, con la fila original de `emb_g` de cada locutor como
vector de entrada, cuatro frases en inglés transcritas por Whisper-small:

| Frase | WER |
|---|---|
| Hello, how are you today? | 0,00 |
| The quick brown fox jumps over the lazy dog. | 0,11 ("Brian Fox") |
| Text to speech in the browser, with voice cloning from a single recording. | 0,31 (guiones y puntuación) |
| Please call Stella. Ask her to bring these things with her from the store. | 0,07 |

WER medio 0,12, casi todo puntuación. **Esto valida dos cosas a la vez:** que el port es exacto y
que mi implementación de VITS (atención relativa, flow, splines, HiFi-GAN, `Tile` del ruido) es
la de referencia, porque unos pesos ajenos hablan a través de ella.

**Bug encontrado por el camino (negativo útil).** Con los pesos entrenados, `tts.onnx` daba
186 frames y la referencia torch 169 para las mismas entradas. No era numérico: `torch.onnx.export`
pone el módulo en `eval` para trazar y al acabar **restaura el modo original del envoltorio**,
que nacía en `train`; la referencia calculada después llevaba el dropout 0,5 del predictor de
duración. Con pesos aleatorios no se veía porque la proyección del predictor arranca a cero.
Arreglado (`SintetizadorExport(...).eval()` y `modelo.eval()` antes de la referencia). Tras el
arreglo: paridad torch ↔ ORT en la onda 3,4·10⁻⁵ en fp32, y **el criterio 2 del terminado pasa**
(log-mel medio < 0,1 sobre las 24 frases del fixture con el checkpoint portado). En fp16 la onda
difiere hasta 0,17 en amplitud con la misma longitud; queda por medir en log-mel.

**Consecuencia.** El entrenamiento pasa de "desde cero" a "afinar": el inglés viene hecho y solo
hay que aprender el condicionamiento con el embedding de WeSpeaker (otro espacio que el `emb_g`
de coqui) y el español. **Coste:** es la configuración base, 61 MB fp16, no los 31 del modelo
reducido: descarga total ≈ 104 MB frente al presupuesto de 80 (ADR 0005). La decisión es de
Marcos y está abierta; las salidas posibles: subir el presupuesto con ADR, recortar espeak a es+en
(−15 MB), o destilar al modelo pequeño después con el grande como maestro.

## 2026-09-04 · El demo en el navegador, medido con Playwright — CRITERIO 5 EN ROJO POR NÚMEROS, NO POR CÓDIGO

**Montaje.** `web/` con Vite (COOP/COEP), ONNX Runtime Web 1.29, espeak-ng WASM con `locateFile`,
página `index.html` (cargar modelos → voz de referencia por fichero o micrófono → texto →
reproducir). `web/e2e/demo.spec.ts` en Chromium headless (sin WebGPU: mide el suelo wasm),
4 hilos wasm, `crossOriginIsolated=true`, referencia de 3,8 s, frase de 10 palabras
(123 tokens con blanks). Modelo portado de coqui, sin afinar.

| | fp32 | fp16 |
|---|---|---|
| Carga encoder + tts | 1,1 + 3,5 s | 1,1 + 5,1 s |
| Frontend (espeak wasm, 1 frase) | 184 ms | 171 ms |
| Modelo (2,68 s de audio) | 2 494 ms · RTF 0,93 | 2 795 ms · RTF 1,04 |
| **Total síntesis** | **2 761 ms** | **3 035 ms** |
| Descarga: tts + encoder | 121,0 + 27,4 | 61,2 + 14,2 |
| ORT wasm (`jsep`, con WebGPU) | 27,8 | 27,8 |
| espeak-ng wasm | 18,5 | 18,5 |
| **Descarga total (bruta)** | 194,7 MB | **121,6 MB** |

Los wasm de ORT: `simd-threaded` 14,0 MB (sin WebGPU), `jspi` 16,0, `jsep` 27,8 (WebGPU),
`asyncify` 25,7. El bundle por defecto de `onnxruntime-web` carga `jsep` aunque se pida solo
wasm. Comprimido (gzip de Vite): espeak 9,3 MB, ORT jsep 6,6 MB; los `.onnx` fp16 apenas
comprimen → transferencia fp16 ≈ **91 MB**.

**Consecuencia.** El criterio 5 (< 3 s y ≤ 110 MB) está en rojo por 35 ms y por 11,6 MB brutos,
con un modelo que aún no se ha afinado y en la máquina de desarrollo. Tres palancas, por coste:
servir comprimido (91 MB, gratis en cualquier hosting estático; habría que decidir si el
presupuesto cuenta bytes transferidos), cargar `onnxruntime-web/wasm` (14 MB) cuando no hay
WebGPU (−14 MB solo para esos usuarios), y recortar espeak a es+en (−15 MB). Y para el tiempo:
WebGPU en un navegador real, que es lo que todavía no se ha medido. **Lo que este montaje no
mide:** calidad (el audio sale del modelo sin afinar, en inglés reconocible), WebGPU, ni un
móvil.

## 2026-09-04 · WebGPU de verdad, en la GTX 1070 — fp32 VA (RTF 0,46), fp16 NO

**Montaje.** Chromium headless con `--enable-unsafe-webgpu` sobre el adaptador `nvidia pascal`
(la 1070), página del demo, proveedor forzado a `webgpu`, log detallado de ORT
(`?ortlog=verbose`), referencia de 3,8 s, frase de 10 palabras (221 tokens con blanks).

| | fp32 | fp16 |
|---|---|---|
| Carga encoder / tts | 3,2 s / 11,6 s (compila shaders) | 10,5 s / 10,5 s |
| Embedding (3,8 s de audio) | 1,6 s, norma 1,000 | **NaN** |
| Síntesis 1ª (5,4 s de audio) | 6,6 s · RTF 1,22 (compila) | — |
| Síntesis 2ª | **2,5 s · RTF 0,46** | — |
| Nodos en CPU | 9 (encoder) · 553 (tts) | 9 · 578 |

Los 553 nodos en CPU del sintetizador son **andamiaje de formas** del trazado (Unsqueeze 131,
Concat 93, Gather 88, Reshape 48, ConstantOfShape 24, Cast 12, Range 2…) más 24 `Transpose` de
los embeddings relativos de la atención, tensores diminutos; todo el cómputo pesado (Conv 226,
MatMul, LayerNormalization 36, Pad 55…) está en el `JsExecutionProvider`. 37 `MemcpyFromHost`
son el precio de ese andamiaje. Marcos vio en su Chrome el mismo cuadro: NaN en el encoder fp16
y un `Reshape` roto en el tts fp16 ("dimension with value zero exceeds…"), ORT avisando de que
no puede plegar `Exp` en fp16 por falta de kernel CPU.

**Consecuencia.** En WebGPU sin `shader-f16` (Pascal) los grafos fp16 no sirven; el demo elige
la precisión en "auto": fp16 solo con WebGPU **y** `shader-f16`, fp32 en los demás casos (en wasm
fp16 es además más lento). El criterio 5 se mide en fp32 en wasm. Queda por comprobar en una GPU
con `shader-f16` (Turing+, Apple, RDNA) si fp16 es correcto ahí o si el fallo es de kernels de ORT;
si es lo segundo, la salida es una conversión mixta que deje el predictor de duración y las
máscaras en fp32. Y el tamaño: fp32 son 148 MB de modelos, muy por encima del presupuesto; el
peso ya no es un detalle, es el siguiente problema de arquitectura.

## 2026-09-05 · Qué significa de verdad un coseno de locutor — RECALIBRA EL CRITERIO 4

**Montaje.** `.scratch/calibrar_secs.py`: 3 000 pares por corpus con los embeddings que ya calcula
`preparar` (el mismo encoder que usa el navegador). Dos preguntas: cuánto se parecen dos audios
REALES del mismo locutor (el techo que este encoder puede dar) y cuánto dos de locutores distintos
(el suelo, o sea "no clonar").

| | OpenSLR es (174 loc., 5,5 s de media) | VCTK (109 loc., 3,4 s de media) |
|---|---|---|
| Mismo locutor, media | **0,757** (p10 0,66 · p50 0,764 · p90 0,845) | **0,685** (p10 0,568 · p50 0,694 · p90 0,791) |
| Mismo locutor, audios ≥ 4 s | 0,767 | 0,750 |
| Distinto locutor, media | 0,164 (p90 0,326 · **p99 0,465**) | 0,110 (p90 0,262 · p99 0,404) |
| Umbral de igual error | 0,488 | 0,415 |

**Consecuencia.** El umbral de 0,70 que llevaba el criterio 4 era **el techo del encoder**, no una
meta alcanzable: pedía que la síntesis se pareciera a la referencia tanto como otra grabación real
de la misma persona. Recalibrado a **≥ 0,55 absoluto y ≥ 0,75 × el techo medido en la misma
tirada**; 0,55 queda por encima del percentil 99 de locutores distintos (0,465), así que un
resultado que lo pase no puede confundirse con "otra voz". La comparación con los 0,75–0,82 de
YourTTS era inválida: otro encoder, otra escala de coseno. El techo también depende de la duración
(VCTK sube de 0,685 a 0,750 con audios de ≥ 4 s), y por eso la evaluación lo mide en cada tirada
en vez de fijarlo.

## 2026-09-05 · El modelo portado con embeddings de WeSpeaker: la clonación ni siquiera llega al suelo — CONFIRMA EL DIAGNÓSTICO

**Montaje.** `ttspro.train.evaluar` sobre `runs/init_coqui_vctk/G_0.pt` (el port sin afinar),
un locutor de VCTK no visto, una frase, con el embedding de WeSpeaker de una referencia suya.

**Resultado.** Coseno de la síntesis **0,085**, con techo 0,680 y suelo 0,197 en ese mismo locutor.
WER 0,60. Es decir: la voz generada se parece **menos** a la referencia que un locutor tomado al
azar. Con la tabla `emb_g` propia de coqui el mismo modelo daba WER 0,12 y voz correcta.

**Consecuencia.** Cuantifica lo que Marcos oyó ("la voz no se clona"): no es un afinado insuficiente,
es que el port habla otro idioma de locutor. El afinado tiene que alinear los dos espacios, y esa
es exactamente la métrica que lo va a medir. Sirve además como línea base: cualquier número por
encima de 0,197 ya es señal de que el modelo empieza a leer el embedding.

## 2026-09-05 · Fase A: 2 000 pasos de afinado solo en español — HABLA ESPAÑOL, CLONA POCO, PIERDE INGLÉS

**Montaje.** Afinado desde el port de coqui (`runs/afinado_es`) sobre las 24 392 frases de
OpenSLR es (174 locutores), batch 16 fp16, lr 1e-4, 500 pasos de calentamiento solo del
discriminador y 1 500 con el generador suelto. 6 934 s de reloj (con una parte a 40 s/paso
porque la extracción de LibriTTS-R saturaba el disco: **no extraer corpus mientras se entrena**).
Evaluado con `ttspro.train.evaluar`: 5 locutores por idioma no vistos como condición, 2 frases
cada uno, embedding de una referencia suya.

| | Port sin afinar | **Afinado es, paso 2 000** | Suelo (otro locutor) | Techo (audio real del mismo) |
|---|---|---|---|---|
| WER es | no hablaba español | **0,272** (mediana 0,225) | — | — |
| SECS es | — | **0,335** | 0,248 | 0,693 |
| WER en | 0,60 | 0,503 | — | — |
| SECS en | 0,085 | 0,172 | 0,150 | 0,671 |

Pérdidas: mel de 25 a 15, KL de 5,1 a 1,9, discriminador clavado en 2,97 (el equilibrio del
LSGAN con seis discriminadores), gradiente del generador estable en 70–90.

**Resultado.** El español pasa de no existir a WER 27 %: se entiende, con errores. La clonación
sube de 0,085 (por debajo del suelo) a 0,335, apenas por encima del suelo de 0,248: el modelo
**empieza** a leer el embedding de WeSpeaker pero aún no lo usa como identidad. El inglés se
degrada, como cabía esperar al entrenar solo con español; se ve además en el demo, donde una
frase inglesa de 10 palabras genera 13,1 s de audio en vez de 2,7 (el predictor de duración se
ha desplazado hacia el español).

**Consecuencia.** Es la línea base de la fase B, que ataca justo esas dos cosas: los dos idiomas
**equilibrados** (`--balancear`, 24 392 frases por idioma para que el inglés no domine) y la
**pérdida de consistencia de locutor** (`--scl 9`). Medida su factura antes de encenderla:
+2,4 % de tiempo por paso (2,525 → 2,586 s) y +0,58 GB de VRAM (3,87 → 4,45), o sea nada.
Al arrancar, la SCL mide 0,62 de coseno entre el segmento generado y el real.

## 2026-09-05 · El demo hablaba el doble de lento: no era el ruido de JS, es el predictor de duración — AJUSTE, NO ARREGLO

**Montaje.** El demo generaba 12,67 s para una frase española de ~4,5 s (155 tokens) y 13,1 s
para una inglesa de 10 palabras, mientras el mismo checkpoint en PyTorch daba 6,1 s. Cadena de
sospechosos, descartados uno a uno con medidas:

| Sospechoso | Comprobación | Veredicto |
|---|---|---|
| El navegador sirve pesos viejos | sha256 de `models/tts.onnx` y `web/public/models/tts.onnx` | idénticos (aunque **sí** había un bug: `preparar-estaticos` comparaba solo el tamaño y un reexport pesa igual; arreglado) |
| El locutor cambia la velocidad | seis presets españoles × tres semillas en PyTorch | 3,17–3,38 frames/token, no explica un 2× |
| Mi ruido de JS no es N(0,1) | 20 000 muestras: media, desviación, autocorrelación 1–4, colas | indistinguible de `torch.randn` |
| **El sorteo del ruido** | el mismo ONNX con el ruido exacto de JS y con el de torch | **12,67 s vs 5,60 s** |

Es decir: el ruido es correcto y el grafo también; lo que pasa es que el **predictor estocástico de
duración amplifica cualquier sorteo**. Veinte semillas sobre la misma frase:

| `noise_w` | mín | mediana | máx | desviación |
|---|---|---|---|---|
| 0,8 (defecto de VITS) | 5,38 s | 6,77 s | **10,08 s** | 1,05 |
| 0,4 | 4,50 s | 5,22 s | 6,01 s | 0,33 |
| 0,0 (determinista) | 4,64 s | 4,64 s | 4,64 s | 0 |

**Consecuencia.** El demo baja su `noise_w` por defecto a 0,5 y el deslizador sigue ahí. Es un
**ajuste, no un arreglo**: con 2 000 pasos el SDP todavía no ha aprendido a poner la varianza
donde toca, y lo que lo corrige es entrenar. Con las mismas frases y locutores, bajar a 0,4 mejora
la **mediana** del WER español (0,225 → 0,167) y empeora la **media** (0,272 → 0,325): con diez
frases eso no decide nada y no se va a fingir que sí. Lo que sí está medido es la dispersión de la
duración, que es lo que se oye como "va lento y raro". Arreglado además el arranque en frío del
generador de ruido de JS (con semilla pequeña su primer valor era de 4 sigma).

## 2026-09-05 · ¿Se puede puentear el espacio de locutor sin entrenar? — NEGATIVO, CIERRA EL ATAJO

**Montaje.** El modelo portado ya clona, pero en el espacio `emb_g` de coqui (109 filas). Para
108 locutores de VCTK tenemos las dos representaciones: la fila de coqui y la media de los
embeddings de WeSpeaker que calculó `preparar`. Si la relación fuera aproximadamente lineal, una
regresión ridge de 256×256 daría clonación **sin entrenar nada**. Validación cruzada de 5
pliegues, con los locutores de prueba fuera del ajuste.

| Predicción | Coseno fuera de muestra |
|---|---|
| Predecir siempre la media de `emb_g` (línea base tonta) | **0,134** |
| Ridge α = 0,01 | 0,005 |
| Ridge α = 1 | 0,026 |
| Ridge α = 100 | 0,066 |

**Resultado.** Ninguna regularización llega siquiera a la línea base. Con 108 puntos en 256
dimensiones el sistema está sin determinar, y sobre todo no hay señal lineal que extraer.

**Consecuencia.** No hay atajo: **enseñar al modelo a leer el embedding es entrenamiento**, y en
esta GPU eso son días. Eso obliga a elegir enfoque en vez de seguir empujando, que es de donde
sale la conversación del 5-sep con Marcos. Lo que sí existe, con licencia MIT y ya entrenado:
MeloTTS español (208 MB fp32) para la voz y el conversor de OpenVoice v2 (131 MB fp32) para el
timbre, ambos **no autorregresivos** y por tanto exportables al navegador. Es clonación como
postproceso en vez de clonación aprendida, y no cuesta un solo paso de entrenamiento.

## 2026-09-05 · El conversor de OpenVoice v2 sobre nuestros propios bloques — CLONA SIN ENTRENAR, DECIDE EL ADR 0007

**Montaje.** `myshell-ai/OpenVoiceV2` (MIT) reimplementado en `ttspro.model.conversor` con los
módulos que ya existían. Configuración: 22 050 Hz, n_fft 1024, hop 256, `gin_channels` 256,
`zero_g` verdadero (toda la identidad vive en el flow). Locutores de OpenSLR es, medido con el
mismo encoder de WeSpeaker que usa el criterio 4.

**El port es exacto:** **486 de 486 tensores** cargados, ninguno sin destino, ninguno sin cargar.
La prueba independiente es la conversión identidad (origen = destino): devuelve el original con
coseno **0,783**, igual que el techo de dos grabaciones reales del mismo locutor (0,779).

| Audio real de A convertido a la voz de B | Coseno |
|---|---|
| Original frente a A (techo de partida) | 0,785 |
| **Convertido frente a B (destino)** | **0,424** con 3 referencias · **0,528** con 5 |
| Convertido frente a A (origen) | 0,257 (bajó desde 0,785) |
| Suelo (dos locutores distintos) | 0,137 |
| **WER del audio convertido** | **0,028** |

El parámetro `tau` casi no influye (0,499–0,528 entre 0,05 y 1,0). Lo que sí importa es cuánta
referencia hay, y **satura pronto**: 0,347 con un audio de 6,5 s, 0,388 con cinco (34 s), y de ahí
no sube con 82 s.

**La cadena completa del producto** (texto → nuestro TTS en voz base fija → conversor), seis
locutores destino:

| | Coseno con el destino |
|---|---|
| Solo TTS | 0,165 (suelo 0,120) |
| **TTS + conversor** | **0,384** |
| Techo (dos audios reales del destino) | 0,771 |

WER: 0,371 solo TTS y 0,443 con conversor de media, pero **0,318 → 0,261 en mediana**: el
conversor no se come las palabras, y la media la mueve un caso suelto.

**Consecuencia.** Clonación funcionando **sin un solo paso de entrenamiento**, y mejor que los
2 000 pasos de la fase A (0,335). El cuello de botella pasa a ser el TTS base, que es donde
existen modelos ya entrenados con licencia MIT (Piper español). Queda escrito en el
[ADR 0007](adr/0007-clonacion-como-postproceso.md). **Lo que esto no es:** una copia de la voz.
0,38–0,53 frente a un techo de 0,78 es un parecido reconocible, no una suplantación, y el criterio
4 (≥ 0,55 y ≥ 0,75 × techo) todavía no pasa.

## 2026-09-05 · La cadena de clonación entera, en el navegador — FUNCIONA; EL CUELLO ES EL TTS BASE Y EL PESO

**Montaje.** Los cuatro grafos exportados y servidos por Vite; Chromium headless, proveedor
`wasm`, fp32. La página carga `tts.onnx`, `voz.onnx` y `conversor.onnx`, elige voz (preset o
clonada de un fichero) y sintetiza. En paralelo, la misma cadena medida en Python sobre los
**mismos ficheros ONNX** con seis locutores destino no vistos.

**Clonación (ONNX, seis locutores):**

| | Coseno con el destino |
|---|---|
| Solo TTS en su voz base | 0,141 (suelo) |
| **TTS + conversor** | **0,432** |
| Techo (dos audios reales del destino) | 0,758 |

Mejor que el 0,384 medido en PyTorch, porque aquí la referencia son los cinco audios encadenados
en vez de la media de sus vectores. **El export no pierde nada.**

**En el navegador**, frase de 10 palabras, wasm:

| | |
|---|---|
| TTS | 3 825 ms |
| Conversor | 2 968 ms |
| Frontend (espeak wasm) | 168 ms |
| Audio generado | 4,14 s |
| RTF total | 1,64 |
| Descarga fp32 | **307 MB** (tts 121 · conversor 132 · voz 7,5 · ORT 27,8 · espeak 18,5) |
| Descarga fp16 | ~178 MB |

**Consecuencia.** El criterio 5 falla por los dos lados: 7 s en vez de < 3, y 178 MB en vez de
≤ 110. Y el WER de la cadena es 0,42–0,49 (mediana 0,37–0,44), que **no es culpa del conversor**:
es nuestro TTS de 2 000 pasos, que ya venía con 0,32. El conversor apenas lo mueve, y en mediana
lo mejora.

Las tres palancas, en orden de rendimiento por esfuerzo:

1. **Cambiar el TTS base por una voz española de Piper** (MIT, VITS con fonemas de espeak, nuestra
   misma familia): WER debería caer a ~0,05, y el `x_low` pesa 28 MB fp32 / ~14 fp16, o sea
   **−47 MB** respecto al nuestro. Arregla calidad y peso a la vez.
2. **WebGPU en vez de wasm** para el tiempo: en la 1070 el TTS solo iba a RTF 0,46 frente a 1,04
   en wasm; el conversor debería escalar igual.
3. **int8 en el conversor** si con lo anterior sigue sin caber.

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
- **Spike XTTS-v2 en el navegador** (pregunta de Marcos, 2026-09-05): exportar solo el GPT con caché
  KV a int4 (`MatMulNBits`) y medir tokens/segundo en la 1070 con WebGPU. Umbral para seguir:
  ~10 tokens/s y calidad int4 audible; por debajo, ADR que lo cierre. Se hace tras el primer
  modelo afinado, con la GPU libre.
- **XTTS-v2 como listón** (ADR 0006): WER y SECS de XTTS sobre `eval/` en español e inglés, para
  saber a qué distancia queda el modelo propio; y si el audio sintético castellano de XTTS mejora
  el afinado (medir con y sin).
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
