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

## 2026-09-05 · La voz base pasa a ser Piper español — WER DE 0,32 A 0,09 Y 29 MB MENOS

**Montaje.** `es_ES-davefx-medium` (MIT, 22 050 Hz, espeak `es`) portado desde su ONNX a nuestro
`Sintetizador` (`ttspro.train.piper`), exportado con nuestro exportador y medido con Whisper-small
sobre cinco frases españolas.

| | Nuestro modelo (2 000 pasos) | **Piper davefx portado** |
|---|---|---|
| WER medio / mediana | 0,42 / 0,32 | **0,17 / 0,091** |
| Parámetros exportados | 30,9 M | **16,4 M** |
| Tamaño fp32 / fp16 | 121 / 61,2 MB | **63,3 / 32,3 MB** |
| Paridad torch ↔ ORT | 1,2·10⁻³ | 3,5·10⁻⁵ |
| RTF en CPU | 0,22 | **0,045** |
| Ops fuera de WebGPU | 0 | **0** |

De las cinco frases, las dos con peor WER son casos donde **Whisper escribe cifras** donde el
texto tenía palabras ("6 entradas" por "seis entradas", "3522" por "tres mil quinientos
veintidós"): el audio es correcto. Sin esas dos, el WER es 0,057. La medición se deja como está
porque los dos modelos se midieron igual y la comparación se sostiene; normalizar números en
`normalizar_para_wer` queda pendiente.

**El port, tensor a tensor.** 349 cargados; lo que falta es solo entrenamiento (`enc_q`,
`dp.post_*`, y el `ConvFlow` que VITS descarta en inferencia). Dos sorpresas que las formas
delataron y las suposiciones habrían escondido:

- el decoder usa **ResBlock2** (dos convoluciones dilatadas por bloque, kernels 3/5/7 con
  dilataciones [1,2] [2,6] [3,12]) y **tres** upsamples de 256 canales, no cuatro de 512;
- `emb_rel_k` es (1, 9, 96) porque las cabezas comparten la tabla: deducir las cabezas del primer
  eje daba 1 donde hay 2, y con eso los doce tensores de atención no cargaban.

Piper además renombra en el export: el embedding de símbolos pasa a llamarse `sid` y las
convoluciones con weight-norm quedan como `onnx::Conv_NNNN`, recuperables porque el **nodo** que
las consume conserva la ruta del módulo.

**En el navegador** (wasm, fp32, frase de 13 palabras): TTS **401 ms** (antes 3 825), conversor
2 048 ms, total 2,7 s, RTF 0,86 (antes 1,64). Descarga fp32 249,4 MB (antes 307); en fp16
serían **148,8 MB**, aún por encima de los 110 del presupuesto, pero ahora quien pesa es el
**conversor** (66,5 MB), no el TTS.

**Consecuencia.** [ADR 0008](adr/0008-la-voz-base-es-una-voz-de-piper.md). El demo queda en
español, que es lo que la voz base sabe. La siguiente palanca de peso ya no es el TTS: es
cuantizar el conversor o recortar espeak.

## 2026-09-05 · La cadena completa con la voz de Piper — CLONA A 0,34 SIN COMERSE LAS PALABRAS

**Montaje.** Los tres grafos exportados (`tts.onnx` de Piper, `voz.onnx`, `conversor.onnx`) en
Python sobre los mismos ficheros que sirve el navegador. Seis locutores destino de OpenSLR es con
cinco audios de referencia cada uno.

| | Coseno con el destino |
|---|---|
| Solo la voz base de Piper | 0,079 |
| **Base + conversor** | **0,338** |
| Techo (dos audios reales del destino) | 0,758 |

**Sobre la inteligibilidad hay que separar dos medidas, porque dicen cosas distintas:**

| Frases | WER base | WER convertida |
|---|---|---|
| Del corpus (conversacionales, nombres propios, cifras) | 0,153 mediana | 0,417 mediana |
| Normales, tres frases limpias | 0,030 | **0,030–0,071 según `tau`, mediana 0,000** |

Con texto normal el conversor **no se come las palabras**: la mediana es cero y el único error de
la base es Whisper escribiendo "8 equipos" donde el texto decía "ocho". Con las frases del corpus,
que ya le costaban a la voz base, el conversor sí las empeora. Barrido de `tau`: 0,15 da el mejor
WER (0,030) y 0,6 la mejor similitud (0,385); 0,3, el defecto, queda en medio (0,063 y 0,380).

**Consecuencia.** El producto funciona de punta a punta: español claro, voz reconocible, en el
navegador y sin entrenar nada. Lo que queda medido y sin resolver es el **peso** (148,8 MB fp16
contra 110 de presupuesto, con el conversor como pieza dominante) y la **similitud**, que en 0,34
frente a un techo de 0,76 es un parecido, no una copia. El criterio 4 sigue sin pasar.

---

## 2026-09-05 · Reconstruir los modelos desde cero, sin corpus y sin entrenar

**Montaje.** Pregunta de Marcos: publicar el repo en GitHub «sin el entrenamiento, para que otro
usuario lo utilice». Eso solo es verdad si un desconocido puede producir los pesos, así que se
escribió `ttspro.export.modelos` y se corrió **borrando el camino previo**: descarga la voz de
Piper de Hugging Face, la porta, exporta el TTS, el conversor, el vector de voz y el speaker
encoder.

| Grafo | fp32 | fp16 | Igual que lo medido antes |
|---|---|---|---|
| `tts.onnx` | 63,3 MB | 32,3 MB | sí |
| `conversor.onnx` | 132,3 MB | 66,5 MB | sí |
| `voz.onnx` | 7,5 MB | 3,7 MB | sí |
| `speaker_encoder.onnx` | 27,4 MB | 14,2 MB | sí |

Repetido en un **entorno limpio** (`.venv-fresh`, solo `--extra dev --extra export`, 40 paquetes) por
`instalar.bat --sin-demo`: port 9,9 s, export del TTS 20,7 s, export del conversor 25,8 s. Menos de
un minuto de CPU una vez descargados los pesos. Antes de esa prueba el extra `export` no llevaba
torch ni huggingface_hub: el README prometía un camino que a un usuario nuevo le habría fallado en
el segundo comando. Ahora `export` es autosuficiente.

Un solo comando, sin GPU y sin un byte de corpus. Lo único que no se regenera así es
`models/voces.json` (los 20 presets se midieron sobre ~12 GB de VCTK y OpenSLR es), y por eso pasa
a viajar en git: 115 KB de vectores, ningún audio.

**El criterio 2 era un falso negativo.** `tests/terminado` fallaba con `zip() argument 2 is longer
than argument 1`: el test alimentaba el grafo con las siete entradas de siempre y desde el ADR 0008
la voz fija tiene cinco, porque el exportador poda `embedding` e `idioma`. Ahora construye el feed
**por nombre desde el contrato**, igual que hace el navegador, y pasa. No cambió el modelo: cambió
un test que llevaba dos ADR mintiendo.

| Criterio de terminado | Estado |
|---|---|
| 1 existe y carga, firma = contrato | pasa |
| 2 paridad PyTorch ↔ ORT (log-mel) | **pasa** (antes: error de firma) |
| 3 paridad frontend Python ↔ JS | pasa |
| 4 WER ≤ 0,10 y SECS ≥ 0,55 | **falla**: similitud 0,338 |
| 5 navegador < 3 s y ≤ 110 MB | **falla**: 3,46 s y 249,4 MB en fp32 wasm |

**`web/e2e/referencia.wav` no tenía procedencia.** Se buscó: 102 124 audios de VCTK, LibriTTS-R y
OpenSLR es filtrados por duración (963 candidatos a ±20 ms) y correlacionados por envolvente; el
mejor da 0,79, que no es una coincidencia sino un parecido. Origen desconocido, así que fuera:
ahora es `p225_003` de VCTK (CC BY 4.0), acreditado en THIRD_PARTY.md. El e2e sigue verde con él.

**Consecuencia.** El repo es publicable en cuanto se elija licencia
([ADR 0009](adr/0009-publicacion-en-github.md)): 2,7 MB en git, ningún peso versionado, ninguna
credencial, y tres comandos desde clonar hasta oír la voz. Los dos criterios rojos se publican
dichos, no escondidos.

---

## 2026-09-05 · Modo chat: dicción sobre mensajes de Twitch y caudal de la cola

**Montaje.** Petición de Marcos: «afinar la dicción» y «procesar muchos mensajes» para un chat de
Twitch ([ADR 0010](adr/0010-modo-chat-para-twitch.md)). Dos medidas: (a) WER con Whisper-small
sobre 25 frases (19 mensajes de `tests/fixtures/frontend/chat.txt` ya normalizados + 6 frases
normales de directo), 3 semillas cada una, `tts.onnx` de Piper en ORT CPU; (b) veinte mensajes
seguidos por la cola en Chromium headless, wasm, fp32, voz base.

**(a) Los parámetros de decodificación no son la palanca.** Barrido de `noise`, `noise_w`,
`length`:

| Ajuste | WER medio | mediana | frases perfectas |
|---|---|---|---|
| **defecto (0,667, 0,5, 1,0)** | **0,282** | 0,143 | 24/75 |
| sobrio (0,5, 0,3, 1,0) | 0,304 | 0,143 | 23/75 |
| seco (0,333, 0,2, 1,0) | 0,330 | 0,182 | 24/75 |
| sobrio rápido (0,5, 0,3, 0,9) | 0,295 | 0,143 | 24/75 |
| sobrio lento (0,5, 0,3, 1,1) | 0,315 | 0,143 | 24/75 |

Bajar el ruido no mejora nada y «seco» empeora. Los valores por defecto se quedan. El WER de 0,28
frente a 0,03 en frases limpias **no es el sintetizador: es el texto**. Frase por frase, lo que
falla es lo que el chat trae y ninguna decodificación arregla: nombres de usuario con cifras
(«Dark Lord99» → «Daclord 99»), alfanuméricos («C3PO R2D2»), inglés entero («Don't stop me now»),
risas («jajaja» → «caja ja ja» para Whisper, que es un juez, no un oyente), y **préstamos del
inglés** que espeak lee con reglas castellanas: «streamer» → «estréamer», «follow» → «follogo».

**Lo que sí es palanca: el texto.** Tras añadir la tabla de préstamos respelados (regla 6b) y las
pausas alrededor de «enlace», Whisper transcribe la palabra inglesa que se quería:

| Escrito | Antes (se leía tal cual) | Ahora (respelado) | Whisper oye |
|---|---|---|---|
| streamer | «estréamer», «estreamer» | estrímer | **streamer** 3/3 |
| spoiler | «esp Organa», «spoiler» 1/3 | espóiler | **spoiler** 3/3 |
| boss | — | bos | **boss** 3/3 |
| speedrun | — | espídran | **speedrun** 2/3 |
| hype | «IP», «it» | jaip | **hype** 2/3 |
| «mira esto enlace brutal» | «en la fatal» | «mira esto, enlace, brutal» | enlace con pausa |

Los nombres de usuario van por `nombre_legible`: «xXDark_Lord99Xx» → «Dark Lord»,
«SuperStreamerTV» → «Super Streamer TV», y «12345» se queda «12345» antes que dejar el mensaje sin
autor.

**(b) Caudal.** Veinte líneas de chat en ráfaga (150 ms entre ellas), una de ellas solo emotes y
otra repetida:

| | |
|---|---|
| Leídas | 18 |
| Saltadas por quedar vacías (solo emotes) | 1 |
| Rechazadas por repetidas | 1 |
| Descartadas por viejas o cola llena | 0 |
| Síntesis media por mensaje (frontend + tts) | 509 ms |
| **Velocidad: segundos de audio por segundo de síntesis** | **×3,3** |
| Tiempo total hasta oír las veinte | 31,8 s |
| Espera media llegada → sonar | 12,7 s |

Con la voz base en wasm el lector produce audio 3,3 veces más deprisa de lo que lo consume: **va al
día** con un chat de unos 30 mensajes por minuto de esta longitud, y la espera de 12,7 s es la de
una ráfaga artificial, no la de un chat real. Con conversor (~2,9 s por mensaje en wasm) la
velocidad cae por debajo de 1 y la cola empieza a descartar: para «una voz por usuario» hace falta
WebGPU o aceptar el retraso. Medición pendiente abajo.

**Falso positivo que salió de la cola.** El primer e2e no detectaba el mensaje repetido: `encolar`
arranca el bucle en el acto, que saca el primer mensaje de `pendientes` para renderizarlo, y el
duplicado llegaba 150 ms después con la cola «vacía». Ahora lo que se está renderizando o sonando
cuenta como en espera. No lo vio la intuición: lo vio el número 0 donde tenía que haber un 1.

---

## 2026-09-05 · Otra aplicación como fuente del chat (streex, Rails)

**Montaje.** Marcos: el flujo real es que una tercera aplicación, streex (Ruby on Rails), es la que
lee el chat. La página no tiene que hablar con Twitch: tiene que recibir líneas. Tres puertas en
`web/src/demo/fuentes.ts`, las tres entregan `{usuario, texto}` con nombres de campo tolerantes:
WebSocket con protocolo **ActionCable** (lo nativo de Rails), **SSE** y **postMessage** (página en
un iframe de la otra aplicación).

**ActionCable sin servidor Rails.** 3 tests con un WebSocket falso que reproduce lo que Rails
manda: `subscribe` solo tras `welcome`, «conectado» solo tras `confirm_subscription`, los `ping`
no producen líneas, el sobre `{identifier, message}` se desenvuelve, `reject_subscription` es un
error y no un silencio.

**SSE de verdad, en otro origen.** El e2e levanta un servidor HTTP en un puerto aleatorio (otro
origen, como estará Rails), que emite tres líneas por `EventSource`; la página, aislada con COEP,
las recibe porque el servidor manda `Access-Control-Allow-Origin`. Sin esa cabecera no llega nada:
es lo que hay que decirle a quien configure el lado Rails, junto con
`config.action_cable.allowed_request_origins`, que rechaza orígenes desconocidos por defecto.
Resultado: 2 líneas leídas, la de solo emote saltada, desconexión limpia.

**Fallo que salió de esa prueba.** `nombre_legible("streex_bot")` decía «stree bot»: la regla que
quita la decoración «xX…Xx» quitaba también una x final de verdad, y «Alex», «Max» o «Felix»
habrían perdido la suya. Ahora solo se van pares «xx»; 4 nombres más en la fixture (76 casos de
paridad). Lo encontró el nombre del bot de prueba, no una revisión.

---

## 2026-09-05 · La librería: `import { TTS }` desde cualquier HTML

**Montaje.** Marcos describe cómo lo va a usar: abre un HTML, importa el módulo, y desde la consola
del navegador `tts.predict(...)`, con el audio de referencia de un `<input type="file">` y un stream
de mensajes. Hecho como `web/src/lib/index.ts`, empaquetado con Vite a `dist/lib/ttspro.js`, y
medido con Playwright sobre **el bundle construido**, servido por `scripts/servir.mjs` (COOP/COEP,
sin Vite), no sobre las fuentes.

**Primer build: 99 MB.** El modo librería de Vite incrusta todos los assets en base64, y dentro
iban las dos variantes del wasm de ORT y el de espeak. Un build normal con entrada TS y
`assetsInlineLimit: 0` deja **ttspro.js en 500 KB** y emite al lado los wasm que hacen falta
(ORT 27,8 MB, espeak 18,5 MB), que se descargan solo cuando se usan.

| Desde la consola, wasm, fp32 | |
|---|---|
| `tts.clonar(fichero)` | vector de 256 desde un wav de 6 s |
| `tts.predict("hola, esta voz sale de un fichero")` con conversor | 1,54 s de audio en 1 913 ms |
| `tts.predict("y esta es la voz base")` sin conversor | 1,10 s de audio en **302 ms** |
| `tts.predict(stream, {chat: true})`, 4 elementos | 4 resultados, en orden, por la misma cola del demo |

**Lo que salió al medir.** «gracias por el stream, nice» no respelaba «stream» porque el token era
«stream,» con la coma pegada, y la tabla mira tokens enteros. Ahora la puntuación pegada se separa,
se busca la palabra y se vuelve a pegar («¿q?» → «¿que?», «(hype)» → «(jaip)»). Es el mismo tipo de
fallo que el de «streex_bot»: lo encuentra un caso real, no una revisión.

---

## 2026-09-06 · ¿Modelos de Hugging Face cargados desde la UI? Lo que cabe y lo que no

**Montaje.** Marcos: «la inferencia tarda demasiado, el modelo base va rápido pero el clonado es
impreciso y lento; ¿podemos incluir modelos de Hugging Face y cargarlos desde la UI?». Tres
comprobaciones antes de contestar.

**1. Hugging Face sí sirve ficheros a un navegador ajeno.** `resolve/main/...` responde con dos
redirecciones; con cabecera `Origin` cada salto devuelve `Access-Control-Allow-Origin` con ese
origen y el CDN final `*`. Sin `Origin` el primer salto dice `https://huggingface.co` y engaña.
Cargar un modelo por URL desde la página, incluso aislada con COEP, es posible.

**2. El port de Piper generaliza, tras arreglarlo.** `es_MX-claude-high` fallaba con
`KeyError: flow.flows.0.enc.in_layers.0.weight`: está exportado con un torch más viejo, los nodos
se llaman `Conv_32` y no `/flow/flows.0/.../Conv`, y la recuperación de pesos anónimos iba por el
nombre del nodo. Ahora va por el **bias hermano** del mismo nodo, que conserva el nombre del módulo
en cualquier export (131 de 132 convs lo tienen; el que no, `dec.conv_post`, trae el peso con
nombre). Comprobado que en davefx no cambia ni un tensor de los 349 cargados (los 154 que difieren
entre dos ports son los de entrenamiento, inicializados al azar).

| Voz de Piper (MIT) | tensores | parámetros | fp32 / fp16 | ms/frase CPU | max diff vs torch |
|---|---|---|---|---|---|
| es_ES-davefx-medium (la actual) | 349/349 | 16,4 M | 63,3 / 32,3 MB | 48 | 3,5e-5 |
| es_MX-claude-high | 349/349 | 16,4 M | 63,3 / 32,3 MB | 60 | 5,3e-6 |
| es_ES-carlfm-x_low | 349/349 | **5,2 M** | **20,7 / 11,0 MB** | **34** | 1,9e-6 |

Cero ops fuera de WebGPU en las tres. «high» resultó ser la misma arquitectura que «medium».

**3. Cada voz trae su tabla de símbolos, y no siempre es compatible.** El export de claude-high
sincronizó el contrato con 5 símbolos más en ids que davefx tenía vacíos: superconjunto, compatible.
carlfm-x_low tiene **130 símbolos** frente a 256 y tokenizar con el contrato actual revienta
(`idx=150 must be within [-130,129]`). Consecuencia de diseño: un paquete de voz es **onnx +
contrato propio**, y el frontend tokeniza con el de la voz que va a hablar.

**WER en 29 frases de chat normalizadas × 3 semillas (Whisper-small, decodificación por defecto):**

| Voz | WER medio | mediana | perfectas |
|---|---|---|---|
| davefx (actual) | 0,319 | 0,200 | 22/87 |
| claude-high (es_MX) | 0,303 | 0,143 | 21/87 |

Misma liga; el acento mexicano de claude es más claro en la mediana. carlfm no se pudo medir por lo
de la tabla.

**Consecuencia.** «Cargar modelos de Hugging Face» en genérico, no: cada modelo necesita un port a
nuestros bloques, un export acotado a WebGPU y un contrato; no existe un cargador universal. Lo que
sí cabe, y es justo lo que pide el uso: **paquetes de voz portados de Piper, servidos desde HF y
elegidos desde la UI**, nueve en español, todos MIT, 34–60 ms por frase. Para un chat con «una voz
por usuario» eso es voz clara y rápida sin conversor. El conversor sigue siendo la única clonación
de verdad y sigue costando ~2,9 s en wasm: la medición en WebGPU en la 1070 sigue pendiente y es la
que decide si la voz clonada es utilizable en directo.

---

## 2026-09-06 · Paquetes de voz bajo demanda y recomendación por hardware

**Montaje.** Marcos: adelante con las voces de Hugging Face desde la UI, y que detecte la gráfica y
recomiende. Hecho como [ADR 0011](adr/0011-paquetes-de-voz-y-hardware.md): `ttspro.export.paquete`
construye un paquete por voz (json con contrato propio y vector de voz base + onnx fp32/fp16),
la página y la librería los descargan bajo demanda, y `hardware.ts` detecta, recomienda y calibra.

**Ocho paquetes construidos con `--todas es`** (las de un solo locutor; sharvard tiene dos y queda
fuera). Cada uno con sus 349 tensores, cero ops fuera de WebGPU:

| Paquete | fp32 / fp16 | ms/frase CPU |
|---|---|---|
| es_ES-carlfm-x_low | 20,7 / 11,0 MB | 33 |
| es_ES-davefx-medium | 63,3 / 32,3 MB | 43 |
| es_MX-claude-high | 63,3 / 32,3 MB | 60 |
| es_ES-mls_9972-low | 63,2 / 32,2 MB | 62 |
| es_MX-ald-medium | 63,3 / 32,3 MB | 63 |
| es_MX-ald-x_low | 20,8 / 11,0 MB | 66 |
| es_ES-mls_10246-low | 63,2 / 32,2 MB | 129 |
| es_AR-daniela-high | 114,0 / 57,6 MB | 210 |

**Dos fallos que salieron construyéndolos.** (1) El contrato de un paquete copiaba las **formas**
del catálogo, no del grafo: carlfm x_low tiene 96 canales de flow y el contrato decía 192, y
`ruido_flow` reventaba al medir su voz base. El exportador escribe ahora las formas del grafo.
(2) `test_terminado` no lo habría visto: mide la voz local. Es el tipo de cosa que solo aparece
al construir la segunda voz.

**Los assets de release de GitHub no valen como host.** `curl -D - -H "Origin: …"` sobre
`releases/download/...`: el 302 y el 200 final de `release-assets.githubusercontent.com` no traen
`Access-Control-Allow-Origin`, así que un `fetch` desde otro origen falla. La release `voces-v1`
queda subida como archivo (25 ficheros) y los paquetes se sirven desde **GitHub Pages** (rama
huérfana `voces`, 572 MB, sin el fp32 de daniela que supera los 100 MB por fichero). Hugging Face
manda CORS correcto y es el destino natural cuando haya sesión.

**En el navegador, Chromium headless, wasm fp32, paquetes servidos en local desde otro origen:**

| | |
|---|---|
| Página: descarga carlfm y habla | 278 ms de TTS para 1,64 s de audio, RTF 0,17 |
| Página: chat «una voz base por usuario» sobre local + carlfm | 4 leídos, dos voces |
| Librería: `cargarVozBase("es_MX-claude-high")` + `predict` | 1,73 s de audio en 552 ms |
| Librería: `predict(…, {vozBase: "local"})` | 0,96 s en 299 ms |
| `recomendar()` en headless (sin WebGPU) | wasm fp32, 4 hilos |

**Calibración.** `calibrar()` abre una sesión por candidato, calienta, mide una frase y descarta
la que dé NaN. En headless solo hay un candidato (wasm). El número que decide, WebGPU fp32 en la
GTX 1070 de Marcos, lo da el botón de la página; pendiente abajo.

---

## 2026-09-06 · Los paquetes viven en Hugging Face

**Montaje.** Marcos hizo `hf auth login` y pidió que subiera el repo. `ttspro.export.publicar
--repo Llicklair/ttspro-voces` creó el repo público, escribió la model card y subió `paquetes/`
(indice, 8 json, 16 onnx). Comprobado desde fuera con `curl -L -D - -H "Origin: …"` sobre un
onnx real: cada salto trae `Access-Control-Allow-Origin` con el origen y el CDN final `*`
(11,0 MB de carlfm fp16). Es el mismo resultado que el sondeo del día 5 sobre `piper-voices`.

**Decisión que sigue:** Hugging Face pasa a ser el **origen por defecto** de `URL_VOCES`, y
GitHub Pages queda como espejo (`URL_VOCES_PAGES`, botón «usar GitHub Pages»). Los ficheros son
los mismos en los dos; con la URL cambia todo. La página tiene un campo «repo de Hugging Face» y
un botón que construye la URL `resolve/main/`: cualquier usuario puede publicar sus paquetes con
el mismo comando y apuntar la página a su repo.

**Prueba de navegador contra el host real** (`TTSPRO_VOCES=https://huggingface.co/Llicklair/ttspro-voces/resolve/main/`,
Chromium headless, wasm fp32): índice cargado, carlfm x_low descargada y abierta en **4,5 s**
(11,0 MB), y la frase «Esta voz ha venido de Hugging Face» sintetizada en 268 ms para 1,24 s de
audio. El e2e queda opcional (se salta sin la variable) porque depende de la red.

---

## 2026-09-06 · Mala pronunciación al clonar: no es `tau`, es la voz base

**Montaje.** Marcos: «las voces clonadas siguen sonando con mala pronunciación». WER con
Whisper-small sobre las 16 primeras frases de chat normalizadas × 3 semillas, `tts.onnx` → 
`conversor.onnx` en ORT CPU, dos destinos: el preset cof_02484 y la grabación p225 de VCTK.

**1. `tau` no manda.** Voz base davefx, WER sin conversor 0,353:

| tau | 0,05 | 0,1 | 0,15 | 0,2 | 0,3 | 0,45 |
|---|---|---|---|---|---|---|
| preset, WER convertida | 0,624 | 0,617 | 0,694 | 1,446 | 0,567 | 0,687 |
| grabación, WER convertida | 0,644 | 0,654 | 0,736 | 0,583 | 0,694 | 0,716 |

Todo en la misma banda: el conversor roba articulación y el parámetro no la devuelve. El 1,446 de
tau 0,2 es una frase que Whisper alucinó largo; no es una tendencia.

**2. La voz base sí manda.** Mismo destino (preset), tau 0,3:

| Voz base | WER base | WER convertida | Parecido con el destino (coseno, 142 refs reales) |
|---|---|---|---|
| davefx, length 1,0 | 0,353 | 0,603 | 0,117 → 0,455 |
| davefx, length 1,15 | 0,369 | 0,577 | — |
| davefx, noise 0,4, length 1,1 | 0,329 | 0,547 | — |
| **claude (es_MX), length 1,0** | 0,362 | **0,397** | 0,248 → **0,484** |

Hablar más despacio o con menos ruido apenas ayuda; cambiar la voz base debajo del conversor
recorta el daño de +0,25 a +0,035 y además acerca más al destino. Coincide con lo que Marcos oyó
antes de ver el número: «claude spanish mexico funciona muy bien».

**Consecuencia.** La voz base local por defecto pasa a `es_MX-claude-high` (enmienda al
[ADR 0011](adr/0011-paquetes-de-voz-y-hardware.md)); `export.modelos` la construye por defecto y
`voces.json` se mide sobre ella. Lo que no cambia: el conversor sigue siendo el mismo, y con
cualquier base la clonación es parecido, no copia.

---

## 2026-09-07 · Lectura de un chat de Twitch real, confirmada por Marcos

**Montaje.** Página levantada con `arrancar.bat`, panel 4 con fuente «Twitch (chat del canal)»,
canal `talk2play`, sin URL ni token (login anónimo `justinfan`). Marcos, con su Chrome y su GTX
1070: «funciona bien». Es la primera confirmación humana del lector de chat con un canal de
verdad; hasta hoy todo era Chromium headless con veinte mensajes simulados.

Lo que sigue sin número: los tiempos de calibración en esa GPU (webgpu fp32 frente a wasm) siguen
pendientes abajo; la página los mide con el botón «calibrar».

## 2026-09-15 · Supertonic 3 como motor: el español, la velocidad y el agujero — MOTOR SÍ, CLONACIÓN NO

**Montaje.** `supertone-oss-archive/supertonic-3` descargado con revisión fijada
(`aafc6e32`), cuatro grafos ONNX, `onnxruntime` 1.29 en CPU, portátil de Marcos.
Motor propio en `src/ttspro/supertonic/` (espejo en `web/src/runtime/supertonic.ts`).
WER con Whisper-small y el mismo `normalizar_para_wer` que `ttspro.train.evaluar`, sobre dos
conjuntos: las 12 frases españolas de `tests/fixtures/frontend/frases.txt` (existen para romper el
frontend) y 10 frases normales, que son las comparables con el 0,030 de la voz base de Piper.
Dos voces, F1 y M1, semilla fija. 44 síntesis.

**Resultado.**

| | Cadena de hoy (2026-09-05) | Supertonic 3 |
|---|---|---|
| WER español, frases normales | 0,030 | **0,008** medio · 0,000 mediana · 0,008 p90 |
| WER español, `frases.txt` | — | 0,178 medio · **0,000 mediana** · 0,675 p90 |
| Frecuencia de salida | 22 050 Hz | 44 100 Hz |
| RTF en CPU (8 pasos) | — | **0,463** (1 703 ms por frase, 3,73 s de audio de media) |
| Reparto del tiempo | conversor 2 861 de 3 460 ms | `vector_estimator` 2 122 de 2 676 ms |
| Grafos | 111,3 MB fp16 + 17,6 de espeak | 398,1 MB fp32, **sin fonemizador** |

Los dos fallos que cargan con casi todo el 0,178 del conjunto difícil tienen causas distintas y
solo una es del modelo:

- **Números leídos como dígitos**: «Tres mil quinientos veintidós euros» sale «3.522 euros». No es
  del modelo, es que le llegó texto sin normalizar. `ttspro.frontend.normalizar` ya arregla esto y
  se queda **delante** del frontend Unicode; la página ya lo hace así.
- **«Pingüino, cigüeña, vergüenza» → WER 1,000 en las dos voces.** Esto sí es del modelo. La
  diéresis está en la tabla (U+0308 → 152) y la NFKD la produce bien, comprobado; el checkpoint
  tropieza con el dígrafo «güe/güi» del español. Tres palabras raras seguidas es el peor caso
  posible y no aparece en frases normales, pero queda escrito.

**El voice builder no existe en el archivo.** Comprobados fichero a fichero los cuatro repos de
pesos (`Supertone/supertonic`, `-2`, `-3` y los espejos del archivo): los cuatro publican
exactamente `text_encoder`, `duration_predictor`, `vector_estimator` y `vocoder`. El `tts.json`
describe un `ae.encoder` y un `ttl.style_encoder` cuyos pesos **no se publicaron nunca**. Los
`voice_styles/*.json` llevan `"source_file": "F1.wav"`: se extrajeron con la herramienta interna, y
esa herramienta era un servicio alojado que cerró el **2026-08-31**.

**Consecuencia.** [ADR 0012](adr/0012-supertonic-como-motor.md): el motor entra, la regla 1 se
reformula, el presupuesto de descarga de la regla 8 se retira, espeak-ng se va con su GPL. El
voice builder hay que reconstruirlo, y lo de abajo es el primer intento.

## 2026-09-15 · Voice builder por mezcla de las diez voces — NO LLEGA, Y EL NÚMERO LO DICE

**Montaje.** `ttspro.supertonic.constructor`: búsqueda por entropía cruzada sobre mezclas de las
diez voces de fábrica, una mezcla distinta por cada una de las 50 fichas de estilo (500
parámetros). Dos etapas, 8 + 18 generaciones, población 16, 4 pasos de flow durante la búsqueda,
dos frases sonda. Referencia: `web/e2e/referencia.wav` (6 s). Objetivo: coseno de WeSpeaker con
`models/speaker_encoder.onnx`, el mismo que usa `tests/terminado`. 904 evaluaciones, 873 s.

**Resultado.**

| | Coseno |
|---|---|
| Mejor voz de fábrica suelta (F1) | 0,100 |
| Mejor de la búsqueda (sobre las frases sonda) | 0,283 |
| **Verificado en tres frases que la búsqueda no vio** | **0,227** |
| Cadena vieja (conversor de OpenVoice), para comparar | 0,338 |
| Objetivo del criterio de terminado | 0,55 |

La etapa 2 dejó de subir a la sexta generación y rebotó entre 0,22 y 0,28 durante doce más: no es
que falten iteraciones, es que **el espacio se agotó**. Diez voces generan un subespacio afín de
nueve dimensiones dentro de uno de 12 928; la envolvente convexa de diez puntos ahí dentro es casi
nada. Y la caída de 0,283 a 0,227 al cambiar de frase dice que parte de lo que subió era ajuste a
las dos sondas, no timbre.

**Consecuencia.** La mezcla queda como lo que es: una forma rápida de sacar voces nuevas y
distintas entre sí, no una clonación. **Peor que lo que el proyecto ya tenía.** Se busca por otro
lado: reconstruir las dos piezas que faltan en vez de esquivarlas — invertir el `vocoder` por
gradiente para obtener el latente de una grabación real (el `ae.encoder` que falta) y ajustar
`latente → estilo` sobre pares que el propio modelo regala al muestrear (el `style_encoder` que
falta). Medición en curso.

## 2026-09-15 · Invertir el vocoder para recuperar el `ae.encoder` — FUNCIONA, Y CON MARGEN

**Montaje.** Lo que falta para clonar son dos piezas que Supertone no publicó: `ae.encoder`
(audio → latente) y `style_encoder` (latente → estilo). La primera no hace falta entrenarla,
porque el **decoder sí está publicado** y es diferenciable: `vocoder.onnx` va de latente
`[1,144,T]` a onda de 44,1 kHz, así que se puede buscar por descenso de gradiente el latente cuya
onda **es** la grabación. Grafo pasado a torch con `onnx2torch` (reutilizando conversores de Cast,
Constant, Pad, Reshape y Shape, que no cambiaron de semántica entre el opset 13-15 y el 19),
pérdida multi-resolución STFT en magnitud lineal y logarítmica, Adam con coseno, 1 500 pasos.
Referencia: `web/e2e/referencia.wav` (VCTK p225, 6 s). GTX 1070.

**Resultado.**

| | |
|---|---|
| Paridad torch ↔ onnxruntime sobre latente aleatorio | max \|dif\| **1,25e-6** — es el mismo grafo |
| Latente a buscar | `[1, 144, 86]` = **12 384 números** para 5,99 s |
| Pérdida | 11,38 → **5,54** (374 s) |
| **Coseno de locutor original ↔ reconstruida** | **0,832** |
| RMS original / reconstruida | 0,1287 / 0,1229 |

Ese 0,832 es el número que importa, y conviene ponerlo al lado de los otros: mezclar las diez
voces daba **0,227**, el conversor de OpenVoice daba **0,338**, el criterio de terminado pide
**0,55**. La identidad del hablante **sobrevive** el viaje de ida y vuelta por el latente del
modelo, así que el latente es un sitio legítimo desde el que construir una voz — cosa que el
embedding de WeSpeaker, con sus 256 números entrenados para ser invariantes al contenido, no puede
ser por sí solo.

**Consecuencia.** `ttspro.supertonic.inversor`. Y el voice builder cambia de entrada: en vez de
ajustar `embedding → estilo`, se ajusta `resumen del latente → estilo`, sobre pares que el modelo
regala al muestrear (el latente ya se calcula, es lo que entra al vocoder). El muestreo guarda las
tres cosas —estilo, embedding y resumen del latente— para poder comparar las tres vías con el mismo
dataset en vez de discutirlas. `motor.lote(..., con_latente=True)` es lo que lo permite.

Lo que **no** demuestra este número: que el `style_encoder` reconstruido conserve esos 0,832. El
estilo es un resumen de 50×256 que tiene que valer para cualquier frase, no para esta; cuánto cae
ahí es la medición siguiente.

## 2026-09-16 · El puente `audio → estilo` por regresión — NO HAY NADA QUE APRENDER

**Montaje.** 4 000 tríos `(estilo, embedding de WeSpeaker, resumen del latente)` generados por el
propio modelo en 49 min de CPU (252 descartados por audio inservible). Estilos muestreados de tres
fuentes: mezclas Dirichlet de las diez voces, mezclas con ruido por ficha, y gaussianas estiradas
alrededor de la media. Ridge sobre las componentes principales del espacio de estilos, tres
entradas posibles y tres tamaños, ajustadas y validadas sobre el mismo dataset con un corte 90/10.

**Resultado.** El error relativo es `‖predicho − real‖ / ‖real − media‖`: **1,0 significa "predecir
la media"**, y por encima de 1,0 es peor que la media.

| entrada | dims | C | var. explicada | err. entrenamiento | err. validación |
|---|---|---|---|---|---|
| embedding | 256 | 64 | 0,811 | 1,014 | **1,045** |
| latente | 432 | 64 | 0,811 | 1,015 | **1,037** |
| ambos | 688 | 64 | 0,811 | 1,060 | **1,126** |

Ninguna combinación de entrada y tamaño baja de 1,0. Y de extremo a extremo, que es lo que
importa, sobre tres frases que el ajuste no vio:

| Voz | Coseno con la referencia |
|---|---|
| La que devuelve el puente | **−0,0399** |
| La voz **media**, sin puente ninguno | **−0,0409** |
| La mejor voz de fábrica suelta (F1) | 0,1063 |

El puente devuelve la voz media. Los dos números coinciden en la tercera cifra: no pasa ni un bit.

**Por qué, que es lo que hay que recordar.** El muestreo recorre un espacio de 12 928 dimensiones
donde la mayoría de las direcciones **no codifican una voz**: el mapa estilo → audio es muy
perdedor, muchos estilos distintos suenan igual, y por tanto el inverso no es una función. Pedirle
a una regresión que recupere 12 928 números desde 256 o 432 es pedirle que adivine ruido, y lo
correcto ante ruido es devolver la media — que es exactamente lo que hace. El fallo no es del
ajuste: es de la pregunta.

**Consecuencia.** Se abandona la regresión `audio → estilo`, en cualquiera de sus entradas. Lo que
queda, y ahora está claro que es el camino, es **optimizar el estilo por gradiente contra el modelo
congelado**, con los latentes de la inversión como objetivo. La diferencia con la búsqueda por
mezclas que ya falló (0,227) no es el optimizador: es el espacio. Las mezclas de diez voces viven
en un subespacio afín de nueve dimensiones; el gradiente se mueve por las 12 928. El código del
puente se conserva porque el muestreo y la inversión valen para eso, y porque un resultado negativo
sin su código no se puede volver a comprobar.

## 2026-09-16 · El estilo por gradiente: el optimizador va, la pérdida no — LA PREGUNTA OTRA VEZ

**Montaje.** Tercer intento del voice builder, y el primero que no está confinado a un subespacio:
los cuatro grafos pasados a torch con `onnx2torch`, congelados, y **el estilo como parámetro**
(12 928 números) movido por Adam a través del modelo entero. Lo que en difusión se llama inversión
textual. El objetivo son los latentes de la grabación real, que da `inversor.py`. Pérdida: MSE
entre los **estadísticos por canal** del latente generado (media, desviación y energía a lo largo
del tiempo, 432 números) y los de la referencia, más un anclaje al estilo de partida. Cuatro frases
de ajuste rotando, tres de verificación que el ajuste no ve nunca. 300 iteraciones, GTX 1070.

**El coseno de locutor NO entró en la pérdida**, a propósito: optimizar la métrica que luego se
publica es la forma más rápida de mentirse.

**Resultado.** Viabilidad primero, que también es un dato:

| | |
|---|---|
| Los cuatro grafos a torch | paridad con ORT: `text_encoder` 1,25e-6; duración 2,9241846 vs 2,9241836 |
| Gradiente que llega a `style_ttl` | media 1,8e-3, máx 3,3e-2 |
| Memoria, 2 pasos de flow + vocoder | **0,70 GB** de 8 |
| Coste | 0,53 s por iteración; 159 s las 300 |

Y el resultado, que es el que duele:

| | |
|---|---|
| Pérdida de ajuste | 0,0337 → **0,0025** (13 veces menos) |
| Coseno de partida (F1, la mejor de fábrica) | 0,0729 |
| **Coseno tras el ajuste** | **0,0719** |

El optimizador hace su trabajo impecablemente y **la voz no se mueve ni una milésima**. Encontró un
estilo que reproduce esos 432 números sin parecerse a la persona, que es exactamente lo que se le
pidió.

**Por qué.** El resumen es demasiado pobre. Media, desviación y energía **por canal** describen
cuánta energía tiene cada canal, no cuáles se encienden juntos, y muchas voces distintas comparten
esos 432 números. No es un fallo del gradiente ni del anclaje: es que la pérdida no discrimina
identidad.

**Consecuencia.** Se cambia el resumen, no el método: a la **matriz de correlaciones entre canales**
(Gram, 20 880 números), que es donde la transferencia de estilo lleva desde 2015 diciendo que vive
la textura. `inversor.resumen(..., forma="gram")` y `ajustador --resumen gram`. La forma
`estadisticos` se conserva, porque un negativo sin su código no se puede volver a comprobar.

**Con el Gram, mismo montaje, 400 iteraciones (211 s):**

| | estadísticos (432) | **Gram (20 880)** |
|---|---|---|
| Pérdida de ajuste | 0,0337 → 0,0025 | 0,0025 → 0,0010 |
| Coseno de partida | 0,0729 | 0,0729 |
| **Coseno tras el ajuste** | 0,0719 | **0,1476** |

Se mueve, y al doble. Es la primera pérdida que toca la identidad **sin meter dentro el medidor que
luego juzga**. Sigue muy lejos del 0,55 y por debajo del 0,227 de las mezclas, pero el diagnóstico
del resumen pobre era correcto: la información de locutor está en las correlaciones.

El ajuste se estanca en la iteración 80 (0,00157 → 0,00104 en las 320 restantes) mientras el
anclaje baja de 0,0117 a 0,0047, así que la hipótesis obvia era que el ancla frenaba. **Se probó y
es falsa:** con el ancla diez veces más floja (0,005), 900 iteraciones y `lr` 0,03 (476 s), la
pérdida baja más —0,00104 → 0,00090— y **el coseno empeora: 0,1476 → 0,1100**.

Pérdida menor, voz peor. Eso no es un detalle de ajuste: es la prueba de que **el Gram del latente
solo correlaciona débilmente con la identidad**, y que apretar el optimizador contra él lleva a
estilos que cuadran las correlaciones sin parecerse a nadie. El ancla no estorbaba, protegía.

**Estado del voice builder, cuatro intentos medidos:**

| Vía | Coseno | Qué descarta |
|---|---|---|
| Mezclar las diez voces | **0,227** | el espacio de mezclas es de 9 dimensiones |
| Regresión `audio → estilo` | −0,040 | el inverso no es una función; devuelve la media |
| Gradiente + estadísticos por canal | 0,072 | el resumen no discrimina identidad |
| Gradiente + Gram | **0,148** (0,110 si se aprieta) | el Gram correlaciona, pero poco |
| *Para comparar:* conversor de OpenVoice (cadena vieja) | 0,338 | |
| *Para comparar:* inversión del vocoder, ida y vuelta | 0,832 | **la información SÍ está ahí** |
| Objetivo del criterio 4 | 0,55 | |

Lo que sobrevive a los cuatro: el gradiente llega, es barato (0,53 s por iteración, 0,7 GB), y la
identidad está en el latente. Lo que falta es **una pérdida que la vea**.

**La tuerca que queda, y por qué no se ha girado sola.** Meter el coseno de WeSpeaker en la pérdida
casi con seguridad funciona — es optimizar directamente el objetivo. Pero el criterio 4 de
[SCOPE.md](../SCOPE.md) **es** ese coseno, así que hacerlo convierte el criterio en una profecía
autocumplida. La salida honesta es ajustar con WeSpeaker y **juzgar con otro encoder distinto**
(ECAPA, por ejemplo), que es una dependencia nueva y una decisión de método, no de código. Eso lo
firma Marcos, no un agente.

Tres intentos, tres negativos, y cada uno descarta una hipótesis distinta: el espacio era
demasiado pequeño (mezclas), el inverso no es una función (regresión), el resumen no discrimina
(estadísticos). Lo que sobrevive a los tres es que el gradiente **sí** llega y **sí** es barato.

## 2026-09-16 · «Las voces no funcionan y clonar tampoco» — UNA ERA CIERTA

**Montaje.** Marcos lo dice de la página; se reproduce con Playwright registrando errores de JS,
peticiones fallidas y el estado, y se mide aparte si las diez voces son de verdad distintas.

**Las voces sí funcionan, y son distintas.** Coseno de locutor entre las diez, misma frase, misma
semilla, 8 pasos:

| | media | mínimo | máximo |
|---|---|---|---|
| Todos los pares distintos | **0,288** | 0,066 | 0,566 |

Y con estructura: el bloque femenino entre sí ronda 0,4, el masculino 0,43, y cruzados 0,18. El p99
de locutores distintos con este encoder es 0,465, así que **son diez voces, no una repetida**.

En el navegador, la misma frase con F1 y con M1 da WAV distintos byte a byte — 248 054 bytes con
hash 2185818538 contra 247 896 y 2943382072. El estilo llega al grafo también ahí, que es lo que
una etiqueta que cambia en la interfaz no demuestra.

**Clonar no funcionaba, y por un bug mío.** `puenteDisponible()` hacía `HEAD` y miraba `r.ok`, pero
**el servidor de desarrollo de Vite responde 200 con el `index.html` a cualquier ruta que no
conoce**. Así que la página creía que había puente, pedía el JSON, y `r.json()` moría con
`Unexpected token '<', "<!doctype "...`. El error subía sin capturar y elegir un fichero no hacía
nada visible.

**Consecuencia.** Tres cosas:

1. `puenteDisponible` abre la cabecera y comprueba que dice lo que tiene que decir, en vez de
   fiarse del código de estado. Lo mismo vale para cualquier otro `fetch` de JSON contra el dev
   server, y por eso queda escrito aquí.
2. `clonar()` captura y lo cuenta en la pastilla y en el registro. Un fallo silencioso es peor que
   uno feo.
3. **El camino «desde una grabación» sale apagado y dice por qué**, porque las cuatro medidas de
   arriba dicen que no llega. En su lugar la página gana **importar una voz `.json`**, que es lo
   que sí funciona hoy: construyes una fuera con `ttspro.supertonic.constructor` y la sueltas ahí.
   Ofrecer un botón que devuelve la voz media no es una funcionalidad, es una trampa.

Y un cuarto, del mismo bug: sin `models/supertonic` copiado, «cargar el motor» fallaba con
`Unexpected token '<'` en vez de decir qué hacer. Ahora nombra las dos salidas — el botón de
Hugging Face, o `descargar` más `npm run preparar`.

**Un quinto, que salió al escribir la prueba.** La primera versión del test comparaba los WAV de F1
y de M1 sin fijar la semilla, y los hashes cambiaban entre tiradas: con ruido distinto, dos audios
distintos no demuestran nada sobre la voz. Al fijarla apareció el bug de verdad —
`Number(campo) || undefined` convertía **la semilla 0 en aleatoria**, que es lo contrario de lo que
dice el propio campo y de la regla 5. Vacío significa aleatorio; 0 es una semilla. El test ahora
comprueba las dos mitades: con semilla fija, **la misma voz repite la toma exacta** y otra voz no,
y lo único que cambió entre ambas es el estilo.

## 2026-09-16 · ¿Hay packs de voces que descargar? — NO EXISTE NINGUNO, Y LOS DE v1/v2 NO SIRVEN

**Montaje.** Marcos pregunta de dónde bajar más voces. Se busca en Hugging Face (`search=supertonic`,
50 repos, y `search=voice_styles`) y se inspecciona el árbol de los candidatos con más tráfico. Y
como los estilos de Supertonic 1 y 2 declaran **las mismas dimensiones** que los de la 3 —
`style_ttl [1,50,256]`, `style_dp [1,8,16]` — se prueba si cargan y si suenan.

**No existe ningún pack de terceros.** Buscar `voice_styles` en Hugging Face devuelve **cero
resultados**. De los 50 repos que mencionan supertonic, los que llevan voces llevan **las mismas
diez** (`ahk-d`, `nik2999`, `IsGarrido`, `Sky-Kim`, `thy025`, `Reza2kn`… todos F1–F5 y M1–M5); el
resto son conversiones del motor a CoreML, MLX, TensorRT, RKNN o Qualcomm, sin voces. Es coherente
con lo demás: la única herramienta que fabricaba una voz era el Voice Builder de Supertone, era un
servicio alojado y cerró el 2026-08-31.

**Y los estilos de v1 y v2 no valen en v3, aunque encajen.** Misma frase, semilla 7, 8 pasos:

| Estilo | Duración | rms | WER | Transcripción |
|---|---|---|---|---|
| v3 F1 (referencia) | 5,03 s | — | — | — |
| v1 F1 | 2,51 s | 0,209 | **1,214** | «oh la distancia es pacific diagonal solución stones…» |
| v1 M1 | 2,46 s | 0,155 | 0,929 | «SÍ知pe» |
| v2 F1 | 2,83 s | 0,104 | **1,000** | «surpassé el ataque, son5712вор» |
| v2 M1 | 2,74 s | 0,038 | 0,786 | «¡Hora que esta es una frase, esta es una suya…» |

Cargan sin quejarse y producen ruido: duraciones a la mitad, el rms cayendo hasta 0,038, y el
reconocedor oyendo chino y francés inventados. **El espacio de estilos no se comparte entre
checkpoints**, aunque las formas coincidan — que es exactamente lo que ya se dijo al descartar los
encoders de locutor ajenos, ahora medido en el caso más favorable posible: el mismo modelo, dos
versiones seguidas, las mismas dimensiones.

**Consecuencia.** No hay nada que descargar y no hay nada que portar, así que las voces hay que
**acuñarlas**: `ttspro.supertonic.pack`. Y ahí conviene separar dos cosas que se venían mezclando:

    parecerse a una persona concreta      0,227 sobre 0,55 — medido, NO llega
    inventar voces distintas entre sí     otro problema, y este sí se resuelve

Para «una voz por usuario» en un chat no hace falta lo primero. `pack` muestrea mezclas con ruido
por ficha de estilo, tira las que no hablan y elige las N **más separadas entre sí** por muestreo
del punto más lejano, que maximiza la distancia mínima en vez de la media. El listón es el del
propio proyecto: el p99 del coseno entre locutores distintos con este encoder es 0,465, y el
comando dice cuántos pares se quedan por encima.

## 2026-09-16 · Por que fallaban los cuatro intentos: el latente lo domina el ruido — CAUSA ENCONTRADA

**Montaje.** Marcos: *«hay que arreglar lo de clonar desde una grabación cuando podamos y que dé un
buen resultado»*. Antes de seguir probando ideas, dos comprobaciones que nunca se habían hecho.

**1. El término de locutor, dentro de la pérdida.** El encoder de WeSpeaker pasa a torch con
paridad **2,0e-7** y coseno **1,000000** contra onnxruntime, y el gradiente lo atraviesa hasta la
onda: la cadena `estilo → latente → onda → embedding` es derivable entera. Se optimiza el coseno
directamente, 400 iteraciones.

| | Coseno |
|---|---|
| Partida (F1) | 0,0729 |
| Durante el ajuste, en las frases de ajuste | 0,148 |
| **Verificado en frases que no vio** | **0,0409** |

Sube donde se le empuja y **baja por debajo del punto de partida** en cualquier otra frase.

**2. La prueba que separa «método roto» de «objetivo inalcanzable».** Todo se venía midiendo contra
VCTK p225, una locutora inglesa a la que la mejor voz de fábrica solo saca 0,073 — si el modelo no
puede expresar esa voz, ningún optimizador la alcanza y los negativos no dicen nada del método. Así
que se cambia el objetivo por **una de sus propias voces**: se sintetiza con M3, se trata ese audio
como una grabación cualquiera y se intenta reconstruir su estilo partiendo de F1. El estilo
verdadero existe y es alcanzable **por construcción**.

| | Coseno |
|---|---|
| Techo (la propia M3 contra sí misma) | **0,9144** |
| Partida (F1) | 0,2306 |
| Inversión del vocoder sobre el audio de M3 | 0,9242 |
| Tras ajustar, sin término de locutor | **−0,0908** |
| Tras ajustar, con término de locutor | **−0,0461** |

El ajuste aleja la voz hasta valores **negativos** mientras la pérdida baja, con la respuesta
correcta al alcance. **El método está roto, y p225 no era la excusa.**

**3. La causa.** Se compara el forward en torch del ajustador con el del motor, dándoles el mismo
estilo de M3:

| | |
|---|---|
| Coseno de locutor ONNX vs torch, mismo estilo | 0,8622 — **el forward es correcto** |
| Duración / rms | 3,69 s / 0,0778 contra 3,76 s / 0,0751 |
| **Correlación entre los dos latentes** | **0,3043** |

Dos latentes del **mismo estilo diciendo la misma frase** se parecen un 30 %. La voz sobrevive
—0,86 de coseno— pero el latente no: **lo domina el ruido del flow matching**, no el estilo. Y todo
lo que se venía optimizando eran estadísticos de **un** latente contra estadísticos de **otro**, así
que el gradiente apuntaba al sorteo de ruido y no a la voz. La pérdida bajaba porque siempre se
puede deformar el estilo hasta cuadrar unas correlaciones; el timbre se iba por el mismo camino.

Esto explica los cuatro negativos anteriores de golpe, y explica también por qué el de locutor solo
funcionaba en las frases que veía: con **un** sorteo por iteración, el gradiente es casi todo ruido.

**Consecuencia.** `ajustar(..., sorteos=N)`: cada iteración promedia sobre N sorteos de ruido y N
frases, con un `backward` por sorteo para que la memoria no se multiplique. Y a partir de ahora el
banco de pruebas del voice builder es **reconstruir una voz de fábrica**, no p225: tiene techo
conocido (0,9144) y respuesta conocida, así que distingue un método que funciona de uno que no.
Medición en curso.

## 2026-09-16 · Sexto intento, con el ruido promediado — RECONSTRUIR EL VOICE BUILDER DE SUPERTONIC, CERRADO

**Montaje.** Corregido lo que decia el diagnostico anterior: cada iteracion promedia sobre 4
sorteos de ruido y 4 frases, con un `backward` por sorteo. Banco de pruebas con verdad conocida —
reconstruir el estilo de M3 desde su propio audio, partiendo de F1. 200 iteraciones.

**Resultado.**

| | Coseno |
|---|---|
| Techo (M3 contra si misma) | 0,9144 |
| Partida (F1) | **0,2306** |
| Inversion del vocoder (otra vez impecable) | 0,9428 |
| Ajustado, con termino de locutor | **0,0519** |
| Ajustado, sin termino de locutor | **−0,0424** |

Promediar el ruido era la hipotesis correcta sobre *por que* fallaba, y **no basta**. Con la
respuesta al alcance y el techo a 0,91, el ajuste sigue alejando la voz por debajo del punto de
partida.

**Consecuencia. Se cierra la linea.** Seis intentos, seis negativos:

| Via | Coseno | Que descarta |
|---|---|---|
| Mezclar las diez voces | 0,227 | el espacio de mezclas es de 9 dimensiones |
| Regresion `audio → estilo` | −0,040 | el inverso no es una funcion |
| Gradiente + estadisticos por canal | 0,072 | el resumen no discrimina |
| Gradiente + Gram | 0,148 | el Gram correlaciona, pero poco |
| Gradiente + coseno en la perdida | 0,041 | no generaliza fuera de sus frases |
| **Lo anterior + ruido promediado, con verdad conocida** | **0,052** | **no es el ruido** |

Lo unico que funciona de todo esto es la **inversion del vocoder** (0,83–0,94, tres veces
reproducida): recuperar el `ae.encoder` si se puede. Lo que no se puede es recuperar el
`style_encoder`, y sin el no hay clonacion. El coste ya gastado son ~40 min de GPU y seis
experimentos; seguir seria tirar mas.

Lo que hay que hacer en su lugar esta en la entrada siguiente: **no reconstruir la pieza que falta,
sino usar un modelo que la publique**.

## 2026-09-16 · Pocket TTS: el voice builder que Supertonic no tiene, publicado — CLONA EN 0,44 s

**Montaje.** Marcos: *«sino busca un voice builder ya hecho que funcione en segundos»*. Lo hay, y es
posterior al corte de conocimiento del agente: **Pocket TTS** de Kyutai (enero de 2026, 100 M
parametros, CC BY 4.0). Lo que a Supertonic le falta, este lo publica: `mimi_encoder.onnx`, el
encoder de audio a latente, en el export de
[`KevinAHM/pocket-tts-onnx`](https://huggingface.co/KevinAHM/pocket-tts-onnx) — no gateado, con
bundles por idioma, **espanol incluido**, y `encode_voice(wav)` como unica llamada para clonar.

Medido con el MISMO arnes que Supertonic: las mismas 10 frases normales, Whisper-small, el mismo
`normalizar_para_wer`, y el coseno de WeSpeaker contra `web/e2e/referencia.wav` (VCTK p225).
Bundle `spanish`, int8, CPU.

**Resultado, barriendo temperatura** (la por defecto, 0,7, es la peor):

| temperatura | pasos de flow | WER medio | mediana | coseno medio | maximo | RTF |
|---|---|---|---|---|---|---|
| 0,7 (por defecto) | 1 | 0,119 | 0,087 | 0,472 | 0,563 | 0,499 |
| **0,5** | 1 | **0,027** | **0,000** | **0,449** | 0,526 | 0,483 |
| 0,3 | 1 | 0,045 | 0,000 | 0,464 | 0,517 | 0,480 |
| 0,3 | 2 | 0,108 | 0,000 | 0,461 | 0,569 | 0,491 |
| 0,1 | 1 | 0,246 | 0,142 | 0,413 | 0,548 | 0,489 |

Con la temperatura por defecto se comia la primera palabra de forma intermitente («Ayer estuvimos»
→ «estuvimos», «Gracias por» → «seas por»), que parecia un artefacto de arranque del streaming y
era muestreo. A 0,5 desaparece. Bajar mas degenera.

**Contra lo que hay:**

| | Supertonic 3 | Pocket TTS (temp 0,5) |
|---|---|---|
| **Clonar desde una grabacion** | **no existe**; 6 intentos, mejor 0,227 | **0,449 medio · 0,526 max** |
| Tiempo de clonar | ~40 min de GPU, y falla | **0,44 s**, una llamada |
| WER espanol, frases normales | **0,008** | 0,027 (mediana 0,000) |
| RTF en CPU | 0,463 | 0,483 |
| Descarga | 398 MB fp32 | **202 MB int8** |
| Frecuencia de salida | **44 100 Hz** | 24 000 Hz |
| Idiomas | **31** | 6 (en, fr, de, it, pt, es) |
| Licencia de pesos | OpenRAIL-M, con restricciones | **CC BY 4.0**, solo atribucion |

Los dos pasan el criterio de inteligibilidad (≤ 0,10). En clonacion, **0,449 dobla el 0,227 de
Supertonic y supera el 0,338 del conversor de OpenVoice**, y el maximo por frase (0,526–0,569) toca
el objetivo de 0,55. Sigue sin llegar de media, y eso hay que decirlo: 0,449 no es 0,55.

**Medido el techo, el numero baja.** Repetido con el montaje del criterio 4 de verdad — seis
locutores de OpenSLR es, una grabacion para clonar, **otra distinta del mismo locutor** para el
techo, y frases del propio corpus — que es exactamente con el que se midio la cadena vieja:

| | Cadena vieja (OpenVoice) | Pocket TTS |
|---|---|---|
| Similitud media | 0,338 | **0,375** |
| Techo (dos grabaciones reales) | 0,758 | 0,752 |
| Razon del techo (el criterio pide 0,75) | 0,446 | **0,498** |
| WER clonando, frases de corpus | 0,417 | **0,218** |
| Tiempo de clonar | — | 752 ms de media (266–1 402) |

**El 0,449 de la tirada anterior era un solo locutor y era optimista.** El numero honesto es 0,375:
mejora el 0,338, pero de forma modesta, y **no pasa el criterio 4** ni por el 0,55 absoluto ni por
el 0,75 del techo. Lo que si mejora claramente es el WER al clonar: 0,218 contra 0,417.

Con 6 locutores parecia que la calidad de la grabacion mandaba (`pef_08784` saco 0,563 con techo
0,867, `clm_02484` 0,297). **Repetido con 14 locutores, esa lectura era falsa** — ver abajo.

**Y la variante de 24 capas mejora las dos cosas.** Mismo montaje, `spanish_24l`:

| | Cadena vieja | Pocket 6L | **Pocket 24L** |
|---|---|---|---|
| Similitud media | 0,338 | 0,375 | **0,428** |
| Mediana / maxima | — | 0,383 / 0,574 | 0,403 / **0,673** |
| Techo | 0,758 | 0,752 | 0,752 |
| Razon del techo (pide 0,75) | 0,446 | 0,498 | **0,569** |
| **WER clonando** | 0,417 | 0,218 | **0,086 ✅ PASA** |
| Tiempo de clonar | — | 752 ms | **488 ms** |
| Descarga (int8) | — | 202 MB | ~376 MB |

**Por primera vez en el proyecto, la voz clonada pasa el criterio de inteligibilidad** (≤ 0,10):
0,086 con mediana 0,000, frente al 0,417 de la cadena vieja al convertir. La similitud sigue sin
pasar —0,428 contra 0,55, y 0,569 del techo contra 0,75— pero es el mejor numero medido nunca aqui,
y un locutor (`pef_08784`, referencia limpia, techo 0,867) llego a **0,626 con WER 0,028**.

**El RTF del 24 capas lo descarta para el chat.** Medido en la misma maquina, int8, CPU:

| bundle | ms por frase | RTF | x tiempo real |
|---|---|---|---|
| `spanish` (6 capas) | 3 148 | 0,753 | 1,33x |
| `spanish_24l` | 7 535 | **2,427** | **0,41x** |
| *Supertonic, para comparar* | 1 703 | 0,463 | 2,16x |

El de 24 capas tarda **7,5 s en decir una frase de 3 s**: leer un chat en vivo con eso es imposible.
El de 6 va justo (1,33x) y Supertonic va holgado (2,16x).

Asi que no hay un ganador unico, hay un dial, y cae solo en los dos modos que ya tiene la pagina:

    Chat     -> `spanish` 6 capas: 1,33x tiempo real, clonacion 0,375, WER 0,218
    Estudio  -> `spanish_24l`:     0,41x,             clonacion 0,428, WER 0,086

Cada bundle trae **su propio** `mimi_encoder.onnx` (hashes distintos entre `spanish` y
`spanish_24l`, aunque el decoder si lo comparten), asi que las voces no se cruzan entre bundles. No
importa: clonar cuesta 0,5 s, se clona con el que vaya a hablar.

**Lo que sigue sin medirse:** la ejecucion en navegador. El export esta pensado para ONNX Runtime
Web y [pocket-tts-raven](https://github.com/pkalogiros/pocket-tts-raven) reporta ~14x tiempo real
con un runtime en C++/WASM, pero eso es su numero, no uno de aqui — y los RTF de arriba dicen que
conviene desconfiar de los numeros ajenos. Y falta el navegador: el export esta pensado para ONNX Runtime Web y
[pocket-tts-raven](https://github.com/pkalogiros/pocket-tts-raven) reporta ~14x tiempo real ahi con
un runtime en C++/WASM, pero aqui no se ha ejecutado en navegador.

**Consecuencia.** La linea de reconstruir el voice builder de Supertonic queda cerrada por la
entrada anterior; esta dice por donde sigue: **no reconstruir la pieza que falta, usar el modelo que
la publica**. Es una decision de motor, con un coste real —24 kHz en vez de 44,1, seis idiomas en
vez de treinta y uno— y la firma Marcos.

## 2026-09-16 · ¿Limita el modelo o la grabacion? — LIMITA EL MODELO, Y SE ESTANCA EN 0,41–0,45

**Montaje.** Con 6 locutores la similitud iba de 0,297 a 0,563 y parecia seguir a la calidad de la
referencia; si fuera asi, con una grabacion limpia se llegaria al 0,55 y la migracion se
justificaria sola. Se repite con **14 locutores** de OpenSLR es, bundle `spanish_24l`, y se
correlaciona la similitud con el **techo** de cada uno (dos grabaciones reales del mismo locutor),
que es una medida directa de lo buena que es su referencia.

**Resultado.**

| | |
|---|---|
| Similitud media (14 locutores, 42 frases) | **0,413** |
| Techo medio | 0,760 |
| Razon del techo | 0,543 |
| WER medio / mediana | 0,184 / 0,101 |
| Tiempo de clonar | 444 ms |

Y la correlacion, que es lo que se venia a buscar:

    correlacion techo <-> similitud:  r = +0,247   (debil)
    ajuste:  similitud = 0,262 x techo + 0,213

| Referencia | Similitud esperada |
|---|---|
| techo 0,70 (regular) | 0,397 |
| techo 0,90 (excelente) | **0,450** |

Los 5 locutores de mejor referencia sacan 0,451 de media; los 5 peores, 0,395. **56 milesimas sobre
un rango enorme de calidad de audio.** La correlacion con el WER es todavia menor (r = −0,109).

**Consecuencia, y correccion.** La lectura anterior —«la calidad de la grabacion decide mas que el
modelo»— **era falsa**: con 6 locutores y 3 frases cada uno, aquel 0,626 contra 0,297 era ruido de
muestreo. Pocket TTS **se estanca en 0,41–0,45 independientemente de la referencia**, y no va a
llegar al 0,55 del criterio 4 con una grabacion mejor.

Eso no lo descarta: 0,413 sigue siendo mejor que el 0,338 de la cadena vieja, clona en 0,44 s contra
seis intentos fallidos, pesa la mitad y su licencia es CC BY 4.0 en vez de OpenRAIL-M. Pero lo que
se compra migrando es **clonacion que funciona a 0,41**, no clonacion que pase el criterio. Quien
decida tiene que decidir sabiendo eso.

## 2026-09-16 · Clonar desde una grabacion, en el navegador — FUNCIONA

**Montaje.** Marcos, mirando la pagina: *«veo el mismo motor y sigo sin poder cargar audios»*.
Tenia razon en las dos: Pocket TTS se habia medido pero no migrado, y aunque el explorador ya
dejaba elegir el audio, la pagina lo rechazaba a proposito porque Supertonic no sabe clonar.

Lo que faltaba era un runtime de Pocket TTS para el navegador, y existe:
[`pocket-tts-onnx`](https://github.com/thewh1teagle/pocket-tts-onnx) — TypeScript, en npm, CC BY
4.0, onnxruntime-web dentro de un **Worker**, siete idiomas con el espanol entre ellos, y
`engine.clone(samples)` como voice builder entero. Los pesos (177 MB, mas 39 del encoder solo si
clonas) vienen de Hugging Face y se quedan en la Cache API.

**Resultado, en Chromium, sin un solo error de JS:**

| | |
|---|---|
| Voces de fabrica del bundle espanol | **2** (`javert`, `lola`) — no las 8 del ingles |
| Clonar desde un wav, **primera vez** | **7 426 ms** (incluye bajar el encoder de 39 MB) |
| Hablar con la voz clonada | 2,96 s de audio en 2 192 ms, **RTF 0,74** (1,35x tiempo real) |

Sueltas un fichero y sale una voz que habla. Es lo que el proyecto llevaba persiguiendo desde el
principio y lo que seis intentos de reconstruir el voice builder de Supertonic no dieron.

**Consecuencia.** La pagina lleva **dos motores** con selector, y arranca en Pocket:

| | Pocket (por defecto) | Supertonic |
|---|---|---|
| Clonar desde audio | **si, y lo ofrece** | no, y lo dice |
| Descarga | 177 MB (+39 al clonar) | 398 MB |
| Velocidad en navegador | RTF 0,74 | RTF 0,37 (1 paso) |
| Calidad | WER 0,027 · 24 kHz · 7 idiomas | WER 0,008 · 44,1 kHz · 31 idiomas |

La temperatura va fijada a **0,5** y no a los 0,7 de fabrica, por la medicion del barrido.

**Y un bug que solo aparece con dos motores.** Cambiar de motor mientras el primero esta cargando
lanzaba **dos cargas a la vez**, y se quedaba la que acabara ultima, que no tiene por que ser la
que pediste. En el registro se ve entero:

    pagina lista; cargando el motor sola     <- arranca solo (pocket)
    motor: supertonic                        <- el selector lanza OTRA carga
    supertonic cargado en wasm               <- termina la segunda
    pocket cargado, 2 voces                  <- ...y luego la PRIMERA, que pisa a la otra

La pagina acababa en Pocket con las tarjetas de los dos motores mezcladas (12 voces) y rechazando
un `.json` de Supertonic por «el motor activo es Pocket». Arreglado con el mismo patron que el
turno de sintesis: cada carga lleva un numero y solo la ultima publica su motor; las demas cierran
el suyo y se van. Lo encontro el e2e, no el ojo.

**Lo que esto se lleva por delante, y hay que decirlo:** `pocket-tts-onnx` depende de `espeak-ng`,
asi que **el wasm de espeak vuelve al bundle** (18,5 MB). El espanol no lo invoca nunca — solo se
usa para palabras latinas dentro de hebreo — pero un build desplegado lo **distribuye**, y la
obligacion GPL-3.0 va con la distribucion, no con la ejecucion. ADR 0012 daba por cerrada esa
puerta al quitar el fonemizador; con este paquete se reabre, y THIRD_PARTY.md tiene que decirlo.

## 2026-09-16 · Los pasos del decodificador de Pocket son gratis — Y VENIAN AL MINIMO

**Montaje.** Marcos, tras probar: *«esta bien pero seria increible si pudieramos subir un poco la
calidad»*. Se mira que palancas hay de verdad. El `assets.json` del bundle espanol lo dice:

    sampler_decode_steps = 1     <- lo que se venia usando
    max_decode_steps     = 4     <- el tope
    temperature          = 0.7

Dos cosas mal a la vez: la pagina pedia 8 pasos y **el motor recortaba a 4 en silencio**, y el
deslizador ofrecia hasta 24, que era mentira. Y el valor efectivo era el de fabrica, 1.

**Resultado**, misma frase, misma voz clonada, Chromium:

| pasos | RTF | segundos de audio |
|---|---|---|
| 1 | 0,81 | 2,82 |
| **4** | **0,72** | 3,11 |

**Subir de 1 a 4 no cuesta tiempo.** La prediccion escrita antes de medir era «unas cuatro veces
mas lento», y es falsa: lo caro es el bucle autorregresivo del modelo de lenguaje, que corre igual
en los dos casos, y los pasos del decodificador de flow son calderilla al lado.

**Consecuencia.** El deslizador pasa a 1–4 (lo que el modelo admite) y arranca en **4**, y el chat
tambien, porque ya no hay nada que pagar. Antes se habia repartido 1 para chat y 4 para estudio
suponiendo un coste que no existe.

Y de paso queda a la vista que el 0,5 de temperatura que se habia forzado en `pocket.ts` venia de
una medicion hecha con **otro** export: este modelo declara 0,7 en su propio `config`. Se quita el
forzado y pasa a deslizador, con el extremo izquierdo como «la del modelo».

## 2026-09-16 · La palanca de calidad no era un parametro: era la duracion — +13 PUNTOS

**Montaje.** El encoder de Pocket lee hasta 20 s y la pagina le daba 8 (y el boton decia 6). Antes
de recomendar «graba mas largo» por intuicion, se mide: para 6 locutores de OpenSLR es se
concatenan sus propias frases hasta 2, 5, 10 y 20 s, se clona con cada tramo, y se mide el coseno
contra un tramo largo **que el clonado no vio**. Bundle `spanish_24l`, temperatura 0,5.

**Resultado.**

| Grabacion que se le da | Similitud |
|---|---|
| 2 s | 0,396 |
| 5 s | 0,409 |
| 10 s | 0,441 |
| **20 s** | **0,527** |

**+13,1 puntos de 2 a 20 segundos**, y monotono. El 0,527 de los 20 s roza el 0,55 que pide el
criterio 4, partiendo de un 0,396 que no se acerca.

Conviene ponerlo al lado de lo otro que se midio sobre la referencia: su **calidad** apenas importa
(correlacion con el techo r = +0,247, 56 milesimas entre las mejores y las peores), pero su
**duracion** importa muchisimo. Son cosas distintas y las dos estan medidas ahora: da igual con que
grabes, importa cuanto.

**Consecuencia.** El boton pasa de «grabar 6 s» —que grababa 8— a **grabar 20 s**, con el contador
a la vista mientras graba, y al clonar desde fichero se usan los 20 primeros segundos en vez de
recortar antes. Es el cambio de mas efecto de toda la sesion y no toca ni un parametro del modelo.

## 2026-09-16 · ¿Catalan? — NO ESTA EN NINGUNO, PERO SUPERTONIC LO LEE Y POCKET NO

**Montaje.** Marcos pregunta si el programa acepta catalan. Ninguno de los dos motores lo lista:
Supertonic tiene 31 idiomas (ni `ca`, ni `gl`, ni `eu`) y Pocket tiene 7. Pero el texto **entra
igual** — el indexador Unicode conoce `ç`, `l·l` y `ny` sin un solo caracter desconocido —, asi que
la pregunta no es si lo acepta sino que sale por el altavoz. Seis frases con lo dificil para un
lector espanol (ç, ela geminada, x inicial, tj, ny, vocales neutras, consonantes finales), y Whisper
transcribiendo **en catalan**.

**Resultado.**

| Motor | WER en catalan | Su WER en espanol |
|---|---|---|
| Pocket, bundle `spanish` | **1,409** | 0,027 |
| Supertonic, idioma `es` | **0,237** | 0,008 |

Sesenta veces peor uno que otro. Lo que dice cada uno de la misma frase:

    Pocket:      «El Xavier va casar un PXR contra el Ponto»
    Supertonic:  «El Xavier va casar un peix al costat del pont»   (WER 0,100)

Y «Bon dia, com estàs? Avui fa molt bon temps» Supertonic la transcribe **perfecta, WER 0,000**.

**Por que.** Supertonic lee **caracteres** y esta entrenado en 31 idiomas, entre ellos espanol,
frances, italiano y portugues; el catalan comparte con ellos casi toda la ortografia, asi que lo
deletrea razonablemente sin que nadie se lo haya ensenado. Pocket tiene un tokenizador **solo
espanol** y se atraganta. Donde Supertonic falla es en lo especificamente catalan: `pa amb
tomàquet` → «pambe, tomàquet i un Gotebi», `Ahir vam anar` → «Air va manar».

Aviso sobre el juez: Whisper-small transcribiendo catalan no es perfecto, asi que parte de ese
0,237 es suyo y no del sintetizador.

**Consecuencia.** 0,237 esta muy por encima del ≤ 0,10 del criterio, asi que **el catalan no esta
soportado** y no se va a anunciar. Pero da un motivo mas para que convivan los dos motores, y ahora
medido: **Pocket clona, Supertonic habla mas idiomas** — incluidos algunos que no figuran en su
lista. Quien quiera leer catalan, gallego o cualquier lengua romanica que no este, que pruebe
Supertonic; con Pocket no hay nada que hacer.

## 2026-09-16 · «Genera mucho ruido estatico de fondo» — ERA EL NIVELADO DE LA REFERENCIA

**Montaje.** Marcos, clonando una voz: *«se clona bien pero genera como mucho ruido estatico de
fondo»*. Habia dos sospechosos, los dos anadidos ese mismo dia, y solo uno podia ser:

    nivelar la SALIDA      ganancia constante -> NO puede cambiar la relacion senal/ruido
    nivelar la REFERENCIA  hasta +4x -> sube su suelo de ruido antes de encodearla

La primera es aritmetica y se descarta sin medir. La segunda se mide: se parte la salida en
ventanas de 50 ms, el percentil 90 de sus RMS es la voz y el percentil 10 el fondo, y la distancia
entre ambos es lo limpio que suena.

Primer intento **no valia**: la referencia del repo ya esta a −17,8 dB, asi que el nivelado le
aplicaba 0,94x y la hipotesis ni se activaba. Repetido simulando lo que de verdad llega — una
grabacion floja y con siseo, x0,12 mas ruido blanco, −36,2 dB RMS, donde el nivelado si aplica
el **4x** completo:

| Referencia | Senal/ruido de la salida | Suelo de ruido |
|---|---|---|
| **Cruda** | **37,7 dB** | −78,0 dB |
| **Nivelada** | **26,0 dB** | −53,8 dB |

**Nivelar la referencia cuesta 11,7 dB de senal/ruido y sube el suelo 24 dB.** Y lo interesante:
la senal/ruido de la propia referencia **no cambia** (28,9 dB en los dos casos, como tiene que ser
con una ganancia constante). Lo que cambia es que **el encoder es sensible al nivel absoluto**: con
la referencia amplificada se queda el siseo como parte de la voz.

**Consecuencia.** Fuera el nivelado de la referencia — era invencion propia, el paquete usa esos
niveladores solo para los marcos de salida. El de la salida se queda: iguala el volumen entre voces,
que en el chat se nota, y no puede empeorar nada.

La leccion, que es la misma de otras veces hoy: una idea razonable («normalizar la entrada ayuda al
encoder») aplicada sin medir, y en la direccion contraria a la que parecia.

---

## Mediciones pendientes que deciden algo

No son tareas: son las preguntas cuyo número cambia una decisión escrita. Cuando se midan, cada una
sube arriba como entrada con fecha.

- **Calibración en la GTX 1070** (ADR 0011): pulsar «calibrar» en la página con Chrome real y anotar
  tts y conversor en webgpu fp32 frente a wasm. Es el número que dice si la voz clonada vale para
  directo; en headless no hay WebGPU y no se puede medir aquí.
- **Modo chat con conversor en WebGPU** (ADR 0010): la política «una voz por usuario» pasa cada
  mensaje por el conversor; en wasm son ~2,9 s y la cola descarta. Medir velocidad (audio/síntesis)
  en la GTX 1070 con WebGPU fp32: si supera ×1 con margen, la voz por usuario es viable en directo.
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
