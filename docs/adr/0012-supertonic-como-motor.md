# 12. Supertonic 3 como motor, y el voice builder que hay que reconstruir

**Estado:** propuesta · **Fecha:** 2026-09-15 · **Decide:** Marcos · **Sustituye a:** [0004](0004-fonemas-con-espeak-ng-en-los-dos-lados.md), [0007](0007-clonacion-como-postproceso.md), [0008](0008-la-voz-base-es-una-voz-de-piper.md) · **Retira:** reglas **1** y **8** de [ARCHITECTURE.md](../../ARCHITECTURE.md) · **Sigue a:** [0011](0011-paquetes-de-voz-y-hardware.md)

## Contexto

Marcos (2026-09-15): *«¿podemos ver si la info de este repo nos sirve para implementar ideas?
necesito un voice builder similar para más calidad de voces»*, sobre
[supertonic-py](https://github.com/supertone-oss-archive/supertonic-py). Y después, cuando el
informe puso por delante el presupuesto de descarga: *«no bajes los grados, trata de adoptar todo.
Sé que el ADR da unas métricas que no hay que superar, sáltatelo, usa los recursos que sean
necesarios»*.

La cadena de hoy tiene tres números rojos y **los tres tienen la misma causa**: el conversor de
OpenVoice. Es 63,4 MB de los 148,8 de descarga, 2,9 s de los 3,46 de latencia, y su 0,338 de
similitud contra un objetivo de 0,55 ([evidencia](../evidencia.md), 2026-09-05). Existe porque la
voz base es un peso fijo ([ADR 0008](0008-la-voz-base-es-una-voz-de-piper.md)) y la identidad
solo se puede añadir **después** ([ADR 0007](0007-clonacion-como-postproceso.md)).

Supertonic 3 no tiene ese problema porque no tiene esa forma. La identidad entra como **dos
tensores de entrada** —`style_ttl` de `[1,50,256]` y `style_dp` de `[1,8,16]`— en tres de sus
cuatro grafos. No hay nada que ejecutar después: la voz que pides es la que sale. Y su frontend
lee **caracteres Unicode**, no fonemas, así que espeak-ng sobra.

Lo medido aquí el 2026-09-15, en este portátil, CPU, `onnxruntime` 1.29 (detalle en
[evidencia](../evidencia.md)):

| | Cadena de hoy | Supertonic 3 |
|---|---|---|
| WER español, frases normales | 0,030 | **0,008** (mediana 0,000) |
| WER español, `fixtures/frases.txt` | — | 0,178 medio, mediana 0,000 |
| Frecuencia de salida | 22 050 Hz | **44 100 Hz** |
| Idiomas | es | **31** |
| RTF en CPU | — | **0,46** (8 pasos) |
| Descarga de grafos | 111,3 MB + 17,6 de espeak | 398,1 MB, sin espeak |
| Clonación | conversor, 0,338 | **no la trae** |

Las dos últimas filas son el precio, y hay que decirlo entero.

**El voice builder no existe en el archivo.** Comprobado fichero a fichero en los cuatro repos de
pesos publicados (`Supertone/supertonic`, `-2`, `-3` y los espejos de `supertone-oss-archive`):
los cuatro publican exactamente `text_encoder`, `duration_predictor`, `vector_estimator` y
`vocoder`. El `tts.json` describe un `ae.encoder` (audio → latente) y un `ttl.style_encoder`
(latente → 50 fichas de estilo), pero **sus pesos no se publicaron nunca**. Los `voice_styles/*.json`
llevan `"source_file": "F1.wav"` y `"extracted_at"`: se extrajeron con una herramienta interna. Esa
herramienta era el Voice Builder, un **servicio alojado** en `supertonic.supertone.ai/voice-builder`,
y el propio README dice que dejó de estar accesible el **2026-08-31**. El repo se archivó el
2026-09-09. `docs/voices.md` lo remata: *«Custom voice service is currently in development»*.

Y no se puede sustituir por un encoder de locutor ya entrenado. `style_ttl` vive en un espacio que
el style encoder y el vector estimator aprendieron **en la misma tirada**; WeSpeaker, ECAPA o el de
XTTS viven cada uno en el suyo. No hay matriz fija entre dos espacios así, igual que no la hay entre
word2vec y GloVe. Tampoco lo resuelve cambiar de modelo: entre los **no autorregresivos** que
corren en el navegador, ninguno trae clonación abierta (F5-TTS pide la transcripción de la
referencia y pesa 1,3 GB; XTTS-v2, Chatterbox y CosyVoice 2 son autorregresivos; Kokoro tiene
exactamente el mismo agujero).

Lo que sí se puede es **aprender el puente, y el propio modelo fabrica los datos**: un estilo
cualquiera produce un audio, y WeSpeaker convierte ese audio en un embedding. Eso es un par
etiquetado, gratis, sin corpus y sin pedirle permiso a nadie.

## Decisión

1. **Supertonic 3 pasa a ser el motor**, descargado ya en ONNX y con la revisión fijada
   (`ttspro.supertonic.descargar`). No se porta, no se exporta, no se reentrena: upstream publica
   ONNX. Con eso desaparecen del camino crítico `ttspro.export.piper`, `ttspro.export.conversor` y
   la cirugía de opsets, y con ellos la clase de fallo en que el navegador y el checkpoint
   discrepan sobre lo que significaba un peso.

2. **Se retira la regla 1** («un forward por frase»). El flow matching es una EDO que se resuelve
   en `pasos` pasos, y el `vector_estimator` corre N veces por frase. La regla existía contra los
   modelos autorregresivos, que decodifican **token a token** con caché KV; esto es otra cosa: N no
   depende de la longitud del texto, es un dial de calidad que el llamante fija, y con N=8 el RTF
   medido es 0,46 en CPU. La regla se reformula, no se borra: **nada en el camino de síntesis puede
   iterar por token ni por frame**. La regla 5 sobrevive intacta: el ruido se genera fuera y entra
   como tensor, así que una semilla reproduce una toma.

   Esto no es una sorpresa: [evidencia](../evidencia.md) ya dejaba escrito, en «lo que falta por
   medir», que *«F5-TTS en WebGPU (ADR 0001): RTF de una alternativa iterativa. Si baja de 1 con la
   misma calidad, la regla 1 se reabre»*. Es la misma puerta, con otro modelo y con el número
   medido.

3. **Se retira el presupuesto de descarga de la regla 8**, por instrucción explícita de Marcos
   (arriba). 398,1 MB contra los 110 MB que decía la regla. Lo que **no** se retira es el resto de
   la regla 8: la latencia y el RTF se siguen midiendo y anotando, porque son lo que decide si la
   página es usable. Y se añade el aviso honesto: en `wasm` sin aislamiento esto va a ser lento;
   la página lo dice antes de descargar, no después.

4. **Se retira espeak-ng** ([ADR 0004](0004-fonemas-con-espeak-ng-en-los-dos-lados.md)). El
   frontend es Unicode: `src/ttspro/supertonic/texto.py` y `web/src/frontend/unicode.ts`, espejo
   exacto con su test de paridad (`tests/test_supertonic_paridad.py`, 26 casos). La regla 3 no se
   toca — sigue habiendo dos frontends y un test que los ata—, pero **se va la obligación GPL-3.0**
   que hasta hoy contaminaba cualquier build desplegado de `web/`, y 17,6 MB de wasm. Lo que **se
   queda** es `ttspro.frontend.normalizar`: Supertonic lee «3.522» como dígitos, y el normalizador
   español que ya existe lo arregla. Va delante del frontend Unicode, no en su lugar.

5. **La regla 2 se reformula, no se retira.** Eran dos grafos y un contrato; ahora son cuatro
   grafos y un contrato, `models/supertonic/contrato.json`, que **se genera leyendo los propios
   `.onnx`** en vez de escribirse a mano. Nadie teclea una forma, así que el contrato no puede
   discrepar de lo que se publicó.

6. **El voice builder se reconstruye, en dos piezas que se miden por separado** y con la misma vara
   que `tests/terminado`: el coseno de WeSpeaker, con `models/speaker_encoder.onnx`.

   - **`ttspro.supertonic.constructor`** — búsqueda directa. Optimiza una mezcla de las voces que
     ya tienes, **una mezcla distinta por cada una de las 50 fichas de estilo**, con búsqueda por
     entropía cruzada en dos etapas (10 números, luego 50×10). Lento (minutos por voz), no
     necesita nada previo, y cada voz construida entra en la base de la siguiente.
   - **`ttspro.supertonic.puente`** — el camino bueno. Muestrea estilos, los sintetiza, los embebe
     con WeSpeaker y **ajusta la flecha inversa**: una regresión ridge sobre las componentes
     principales del espacio de estilos. No es entrenar un modelo: es `np.linalg.solve` sobre una
     matriz de 257×64, en un segundo de CPU. Clonar pasa a ser un forward de milisegundos, que
     **cabe en el navegador**, y a diferencia de la búsqueda puede salir del casco convexo de las
     diez voces.

   Los dos se validan contra una **segunda grabación del mismo locutor que el ajuste no vio**, y
   contra el techo del propio encoder (dos grabaciones reales). El número honesto es ese, no el de
   la función que se optimizó.

7. **La licencia cambia de naturaleza y hay que decirlo.** El código de upstream es MIT; **los
   pesos son OpenRAIL-M**, que lleva restricciones de uso y no es permisiva. El `LICENSE` se
   descarga junto a los grafos para que sea imposible repartirlos sin él, y
   [THIRD_PARTY.md](../../THIRD_PARTY.md) y [MODEL_CARD.md](../../MODEL_CARD.md) recogen qué
   implica. El repo de upstream está **archivado**: no habrá parches, así que la revisión va fijada.

## Consecuencias

**A favor.** Español más inteligible que el de hoy (0,008 contra 0,030) a 44,1 kHz y en 31 idiomas.
Desaparece el conversor y con él la pieza lenta, pesada e imprecisa. Desaparece espeak-ng y su GPL.
La UI se simplifica sola: se van «voz base» contra «voz destino», `tau`, y la mitad de los
conceptos que hacían falta explicar.

**En contra.** 398 MB, contra 148,8. En `wasm` sin aislamiento la primera carga será larga y la
síntesis lenta; hay que medirlo en el navegador y decirlo en la página. Los pesos ya no son
permisivos. Y **la clonación deja de venir de fábrica**: hasta que el puente esté medido, lo que
hay son diez voces excelentes y un constructor que las mezcla.

**Lo que queda abierto.** Cuánto alcanza el puente. El espacio que cubre es el que el muestreo
alcanzó, y si un timbre cae fuera, cae fuera. Eso no se decide en un ADR: se mide, y va a
[evidencia](../evidencia.md) con su fecha, incluido si el número sale malo.

**Lo que sigue vivo del trabajo anterior.** El normalizador español y el de chat, la cola, el modo
chat entero ([ADR 0010](0010-modo-chat-para-twitch.md)), la detección de hardware y la calibración
([ADR 0011](0011-paquetes-de-voz-y-hardware.md)), y el speaker encoder, que ahora tiene **dos**
oficios: medir, como siempre, y ser la entrada del puente.
