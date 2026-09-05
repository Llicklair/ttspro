# 10. Modo chat: leer un chat de Twitch, muchos mensajes, dicción cuidada

**Estado:** aceptada · **Fecha:** 2026-09-05 · **Decide:** Marcos · **Toca:** [SCOPE.md](../../SCOPE.md) (streaming, «sin red»), [ARCHITECTURE.md](../../ARCHITECTURE.md) regla 3

## Contexto

Marcos (2026-09-05): *«¿podemos afinarlo un poco más? Dicción. Ha de procesar muchos mensajes,
ya que es para un chat de Twitch.»* Eso fija el uso real: un lector de chat, no una frase suelta.

Un chat no es prosa. Medido sobre los mensajes de `tests/fixtures/frontend/chat.txt`: enlaces,
`@menciones`, emotes (`KEKW`, `Kappa`), emoji, «jajajaja», «holaaaa», «q tal», «xq», y el mismo
mensaje tres veces seguidas. Pasado tal cual al sintetizador, cada uno de esos es o ruido o un
deletreo de cinco segundos. **«Afinar la dicción» aquí no es entrenar**: la voz base ya tiene WER
0,030 en frases limpias ([ADR 0008](0008-la-voz-base-es-una-voz-de-piper.md)). Es darle texto que
una persona diría en voz alta y elegir los parámetros de decodificación que menos tartamudean.

Y «muchos mensajes» es un problema de cola, no de modelo: en wasm la voz base tarda ~0,5 s por
frase y el conversor ~2,9 s. Sin cola, el lector va dos minutos por detrás del chat a los cinco
minutos de directo.

## Decisión

Tres piezas, todas en la capa que les corresponde:

1. **Etapa `chat` del frontend**, antes de `normalizar`, espejada en Python
   (`ttspro.frontend.chat`) y TypeScript (`web/src/frontend/chat.ts`) con paridad por fixture,
   como manda la regla 3. Once reglas en orden fijo: enlaces → «enlace»; emoji fuera; símbolos
   fuera; risas en todas sus grafías → «jajaja»; emotes fuera (lista declarada, sensible a
   mayúsculas como Twitch); abreviaturas expandidas (tabla declarada); letras repetidas 3+ → una;
   signos repetidos → uno; gritos en mayúsculas → minúsculas; palabras repetidas → una; corte a
   200 caracteres en un límite de palabra. Un mensaje que se queda vacío (solo emotes) **no se
   lee**: devuelve `""` y el que llama lo salta. Sin `\b` ni `\w` en ninguno de los dos lados,
   porque en JavaScript son ASCII y en Python Unicode: es la trampa de paridad más fácil de caer.
2. **Cola en el runtime** (`web/src/runtime/cola.ts`): sintetiza el siguiente mientras suena el
   actual; descarta lo que lleva más de 30 s esperando, lo que repite un texto que ya está en cola
   y lo que llega con la cola llena (20). **Todo lo descartado se cuenta** y se muestra: el retraso
   real del lector es un número en pantalla, no una sensación.
3. **Twitch como fuente, en el demo**, no en el runtime (`web/src/demo/twitch.ts`): IRC por
   WebSocket con el login anónimo `justinfan`, sin cuenta ni token. Es la única red del demo tras
   la carga, y **solo si el visitante escribe un canal y pulsa conectar**: la regla 9 («el navegador
   sin red sintetiza») sigue en pie, esto es una fuente opcional de texto.

Política de voz por mensaje, elegida en la página: **voz base** (rápida), **la voz elegida** o
**una por usuario** (hash del nombre sobre los presets). Las dos últimas pasan por el conversor y
por tanto cuestan ~3 s por mensaje en wasm: la cola las aguanta, pero la estadística de velocidad
(segundos de audio por segundo de síntesis) dice si el lector va al día.

## Consecuencias

- El frontend tiene una etapa más y una fixture más (32 mensajes). Añadir una abreviatura o un
  emote es tocar **las dos tablas** y correr la paridad; es exactamente el coste que la regla 3
  quiere que se pague.
- Los números de dicción (barrido de `noise`, `noise_w`, `length` con Whisper sobre los mensajes
  normalizados) y de caudal (veinte mensajes por la cola en Chromium headless, wasm) están en
  [evidencia](../evidencia.md) y el e2e `modo chat` los recomprueba.
- Lo que este ADR **no** hace: no reentrena, no añade inglés, no cambia el conversor. Si el lector
  con voz por usuario no va al día en wasm, la palanca es WebGPU o la voz base, no un modelo nuevo.
- SCOPE ya excluía «streaming dentro de una frase»; esto es streaming **entre** mensajes, que
  siempre estuvo dentro.
- **Enmienda del mismo día.** El uso real no es la página: es `import { TTS } from "./ttspro.js"`
  desde un HTML cualquiera y `tts.clonar(fichero)` + `tts.predict(texto | stream)` desde la consola,
  con los mensajes llegando de streex (Rails) por ActionCable, SSE o postMessage. La librería vive en
  `web/src/lib`, sale de `npm run build:lib` (500 KB más los wasm al lado) y `scripts/servir.mjs` la
  sirve con COOP/COEP sin Vite. Las tres puertas para otra aplicación están en
  `web/src/demo/fuentes.ts`. Todo medido sobre el bundle construido, en [evidencia](../evidencia.md).
