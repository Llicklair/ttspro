# 11. Paquetes de voz bajo demanda y recomendación por hardware

**Estado:** aceptada · **Fecha:** 2026-09-06 · **Decide:** Marcos · **Sigue a:** [0008](0008-la-voz-base-es-una-voz-de-piper.md), [0010](0010-modo-chat-para-twitch.md)

## Contexto

Marcos (2026-09-06): *«la inferencia tarda demasiado; el modelo base va rápido pero el clonado
sigue siendo impreciso y lento. ¿Podemos incluir modelos de Hugging Face y cargarlos desde la UI?»*
Y después: *«adelante, y adáptalo para gráficas superiores: que detecte el sistema y recomiende el
mejor»*.

Dos preguntas distintas con dos respuestas medidas ([evidencia](../evidencia.md), 2026-09-06):

- **«Modelos de Hugging Face» en genérico, no.** Cada modelo es un port a nuestros bloques, un
  export acotado a las ops de WebGPU y un contrato; no hay cargador universal. Lo que sí hay son
  **nueve voces españolas de Piper, MIT**, que el port ya convierte (349/349 tensores, paridad
  ~1e-6) a 34–210 ms por frase en CPU. Para «una voz por usuario» en un chat, eso es voz clara y
  rápida **sin conversor**, que es la pieza lenta (2,9 s en wasm) e imprecisa (0,34 de similitud).
- **El hardware decide más que el modelo.** En WebGPU con `shader-f16` (RTX 20xx en adelante,
  Apple Silicon) el conversor cabe en la GPU y en fp16; en Pascal (GTX 1070) fp16 da NaN y hay
  que ir en fp32; sin WebGPU, wasm fp32 y con hilos solo bajo COOP/COEP. Adivinarlo es una regla;
  medirlo es una calibración.

## Decisión

1. **Paquete de voz = `<clave>.json` + `<clave>.tts.onnx` + `<clave>.tts.fp16.onnx`**, ficheros
   planos para que sirva cualquier host estático, más un `indice.json`. Los construye
   `ttspro.export.paquete` (descarga de Piper → port → export → contrato propio → vector de voz
   base medido con `voz.onnx`). **Cada voz lleva su contrato**, porque las tablas de símbolos
   difieren (carlfm 130, davefx 256, claude 261) y el frontend tokeniza con la de la voz que va a
   hablar; el ADR 0002 decía «un contrato» y aquí son uno por voz base más el común del conversor.
2. **Se sirven desde GitHub Pages** (rama huérfana `voces`, `https://llicklair.github.io/ttspro/`).
   No desde la release: los assets de release **no mandan `Access-Control-Allow-Origin`** y el
   navegador no puede bajarlos desde otro origen; medido con `curl` y `Origin`. Hugging Face sí
   manda CORS y sería el host natural, pero requiere una cuenta con sesión que hoy no hay: es un
   cambio de URL cuando la haya. La `es_AR-daniela-high` en fp32 pesa 114 MB, más del límite de
   100 MB por fichero de GitHub, y en ese host su fp32 apunta al fp16.
3. **En la página y en la librería**: selector de voz base con el índice, descarga bajo demanda
   con progreso, `tts.cargarVozBase(clave)` / `tts.elegirVozBase` / `predict({vozBase})`, y en el
   chat la política **«una voz base por usuario»**, que reparte los usuarios entre las voces
   descargadas. `?voces=<url>` y `TTS.cargar({vocesBase})` apuntan a otro host (el e2e sirve la
   carpeta local).
4. **Hardware**: `detectar()` lee (WebGPU, adaptador, `shader-f16`, hilos, aislamiento);
   `recomendar()` aplica la regla de arriba y la dice en la página al cargar; **`calibrar()` mide**
   una frase por el TTS y por el conversor en cada candidato y elige el más rápido, que se guarda
   en `localStorage` y se aplica al recargar. La regla puede equivocarse en hardware que nadie
   midió; la calibración no.

## Consecuencias

- El port de Piper dejó de depender del nombre de los nodos (el bias hermano nombra al peso) y
  el exportador escribe las **formas** del contrato desde el grafo, no desde el catálogo: una
  voz x_low tiene 96 canales de flow, no 192, y el contrato lo decía mal.
- Ocho paquetes publicados; 572 MB en la rama `voces`. Quien clone el repo no los necesita: el
  demo y la librería funcionan con la voz local y el índice es opcional (sin red, sin paquetes).
- El conversor no cambia. Sigue siendo la única clonación real y sigue costando lo que cuesta;
  lo que cambia es que ya no es el único camino a «muchas voces».
- Pendiente que decide algo: la calibración en la GTX 1070 de Marcos con WebGPU fp32, que es el
  número que dice si la voz clonada vale para directo. La página ya la mide con un botón.

---

**Enmienda (2026-09-06, misma tarde):** Marcos: *«las voces clonadas siguen sonando con mala
pronunciación, ¿se puede mejorar?»* y, probando, *«esta funciona muy bien: claude spanish mexico»*.
Medido ([evidencia](../evidencia.md)): `tau` no es la palanca (la WER convertida queda en 0,55–0,72
para cualquier valor); la **voz base debajo del conversor sí lo es**. Con davefx la conversión
duplica los errores (0,353 → 0,603); con claude casi no los toca (0,362 → 0,397) y además clona
más cerca (parecido 0,484 frente a 0,455). **La voz base local por defecto pasa a
`es_MX-claude-high`**: `ttspro.export.modelos` la construye por defecto, `models/contrato.json` y
`models/voces.json` se regeneran con ella, y davefx sigue disponible como paquete. El ADR 0008
eligió davefx por WER de la voz base; este dato es sobre lo que sale del conversor, que es lo que
el usuario oye cuando clona.

