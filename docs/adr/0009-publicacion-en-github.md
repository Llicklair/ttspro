# 9. Publicar en GitHub: qué sale, con qué licencia y sin pesos

**Estado:** propuesta · **Fecha:** 2026-09-05 · **Decide:** Marcos · **Toca:** [SCOPE.md](../../SCOPE.md), [ADR 0006](0006-uso-no-comercial-y-xtts-como-maestro.md)

## Contexto

Pregunta de Marcos (2026-09-05): *«¿podemos subirlo a GitHub sin el entrenamiento, para que otro
usuario lo utilice?»*. Técnicamente sí, y el repo está más cerca de lo que parece: 2,7 MB en git,
42 commits, ningún peso versionado, ninguna credencial. Lo que faltaba no era código.

El propio SCOPE puso dos condiciones antes de publicar, y las puso cuando publicar era una idea
lejana:

- **Watermarking y anti-spoofing**: fuera del alcance *«no porque no importe: porque hoy no se
  publica ningún peso. Se reabre en el ADR de publicación, antes de publicar»*.
- **Clonar sin consentimiento**: *«la política de uso del modelo se escribe en el model card antes
  de publicar»*.

Y hay tres hechos que mandan sobre la forma de publicarlo:

1. **espeak-ng es GPL-3.0-or-later** y el navegador lo carga. Una build de `web/` distribuye ese
   WASM. Eso no afecta al código de este repo mientras nadie sirva la página, pero sí a quien la
   sirva.
2. **Los pesos no caben en git**: `conversor.onnx` son 132,3 MB y GitHub rechaza ficheros de más de
   100 MB. Los tres que van al navegador suman 102,5 MB en fp16 y 203,1 en fp32.
3. **Los pesos ya no hacen falta en el repo**: todo lo que se sirve es un port de pesos públicos
   (Piper MIT, OpenVoice v2 MIT), así que un comando los reconstruye desde la fuente en minutos.

## Decisión

Publicar **código, contrato, documentación y demo; los pesos no**. Quien clone el repo corre
`uv run python -m ttspro.export.modelos` y obtiene exactamente los mismos ficheros que se midieron,
desde Hugging Face. Se versiona `models/voces.json` (115 KB de vectores, no audio) porque es lo
único que no se puede regenerar sin ~12 GB de corpus.

Tres cosas cambian para que eso sea verdad y no un párrafo optimista:

- El port de Piper pasa de `ttspro.train.piper` a **`ttspro.export.piper`**. Nunca importó
  entrenamiento; estaba en la carpeta equivocada y hacía falsa la frase «la inferencia no depende
  del entrenamiento» justo en el paso que produce la voz base.
- `web/e2e/referencia.wav` no tenía procedencia escrita en ninguna parte. Se sustituye por
  `p225_003` de VCTK (CC BY 4.0), acreditado en THIRD_PARTY.md. Un repo que se publica no
  redistribuye audio de origen desconocido.
- `models/contrato.json` decía que el TTS venía del VITS de coqui. Viene de Piper desde el ADR 0008.

**No publicar pesos mantiene cerrada la puerta del watermarking**, que es exactamente la condición
que escribió SCOPE: nadie recibe de aquí un modelo listo para clonar, recibe una receta sobre pesos
que ya son públicos. El día que se publique un peso, o se sirva la demo online, esta decisión se
reabre con datos, no con opinión.

### Lo que queda por decidir (lo firma Marcos)

1. **Licencia del repo.** Dos opciones coherentes:
   - **MIT para el código propio**, con THIRD_PARTY.md diciendo que la build incluye espeak-ng
     GPL-3.0 y que quien la sirva asume esa obligación. Es lo que hacen los proyectos que cargan
     espeak-ng como módulo separado, y deja el código reutilizable fuera del navegador.
   - **GPL-3.0 para todo**, que evita la discusión de si el WASM cargado en tiempo de ejecución
     forma obra combinada, a cambio de contagiar la licencia a quien reutilice solo el frontend
     de texto o el exportador.

   Recomendación: **MIT + aviso**, porque lo que este repo aporta (el port, el exportador acotado
   a las ops de WebGPU, el contrato, el frontend espejado) tiene valor fuera del navegador, y
   porque el vínculo con espeak-ng es una llamada a un módulo separado, no un enlace estático.
2. **Demo online.** Servir la página en GitHub Pages significa distribuir espeak-ng y los pesos, no
   solo el código; entra entonces el watermarking y una política de abuso. Sin decisión, la demo se
   corre en local, que es donde hoy está medida.
3. **Uso no comercial (ADR 0006).** Hoy la cadena entera es MIT + CC0: la restricción es nuestra,
   no de los pesos. Si se publica, o se justifica en el model card o se levanta.

## Consecuencias

- Otro usuario arranca con tres comandos y sin GPU. Eso es nuevo: hasta hoy el README decía «day
  zero, no model yet» y el camino real estaba solo en AGENTS.md.
- El repo se publica con dos criterios en rojo y dichos en voz alta: similitud 0,338 contra 0,55 y
  148,8 MB contra 110. Publicar no los arregla; esconderlos sí los empeora.
- `ttspro.train` se queda. Es el camino no tomado, con sus números en evidencia, y la frontera
  `INFERENCIA -/-> ENTRENAMIENTO` de `gb graph --gate` garantiza que nada de lo que se publica
  depende de él.
- Mientras la licencia no esté elegida, el repo **no** puede hacerse público: sin fichero LICENSE
  el código es «todos los derechos reservados» y nadie puede usarlo, que es justo lo contrario de
  lo que se pretende.
