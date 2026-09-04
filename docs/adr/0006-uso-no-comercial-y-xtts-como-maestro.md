# 6. El proyecto es no comercial y corre en el navegador; XTTS-v2 entra como maestro, no como runtime

**Estado:** aceptada · **Fecha:** 2026-09-05 · **Decide:** Marcos · **Toca:** [SCOPE.md](../../SCOPE.md), [ADR 0001](0001-sintetizador-no-autorregresivo.md)

## Contexto

XTTS-v2 (coqui) habla español, clona con 6 s y suena mejor que un VITS. Estaba fuera por dos
hechos comprobados en su repositorio el 2026-09-05:

- licencia **CPML 1.0.0**: solo uso no comercial ("any direct or indirect payment arising from
  the use of the model or its output" lo excluye), sin sublicencia, y "use of the model to train
  other models for commercial use is not a non-commercial purpose";
- `model.pth` de 1,87 GB, GPT autorregresivo: modelo de servidor con GPU, no de navegador
  (presupuesto 110 MB, un forward por frase: ADR 0001 y 0005).

Marcos fijó el uso: **no comercial, en el navegador, sin servidor.**

## Decisión

1. El runtime sigue siendo el sintetizador propio (VITS afinado desde el VITS de VCTK de coqui,
   Apache 2.0) con el encoder de WeSpeaker (CC-BY-4.0), en el navegador. XTTS-v2 no se ejecuta
   en el producto.
2. XTTS-v2 **sí** puede usarse, dentro de la CPML, para dos cosas:
   - **maestro**: generar audio sintético de español (en particular castellano, que OpenSLR no
     cubre) y de locutores variados para afinar el modelo propio, que es no comercial;
   - **listón**: medir WER y SECS de XTTS sobre `eval/` como techo de calidad contra el que se
     lee el criterio de terminado.
3. Todo lo que se publique (pesos, demo, model card) lleva la condición **no comercial**
   heredada de la CPML si en su entrenamiento entró audio de XTTS. Si un día se quiere uso
   comercial, hay que reentrenar sin ese audio: se anota en el índice de datos qué frases son
   sintéticas para poder excluirlas.

## Consecuencias

- El model card del modelo propio dirá: código MIT; pesos no comerciales (CPML aguas arriba si
  hay datos de XTTS; CC BY-SA 4.0 de OpenSLR en cualquier caso).
- `ttspro.data` distinguirá `origen: real | xtts` por frase, y `preparar` lo conserva.
- Generar datos con XTTS necesita GPU: se hace entre entrenamientos, no a la vez (8 GB).
- Lo que esto no cambia: el criterio de terminado, el presupuesto de descarga, ni la regla 9
  (cero modelos en el camino de síntesis del navegador).
