# 3. Implementación propia en PyTorch, no fork de coqui TTS

**Estado:** aceptada · **Fecha:** 2026-09-04 · **Relacionada:** [SCOPE.md](../../SCOPE.md) (lo que no entra)

## Contexto

[coqui TTS](https://github.com/coqui-ai/tts) es el referente y tiene la receta que se quiere
(YourTTS). Tres formas de apoyarse en él:

| Vía | Qué pasa |
|---|---|
| Fork de `coqui-ai/TTS` | Archivado en enero de 2024. Se hereda un repo de ~60 k líneas con 20 arquitecturas, y ninguna pensada para exportarse: el `export_onnx` de VITS existía pero cubría un subconjunto y sin speaker encoder |
| Dependencia de `coqui-tts` (fork mantenido por Idiap) | Vivo, pero la misma base: config de cientos de campos, `phonemizer` acoplado, `monotonic_align` compilado en Cython, y el modelo entrenado sale con un grafo que hay que desmontar para exportar |
| **Implementación propia** | La referencia de VITS ([jaywalnut310/vits](https://github.com/jaywalnut310/vits), MIT) son ~2 000 líneas; YourTTS añade embedding de idioma, embedding de locutor externo y una pérdida de consistencia del locutor. Es un paquete pequeño que se escribe pensando en el export desde el primer módulo |

## Decisión

**Paquete propio `ttspro`**, escrito con el export como restricción de diseño: cada bloque del
modelo se prueba con `torch.onnx.export` en cuanto existe, no al final. La referencia se cita, no
se copia: la arquitectura se reimplementa siguiendo el paper y el código MIT, y cada desviación
(qué se quitó, qué se cambió) va en el docstring del módulo.

Lo que se toma de coqui es **la receta y los números**: hiperparámetros de YourTTS, el conjunto de
datos con el que se entrenó, sus métricas publicadas como baseline del criterio de terminado.

## Consecuencias

- Más trabajo al principio: no hay zoo de recetas ni de pesos. Se paga una vez.
- El export es nuestro de punta a punta: cuando falle, el fallo está en un módulo que se escribió
  aquí, y `gb show` lo trae con su estado.
- Sin `monotonic_align` en Cython: MAS se implementa en `numpy`/`numba` opcional y solo en
  entrenamiento (regla 7 lo mantiene fuera de la inferencia).
- Se pierde la compatibilidad con checkpoints de coqui. Se acepta: convertirlos sería mantener dos
  formatos de peso y no es el objetivo.
