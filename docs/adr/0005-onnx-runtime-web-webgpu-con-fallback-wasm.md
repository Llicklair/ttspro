# 5. ONNX Runtime Web: WebGPU primero, WASM como suelo garantizado

**Estado:** aceptada · **Fecha:** 2026-09-04 · **Regla:** [ARCHITECTURE 4 y 8](../../ARCHITECTURE.md)

## Contexto

El runtime en el navegador está fijado por el objetivo (ONNX Runtime Web). Lo que hay que decidir
es contra qué *execution provider* se diseña el modelo, porque los dos que ofrece no ejecutan el
mismo conjunto de ops ni con el mismo coste:

| EP | Velocidad | Cobertura de ops | Disponibilidad |
|---|---|---|---|
| `webgpu` | La que hace posible RTF < 1 | Parcial: cada versión de ORT Web añade ops; las que faltan caen a CPU con copia de memoria de por medio | Chrome/Edge sí; Firefox y Safari a medias en 2026 |
| `wasm` (SIMD + hilos) | 5–20× más lento que WebGPU en convoluciones grandes | Prácticamente completa | Todos los navegadores; los hilos exigen *cross-origin isolation* (cabeceras COOP/COEP) |

Un modelo que use una op que WebGPU no cubre no falla: **se vuelve lento en silencio** por el
fallback op a op. Eso es lo que hay que evitar por diseño, no por perfilado.

## Decisión

- El modelo se diseña **para la intersección**: solo ops que ejecutan los dos EP, opset 17. El
  test de export carga el grafo con `onnxruntime` y comprueba la lista de ops contra la que
  declara ORT Web para la versión pinneada en `web/package.json`. Op fuera de la lista → test rojo.
- Pesos en **fp16** en los `.onnx` publicados (mitad de descarga); la paridad PyTorch ↔ ORT del
  criterio de terminado se mide sobre el fp16, que es lo que corre.
- El runtime pide `webgpu` y cae a `wasm` **avisando** (la consola del navegador y el objeto de
  sesión dicen qué EP tocó). El presupuesto de la regla 8 se mide en `wasm` porque es el suelo que
  siempre existe; WebGPU es la mejora, no la garantía.
- Hilos WASM sí: el demo se sirve con COOP/COEP. Está documentado en `web/README.md` porque es la
  causa número uno de "en local va y en producción no".

## Consecuencias

- Nada de `STFT`, `RandomNormal`, ops de `com.microsoft` ni bucles `Loop`/`Scan` en el grafo
  (reglas 1, 4 y 5). Lo que necesite algo así se expresa con `Conv`, `MatMul` y entradas.
- Actualizar `onnxruntime-web` puede ampliar la lista de ops: es una mejora que se mide y se anota
  en evidencia, nunca una suposición.
- Playwright headless ejecuta `wasm` pero WebGPU headless es inestable: el criterio de terminado
  mide `wasm`; el RTF en WebGPU se mide a mano y se anota en evidencia con navegador y GPU.
- Presupuesto de descarga (≤ 80 MB) contando ORT Web (~10 MB de wasm), espeak-ng WASM (~2 MB) y
  los dos grafos: el sintetizador tiene ~60 MB en fp16, lo que acota el modelo a unos 30 M
  parámetros. YourTTS base cabe; una versión más grande necesita ADR.
