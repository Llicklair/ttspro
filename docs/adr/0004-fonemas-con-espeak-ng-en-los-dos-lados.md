# 4. Fonemas con espeak-ng en los dos lados, y la paridad la vigila un test

**Estado:** aceptada · **Fecha:** 2026-09-04 · **Regla:** [ARCHITECTURE 3](../../ARCHITECTURE.md)

## Contexto

El texto tiene que llegar al modelo como una secuencia de ids, y la misma frase tiene que dar los
**mismos ids** en Python (entrenamiento) y en el navegador (inferencia). Si difieren, el modelo
recibe en producción algo que nunca vio y la calidad cae sin que ningún test lo diga.

Tres formas de representar el texto:

| Entrada | A favor | En contra |
|---|---|---|
| Caracteres | Sin dependencias; paridad trivial | Necesita más datos, pronuncia peor lo que no ha visto, y en multilingüe mezcla ortografías con sonidos distintos para la misma letra |
| **Fonemas por espeak-ng** | Es lo que usaba YourTTS/coqui; cubre es, en y ~100 idiomas más; determinista | Dependencia externa; en el navegador hace falta la versión WASM; la salida cambia entre versiones de espeak-ng |
| G2P neuronal exportado a ONNX | Todo en el grafo | Un tercer modelo que descargar y entrenar; para es/en no aporta sobre espeak-ng |

## Decisión

**Fonemas IPA de espeak-ng en los dos lados.** En Python, el binario `espeak-ng` instalado por su
canal oficial (regla 11), detectado en ejecución. En `web/`, espeak-ng compilado a WASM como
dependencia npm **pinneada a la misma versión** que el binario. Sobre los fonemas, el mismo
tokenizador (tabla de símbolos en `models/contrato.json`) en ambos lados.

La paridad **no se supone: se comprueba**. `tests/test_frontend_paridad.py` pasa las frases de
`tests/fixtures/frontend/` por `ttspro.frontend` y por `node web/src/frontend/cli.ts` y compara
la salida. Cualquier regla nueva de normalización (números, abreviaturas, símbolos) se escribe dos
veces y el test la aprueba o la tumba.

Si espeak-ng no está en Python, el frontend **lo dice y falla**; no degrada a caracteres en
silencio, porque un modelo entrenado con fonemas y alimentado con letras produce audio que "casi"
funciona, que es la peor clase de fallo.

## Enmienda 2026-09-04: por qué la CLI y no la API, y qué paquete

Medido el mismo día ([evidencia](../evidencia.md)), tres cosas que la decisión de arriba no
sabía:

1. **La API C no vale.** `espeak_TextToPhonemes` omite la fase de entonación y devuelve acentos
   distintos a los de la CLI en palabras función (`ˈænd` frente a `ænd`, 3 de 24 frases). El WASM
   solo ofrece la CLI, así que Python usa **el ejecutable** también, por `--stdin` y en lotes.
2. **`-f fichero` está prohibido.** En Windows añade una cláusula de basura de forma no
   determinista (14 de 20). Texto por stdin o por argumento: 0 de 20.
3. **El paquete es `espeak-ng` 1.0.2** (npm, ianmarmour): 18,5 MB, GPL-3.0, todos los idiomas,
   fonemas idénticos al binario 1.52.0 en 24/24. `phonemizer` (2,6 MB, Apache) quedó descartado
   por llevar solo inglés.

El protocolo compartido queda en `models/contrato.json` (`fonemizador`): un trozo por línea con
" ." añadido, una línea de salida por trozo, y error si no cuadra.

## Consecuencias

- espeak-ng es GPL-3. El texto fonémico que produce no es obra derivada, así que los pesos no
  heredan la licencia. El bundle web **sí enlaza** el WASM: la licencia del paquete `web/` tiene
  que ser compatible o el WASM cargarse como módulo aparte. Esto se resuelve en el ADR de
  publicación, antes de publicar; hasta entonces no bloquea nada.
- Un cambio de versión de espeak-ng obliga a reentrenar o, como mínimo, a rehacer los fixtures y
  medir la calidad. El pin en `pyproject.toml` y en `web/package.json` lleva un comentario que lo
  dice.
- Idiomas nuevos: espeak-ng cubre muchos, pero cubrir no es sonar bien. Un idioma entra al
  conjunto entrenado con datos y con una medición, no porque el fonemizador lo liste.
