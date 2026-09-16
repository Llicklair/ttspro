/**
 * Que un build publicado **no reparta** espeak-ng, que es GPL-3.0-or-later.
 *
 * La obligacion de esa licencia la contrae quien **distribuye** el binario, no
 * quien lo ejecuta. Un build que lo lleva dentro se lo entrega a cada visitante
 * y con eso hereda la GPL: hay que dar el aviso, ofrecer el fuente de esa
 * version exacta, y queda abierta la pregunta incomoda de si la aplicacion y
 * espeak-ng cuentan como una sola obra combinada.
 *
 * Y lo repartia sin que nadie lo usara. `pocket-tts-onnx` solo llama a espeak
 * cuando una linea mezcla hebreo con letras latinas, y el paquete espanol ni
 * siquiera trae el adaptador de fonemas que activaria esa rama. Aun asi el build
 * emitia 18,5 MB de wasm y 68 KB de pegamento, porque dentro del codigo generado
 * por Emscripten hay un
 *
 *     wasmBinaryFile = new URL('espeak-ng.wasm', import.meta.url).href;
 *
 * —el camino de repuesto para cuando no te pasan los bytes, que aqui si te los
 * pasan— y Vite lee ese `new URL` como «emite este fichero».
 *
 * Asi que en produccion el paquete entero se sustituye por
 * `src/runtime/espeak-remoto.ts`, que lo pide a jsDelivr cuando de verdad haga
 * falta. El hebreo mezclado sigue funcionando; el build no reparte nada y quien
 * lo reparte es el CDN, que ya lo hacia. En `dev` y en los tests no se toca:
 * ahi no se distribuye a nadie.
 */

import { fileURLToPath } from "node:url";

const REMOTO = fileURLToPath(new URL("../src/runtime/espeak-remoto.ts", import.meta.url));

/** @returns {import("vite").Plugin} */
export function espeakFueraDelBuild() {
  return {
    name: "ttspro:espeak-fuera-del-build",
    apply: "build",
    enforce: "pre",
    config() {
      return { resolve: { alias: { "espeak-ng": REMOTO } } };
    },
  };
}
