/**
 * espeak-ng, cargado desde el CDN en vez de empaquetado.
 *
 * El build sustituye el paquete `espeak-ng` por este modulo (alias en
 * `vite.config.ts` y `vite.lib.config.ts`). El motivo no es el tamano: es que
 * espeak-ng es **GPL-3.0-or-later**, y la obligacion de esa licencia la contrae
 * quien **distribuye** el binario, no quien lo ejecuta. Un build que lo lleva
 * dentro lo reparte a cada visitante; uno que lo pide a jsDelivr no reparte
 * nada, y quien lo reparte es el CDN, que ya lo hacia.
 *
 * Aqui no se pierde nada, porque **este camino no se usa**: `pocket-tts-onnx`
 * solo llama a espeak cuando una linea mezcla hebreo con letras latinas, y el
 * paquete espanol ni siquiera trae el adaptador de fonemas que lo activaria. Lo
 * que se quitaba de en medio eran 18,5 MB de wasm y 68 KB de pegamento que
 * nadie descargaba nunca.
 *
 * Si el CDN no responde, falla **el hebreo mezclado** y nada mas. Se dice por
 * que, en vez de dejar un error de modulo sin resolver.
 */

const VERSION = "1.0.2";
const CDN = `https://cdn.jsdelivr.net/npm/espeak-ng@${VERSION}/dist/espeak-ng.js`;

type Fabrica = (opciones?: unknown) => Promise<unknown>;

let pedido: Promise<Fabrica> | null = null;

function traer(): Promise<Fabrica> {
  // `@vite-ignore` porque la URL es externa: sin esto Vite intenta resolverla en
  // el build, que es justo lo que se quiere evitar.
  pedido ??= import(/* @vite-ignore */ CDN)
    .then((m) => (m.default ?? m) as Fabrica)
    .catch((e) => {
      pedido = null;
      throw new Error(
        `no se pudo cargar espeak-ng desde ${CDN}: ${String(e)}. Solo hace falta para texto que mezcla hebreo con letras latinas; el resto no lo usa.`,
      );
    });
  return pedido;
}

export default async function espeak(opciones?: unknown): Promise<unknown> {
  return (await traer())(opciones);
}
