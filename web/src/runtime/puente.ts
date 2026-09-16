/**
 * Clonar una voz en el navegador: grabación -> embedding -> estilo, en un forward.
 * Espejo de `src/ttspro/supertonic/puente.py` (la parte de inferencia).
 *
 * Supertonic no publicó el encoder que convertía un audio en `style_ttl`, así que
 * el puente lo reconstruye: WeSpeaker embebe la grabación en 256 números, y una
 * matriz ajustada offline los lleva al espacio de estilos (ADR 0012, decisión 6).
 * Aquí eso es una multiplicación — 3,4 MB de pesos y unos milisegundos — mientras
 * que lo caro (muestrear el modelo para ajustar esa matriz) pasó una sola vez, en
 * Python, y para siempre.
 *
 * El coste de clonar se paga UNA vez, no en cada frase: es la diferencia con el
 * conversor que esto sustituye, que costaba 2,9 s por frase.
 */

import { type Sesion, crearSesion, ort } from "./ort.ts";
import type { Estilo } from "./supertonic.ts";

interface Pieza {
  forma: number[];
  desde: number;
  n: number;
}

interface Cabecera {
  piezas: Record<"media_estilo" | "componentes" | "media_emb" | "pesos", Pieza>;
  floats: number;
  error: Record<string, unknown>;
}

interface Cargado {
  cabecera: Cabecera;
  datos: Float32Array;
  encoder: Sesion;
}

let cargado: Cargado | null = null;

/**
 * ¿Está el puente publicado en este origen? Sin él, la página lo dice en vez de
 * ofrecer un botón que no puede funcionar.
 *
 * No vale con mirar el código de estado: el servidor de desarrollo de Vite
 * responde **200 con el index.html** a cualquier ruta que no conoce, así que un
 * `HEAD` a un fichero inexistente sale "ok" y lo siguiente que pasa es que
 * `r.json()` se atraganta con `<!doctype` (medido el 2026-09-16). Hay que abrir
 * la cabecera y comprobar que dice lo que tiene que decir.
 */
export async function puenteDisponible(raiz: string): Promise<boolean> {
  try {
    const r = await fetch(`${raiz}puente.json`);
    if (!r.ok) return false;
    const c = (await r.json()) as Partial<Cabecera>;
    return Boolean(c?.piezas?.componentes && c?.floats);
  } catch {
    return false;
  }
}

async function cargar(raiz: string): Promise<Cargado> {
  if (cargado) return cargado;
  const [cabecera, blob, encoder] = await Promise.all([
    fetch(`${raiz}puente.json`).then((r) => r.json() as Promise<Cabecera>),
    fetch(`${raiz}puente.bin`).then((r) => r.arrayBuffer()),
    crearSesion(`${raiz}speaker_encoder.onnx`, ["wasm"]),
  ]);
  const datos = new Float32Array(blob);
  if (datos.length !== cabecera.floats) {
    throw new Error(`puente.bin trae ${datos.length} floats y la cabecera dice ${cabecera.floats}`);
  }
  cargado = { cabecera, datos, encoder };
  return cargado;
}

const trozo = (c: Cargado, nombre: keyof Cabecera["piezas"]): Float32Array => {
  const p = c.cabecera.piezas[nombre];
  return c.datos.subarray(p.desde, p.desde + p.n);
};

/** A 16 kHz mono, que es lo que el contrato del encoder pide. */
async function a16k(onda: Float32Array, sampleRate: number): Promise<Float32Array> {
  if (sampleRate === 16000) return onda;
  const destino = new OfflineAudioContext(1, Math.ceil((onda.length * 16000) / sampleRate), 16000);
  const buffer = new AudioContext({ sampleRate }).createBuffer(1, onda.length, sampleRate);
  // El tipado de lib.dom pide Float32Array<ArrayBuffer> y aqui llega el ancho
  // (puede venir de un SharedArrayBuffer bajo COOP/COEP); los bytes son los mismos.
  buffer.copyToChannel(onda as Float32Array<ArrayBuffer>, 0);
  const fuente = destino.createBufferSource();
  fuente.buffer = buffer;
  fuente.connect(destino.destination);
  fuente.start();
  return (await destino.startRendering()).getChannelData(0);
}

/** Grabación -> voz. El embedding sale normalizado L2, así que no hay que tocarlo. */
export async function construirVoz(
  raiz: string,
  onda: Float32Array,
  sampleRate: number,
  nombre: string,
): Promise<Estilo> {
  const c = await cargar(raiz);
  let corta = await a16k(onda, sampleRate);
  if (corta.length < 16000) {
    const relleno = new Float32Array(16000);
    relleno.set(corta);
    corta = relleno;
  }
  const salida = await c.encoder.sesion.run({
    onda: new ort.Tensor("float32", corta, [1, corta.length]),
  });
  const embedding = salida[c.encoder.sesion.outputNames[0]].data as Float32Array;

  const mediaEmb = trozo(c, "media_emb");
  const pesos = trozo(c, "pesos");
  const componentes = trozo(c, "componentes");
  const mediaEstilo = trozo(c, "media_estilo");
  const [filas, nComp] = c.cabecera.piezas.pesos.forma;
  const dimEstilo = c.cabecera.piezas.media_estilo.forma[0];

  // x = [embedding - media, 1];  coef = x @ pesos;  vector = media + coef @ componentes
  const x = new Float32Array(filas);
  for (let i = 0; i < filas - 1; i++) x[i] = embedding[i] - mediaEmb[i];
  x[filas - 1] = 1;
  const coef = new Float32Array(nComp);
  for (let i = 0; i < filas; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    for (let j = 0; j < nComp; j++) coef[j] += xi * pesos[i * nComp + j];
  }
  const vector = Float32Array.from(mediaEstilo);
  for (let j = 0; j < nComp; j++) {
    const cj = coef[j];
    if (cj === 0) continue;
    const fila = j * dimEstilo;
    for (let k = 0; k < dimEstilo; k++) vector[k] += cj * componentes[fila + k];
  }

  const corte = 50 * 256;
  return {
    nombre,
    ttl: vector.slice(0, corte),
    dp: vector.slice(corte, corte + 8 * 16),
    metadatos: { construida_por: "puente", ...c.cabecera.error },
  };
}
