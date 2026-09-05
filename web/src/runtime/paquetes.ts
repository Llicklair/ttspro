/**
 * Voice packs: base voices downloaded on demand from any static host (ADR 0011).
 *
 * A pack is `<clave>.json` (meta, the voice's own contract, its base-voice
 * vector) plus `<clave>.tts.onnx` / `<clave>.tts.fp16.onnx`; `indice.json` lists
 * them. Built by `ttspro.export.paquete`, hosted on GitHub Pages today (the
 * orphan branch `voces`), a Hugging Face repo tomorrow: the base URL is the only
 * thing that changes.
 *
 * Why a contract per voice: the symbol table differs between Piper voices, so a
 * loaded pack carries the table the frontend must tokenize with. The converter
 * needs the pack's base-voice vector as `voz_origen`, measured at build time.
 */

import type { Contrato } from "./contrato.ts";
import { type AlDescargar, type Proveedor, type Sesion, crearSesion } from "./ort.ts";

export interface MetaPaquete {
  clave: string;
  nombre: string;
  idioma: string;
  region: string;
  calidad: string;
  licencia: string;
  parametros_M: number;
  MB: { fp32: number; fp16: number };
  ms_frase_cpu: number;
  ficheros: { fp32: string; fp16: string };
}

export interface Paquete extends MetaPaquete {
  contrato: Pick<Contrato, "simbolos" | "idiomas" | "grafos">;
  voz_base: number[];
}

/** A pack that is ready to speak. */
export interface VozBase {
  meta: MetaPaquete;
  sesion: Sesion;
  contrato: Contrato;
  vozBase: Float32Array;
}

// Hugging Face by default (CORS along its whole redirect chain, measured, ADR 0011);
// GitHub Pages carries the same files as a mirror. Not the GitHub release: its
// assets send no Access-Control-Allow-Origin and a browser cannot fetch them.
export const URL_VOCES = "https://huggingface.co/Llicklair/ttspro-voces/resolve/main/";
export const URL_VOCES_PAGES = "https://llicklair.github.io/ttspro/";

const conBarra = (url: string) => url.replace(/\/?$/, "/");

export async function cargarIndice(base: string): Promise<Record<string, MetaPaquete>> {
  const r = await fetch(`${conBarra(base)}indice.json`);
  if (!r.ok) throw new Error(`no hay indice.json en ${base}: ${r.status}`);
  return (await r.json()) as Record<string, MetaPaquete>;
}

/**
 * Download one pack and open its graph. `plantilla` is the local contract: the
 * pack only brings what differs (symbols, languages, the tts signature), the
 * rest (converter, voice-extractor graphs) is shared.
 */
export async function cargarPaquete(
  base: string,
  clave: string,
  plantilla: Contrato,
  precision: "" | ".fp16",
  proveedores: Proveedor[],
  alDescargar?: AlDescargar,
): Promise<VozBase> {
  const raiz = conBarra(base);
  const r = await fetch(`${raiz}${clave}.json`);
  if (!r.ok) throw new Error(`no hay paquete ${clave} en ${base}: ${r.status}`);
  const p = (await r.json()) as Paquete;
  const fichero = precision === ".fp16" ? p.ficheros.fp16 : p.ficheros.fp32;
  const sesion = await crearSesion(`${raiz}${fichero}`, proveedores, alDescargar);
  const { contrato: propio, voz_base, ...meta } = p;
  const contrato: Contrato = {
    ...plantilla,
    simbolos: propio.simbolos,
    idiomas: propio.idiomas,
    grafos: { ...plantilla.grafos, tts: propio.grafos.tts },
  };
  return { meta, sesion, contrato, vozBase: Float32Array.from(voz_base) };
}
