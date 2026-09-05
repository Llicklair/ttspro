/**
 * What this machine can run, and what it should run (ADR 0011).
 *
 * Two different things, kept apart on purpose:
 *
 *   - `detectar()` READS: is there WebGPU, which adapter, does it do shader-f16,
 *     how many threads, is the page cross-origin isolated. No inference.
 *   - `recomendar(info)` DEDUCES a provider and precision from that. It is a
 *     rule, so it can be wrong on hardware nobody measured.
 *   - `calibrar(...)` MEASURES: runs the same sentence through the same graph on
 *     each candidate and picks the fastest. It is the only one of the three that
 *     is true on the GPU in front of it, which is why the page offers it.
 *
 * The rule of thumb behind `recomendar`: WebGPU with shader-f16 (RTX 20xx and
 * later, Apple Silicon, RDNA2+) -> webgpu fp16; WebGPU without it (Pascal such
 * as a GTX 1070) -> webgpu fp32, because fp16 there produces NaN; no WebGPU ->
 * wasm fp32, where fp16 is slower than fp32.
 */

import { type Proveedor, type Sesion, crearSesion } from "./ort.ts";

export interface Hardware {
  webgpu: boolean;
  /** Adapter description as the browser reports it; empty when it will not say. */
  adaptador: { vendor: string; architecture: string; device: string; description: string };
  f16: boolean;
  hilos: number;
  aislado: boolean;
  /** Approximate device memory in GB when the browser exposes it (Chrome only). */
  memoriaGB: number | null;
}

export interface Recomendacion {
  proveedor: Proveedor;
  precision: "" | ".fp16";
  motivo: string;
}

export async function detectar(): Promise<Hardware> {
  const nav = navigator as Navigator & {
    gpu?: {
      requestAdapter(): Promise<
        | (GPUAdapterLike & {
            requestAdapterInfo?: () => Promise<Record<string, string>>;
            info?: Record<string, string>;
          })
        | null
      >;
    };
    deviceMemory?: number;
  };
  const salida: Hardware = {
    webgpu: false,
    adaptador: { vendor: "", architecture: "", device: "", description: "" },
    f16: false,
    hilos: navigator.hardwareConcurrency || 1,
    aislado: typeof crossOriginIsolated !== "undefined" && crossOriginIsolated,
    memoriaGB: nav.deviceMemory ?? null,
  };
  if (!nav.gpu) return salida;
  try {
    const adaptador = await nav.gpu.requestAdapter();
    if (!adaptador) return salida;
    salida.webgpu = true;
    salida.f16 = adaptador.features.has("shader-f16");
    const info = adaptador.info ?? (await adaptador.requestAdapterInfo?.()) ?? {};
    salida.adaptador = {
      vendor: info.vendor ?? "",
      architecture: info.architecture ?? "",
      device: info.device ?? "",
      description: info.description ?? "",
    };
  } catch {
    // an adapter that refuses to answer is reported as "no WebGPU", not as a crash
  }
  return salida;
}

interface GPUAdapterLike {
  features: { has(f: string): boolean };
}

export function recomendar(h: Hardware): Recomendacion {
  if (h.webgpu && h.f16) {
    return {
      proveedor: "webgpu",
      precision: ".fp16",
      motivo: "WebGPU con shader-f16: la mitad de descarga y el conversor en la GPU",
    };
  }
  if (h.webgpu) {
    return {
      proveedor: "webgpu",
      precision: "",
      motivo: "WebGPU sin shader-f16 (Pascal o similar): fp32 en la GPU, fp16 daría NaN",
    };
  }
  const hilos = h.aislado ? `${Math.min(4, h.hilos)} hilos` : "un hilo, sin aislamiento COOP/COEP";
  return {
    proveedor: "wasm",
    precision: "",
    motivo: `sin WebGPU: wasm fp32 con ${hilos}; fp16 en wasm es más lento que fp32`,
  };
}

export interface Medida {
  proveedor: Proveedor;
  precision: "" | ".fp16";
  /** ms of one run of the graph after a warm-up run; null when it failed or gave NaN. */
  ms: number | null;
  error?: string;
}

/**
 * Time one graph on each candidate. `correr` receives the session and must run
 * ONE synthesis and return its output; the caller decides which graph and which
 * inputs, so the same function times the TTS and the converter.
 */
export async function calibrar(
  url: (precision: "" | ".fp16") => string,
  correr: (s: Sesion) => Promise<Float32Array>,
  candidatos: Array<[Proveedor, "" | ".fp16"]>,
  alProgresar?: (m: Medida) => void,
): Promise<{ medidas: Medida[]; mejor: Medida | null }> {
  const medidas: Medida[] = [];
  for (const [proveedor, precision] of candidatos) {
    const medida: Medida = { proveedor, precision, ms: null };
    try {
      const sesion = await crearSesion(url(precision), [proveedor]);
      if (sesion.proveedor !== proveedor) {
        medida.error = `cayó a ${sesion.proveedor}`;
      } else {
        await correr(sesion); // warm-up: the first run compiles shaders / allocates
        const t0 = performance.now();
        const salida = await correr(sesion);
        const ms = performance.now() - t0;
        if (salida.some((x) => !Number.isFinite(x))) medida.error = "NaN en la salida";
        else medida.ms = ms;
      }
      await sesion.sesion.release?.();
    } catch (err) {
      medida.error = String(err);
    }
    medidas.push(medida);
    alProgresar?.(medida);
  }
  const validas = medidas.filter((m) => m.ms !== null) as Array<Medida & { ms: number }>;
  const mejor = validas.length ? validas.reduce((a, b) => (b.ms < a.ms ? b : a)) : null;
  return { medidas, mejor };
}

export function describir(h: Hardware): string {
  const gpu = h.webgpu
    ? `WebGPU: ${[h.adaptador.description, h.adaptador.device, h.adaptador.vendor, h.adaptador.architecture].filter(Boolean).join(" · ") || "adaptador sin nombre"} · shader-f16 ${h.f16 ? "sí" : "no"}`
    : "sin WebGPU";
  return `${gpu} · ${h.hilos} hilos · aislado ${h.aislado ? "sí" : "no"}${h.memoriaGB ? ` · ${h.memoriaGB} GB` : ""}`;
}
