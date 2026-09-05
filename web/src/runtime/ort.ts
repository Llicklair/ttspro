/**
 * ONNX Runtime Web sessions: WebGPU first, wasm as the guaranteed floor
 * (ADR 0005). Which provider actually ran is returned, never guessed.
 */

import * as ort from "onnxruntime-web";

export type Proveedor = "webgpu" | "wasm";

let configurado = false;
function configurar(): void {
  if (configurado) return;
  // No `wasmPaths`: ORT resolves its .mjs/.wasm next to its own module via
  // `import.meta.url`, which Vite serves in dev and emits as assets in build.
  // Pointing at /public/ort/ breaks in dev (Vite refuses JS imports from public/).
  // Threads only work under cross-origin isolation (COOP/COEP); otherwise ORT
  // would silently run single-threaded — say so instead.
  // `?ortlog=verbose` in the page URL prints node placement (which ops fell back to CPU).
  const params = new URLSearchParams(globalThis.location?.search ?? "");
  if (params.get("ortlog") === "verbose") ort.env.logLevel = "verbose";
  const aislado = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  ort.env.wasm.numThreads = aislado ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  configurado = true;
}

export interface Sesion {
  sesion: ort.InferenceSession;
  proveedor: Proveedor;
  ms_carga: number;
  bytes: number;
}

/** Bytes so far and total (0 when the server sends no length). */
export type AlDescargar = (recibidos: number, total: number) => void;

async function descargar(url: string, alDescargar?: AlDescargar): Promise<Uint8Array> {
  const respuesta = await fetch(url);
  if (!respuesta.ok) throw new Error(`no se pudo descargar ${url}: ${respuesta.status}`);
  const total = Number(respuesta.headers.get("content-length") ?? 0);
  if (!alDescargar || !respuesta.body) return new Uint8Array(await respuesta.arrayBuffer());
  // Read the stream so the page can show real progress: these models are tens of
  // megabytes and a frozen button for twenty seconds looks like a crash.
  const lector = respuesta.body.getReader();
  const trozos: Uint8Array[] = [];
  let recibidos = 0;
  while (true) {
    const { done, value } = await lector.read();
    if (done) break;
    trozos.push(value);
    recibidos += value.length;
    alDescargar(recibidos, total);
  }
  const bytes = new Uint8Array(recibidos);
  let pos = 0;
  for (const t of trozos) {
    bytes.set(t, pos);
    pos += t.length;
  }
  return bytes;
}

export async function crearSesion(
  url: string,
  preferidos: Proveedor[] = ["webgpu", "wasm"],
  alDescargar?: AlDescargar,
): Promise<Sesion> {
  configurar();
  const t0 = performance.now();
  const bytes = await descargar(url, alDescargar);
  const errores: string[] = [];
  for (const proveedor of preferidos) {
    if (proveedor === "webgpu" && !("gpu" in navigator)) {
      errores.push("webgpu: navigator.gpu no existe");
      continue;
    }
    try {
      const sesion = await ort.InferenceSession.create(bytes, {
        executionProviders: [proveedor],
        graphOptimizationLevel: "all",
        ...(ort.env.logLevel === "verbose" ? { logSeverityLevel: 0, logVerbosityLevel: 0 } : {}),
      });
      return { sesion, proveedor, ms_carga: performance.now() - t0, bytes: bytes.byteLength };
    } catch (e) {
      errores.push(`${proveedor}: ${String(e).slice(0, 200)}`);
    }
  }
  throw new Error(`ningún proveedor pudo cargar ${url}:\n${errores.join("\n")}`);
}

/** Does this WebGPU adapter run half precision in shaders? Without `shader-f16`
 * the fp16 graphs return NaN (measured 2026-09-04 on a GTX 1070, Pascal). */
export async function soportaF16(): Promise<boolean> {
  const gpu = (
    navigator as unknown as {
      gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> };
    }
  ).gpu;
  if (!gpu) return false;
  try {
    const adaptador = await gpu.requestAdapter();
    return !!adaptador && adaptador.features.has("shader-f16");
  } catch {
    return false;
  }
}

export function hilos(): number {
  configurar();
  return ort.env.wasm.numThreads ?? 1;
}

export { ort };
