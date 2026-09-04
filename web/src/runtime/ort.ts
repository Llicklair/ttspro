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

export async function crearSesion(
  url: string,
  preferidos: Proveedor[] = ["webgpu", "wasm"],
): Promise<Sesion> {
  configurar();
  const t0 = performance.now();
  const respuesta = await fetch(url);
  if (!respuesta.ok) throw new Error(`no se pudo descargar ${url}: ${respuesta.status}`);
  const bytes = new Uint8Array(await respuesta.arrayBuffer());
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
      });
      return { sesion, proveedor, ms_carga: performance.now() - t0, bytes: bytes.byteLength };
    } catch (e) {
      errores.push(`${proveedor}: ${String(e).slice(0, 200)}`);
    }
  }
  throw new Error(`ningún proveedor pudo cargar ${url}:\n${errores.join("\n")}`);
}

export function hilos(): number {
  configurar();
  return ort.env.wasm.numThreads ?? 1;
}

export { ort };
