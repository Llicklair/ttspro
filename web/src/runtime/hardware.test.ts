import { describe, expect, it } from "vitest";
import { type Hardware, describir, recomendar } from "./hardware.ts";

const base: Hardware = {
  webgpu: false,
  adaptador: { vendor: "", architecture: "", device: "", description: "" },
  f16: false,
  hilos: 8,
  aislado: true,
  memoriaGB: 16,
};

describe("recomendar: la regla que deduce proveedor y precisión del hardware", () => {
  it("WebGPU con shader-f16 (RTX 20xx+, Apple Silicon) -> webgpu fp16", () => {
    const r = recomendar({ ...base, webgpu: true, f16: true });
    expect([r.proveedor, r.precision]).toEqual(["webgpu", ".fp16"]);
  });

  it("WebGPU sin shader-f16 (una GTX 1070) -> webgpu fp32, nunca fp16", () => {
    const r = recomendar({ ...base, webgpu: true, f16: false });
    expect([r.proveedor, r.precision]).toEqual(["webgpu", ""]);
    expect(r.motivo).toContain("NaN");
  });

  it("sin WebGPU -> wasm fp32, y dice con cuántos hilos", () => {
    expect(recomendar(base).motivo).toContain("4 hilos");
    expect(recomendar({ ...base, aislado: false }).motivo).toContain("un hilo");
    expect(recomendar({ ...base, hilos: 2 }).motivo).toContain("2 hilos");
  });

  it("describir nombra la GPU cuando el navegador la nombra", () => {
    expect(
      describir({
        ...base,
        webgpu: true,
        adaptador: {
          vendor: "nvidia",
          architecture: "pascal",
          device: "",
          description: "GTX 1070",
        },
      }),
    ).toContain("GTX 1070");
    expect(describir(base)).toContain("sin WebGPU");
  });
});
