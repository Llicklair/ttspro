import { describe, expect, it } from "vitest";
import { cambiarVelocidad } from "./velocidad.ts";

/** Un tono puro: si el estirador tocara el tono, aquí se vería. */
function tono(segundos: number, hz = 220, sampleRate = 24000): Float32Array {
  const n = Math.round(segundos * sampleRate);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin((2 * Math.PI * hz * i) / sampleRate) * 0.5;
  return x;
}

/** Cruces por cero por segundo: el doble de la frecuencia dominante. */
function cruces(x: Float32Array, sampleRate = 24000): number {
  let n = 0;
  for (let i = 1; i < x.length; i++) if (x[i - 1] < 0 !== x[i] < 0) n++;
  return (n * sampleRate) / x.length;
}

describe("cambiarVelocidad", () => {
  it("no toca nada con factor 1", () => {
    const x = tono(0.5);
    expect(cambiarVelocidad(x, 1)).toBe(x);
  });

  it("acorta al acelerar y alarga al frenar", () => {
    const x = tono(1);
    const rapido = cambiarVelocidad(x, 1.5);
    const lento = cambiarVelocidad(x, 0.75);
    // Con margen: WSOLA no clava la longitud al milisegundo.
    expect(rapido.length).toBeGreaterThan(x.length / 1.5 - 3000);
    expect(rapido.length).toBeLessThan(x.length / 1.5 + 3000);
    expect(lento.length).toBeGreaterThan(x.length / 0.75 - 3000);
  });

  it("mantiene el tono, que es lo que playbackRate rompe", () => {
    const x = tono(1, 220);
    const antes = cruces(x);
    for (const f of [0.75, 1.5]) {
      const y = cambiarVelocidad(x, f);
      // playbackRate movería esto de 440 a 330 o a 660; WSOLA lo deja donde está.
      expect(Math.abs(cruces(y) - antes) / antes).toBeLessThan(0.08);
    }
  });

  it("no mete silencios ni clips", () => {
    const y = cambiarVelocidad(tono(1), 1.3);
    let pico = 0;
    let suma = 0;
    for (const v of y) {
      pico = Math.max(pico, Math.abs(v));
      suma += v * v;
    }
    expect(pico).toBeLessThan(1);
    expect(Math.sqrt(suma / y.length)).toBeGreaterThan(0.2);
  });
});
