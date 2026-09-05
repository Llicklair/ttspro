/**
 * Criterion 5 (SCOPE.md) on the real page, with the chain of ADR 0007:
 * text -> tts.onnx (base voice) -> conversor.onnx (chosen voice), wasm provider.
 * Exercises both ways of picking a voice (preset and cloned from a recording)
 * and writes test-results/criterio5.json with every measured number.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const AQUI = dirname(fileURLToPath(import.meta.url));
const REFERENCIA = process.env.TTSPRO_REFERENCIA ?? resolve(AQUI, "referencia.wav");
const FRASE = "The quick brown fox jumps over the lazy dog today.";
// "" = fp32 (what wasm should run: faster than fp16 there and no NaN)
const PRECISION = process.env.TTSPRO_PRECISION ?? "";

test("carga, elige voz, sintetiza y convierte en wasm", async ({ page }) => {
  const descargas: Record<string, number> = {};
  page.on("response", async (r) => {
    const url = new URL(r.url()).pathname;
    if (/\.(onnx|wasm)$/.test(url)) {
      const largo = Number(r.headers()["content-length"] ?? 0);
      descargas[url] = largo || (await r.body().catch(() => Buffer.alloc(0))).length;
    }
  });
  await page.goto("/");
  await page.selectOption("#precision", PRECISION);
  await page.selectOption("#proveedor", "wasm");
  await page.click("#cargar");
  await expect(page.locator("#estado")).toContainText("listo", { timeout: 240_000 });
  const estado = (await page.locator("#estado").textContent()) ?? "";
  expect(estado).toContain("aislado=true");

  await page.waitForFunction(() => document.querySelectorAll("#preset option").length > 1, null, {
    timeout: 120_000,
  });
  await page.selectOption("#preset", "0");
  await expect(page.locator("#infoVoz")).toContainText("preset");

  await page.setInputFiles("#fichero", REFERENCIA);
  await expect(page.locator("#infoVoz")).toContainText("voz clonada", { timeout: 120_000 });

  await page.fill("#texto", FRASE);
  await page.selectOption("#idioma", "en");
  const t0 = Date.now();
  await page.click("#sintetizar");
  await expect(page.locator("#medidas")).toContainText("RTF", { timeout: 180_000 });
  const ms_total_sintesis = Date.now() - t0;
  const r = (await page.evaluate(() => (window as unknown as { __ttspro: unknown }).__ttspro)) as {
    muestras: number;
    sampleRate: number;
    ms_modelo: number;
    ms_conversor: number;
    ms_frontend: number;
    proveedor: string;
    tokens: number;
    convertido: boolean;
  };
  expect(r.proveedor).toBe("wasm");
  expect(r.convertido).toBe(true);
  expect(r.ms_conversor).toBeGreaterThan(0);
  expect(r.muestras / r.sampleRate).toBeGreaterThan(1.5);

  const mb_descarga_total = Object.values(descargas).reduce((a, b) => a + b, 0) / 1e6;
  const medido = {
    frase: FRASE,
    precision: PRECISION || "fp32",
    palabras: FRASE.split(/\s+/).length,
    ms_total_sintesis,
    ms_modelo: r.ms_modelo,
    ms_conversor: r.ms_conversor,
    ms_frontend: r.ms_frontend,
    segundos_audio: r.muestras / r.sampleRate,
    rtf: (r.ms_modelo + r.ms_conversor) / 1000 / (r.muestras / r.sampleRate),
    proveedor: r.proveedor,
    tokens: r.tokens,
    estado,
    descargas_MB: Object.fromEntries(
      Object.entries(descargas).map(([k, v]) => [k, +(v / 1e6).toFixed(2)]),
    ),
    mb_descarga_total: +mb_descarga_total.toFixed(1),
  };
  mkdirSync(resolve(AQUI, "../test-results"), { recursive: true });
  writeFileSync(resolve(AQUI, "../test-results/criterio5.json"), JSON.stringify(medido, null, 1));
  console.log(JSON.stringify(medido));
});
