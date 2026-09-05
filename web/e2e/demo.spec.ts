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
const FRASE = "Hola, hoy hace un día muy bueno para salir a pasear un rato.";
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

  await page.waitForFunction(() => document.querySelectorAll(".voz").length > 0, null, {
    timeout: 120_000,
  });
  await page.click('[data-voz="0"]');
  await expect(page.locator("#infoVoz")).toContainText("preset");
  await expect(page.locator('[data-voz="0"]')).toHaveAttribute("aria-checked", "true");

  await page.setInputFiles("#fichero", REFERENCIA);
  await expect(page.locator("#infoVoz")).toContainText("voz clonada", { timeout: 120_000 });

  await page.fill("#texto", FRASE);
  await page.selectOption("#idioma", "es");
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
  await page.screenshot({ path: resolve(AQUI, "../test-results/demo.png"), fullPage: true });
  writeFileSync(resolve(AQUI, "../test-results/criterio5.json"), JSON.stringify(medido, null, 1));
  console.log(JSON.stringify(medido));
});

// ---------------------------------------------------------------- chat mode (ADR 0010)
//
// A burst of twenty chat lines through the normalizer and the queue, base voice,
// wasm. What matters is not one latency but whether the reader keeps up: seconds
// of audio produced per second of synthesis, and how many lines it had to drop.

test("modo chat: veinte mensajes seguidos por la cola, voz base, wasm", async ({ page }) => {
  page.on("pageerror", (e) => console.log("pageerror:", e.message));
  await page.goto("/");
  await page.selectOption("#precision", PRECISION);
  await page.selectOption("#proveedor", "wasm");
  await page.click("#cargar");
  await page.waitForFunction(() => document.querySelectorAll(".voz").length > 0, null, {
    timeout: 240_000,
  });
  await page.selectOption("#politica", "base");
  const t0 = Date.now();
  await page.click("#simular");
  // 20 lines: 19 distinct plus one exact repeat, which the queue must refuse.
  await page.waitForFunction(
    () => {
      const s = (
        window as unknown as { __ttspro_chat: { estadisticas: () => Record<string, number> } }
      ).__ttspro_chat.estadisticas();
      return (
        s.pendientes === 0 &&
        s.reproducidos + s.descartadosViejos + s.descartadosLlenos + s.descartadosRepetidos >= 18
      );
    },
    null,
    { timeout: 300_000 },
  );
  // let the last one finish playing
  await page.waitForFunction(
    () => document.getElementById("pastilla-chat")?.dataset.estado !== "trabajando",
    null,
    { timeout: 60_000 },
  );
  const segundosTotales = (Date.now() - t0) / 1000;
  const stats = await page.evaluate(() =>
    (
      window as unknown as { __ttspro_chat: { estadisticas: () => Record<string, number> } }
    ).__ttspro_chat.estadisticas(),
  );
  const medido = { precision: PRECISION || "fp32", proveedor: "wasm", segundosTotales, ...stats };
  console.log(JSON.stringify(medido));
  mkdirSync(resolve(AQUI, "../test-results"), { recursive: true });
  writeFileSync(resolve(AQUI, "../test-results/chat.json"), JSON.stringify(medido, null, 1));
  expect(stats.reproducidos).toBeGreaterThanOrEqual(15);
  expect(stats.descartadosRepetidos).toBe(1);
  // The queue must at least keep up with itself: audio out faster than synthesis in.
  expect(stats.velocidad).toBeGreaterThan(1);
});

// ---------------------------------------------------------------- another app as the source
//
// What an application like streex would do: stream chat lines to the page. Here a
// throwaway SSE server stands in for it, on another port and therefore another
// origin, which is exactly the CORS/COEP situation a Rails app is in.

import { createServer } from "node:http";

test("fuente externa: un servidor SSE en otro origen alimenta la cola", async ({ page }) => {
  const lineas = [
    { usuario: "streex_bot", texto: "hola desde la otra aplicación" },
    { user: "Ana", text: "esto viene por SSE, nice" },
    { display_name: "Pepe_Gamer", message: "Kappa" },
  ];
  const servidor = createServer((req, res) => {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      // the page is cross-origin isolated: an EventSource from it is a CORS request
      "Access-Control-Allow-Origin": "*",
    });
    let i = 0;
    const tic = setInterval(() => {
      if (i >= lineas.length) return clearInterval(tic);
      res.write(`data: ${JSON.stringify(lineas[i++])}\n\n`);
    }, 200);
    req.on("close", () => clearInterval(tic));
  });
  await new Promise<void>((r) => servidor.listen(0, "127.0.0.1", r));
  const puerto = (servidor.address() as { port: number }).port;

  try {
    await page.goto("/");
    await page.selectOption("#precision", PRECISION);
    await page.selectOption("#proveedor", "wasm");
    await page.click("#cargar");
    await page.waitForFunction(() => document.querySelectorAll(".voz").length > 0, null, {
      timeout: 240_000,
    });
    await page.selectOption("#fuente", "sse");
    await page.fill("#url", `http://127.0.0.1:${puerto}/eventos`);
    await page.click("#conectar");
    await expect(page.locator("#pastilla-chat")).toHaveText("escuchando", { timeout: 10_000 });
    // two readable lines and one emote-only (skipped before the queue)
    await page.waitForFunction(
      () =>
        (
          window as unknown as { __ttspro_chat: { estadisticas: () => { reproducidos: number } } }
        ).__ttspro_chat.estadisticas().reproducidos >= 2,
      null,
      { timeout: 120_000 },
    );
    const registro = await page.locator("#chatLog").innerText();
    expect(registro).toContain("streex_bot");
    expect(registro).toContain("nada que leer");
    await page.click("#conectar");
    await expect(page.locator("#conectar")).toHaveText("conectar");
  } finally {
    servidor.close();
  }
});
