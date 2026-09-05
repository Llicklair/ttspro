/**
 * Criterion 5 (SCOPE.md) on the real page, with the chain of ADR 0007:
 * text -> tts.onnx (base voice) -> conversor.onnx (chosen voice), wasm provider.
 * Exercises both ways of picking a voice (preset and cloned from a recording)
 * and writes test-results/criterio5.json with every measured number.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
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

// ---------------------------------------------------------------- the library
//
// What Marcos described: open an HTML, import ttspro, and from the console call
// tts.clonar(file) and tts.predict(text | stream). The example page is served by
// scripts/servir.mjs (COOP/COEP, no Vite) on a random port, so this exercises the
// built bundle in dist/lib, not the dev sources.

// @ts-expect-error plain JS, no declaration: it is the same server `npm run servir` starts
import { servir } from "../scripts/servir.mjs";

test("librería: import ttspro.js, clonar desde un fichero, predict de texto y de stream", async ({
  page,
}) => {
  test.skip(!existsSync(resolve(AQUI, "../dist/lib/ttspro.js")), "run `npm run build:lib` first");
  const servidor = (await servir(0)) as { address: () => { port: number }; close: () => void };
  const base = `http://127.0.0.1:${servidor.address().port}`;
  try {
    page.on("pageerror", (e) => console.log("pageerror:", e.message));
    await page.goto(`${base}/`);
    await page.waitForFunction(() => Boolean((window as unknown as { tts?: unknown }).tts), null, {
      timeout: 240_000,
    });
    // the file input, as a person would use it
    await page.setInputFiles("#audio", {
      name: "referencia.wav",
      mimeType: "audio/wav",
      buffer: readFileSync(REFERENCIA),
    });
    const medido = await page.evaluate(async () => {
      type R = {
        texto: string;
        onda: Float32Array;
        sampleRate: number;
        ms: number;
        wav: () => Blob;
      };
      const w = window as unknown as {
        tts: {
          voces: unknown[];
          proveedores: Record<string, string>;
          clonar: (f: File) => Promise<Float32Array>;
          elegirVoz: (v: string | null) => Float32Array | null;
          predict: (e: unknown, o?: unknown) => Promise<R> & AsyncIterable<R>;
        };
        audio: HTMLInputElement;
        mensajes: () => AsyncIterable<unknown>;
      };
      const vector = await w.tts.clonar(w.audio.files?.[0] as File);
      const clonado = await w.tts.predict("hola, esta voz sale de un fichero");
      w.tts.elegirVoz(null);
      const base = await w.tts.predict("y esta es la voz base");
      const stream: Array<{ texto: string; segundos: number }> = [];
      for await (const r of w.tts.predict(w.mensajes(), { chat: true })) {
        stream.push({ texto: r.texto, segundos: r.onda.length / r.sampleRate });
      }
      return {
        voces: w.tts.voces.length,
        proveedores: w.tts.proveedores,
        vector: vector.length,
        clonado: {
          segundos: clonado.onda.length / clonado.sampleRate,
          ms: clonado.ms,
          wavBytes: clonado.wav().size,
        },
        base: { segundos: base.onda.length / base.sampleRate, ms: base.ms },
        stream,
      };
    });
    console.log(JSON.stringify(medido));
    expect(medido.voces).toBe(20);
    expect(medido.vector).toBe(256);
    expect(medido.clonado.segundos).toBeGreaterThan(1);
    expect(medido.clonado.wavBytes).toBeGreaterThan(44);
    expect(medido.base.segundos).toBeGreaterThan(1);
    // four items: three readable (one with a user, one bare) and "KEKW" -> "jajaja"
    expect(medido.stream.map((s) => s.texto)).toEqual([
      "Pepe Gamer: hola que tal todos!",
      "Ana: gracias por el estrim, nais",
      "Luci: jajaja",
      "y un texto suelto sin usuario",
    ]);
  } finally {
    servidor.close();
  }
});

// ---------------------------------------------------------------- voice packs (ADR 0011)
//
// The packs built by ttspro.export.paquete, served from a local folder on another
// origin (as the GitHub release will be): the page lists them, downloads one on
// demand, speaks with it, and "una voz base por usuario" spreads users over them.

function servirCarpeta(carpeta: string) {
  const servidor = createServer((req, res) => {
    const nombre = decodeURIComponent((req.url ?? "/").split("?")[0].slice(1));
    const ruta = resolve(carpeta, nombre);
    if (!nombre || !existsSync(ruta)) {
      res.writeHead(404, { "Access-Control-Allow-Origin": "*" }).end();
      return;
    }
    const datos = readFileSync(ruta);
    res.writeHead(200, {
      "Content-Type": nombre.endsWith(".json") ? "application/json" : "application/octet-stream",
      "Content-Length": datos.length,
      "Access-Control-Allow-Origin": "*",
    });
    res.end(datos);
  });
  return new Promise<typeof servidor>((r) => servidor.listen(0, "127.0.0.1", () => r(servidor)));
}

const PAQUETES = resolve(AQUI, "../../paquetes");

test("paquetes de voz: la página descarga una voz base y habla con ella", async ({ page }) => {
  test.skip(!existsSync(resolve(PAQUETES, "indice.json")), "run `ttspro.export.paquete` first");
  const servidor = await servirCarpeta(PAQUETES);
  const puerto = (servidor.address() as { port: number }).port;
  try {
    page.on("pageerror", (e) => console.log("pageerror:", e.message));
    await page.goto(`/?voces=http://127.0.0.1:${puerto}/`);
    await page.selectOption("#precision", PRECISION);
    await page.selectOption("#proveedor", "wasm");
    await page.click("#cargar");
    await page.waitForFunction(() => document.querySelectorAll(".voz").length > 0, null, {
      timeout: 240_000,
    });
    // the index arrived: the selector lists the packs
    await page.waitForFunction(
      () => document.querySelectorAll("#vozBase option").length > 1,
      null,
      {
        timeout: 30_000,
      },
    );
    const hardware = await page.locator("#hardware").innerText();
    expect(hardware).toContain("recomendado");
    await page.selectOption("#vozBase", "es_ES-carlfm-x_low");
    await page.click("#descargarVozBase");
    await expect(page.locator("#infoVozBase")).toContainText("carlfm", { timeout: 120_000 });
    await page.fill("#texto", "Esta frase la dice una voz base descargada.");
    await page.click("#sinConvertir");
    await page.waitForFunction(
      () => document.getElementById("medidas")?.textContent?.includes("RTF"),
      null,
      {
        timeout: 60_000,
      },
    );
    const medidas = await page.locator("#medidas").innerText();
    console.log("carlfm:", medidas);
    expect(medidas).toContain("sin convertir");
    // chat: one base voice per user, over the two loaded (local + carlfm)
    await page.selectOption("#politica", "baseUsuario");
    await page.click("#simular");
    await page.waitForFunction(
      () =>
        (
          window as unknown as { __ttspro_chat: { estadisticas: () => { reproducidos: number } } }
        ).__ttspro_chat.estadisticas().reproducidos >= 4,
      null,
      { timeout: 120_000 },
    );
  } finally {
    servidor.close();
  }
});

test("librería: paquetes de voz por URL, cargarVozBase y predict con vozBase", async ({ page }) => {
  test.skip(!existsSync(resolve(AQUI, "../dist/lib/ttspro.js")), "run `npm run build:lib` first");
  test.skip(!existsSync(resolve(PAQUETES, "indice.json")), "run `ttspro.export.paquete` first");
  const paquetes = await servirCarpeta(PAQUETES);
  const servidor = (await servir(0)) as { address: () => { port: number }; close: () => void };
  const base = `http://127.0.0.1:${servidor.address().port}`;
  const urlPaquetes = `http://127.0.0.1:${(paquetes.address() as { port: number }).port}/`;
  try {
    page.on("pageerror", (e) => console.log("pageerror:", e.message));
    await page.goto(`${base}/`);
    await page.waitForFunction(() => Boolean((window as unknown as { tts?: unknown }).tts), null, {
      timeout: 240_000,
    });
    const medido = await page.evaluate(async (url) => {
      const { TTS } = (await import("/ttspro.js")) as {
        TTS: {
          cargar: (o: unknown) => Promise<{
            hardware: { webgpu: boolean };
            recomendacion: { proveedor: string; precision: string };
            vocesBase: Record<string, { MB: { fp16: number } }>;
            cargarVozBase: (c: string) => Promise<{ clave: string; ms_frase_cpu: number }>;
            elegirVozBase: (c: string) => void;
            vocesBaseCargadas: string[];
            predict: (
              t: string,
              o?: unknown,
            ) => Promise<{ onda: Float32Array; sampleRate: number; ms: number }>;
          }>;
        };
      };
      const t = await TTS.cargar({
        modelos: "/models/",
        espeak: "/espeak/espeak-ng.wasm",
        proveedor: "wasm",
        precision: "",
        vocesBase: url,
      });
      const meta = await t.cargarVozBase("es_MX-claude-high");
      t.elegirVozBase("es_MX-claude-high");
      const claude = await t.predict("hola desde una voz base descargada");
      const local = await t.predict("y esta es la local", { vozBase: "local" });
      return {
        indice: Object.keys(t.vocesBase).length,
        meta,
        cargadas: t.vocesBaseCargadas,
        recomendacion: t.recomendacion,
        webgpu: t.hardware.webgpu,
        claude: { segundos: claude.onda.length / claude.sampleRate, ms: claude.ms },
        local: { segundos: local.onda.length / local.sampleRate, ms: local.ms },
      };
    }, urlPaquetes);
    console.log(JSON.stringify(medido));
    expect(medido.indice).toBeGreaterThanOrEqual(8);
    expect(medido.cargadas).toEqual(["local", "es_MX-claude-high"]);
    expect(medido.claude.segundos).toBeGreaterThan(1);
    expect(medido.local.segundos).toBeGreaterThan(0.5);
    expect(["webgpu", "wasm"]).toContain(medido.recomendacion.proveedor);
  } finally {
    paquetes.close();
    servidor.close();
  }
});
