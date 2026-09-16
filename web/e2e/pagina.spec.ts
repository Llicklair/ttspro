/**
 * La página nueva, en un navegador de verdad (ADR 0012).
 *
 * La promesa que se comprueba aquí es "abrirla y que hable": el motor se carga
 * solo, y si los grafos no están en este servidor se bajan de Hugging Face sin
 * que nadie pulse nada. Lo demás que se prueba es lo que puede romperse en
 * silencio — que el estilo llega al grafo y no solo a la etiqueta, y que lo que
 * no funciona (clonar) sale apagado y lo dice.
 *
 * `demo.spec.ts` sigue apuntando a la página anterior y sus identificadores ya no
 * existen: está pendiente de reescritura, no de arreglo.
 */

import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { type Page, expect, test } from "@playwright/test";

const AQUI = dirname(fileURLToPath(import.meta.url));
const MODELOS = resolve(AQUI, "../public/models/supertonic/onnx/vector_estimator.onnx");
const HF = "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/main/";

const hayModelos = existsSync(MODELOS);

/** Espera a que el arranque automático termine. */
async function listo(page: Page) {
  await expect(page.locator("#estado")).toContainText("listo", { timeout: 600_000 });
}

/**
 * Supertonic es el motor que arranca con la página: es el que lee. El que clona
 * se baja solo cuando alguien suelta una grabación (ver `pocket.spec.ts`), así
 * que aquí no hay nada que elegir — solo esperar.
 */
async function conSupertonic(page: Page) {
  await expect(page.locator("#e-motor")).toContainText("supertonic", { timeout: 600_000 });
  await listo(page);
}

/** Menos pasos de flow: estos tests miran que la cadena funcione, no cuánto tarda.
 *
 * Los matices viven plegados en un `<details>` — es progressive disclosure, no un
 * descuido —, así que hay que abrirlo antes de tocarlos o `fill` espera por algo
 * invisible hasta agotar el tiempo.
 */
async function rapido(page: Page, pasos = 2) {
  await page.locator("details.matices").evaluate((d) => {
    (d as HTMLDetailsElement).open = true;
  });
  await page.fill("#pasos", String(pasos));
  await page.locator("#pasos").dispatchEvent("input");
  await page.fill("#semilla", "1234");
}

test("la página monta: dos modos, 31 idiomas y los ajustes", async ({ page }) => {
  const errores: string[] = [];
  page.on("pageerror", (e) => errores.push(e.message));
  await page.goto("/");

  await expect(page.locator("h1")).toHaveText("ttspro");
  await expect(page.locator("#tab-estudio")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#modo-chat")).toBeHidden();
  await page.click("#tab-chat");
  await expect(page.locator("#modo-chat")).toBeVisible();
  await expect(page.locator("#modo-estudio")).toBeHidden();
  await page.click("#tab-estudio");

  // Los 31 idiomas del motor, menos el comodín `na`.
  expect(await page.locator("#idioma option").count()).toBe(31);

  await page.click("#abrir-ajustes");
  await expect(page.locator("#origenModelos")).toHaveValue(hayModelos ? "/models/supertonic/" : HF);
  await page.click("#cerrar-ajustes");
  expect(errores, `errores de JS en la página: ${errores.join(" | ")}`).toEqual([]);
});

test("supertonic: arranca solo y queda cargado", async ({ page }) => {
  test.skip(!hayModelos, "corre `ttspro.supertonic.descargar` y `npm run preparar`");
  test.setTimeout(900_000);
  page.on("pageerror", (e) => console.log("pageerror:", e.message));
  await page.goto("/");
  await conSupertonic(page);
  await expect(page.locator("#e-voz")).not.toHaveText("—");
  await expect(page.locator("#e-proveedor")).not.toHaveText("—");
  expect(await page.locator(".voz").count()).toBe(10);

  await rapido(page, 1);
  await page.fill("#texto", "Hola, esto lo dice el motor nuevo.");
  await page.click("#hablar");
  await expect(page.locator("#medidas")).toContainText("RTF", { timeout: 600_000 });
  console.log("medidas:", await page.locator("#medidas").innerText());
  await expect(page.locator("#descargar")).toBeVisible();
});

test("sin modelos locales cae sola en Hugging Face", async ({ page }) => {
  test.setTimeout(180_000);
  // Se simula el servidor sin grafos: lo mismo que ve quien no ha corrido
  // `descargar`. Vite responde 200 con el index.html a lo que no encuentra, que
  // es justo la trampa que `sirveModelos` tiene que detectar.
  await page.route("**/models/supertonic/**", (ruta) =>
    ruta.fulfill({ status: 200, contentType: "text/html", body: "<!doctype html><html></html>" }),
  );
  await page.goto("/");
  await expect(page.locator("#estado")).toContainText("Hugging Face", { timeout: 60_000 });
  await expect(page.locator("#origenModelos")).toHaveValue(HF);
  expect(await page.locator("#registro").innerText()).toContain("no tiene los grafos");
  // Y sigue: las diez voces se pintan desde el contrato mínimo, porque en
  // Hugging Face no hay contrato.json.
  await expect(page.locator(".voz")).toHaveCount(10, { timeout: 120_000 });
  expect(await page.locator("#registro").innerText()).toContain("sin contrato.json");
});

test("elegir una voz la dice al momento, y dos voces suenan distinto", async ({ page }) => {
  test.skip(!hayModelos, "corre `ttspro.supertonic.descargar` y `npm run preparar`");
  test.setTimeout(900_000);
  const errores: string[] = [];
  page.on("pageerror", (e) => errores.push(e.message));
  await page.goto("/");
  await conSupertonic(page);
  await rapido(page);
  await page.fill("#texto", "Una frase para comparar las voces.");

  /** Elige una voz SIN tocar «hablar» y devuelve el WAV que dejó, resumido. */
  const elegir = async (voz: string) => {
    const nombres = await page.locator(".voz .nombre").allInnerTexts();
    await page.locator(".voz").nth(nombres.indexOf(voz)).click();
    await expect(page.locator("#e-voz")).toHaveText(voz);
    await expect(page.locator("#medidas")).toContainText(voz, { timeout: 600_000 });
    return page.evaluate(async () => {
      const a = document.getElementById("descargar") as HTMLAnchorElement;
      const b = new Uint8Array(await (await fetch(a.href)).arrayBuffer());
      let h = 0;
      for (let i = 0; i < b.length; i += 97) h = (h * 31 + b[i]) >>> 0;
      return { bytes: b.length, hash: h };
    });
  };

  const f1 = await elegir("F1");
  const m1 = await elegir("M1");
  const otraVez = await elegir("F1");
  console.log("F1:", JSON.stringify(f1), "M1:", JSON.stringify(m1), "F1:", JSON.stringify(otraVez));
  // Con la semilla fija, la misma voz repite la toma exacta (regla 5)...
  expect(otraVez.hash).toBe(f1.hash);
  // ...y otra voz no. Lo único que cambió entre las dos es el estilo, así que
  // esto es la prueba de que el estilo llega al grafo y no solo a la etiqueta.
  expect(m1.hash).not.toBe(f1.hash);

  expect(errores, `errores de JS: ${errores.join(" | ")}`).toEqual([]);
});

test("añadir una voz: un .json de estilo entra por la misma puerta que un audio", async ({
  page,
}) => {
  test.skip(!hayModelos, "corre `ttspro.supertonic.descargar` y `npm run preparar`");
  test.setTimeout(900_000);
  await page.goto("/");
  await conSupertonic(page);
  await page.uncheck("#escucharAlElegir");

  // La entrada de fichero no lleva `accept`: con `audio/*` el explorador de
  // Windows escondia los audios, y con `.json` escondia todo lo demas. Hay UNA
  // puerta y decide el codigo por el contenido, no el sistema por el MIME.
  await expect(page.locator("#importar")).not.toHaveAttribute("accept", /.+/);
  await expect(page.locator("#importar")).toBeEnabled();

  // Y clonar esta ofrecido desde el arranque, sin depender de que Supertonic
  // haya cargado: son motores distintos y el que clona se baja aparte.
  await expect(page.locator("#p-clonar")).toHaveText("clona desde audio");
  await expect(page.locator("#grabar")).toBeEnabled();

  // Un .json entra: se coge una voz ya servida y se sube con otro nombre, sin
  // fixture nueva, por el mismo camino que un .json salido de `constructor`.
  const crudo = await (await page.request.get("/models/supertonic/voice_styles/M5.json")).body();
  await page.setInputFiles("#importar", {
    name: "mi-voz.json",
    mimeType: "application/json",
    buffer: crudo,
  });
  await expect(page.locator("#e-voz")).toHaveText("mi-voz", { timeout: 30_000 });
  expect(await page.locator(".voz").count()).toBe(11);
  await expect(page.locator(".marca")).toHaveText("clonada");

  // Soltar un audio aqui bajaria los 216 MB del motor que clona; ese camino se
  // mide entero en `pocket.spec.ts`, que es donde se paga esa descarga una vez.
});
