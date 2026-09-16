/**
 * El motor que sí clona, en un navegador de verdad.
 *
 * Esto es lo que Marcos pedía y no había: soltar una grabación y que salga una
 * voz. Supertonic no puede — su encoder no se publicó y reconstruirlo no llega,
 * seis medidas en docs/evidencia.md —, así que la página trae los dos motores y
 * este test comprueba el camino entero del que clona.
 *
 * Los pesos (177 MB, más 39 del encoder al clonar) vienen de Hugging Face y se
 * quedan en la Cache API, así que la primera tirada es lenta y las siguientes no.
 */

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const AQUI = dirname(fileURLToPath(import.meta.url));
const REFERENCIA = resolve(AQUI, "referencia.wav");

test("pocket: carga, clona desde un wav y habla con esa voz", async ({ page }) => {
  test.setTimeout(1_800_000);
  const errores: string[] = [];
  page.on("pageerror", (e) => errores.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errores.push(`console: ${m.text()}`);
  });

  await page.goto("/");
  // El selector arranca en pocket: es el único que clona.
  await expect(page.locator("#motor")).toHaveValue("pocket");
  await expect(page.locator("#estado")).toContainText("listo", { timeout: 1_500_000 });
  await expect(page.locator("#e-motor")).toContainText("pocket");

  const deFabrica = await page.locator(".voz").count();
  expect(deFabrica).toBeGreaterThan(0);
  const nombres = await page.locator(".voz .nombre").allInnerTexts();
  const detalles = await page.locator(".voz .detalle").allInnerTexts();
  console.log(`voces de fábrica: ${deFabrica}`, nombres, detalles);
  // Pocket no dice el sexo de sus voces en ningún sitio: su manifiesto trae el
  // idioma y nada más. Deducirlo del nombre daba «lola: masculina», así que
  // aquí se fija que no se invente.
  expect(detalles.join(" ")).not.toMatch(/masculina|femenina/);

  // Y clonar está ofrecido, no apagado.
  await expect(page.locator("#p-clonar")).toHaveText("clona desde audio");
  await expect(page.locator("#grabar")).toBeEnabled();

  // Soltar un wav: la voz tiene que aparecer y quedar elegida.
  await page.uncheck("#escucharAlElegir");
  const t0 = Date.now();
  await page.setInputFiles("#importar", REFERENCIA);
  await expect(page.locator("#e-voz")).toHaveText("referencia", { timeout: 600_000 });
  console.log(`clonado en ${Date.now() - t0} ms`);
  expect(await page.locator(".voz").count()).toBe(deFabrica + 1);
  await expect(page.locator(".marca")).toHaveText("clonada");

  // Y habla con ella.
  await page.fill("#texto", "Hola, esto lo dice una voz clonada de un fichero.");
  await page.click("#hablar");
  await expect(page.locator("#medidas")).toContainText("pocket", { timeout: 600_000 });
  const medidas = await page.locator("#medidas").innerText();
  console.log("medidas:", medidas);
  expect(medidas).toContain("referencia");
  await expect(page.locator("#descargar")).toBeVisible();

  expect(errores, `errores de JS: ${errores.join(" | ")}`).toEqual([]);
});
