/**
 * El motor que sí clona, en un navegador de verdad.
 *
 * Esto es lo que Marcos pedía y no había: soltar una grabación y que salga una
 * voz. Supertonic no puede — su encoder no se publicó y reconstruirlo no llega,
 * seis medidas en docs/evidencia.md —, así que la página trae un segundo motor
 * que **solo** clona, y este test comprueba ese camino entero.
 *
 * No hay selector de motor: se suelta el fichero y la página baja lo que le hace
 * falta (216 MB) en ese momento. Vienen de Hugging Face y se quedan en la Cache
 * API, así que la primera tirada es lenta y las siguientes no.
 */

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const AQUI = dirname(fileURLToPath(import.meta.url));
const REFERENCIA = resolve(AQUI, "referencia.wav");
/** La sonda que decide si hay grafos servidos aqui; 404 cuando no los hay. */
const SONDA_LOCAL = /\/models\/supertonic\/onnx\/tts\.json$/;

test("pocket: carga, clona desde un wav y habla con esa voz", async ({ page }) => {
  test.setTimeout(1_800_000);
  const errores: string[] = [];
  page.on("pageerror", (e) => errores.push(e.message));
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    // En una instalacion recien hecha no hay copia local de los grafos, y la
    // pagina lo averigua **pidiendo uno**: ese 404 es como decide pasarse a
    // Hugging Face, y el navegador lo escribe en la consola sin que la pagina
    // pueda evitarlo. Se tolera por su URL exacta, no bajando la exigencia:
    // cualquier otro error de consola sigue tumbando el test.
    if (SONDA_LOCAL.test(m.location()?.url ?? "")) return;
    errores.push(`console: ${m.text()}`);
  });

  await page.goto("/");
  await expect(page.locator("#estado")).toContainText("listo", { timeout: 1_500_000 });
  await expect(page.locator("#e-motor")).toContainText("predeterminada");

  // Las voces que se ofrecen son las diez predeterminadas y ninguna más: las dos
  // de fábrica del modelo que clona (javert, lola) no se listan, porque suenan
  // peor y porque ese modelo ni siquiera está cargado todavía.
  const deFabrica = await page.locator(".voz").count();
  const nombres = await page.locator(".voz .nombre").allInnerTexts();
  console.log(`voces de fábrica: ${deFabrica}`, nombres);
  expect(nombres).not.toContain("javert");
  expect(nombres).not.toContain("lola");

  // Y clonar está ofrecido desde el principio, no apagado a la espera de nada.
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
  await expect(page.locator("#medidas")).toContainText("clonada", { timeout: 1_500_000 });
  const medidas = await page.locator("#medidas").innerText();
  console.log("medidas:", medidas);
  expect(medidas).toContain("referencia");
  await expect(page.locator("#descargar")).toBeVisible();

  expect(errores, `errores de JS: ${errores.join(" | ")}`).toEqual([]);
});
