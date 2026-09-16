/**
 * El criterio de terminado, la mitad que sólo se puede medir en un navegador.
 *
 * `tests/terminado` es lo que SCOPE.md define como «terminado», y dos de sus
 * cinco puntos —la calidad del audio y la latencia— no se pueden medir en Python
 * sin **reimplementar el producto**: el que clona corre dentro de un Worker con
 * onnxruntime-web, y una copia en Python mediría otra cosa parecida. Así que este
 * spec ejercita la página de verdad, guarda lo que salió, y `tests/terminado`
 * puntúa esos ficheros con Whisper y WeSpeaker.
 *
 * Quien mide y quien decide están separados a propósito: aquí no hay ni un
 * `expect` sobre la calidad. Si este spec pasa significa «la página hizo las dos
 * cosas y hay wav que puntuar», no «el audio es bueno». Eso lo dice el criterio.
 *
 * Es lento por diseño: 398 MB para leer y 216 más para clonar, sin caché porque
 * cada tirada abre un navegador limpio.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const AQUI = dirname(fileURLToPath(import.meta.url));
const REFERENCIA = resolve(AQUI, "referencia.wav");
/**
 * **No** `test-results/`: ese directorio lo gestiona Playwright y lo **borra al
 * empezar** cualquier tirada, asi que correr otro spec cualquiera se llevaba por
 * delante lo que este habia medido y el criterio de Python fallaba con un «vuelve
 * a correr la pagina» que no venia a cuento.
 */
const SALIDA = resolve(AQUI, "../medido");

/**
 * Fijas y no del textarea: el criterio compara contra ESTE texto.
 *
 * **Cuatro frases y no una.** Con una sola de diez palabras, cada palabra que
 * Whisper oiga mal vale 0,1 de WER — es decir, el criterio «≤ 0,10» lo decide un
 * unico fallo de transcripcion. Y los hay que no son del sintetizador: la primera
 * tirada dio 0,100 exacto porque Whisper transcribio «zarpó» como «sarpó», que es
 * seseo suyo. Cuarenta palabras no quitan el ruido, pero lo dividen por cuatro.
 *
 * La PRIMERA tiene diez palabras exactas y es la que cronometra el criterio 5,
 * que SCOPE.md escribio sobre «una frase de 10 palabras»: cambiarla moveria el
 * umbral por la puerta de atras.
 */
const FRASES_LECTURA = [
  "El barco zarpó del puerto al amanecer con mucha niebla.",
  "Mi hermana trabaja en una librería cerca de la plaza mayor.",
  "El jueves que viene tenemos una reunión importante a las nueve.",
  "Guarda las llaves en el cajón de la mesa del salón.",
];
const FRASES_CLONADAS = [
  "Mañana por la tarde iremos juntos a la playa.",
  "No me acuerdo de dónde dejé el paraguas azul.",
];

/** El wav que la página deja en el enlace de descarga, tal cual, en base64. */
async function wavDeLaPagina(page: import("@playwright/test").Page): Promise<Buffer> {
  const b64 = await page.evaluate(async () => {
    const a = document.getElementById("descargar") as HTMLAnchorElement;
    const bytes = new Uint8Array(await (await fetch(a.href)).arrayBuffer());
    let s = "";
    for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return btoa(s);
  });
  return Buffer.from(b64, "base64");
}

test("criterio: la página lee y clona, y se guarda lo que salió", async ({ page }) => {
  test.setTimeout(1_800_000);

  // Lo que el navegador se baja de verdad, sumado por el cable. El presupuesto de
  // descarga lo retiró el ADR 0012, así que esto se anota y no se juzga: un
  // número que nadie mira se inventa solo, y este sigue decidiendo si la página
  // es usable con una conexión normal.
  let bytes = 0;
  page.on("response", (r) => {
    const n = Number(r.headers()["content-length"] ?? 0);
    if (Number.isFinite(n)) bytes += n;
  });

  await page.goto("/");
  await expect(page.locator("#estado")).toContainText("listo", { timeout: 1_500_000 });
  await page.uncheck("#escucharAlElegir");

  // Ocho pasos (la rodilla medida) y semilla fija: el criterio tiene que dar el
  // mismo número dos veces seguidas o no es un criterio.
  await page.locator("details.matices").evaluate((d) => {
    (d as HTMLDetailsElement).open = true;
  });
  await page.fill("#pasos", "8");
  await page.locator("#pasos").dispatchEvent("input");
  await page.fill("#velocidad", "1");
  await page.locator("#velocidad").dispatchEvent("input");
  await page.fill("#semilla", "7");

  // --- lectura: una voz predeterminada
  mkdirSync(SALIDA, { recursive: true });
  const lectura: { frase: string; wav: string; ms: number; medidas: string }[] = [];
  for (const [i, frase] of FRASES_LECTURA.entries()) {
    await page.fill("#texto", frase);
    await page.click("#hablar");
    await expect(page.locator("#medidas")).toContainText("predeterminada", { timeout: 900_000 });
    const medidas = await page.locator("#medidas").innerText();
    const wav = `criterio-lectura-${i}.wav`;
    writeFileSync(resolve(SALIDA, wav), await wavDeLaPagina(page));
    lectura.push({
      frase,
      wav,
      ms: Number(/·\s*(\d+)\s*ms/.exec(medidas)?.[1] ?? Number.NaN),
      medidas,
    });
    // Que no se cuele la medida anterior en la siguiente espera.
    await page.locator("#medidas").evaluate((e) => {
      e.textContent = "—";
    });
  }

  // --- clonado: la misma referencia que usa el resto de la suite
  await page.setInputFiles("#importar", REFERENCIA);
  await expect(page.locator("#e-voz")).toHaveText("referencia", { timeout: 900_000 });
  const clonada: { frase: string; wav: string; medidas: string }[] = [];
  for (const [i, frase] of FRASES_CLONADAS.entries()) {
    await page.fill("#texto", frase);
    await page.click("#hablar");
    await expect(page.locator("#medidas")).toContainText("clonada", { timeout: 900_000 });
    const medidas = await page.locator("#medidas").innerText();
    const wav = `criterio-clonada-${i}.wav`;
    writeFileSync(resolve(SALIDA, wav), await wavDeLaPagina(page));
    clonada.push({ frase, wav, medidas });
    await page.locator("#medidas").evaluate((e) => {
      e.textContent = "—";
    });
  }

  const medido = {
    proveedor: await page.locator("#e-proveedor").innerText(),
    // La primera, la de diez palabras, es la que cronometra el criterio 5.
    ms_lectura: lectura[0].ms,
    lectura,
    clonada,
    mb_descarga_total: Number((bytes / 1e6).toFixed(1)),
    referencia: "e2e/referencia.wav",
  };
  writeFileSync(resolve(SALIDA, "criterio.json"), `${JSON.stringify(medido, null, 1)}\n`);
  console.log(JSON.stringify(medido, null, 1));

  // Lo unico que se afirma aqui: que hay algo que puntuar.
  expect(
    lectura.every((l) => Number.isFinite(l.ms) && l.ms > 0),
    JSON.stringify(lectura),
  ).toBe(true);
  expect(clonada).toHaveLength(FRASES_CLONADAS.length);
});
