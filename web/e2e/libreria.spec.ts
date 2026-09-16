/**
 * La librería, desde una página cualquiera: `import { TTS } from "./ttspro.js"`.
 *
 * Lo que Marcos pidió en su día — abrir un HTML, importar ttspro y hablar desde
 * la consola — pero contra el motor de ADR 0012. Se ejercita el **bundle
 * construido** en `dist/lib`, no las fuentes, servido por `scripts/servir.mjs`
 * con COOP/COEP, que es la situación real de quien la empotra en su sitio.
 *
 * Sustituye a `demo.spec.ts`, que apuntaba a la cadena anterior: sus nueve tests
 * probaban paquetes de voz de Piper, el conversor y `tau`, que ya no existen.
 */

import { existsSync, readdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const AQUI = dirname(fileURLToPath(import.meta.url));
const BUNDLE = resolve(AQUI, "../dist/lib/ttspro.js");
const MODELOS = resolve(AQUI, "../public/models/supertonic/onnx/vector_estimator.onnx");
const DIST = resolve(AQUI, "../dist/lib");

// @ts-expect-error JS plano y sin declaración: es el mismo servidor que `npm run servir`
import { servir } from "../scripts/servir.mjs";

/**
 * Lo que el build **reparte**, que es donde vive la obligacion de la GPL: no en
 * ejecutar espeak-ng, en entregarselo a quien abre la pagina. El paquete llega
 * por `pocket-tts-onnx`, solo lo usa el hebreo mezclado con letras latinas —una
 * rama que el modelo espanol ni siquiera tiene— y aun asi el build emitia 18,5 MB
 * de wasm y 68 KB de pegamento GPL. Ahora se pide al CDN cuando hace falta.
 *
 * Esto lo fija como test y no como comentario porque vuelve solo: basta con que
 * alguien quite el alias de `vite.config.ts`, o con que otra dependencia tire de
 * espeak, para que el fichero reaparezca sin que nadie lo note.
 */
test("el build no reparte espeak-ng, que es GPL-3.0", () => {
  test.skip(!existsSync(BUNDLE), "corre `npm run build:lib` primero");
  const ficheros: string[] = [];
  const mirar = (dir: string) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      if (e.isDirectory()) mirar(resolve(dir, e.name));
      else ficheros.push(e.name);
    }
  };
  mirar(DIST);
  expect(ficheros.filter((f) => /espeak-ng/.test(f))).toEqual([]);
});

test("librería: elegir voz, importar un .json, predict de texto y de stream", async ({ page }) => {
  test.skip(!existsSync(BUNDLE), "corre `npm run build:lib` primero");
  test.skip(!existsSync(MODELOS), "corre `ttspro.supertonic.descargar` y `npm run preparar`");
  test.setTimeout(900_000);

  const servidor = (await servir(0)) as { address: () => { port: number }; close: () => void };
  const base = `http://127.0.0.1:${servidor.address().port}`;
  try {
    const errores: string[] = [];
    page.on("pageerror", (e) => errores.push(e.message));
    await page.goto(`${base}/`);
    await page.waitForFunction(() => Boolean((window as unknown as { tts?: unknown }).tts), null, {
      timeout: 600_000,
    });

    const medido = await page.evaluate(async () => {
      type R = {
        texto: string;
        onda: Float32Array;
        sampleRate: number;
        ms: number;
        voz: string;
        wav: () => Blob;
      };
      const t = (
        window as unknown as {
          tts: {
            voces: string[];
            voz: string | null;
            idiomas: string[];
            sampleRate: number;
            proveedor: string;
            ajustes: { pasos: number; semilla?: number };
            elegirVoz: (n: string) => Promise<string>;
            importar: (o: unknown, n?: string) => Promise<string>;
            clonar: (f: unknown, nombre?: string) => Promise<string>;
            predict: (e: unknown, o?: unknown) => Promise<R> & AsyncIterable<R>;
          };
        }
      ).tts;

      // Pocos pasos y semilla fija: esto mira que la cadena funcione, no cuánto tarda.
      t.ajustes.pasos = 2;
      t.ajustes.semilla = 1234;

      await t.elegirVoz("F1");
      const f1 = await t.predict("una frase para comparar las voces");
      await t.elegirVoz("M1");
      const m1 = await t.predict("una frase para comparar las voces");

      // Importar un .json: se coge una voz servida y entra con otro nombre.
      const crudo = await (await fetch("/models/supertonic/voice_styles/M5.json")).json();
      const importada = await t.importar(crudo, "mi-voz");
      const suya = await t.predict("y esta la dice la voz importada");

      // `clonar` existe y no es un método muerto. Lo que hace de verdad se mide
      // en `pocket.spec.ts`, por la página: aquí llamarlo bajaría 216 MB del otro
      // motor, y esto es la prueba rápida de la librería.
      const sabeClonar = typeof t.clonar === "function";

      // Un stream: se sintetiza por delante y sale en orden.
      const stream: string[] = [];
      for await (const r of t.predict(["hola que tal", "KEKW", "segunda linea"], { chat: true })) {
        stream.push(r.texto);
      }

      const resumen = (r: R) => ({
        segundos: r.onda.length / r.sampleRate,
        voz: r.voz,
        wavBytes: r.wav().size,
        suma: r.onda.reduce((a, b) => a + Math.abs(b), 0),
      });
      return {
        voces: t.voces.length,
        idiomas: t.idiomas.length,
        sampleRate: t.sampleRate,
        proveedor: t.proveedor,
        importada,
        vozActual: t.voz,
        f1: resumen(f1),
        m1: resumen(m1),
        suya: resumen(suya),
        sabeClonar,
        stream,
      };
    });

    console.log(JSON.stringify(medido, null, 1));
    expect(medido.voces).toBeGreaterThanOrEqual(10);
    expect(medido.idiomas).toBe(31);
    expect(medido.sampleRate).toBe(44100);
    expect(medido.f1.voz).toBe("F1");
    expect(medido.m1.voz).toBe("M1");
    expect(medido.f1.segundos).toBeGreaterThan(0.5);
    // Misma frase, mismos pasos, misma semilla: lo único distinto es el estilo.
    expect(medido.m1.suma).not.toBe(medido.f1.suma);
    expect(medido.f1.wavBytes).toBeGreaterThan(44);

    expect(medido.importada).toBe("mi-voz");
    expect(medido.vozActual).toBe("mi-voz");
    expect(medido.suya.voz).toBe("mi-voz");

    expect(medido.sabeClonar).toBe(true);

    // "KEKW" -> "jajaja" por el normalizador de chat; las tres salen y en orden.
    expect(medido.stream).toEqual(["hola que tal", "jajaja", "segunda linea"]);

    expect(errores, `errores de JS: ${errores.join(" | ")}`).toEqual([]);
  } finally {
    servidor.close();
  }
});
