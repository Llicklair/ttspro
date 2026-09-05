// Static server for the library and the example page, with the two headers
// that make the page cross-origin isolated (multithreaded wasm needs them).
//
//   node scripts/servir.mjs [puerto]        default 8080
//
// Serves, in this order of precedence: ejemplo/ (the page), dist/lib/ (the
// bundle), public/ (models, espeak, ORT wasm — run `npm run preparar` first).
// Dependency-free on purpose: it is the "open an html" path, not a build tool.

import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const raices = [join(web, "ejemplo"), join(web, "dist", "lib"), join(web, "public")];
const tipos = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".wav": "audio/wav",
  ".css": "text/css",
};

export function servir(puerto = 0) {
  const servidor = createServer((req, res) => {
    const ruta = normalize(decodeURIComponent(new URL(req.url ?? "/", "http://x").pathname));
    const relativa = ruta === "/" || ruta === "\\" ? "/index.html" : ruta;
    const fichero = raices
      .map((r) => join(r, relativa))
      .find((f) => existsSync(f) && statSync(f).isFile());
    if (!fichero || !fichero.startsWith(web)) {
      res.writeHead(404).end(`no encontrado: ${relativa}`);
      return;
    }
    res.writeHead(200, {
      "Content-Type": tipos[extname(fichero)] ?? "application/octet-stream",
      "Content-Length": statSync(fichero).size,
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cache-Control": "no-cache",
    });
    createReadStream(fichero).pipe(res);
  });
  return new Promise((listo) => servidor.listen(puerto, "127.0.0.1", () => listo(servidor)));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const puerto = Number(process.argv[2] ?? 8080);
  const servidor = await servir(puerto);
  console.log(
    `ttspro: http://127.0.0.1:${servidor.address().port}/  (ejemplo/, dist/lib/, public/)`,
  );
}
