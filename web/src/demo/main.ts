/**
 * The test page: load the two graphs, clone a voice from a file or the mic,
 * type text, listen. Every number shown is measured in this browser.
 */

import { configurarEspeak, versionEspeak } from "../frontend/fonemas.ts";
import { aWav, decodificar, grabarPCM, reproducir } from "../runtime/audio.ts";
import type { Contrato } from "../runtime/contrato.ts";
import { type Locutor, calcularEmbedding } from "../runtime/locutor.ts";
import { type Proveedor, type Sesion, crearSesion, hilos, soportaF16 } from "../runtime/ort.ts";
import { sintetizar } from "../runtime/sintetizador.ts";

configurarEspeak({
  locateFile: (f: string) => (f.endsWith(".wasm") ? "/espeak/espeak-ng.wasm" : f),
});

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const estado = $("estado");
const registro = $("registro");
const log = (msg: string) => {
  registro.textContent = `${new Date().toLocaleTimeString()}  ${msg}\n${registro.textContent}`;
};

let contrato: Contrato;
interface Voz {
  id: string;
  nombre: string;
  idioma: string;
  embedding: number[];
}
let voces: { espacio: string; voces: Voz[] } | null = null;
let espacioModelo = "wespeaker-resnet34-LM";
let encoder: Sesion | null = null;
let tts: Sesion | null = null;
let locutor: Locutor | null = null;
let ultimaOnda: { onda: Float32Array; sr: number } | null = null;

async function cargarModelos(): Promise<void> {
  const preferido = ($("proveedor") as HTMLSelectElement).value as Proveedor | "auto";
  const proveedores: Proveedor[] = preferido === "auto" ? ["webgpu", "wasm"] : [preferido];
  let precision = ($("precision") as HTMLSelectElement).value; // "auto" | "" | ".fp16"
  if (precision === "auto") {
    // fp16 only where it is both correct and faster: WebGPU with shader-f16.
    // wasm runs fp16 slower than fp32, and WebGPU without f16 returns NaN.
    const f16 = proveedores[0] === "webgpu" && (await soportaF16());
    precision = f16 ? ".fp16" : "";
    log(`precisión auto → ${precision || "fp32"} (shader-f16: ${f16})`);
  }
  estado.textContent = "cargando…";
  ($("cargar") as HTMLButtonElement).disabled = true;
  try {
    contrato = await (await fetch("/models/contrato.json")).json();
    const [e, t] = await Promise.all([
      crearSesion(`/models/speaker_encoder${precision}.onnx`, proveedores),
      crearSesion(`/models/tts${precision}.onnx`, proveedores),
    ]);
    encoder = e;
    tts = t;
    const mb = ((e.bytes + t.bytes) / 1e6).toFixed(1);
    estado.textContent = `listo · encoder ${e.proveedor} ${e.ms_carga.toFixed(0)} ms · tts ${t.proveedor} ${t.ms_carga.toFixed(0)} ms · ${mb} MB de modelos · ${hilos()} hilo(s) wasm · crossOriginIsolated=${crossOriginIsolated}`;
    log(`modelos cargados: ${estado.textContent}`);
    log(`espeak-ng: ${await versionEspeak()}`);
    await cargarPresets();
    for (const id of ["fichero", "grabar", "sintetizar", "preset"])
      ($(id) as HTMLButtonElement).disabled = false;
  } catch (err) {
    estado.textContent = `error: ${String(err)}`;
    log(String(err));
  } finally {
    ($("cargar") as HTMLButtonElement).disabled = false;
  }
}

const srEncoder = () => contrato.grafos.speaker_encoder.frecuencia_entrada_hz ?? 16000;

/** Presets are speaker vectors in a NAMED space; the un-finetuned port speaks
 * coqui's emb_g space, the fine-tuned model WeSpeaker's. A mismatch is not a
 * worse voice, it is noise, so the page says which space each side is in. */
async function cargarPresets(): Promise<void> {
  const exportado = await fetch("/models/tts.export.json")
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
  espacioModelo = exportado?.espacio_locutor ?? espacioModelo;
  voces = await fetch("/models/voces.json")
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
  const sel = $("preset") as HTMLSelectElement;
  sel.innerHTML = '<option value="">— elige una voz —</option>';
  if (!voces) {
    $("espacio").textContent = `sin models/voces.json (modelo: ${espacioModelo})`;
    return;
  }
  for (const [i, v] of voces.voces.entries()) {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = `${v.idioma} · ${v.nombre}`;
    sel.appendChild(o);
  }
  const ok = voces.espacio === espacioModelo;
  $("espacio").textContent = ok
    ? `${voces.voces.length} voces (${voces.espacio})`
    : `AVISO: presets en ${voces.espacio}, modelo en ${espacioModelo}: no coinciden`;
  const clonable = espacioModelo === "wespeaker-resnet34-LM";
  if (!clonable)
    log(
      `el modelo cargado espera el espacio ${espacioModelo}: la clonación desde audio (WeSpeaker) no le sirve todavía; usa un preset`,
    );
}

$("preset").addEventListener("change", () => {
  const i = ($("preset") as HTMLSelectElement).value;
  if (!voces || i === "") return;
  const v = voces.voces[Number(i)];
  locutor = { embedding: Float32Array.from(v.embedding), segundos: 0, ms: 0 };
  ($("idioma") as HTMLSelectElement).value = v.idioma;
  $("locutor").textContent = `preset ${v.nombre} (${voces.espacio})`;
  log($("locutor").textContent ?? "");
});

async function referenciaDesde(datos: ArrayBuffer, origen: string): Promise<void> {
  await referenciaDesdeOnda(await decodificar(datos, srEncoder()), origen);
}

async function referenciaDesdeOnda(onda: Float32Array, origen: string): Promise<void> {
  if (!encoder) return;
  if (onda.length < srEncoder()) {
    log(
      `${origen}: solo ${(onda.length / srEncoder()).toFixed(1)} s de audio; hacen falta al menos 3 s`,
    );
    return;
  }
  locutor = await calcularEmbedding(encoder, contrato, onda);
  const norma = Math.sqrt(locutor.embedding.reduce((s, x) => s + x * x, 0));
  if (!Number.isFinite(norma)) {
    log(
      `${origen}: el encoder devolvió NaN/inf con el proveedor ${encoder.proveedor}; prueba fp32 o wasm`,
    );
    locutor = null;
    return;
  }
  $("locutor").textContent =
    `${origen}: ${locutor.segundos.toFixed(1)} s de audio → embedding en ${locutor.ms.toFixed(0)} ms (norma ${norma.toFixed(3)})`;
  log($("locutor").textContent ?? "");
}

$("cargar").addEventListener("click", cargarModelos);

($("fichero") as HTMLInputElement).addEventListener("change", async (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (f) await referenciaDesde(await f.arrayBuffer(), f.name);
});

$("grabar").addEventListener("click", async () => {
  const boton = $("grabar") as HTMLButtonElement;
  boton.disabled = true;
  try {
    const onda = await grabarPCM(5, srEncoder(), (s) => {
      boton.textContent = `grabando… ${s}/5 s`;
    });
    await referenciaDesdeOnda(onda, "micrófono");
  } catch (err) {
    log(`micrófono: ${String(err)}`);
  } finally {
    boton.textContent = "grabar 5 s con el micrófono";
    boton.disabled = false;
  }
});

$("sintetizar").addEventListener("click", async () => {
  if (!tts) return;
  if (!locutor) {
    log("falta la voz de referencia: sube un audio o graba 5 s");
    return;
  }
  const boton = $("sintetizar") as HTMLButtonElement;
  boton.disabled = true;
  try {
    const r = await sintetizar(
      tts,
      contrato,
      ($("texto") as HTMLTextAreaElement).value,
      ($("idioma") as HTMLSelectElement).value,
      locutor.embedding,
      {
        noise_scale: Number(($("noise") as HTMLInputElement).value),
        noise_scale_w: Number(($("noise_w") as HTMLInputElement).value),
        length_scale: Number(($("length") as HTMLInputElement).value),
        semilla: Number(($("semilla") as HTMLInputElement).value),
      },
    );
    ultimaOnda = { onda: r.onda, sr: r.sampleRate };
    // for the e2e test (tests/terminado criterion 5): the last result, measurable
    (window as unknown as { __ttspro: unknown }).__ttspro = {
      muestras: r.onda.length,
      sampleRate: r.sampleRate,
      ms_modelo: r.ms_modelo,
      ms_frontend: r.ms_frontend,
      proveedor: tts.proveedor,
      tokens: r.tokens,
    };
    $("fonemas").textContent = r.fonemas;
    $("medidas").textContent =
      `${r.tokens} tokens · frontend ${r.ms_frontend.toFixed(0)} ms · modelo ${r.ms_modelo.toFixed(0)} ms (${tts.proveedor}) · ${(r.onda.length / r.sampleRate).toFixed(2)} s de audio · RTF ${r.rtf.toFixed(3)}`;
    log($("medidas").textContent ?? "");
    reproducir(r.onda, r.sampleRate);
    const enlace = $("descargar") as HTMLAnchorElement;
    enlace.href = URL.createObjectURL(aWav(r.onda, r.sampleRate));
    enlace.download = "ttspro.wav";
    enlace.hidden = false;
  } catch (err) {
    log(`síntesis: ${String(err)}`);
  } finally {
    boton.disabled = false;
  }
});

$("repetir").addEventListener("click", () => {
  if (ultimaOnda) reproducir(ultimaOnda.onda, ultimaOnda.sr);
});

for (const id of ["noise", "noise_w", "length"]) {
  const entrada = $(id) as HTMLInputElement;
  const etiqueta = $(`${id}_v`);
  const pintar = () => {
    etiqueta.textContent = entrada.value;
  };
  entrada.addEventListener("input", pintar);
  pintar();
}
