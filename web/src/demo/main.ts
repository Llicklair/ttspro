/**
 * The test page. Chain (ADR 0007): text -> tts.onnx in a fixed base voice ->
 * conversor.onnx in the chosen voice. The voice is either a preset shipped in
 * models/voces.json or one cloned from a recording through voz.onnx.
 *
 * Every number shown is measured in this browser.
 */

import { configurarEspeak, versionEspeak } from "../frontend/fonemas.ts";
import { aWav, decodificar, grabarPCM, reproducir } from "../runtime/audio.ts";
import type { Contrato } from "../runtime/contrato.ts";
import { convertir, vectorVoz } from "../runtime/conversor.ts";
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

interface Voz {
  id: string;
  nombre: string;
  idioma: string;
  voz: number[];
}
interface Voces {
  espacio: string;
  base: { locutor: string; idioma: string; embedding_tts: number[]; voz: number[] };
  voces: Voz[];
}

let contrato: Contrato;
let voces: Voces | null = null;
let tts: Sesion | null = null;
let grafoVoz: Sesion | null = null;
let grafoConversor: Sesion | null = null;
let vozDestino: Float32Array | null = null;
let ultima: { onda: Float32Array; sr: number } | null = null;

async function cargarModelos(): Promise<void> {
  const preferido = ($("proveedor") as HTMLSelectElement).value as Proveedor | "auto";
  const proveedores: Proveedor[] = preferido === "auto" ? ["webgpu", "wasm"] : [preferido];
  let precision = ($("precision") as HTMLSelectElement).value;
  if (precision === "auto") {
    const f16 = proveedores[0] === "webgpu" && (await soportaF16());
    precision = f16 ? ".fp16" : "";
    log(`precisión auto → ${precision || "fp32"} (shader-f16: ${f16})`);
  }
  estado.textContent = "cargando…";
  ($("cargar") as HTMLButtonElement).disabled = true;
  try {
    contrato = await (await fetch("/models/contrato.json")).json();
    voces = await (await fetch("/models/voces.json")).json();
    const [t, c] = await Promise.all([
      crearSesion(`/models/tts${precision}.onnx`, proveedores),
      crearSesion(`/models/conversor${precision}.onnx`, proveedores),
    ]);
    // The voice extractor carries a GRU, which WebGPU has no kernel for: wasm.
    grafoVoz = await crearSesion(`/models/voz${precision}.onnx`, ["wasm"]);
    tts = t;
    grafoConversor = c;
    const mb = ((t.bytes + c.bytes + grafoVoz.bytes) / 1e6).toFixed(1);
    const resumen = `tts ${t.proveedor} · conversor ${c.proveedor} · voz ${grafoVoz.proveedor} · ${mb} MB · ${hilos()} hilo(s) wasm · aislado=${crossOriginIsolated}`;
    estado.textContent = `preparando fonemizador y voces… (${resumen})`;
    log(`espeak-ng: ${await versionEspeak()}`);
    rellenarPresets();
    for (const id of ["fichero", "grabar", "sintetizar", "sinConvertir", "preset"])
      ($(id) as HTMLButtonElement).disabled = false;
    estado.textContent = `listo · ${resumen}`;
    log(`modelos cargados: ${estado.textContent}`);
  } catch (err) {
    estado.textContent = `error: ${String(err)}`;
    log(String(err));
  } finally {
    ($("cargar") as HTMLButtonElement).disabled = false;
  }
}

function rellenarPresets(): void {
  const sel = $("preset") as HTMLSelectElement;
  sel.innerHTML = '<option value="">— elige una voz —</option>';
  if (!voces) {
    $("infoVoz").textContent = "sin models/voces.json";
    return;
  }
  for (const [i, v] of voces.voces.entries()) {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = `${v.idioma} · ${v.nombre}`;
    sel.appendChild(o);
  }
  $("infoVoz").textContent = `${voces.voces.length} presets · voz base ${voces.base.locutor}`;
}

const srConversor = () => contrato.grafos.voz.frecuencia_entrada_hz ?? 22050;

async function vozDesdeOnda(onda: Float32Array, origen: string): Promise<void> {
  if (!grafoVoz) return;
  if (onda.length < srConversor() * 2) {
    log(`${origen}: solo ${(onda.length / srConversor()).toFixed(1)} s; hacen falta 3 s o más`);
    return;
  }
  const t0 = performance.now();
  vozDestino = await vectorVoz(grafoVoz, contrato, onda);
  ($("preset") as HTMLSelectElement).value = "";
  $("infoVoz").textContent =
    `voz clonada de ${origen}: ${(onda.length / srConversor()).toFixed(1)} s de audio → vector en ${(performance.now() - t0).toFixed(0)} ms`;
  log($("infoVoz").textContent ?? "");
}

$("cargar").addEventListener("click", cargarModelos);

$("preset").addEventListener("change", () => {
  const i = ($("preset") as HTMLSelectElement).value;
  if (!voces || i === "") return;
  const v = voces.voces[Number(i)];
  vozDestino = Float32Array.from(v.voz);
  ($("idioma") as HTMLSelectElement).value = v.idioma;
  $("infoVoz").textContent = `preset ${v.nombre}`;
  log($("infoVoz").textContent ?? "");
});

($("fichero") as HTMLInputElement).addEventListener("change", async (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (f) await vozDesdeOnda(await decodificar(await f.arrayBuffer(), srConversor()), f.name);
});

$("grabar").addEventListener("click", async () => {
  const boton = $("grabar") as HTMLButtonElement;
  boton.disabled = true;
  try {
    const onda = await grabarPCM(6, srConversor(), (s) => {
      boton.textContent = `grabando… ${s}/6 s`;
    });
    await vozDesdeOnda(onda, "micrófono");
  } catch (err) {
    log(`micrófono: ${String(err)}`);
  } finally {
    boton.textContent = "grabar 6 s con el micrófono";
    boton.disabled = false;
  }
});

async function hablar(conConversor: boolean): Promise<void> {
  if (!tts || !voces) return;
  if (conConversor && !vozDestino) {
    log("elige un preset o clona una voz antes de sintetizar");
    return;
  }
  const botones = ["sintetizar", "sinConvertir"].map((id) => $(id) as HTMLButtonElement);
  for (const b of botones) b.disabled = true;
  try {
    const semilla = Number(($("semilla") as HTMLInputElement).value);
    const base = await sintetizar(
      tts,
      contrato,
      ($("texto") as HTMLTextAreaElement).value,
      ($("idioma") as HTMLSelectElement).value,
      Float32Array.from(voces.base.embedding_tts),
      {
        noise_scale: Number(($("noise") as HTMLInputElement).value),
        noise_scale_w: Number(($("noise_w") as HTMLInputElement).value),
        length_scale: Number(($("length") as HTMLInputElement).value),
        semilla,
      },
    );
    let onda = base.onda;
    let sr = base.sampleRate;
    let msConv = 0;
    if (conConversor && grafoConversor && vozDestino) {
      const r = await convertir(
        grafoConversor,
        contrato,
        base.onda,
        Float32Array.from(voces.base.voz),
        vozDestino,
        { tau: Number(($("tau") as HTMLInputElement).value), semilla },
      );
      onda = r.onda;
      sr = r.sampleRate;
      msConv = r.ms;
    }
    ultima = { onda, sr };
    $("fonemas").textContent = base.fonemas;
    const segundos = onda.length / sr;
    $("medidas").textContent =
      `${base.tokens} tokens · frontend ${base.ms_frontend.toFixed(0)} ms · tts ${base.ms_modelo.toFixed(0)} ms` +
      (conConversor ? ` · conversor ${msConv.toFixed(0)} ms` : " · sin convertir") +
      ` · ${segundos.toFixed(2)} s · RTF ${((base.ms_modelo + msConv) / 1000 / segundos).toFixed(3)}`;
    log($("medidas").textContent ?? "");
    (window as unknown as { __ttspro: unknown }).__ttspro = {
      muestras: onda.length,
      sampleRate: sr,
      ms_modelo: base.ms_modelo,
      ms_conversor: msConv,
      ms_frontend: base.ms_frontend,
      proveedor: tts.proveedor,
      tokens: base.tokens,
      convertido: conConversor,
    };
    reproducir(onda, sr);
    const enlace = $("descargar") as HTMLAnchorElement;
    enlace.href = URL.createObjectURL(aWav(onda, sr));
    enlace.download = conConversor ? "ttspro-voz.wav" : "ttspro-base.wav";
    enlace.hidden = false;
  } catch (err) {
    log(`síntesis: ${String(err)}`);
  } finally {
    for (const b of botones) b.disabled = false;
  }
}

$("sintetizar").addEventListener("click", () => hablar(true));
$("sinConvertir").addEventListener("click", () => hablar(false));
$("repetir").addEventListener("click", () => {
  if (ultima) reproducir(ultima.onda, ultima.sr);
});

for (const id of ["noise", "noise_w", "length", "tau"]) {
  const entrada = $(id) as HTMLInputElement;
  const etiqueta = $(`${id}_v`);
  const pintar = () => {
    etiqueta.textContent = entrada.value;
  };
  entrada.addEventListener("input", pintar);
  pintar();
}
