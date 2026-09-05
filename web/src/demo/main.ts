/**
 * The test page. Chain (ADR 0007 and 0008): text -> tts.onnx in a fixed base
 * voice -> conversor.onnx in the chosen voice. The voice is either a preset
 * shipped in models/voces.json or one cloned from a recording through voz.onnx.
 *
 * Every number shown is measured in this browser.
 */

import { configurarEspeak, versionEspeak } from "../frontend/fonemas.ts";
import { aWav, decodificar, grabarPCM, reproducir } from "../runtime/audio.ts";
import type { Contrato } from "../runtime/contrato.ts";
import { convertir, vectorVoz } from "../runtime/conversor.ts";
import { type Proveedor, type Sesion, crearSesion, hilos, soportaF16 } from "../runtime/ort.ts";
import { sintetizar } from "../runtime/sintetizador.ts";
import { Visor, pintarNivel } from "./visor.ts";

configurarEspeak({
  locateFile: (f: string) => (f.endsWith(".wasm") ? "/espeak/espeak-ng.wasm" : f),
});

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const estado = $("estado");
const registro = $("registro");
const log = (msg: string) => {
  registro.textContent = `${new Date().toLocaleTimeString()}  ${msg}\n${registro.textContent}`;
};
const pastilla = (id: string, texto: string, marca = "") => {
  const el = $(`pastilla-${id}`);
  el.textContent = texto;
  if (marca) el.dataset.estado = marca;
  else el.removeAttribute("data-estado");
};

interface Voz {
  id: string;
  nombre: string;
  idioma: string;
  voz: number[];
}
interface Voces {
  espacio: string;
  /** `embedding_tts` only exists when the base TTS takes a speaker vector. */
  base: { locutor: string; idioma: string; embedding_tts?: number[]; voz: number[] };
  voces: Voz[];
}

let contrato: Contrato;
let voces: Voces | null = null;
let tts: Sesion | null = null;
let grafoVoz: Sesion | null = null;
let grafoConversor: Sesion | null = null;
let vozDestino: Float32Array | null = null;
let ultima: { onda: Float32Array; sr: number } | null = null;
const visor = new Visor($<HTMLCanvasElement>("onda"));
visor.poner("vacio");

// ---------------------------------------------------------------- models

async function cargarModelos(): Promise<void> {
  const preferido = ($("proveedor") as HTMLSelectElement).value as Proveedor | "auto";
  const proveedores: Proveedor[] = preferido === "auto" ? ["webgpu", "wasm"] : [preferido];
  let precision = ($("precision") as HTMLSelectElement).value;
  if (precision === "auto") {
    const f16 = proveedores[0] === "webgpu" && (await soportaF16());
    precision = f16 ? ".fp16" : "";
    log(`precisión auto → ${precision || "fp32"} (shader-f16: ${f16})`);
  }
  pastilla("modelos", "descargando…", "trabajando");
  ($("cargar") as HTMLButtonElement).disabled = true;
  const barra = $("progreso");
  // One bar for three files: each reports its own bytes, so they are summed.
  const bytes: Record<string, [number, number]> = {};
  const progreso = (clave: string) => (recibidos: number, total: number) => {
    bytes[clave] = [recibidos, total];
    const suma = Object.values(bytes);
    const hechos = suma.reduce((a, [r]) => a + r, 0);
    const todos = suma.reduce((a, [, t]) => a + t, 0);
    if (todos) barra.style.width = `${Math.min(100, (100 * hechos) / todos).toFixed(1)}%`;
    estado.textContent = `descargando modelos… ${(hechos / 1e6).toFixed(0)} MB`;
  };
  try {
    contrato = await (await fetch("/models/contrato.json")).json();
    voces = await (await fetch("/models/voces.json")).json();
    const [t, c] = await Promise.all([
      crearSesion(`/models/tts${precision}.onnx`, proveedores, progreso("tts")),
      crearSesion(`/models/conversor${precision}.onnx`, proveedores, progreso("conv")),
    ]);
    // The voice extractor carries a GRU, which WebGPU has no kernel for: wasm.
    grafoVoz = await crearSesion(`/models/voz${precision}.onnx`, ["wasm"], progreso("voz"));
    tts = t;
    grafoConversor = c;
    barra.style.width = "100%";
    const mb = ((t.bytes + c.bytes + grafoVoz.bytes) / 1e6).toFixed(1);
    estado.textContent = "preparando fonemizador y voces…";
    log(`espeak-ng: ${await versionEspeak()}`);
    pintarVoces();
    for (const id of ["fichero", "grabar", "sintetizar", "sinConvertir"])
      ($(id) as HTMLButtonElement).disabled = false;
    const resumen = `tts ${t.proveedor} · conversor ${c.proveedor} · voz ${grafoVoz.proveedor} · ${mb} MB · ${hilos()} hilo(s) wasm · aislado=${crossOriginIsolated}`;
    estado.textContent = `listo · ${resumen}`;
    pastilla("modelos", `${precision || "fp32"} · ${t.proveedor}`, "listo");
    log(`modelos cargados: ${resumen}`);
  } catch (err) {
    estado.textContent = `error: ${String(err)}`;
    pastilla("modelos", "error", "error");
    log(String(err));
  } finally {
    ($("cargar") as HTMLButtonElement).disabled = false;
  }
}

// ---------------------------------------------------------------- voice

function pintarVoces(): void {
  const caja = $("voces");
  caja.textContent = "";
  if (!voces) {
    $("infoVoz").textContent = "sin models/voces.json";
    return;
  }
  voces.voces.forEach((v, i) => {
    const [id, ...resto] = v.nombre.split(" · ");
    const tarjeta = document.createElement("button");
    tarjeta.className = "voz";
    tarjeta.type = "button";
    tarjeta.setAttribute("role", "radio");
    tarjeta.setAttribute("aria-checked", "false");
    tarjeta.dataset.voz = String(i);
    const nombre = document.createElement("b");
    nombre.textContent = id;
    const detalle = document.createElement("span");
    detalle.textContent = `${v.idioma} · ${resto.join(" · ") || "voz"}`;
    tarjeta.append(nombre, detalle);
    tarjeta.addEventListener("click", () => elegirVoz(i));
    caja.appendChild(tarjeta);
  });
  $("infoVoz").textContent = `${voces.voces.length} presets · voz base ${voces.base.locutor}`;
}

function marcar(indice: number | null): void {
  for (const el of Array.from(document.querySelectorAll<HTMLElement>(".voz")))
    el.setAttribute("aria-checked", String(el.dataset.voz === String(indice)));
}

function elegirVoz(indice: number): void {
  if (!voces) return;
  const v = voces.voces[indice];
  vozDestino = Float32Array.from(v.voz);
  marcar(indice);
  // The base TTS is monolingual: a preset in another language only changes the
  // timbre, never what language is spoken.
  if (contrato.idiomas.includes(v.idioma)) ($("idioma") as HTMLSelectElement).value = v.idioma;
  $("infoVoz").textContent = `preset ${v.nombre}`;
  pastilla("voz", v.id, "listo");
  log(`voz: ${v.nombre}`);
}

const srConversor = () => contrato.grafos.voz.frecuencia_entrada_hz ?? 22050;

async function vozDesdeOnda(onda: Float32Array, origen: string): Promise<void> {
  if (!grafoVoz) return;
  if (onda.length < srConversor() * 2) {
    log(`${origen}: solo ${(onda.length / srConversor()).toFixed(1)} s; hacen falta 3 s o más`);
    pastilla("voz", "muy corta", "error");
    return;
  }
  pastilla("voz", "analizando…", "trabajando");
  const t0 = performance.now();
  vozDestino = await vectorVoz(grafoVoz, contrato, onda);
  marcar(null);
  $("infoVoz").textContent =
    `voz clonada de ${origen}: ${(onda.length / srConversor()).toFixed(1)} s de audio → vector en ${(performance.now() - t0).toFixed(0)} ms`;
  pastilla("voz", "clonada", "listo");
  log($("infoVoz").textContent ?? "");
}

($("fichero") as HTMLInputElement).addEventListener("change", async (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (f) await vozDesdeOnda(await decodificar(await f.arrayBuffer(), srConversor()), f.name);
});

$("grabar").addEventListener("click", async () => {
  const boton = $("grabar") as HTMLButtonElement;
  const medidor = $<HTMLCanvasElement>("nivel");
  boton.disabled = true;
  try {
    const onda = await grabarPCM(6, srConversor(), (s, nivel) => {
      boton.textContent = `grabando… ${Math.max(0, 6 - s).toFixed(1)} s`;
      pintarNivel(medidor, nivel);
    });
    pintarNivel(medidor, 0);
    await vozDesdeOnda(onda, "micrófono");
  } catch (err) {
    log(`micrófono: ${String(err)}`);
    pastilla("voz", "sin micrófono", "error");
  } finally {
    boton.textContent = "grabar 6 s";
    boton.disabled = false;
  }
});

// ---------------------------------------------------------------- speech

async function hablar(conConversor: boolean): Promise<void> {
  if (!tts || !voces) return;
  if (conConversor && !vozDestino) {
    log("elige un preset o clona una voz antes de sintetizar");
    pastilla("audio", "falta la voz", "error");
    return;
  }
  const botones = ["sintetizar", "sinConvertir"].map((id) => $(id) as HTMLButtonElement);
  for (const b of botones) b.disabled = true;
  pastilla("audio", "generando…", "trabajando");
  visor.poner("trabajando");
  try {
    const semilla = Number(($("semilla") as HTMLInputElement).value);
    const base = await sintetizar(
      tts,
      contrato,
      ($("texto") as HTMLTextAreaElement).value,
      ($("idioma") as HTMLSelectElement).value,
      voces.base.embedding_tts ? Float32Array.from(voces.base.embedding_tts) : null,
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
    const tramoConversor = conConversor
      ? ` · conversor ${msConv.toFixed(0)} ms`
      : " · sin convertir";
    const rtf = ((base.ms_modelo + msConv) / 1000 / segundos).toFixed(3);
    $("medidas").textContent =
      `${base.tokens} tokens · frontend ${base.ms_frontend.toFixed(0)} ms · ` +
      `tts ${base.ms_modelo.toFixed(0)} ms${tramoConversor} · ${segundos.toFixed(2)} s · RTF ${rtf}`;
    log($("medidas").textContent ?? "");
    pastilla("audio", `${segundos.toFixed(1)} s`, "listo");
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
    const enlace = $("descargar") as HTMLAnchorElement;
    enlace.href = URL.createObjectURL(aWav(onda, sr));
    enlace.download = conConversor ? "ttspro-voz.wav" : "ttspro-base.wav";
    enlace.hidden = false;
    visor.mostrar(onda, sr);
    // Re-enabled BEFORE playback: the audio lasts seconds and there is no reason
    // to keep the page frozen while it sounds.
    for (const b of botones) b.disabled = false;
    await visor.reproducir(reproducir);
  } catch (err) {
    log(`síntesis: ${String(err)}`);
    pastilla("audio", "error", "error");
    visor.poner("vacio");
  } finally {
    for (const b of botones) b.disabled = false;
  }
}

$("cargar").addEventListener("click", cargarModelos);
$("sintetizar").addEventListener("click", () => hablar(true));
$("sinConvertir").addEventListener("click", () => hablar(false));
$("repetir").addEventListener("click", () => {
  if (ultima) visor.reproducir(reproducir);
});

// Theme: the page follows the system unless the visitor says otherwise, and the
// choice sticks. The waveform reads its colours from CSS, so it repaints too.
const tema = $("tema") as HTMLSelectElement;
tema.value = localStorage.getItem("ttspro-tema") ?? "auto";
const aplicarTema = () => {
  if (tema.value === "auto") document.documentElement.removeAttribute("data-tema");
  else document.documentElement.dataset.tema = tema.value;
  localStorage.setItem("ttspro-tema", tema.value);
  visor.pintar();
};
tema.addEventListener("change", aplicarTema);
aplicarTema();

for (const id of ["noise", "noise_w", "length", "tau"]) {
  const entrada = $(id) as HTMLInputElement;
  const etiqueta = $(`${id}_v`);
  const pintar = () => {
    etiqueta.textContent = entrada.value;
  };
  entrada.addEventListener("input", pintar);
  pintar();
}
