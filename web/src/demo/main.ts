/**
 * The test page. Chain (ADR 0007 and 0008): text -> tts.onnx in a fixed base
 * voice -> conversor.onnx in the chosen voice. The voice is either a preset
 * shipped in models/voces.json or one cloned from a recording through voz.onnx.
 *
 * Every number shown is measured in this browser.
 */

import { nombreLegible, normalizarChat } from "../frontend/chat.ts";
import { configurarEspeak, versionEspeak } from "../frontend/fonemas.ts";
import { aWav, decodificar, grabarPCM, reproducir } from "../runtime/audio.ts";
import { Cola, type Estadisticas, type Mensaje } from "../runtime/cola.ts";
import type { Contrato } from "../runtime/contrato.ts";
import { convertir, vectorVoz } from "../runtime/conversor.ts";
import { type Proveedor, type Sesion, crearSesion, hilos, soportaF16 } from "../runtime/ort.ts";
import { sintetizar } from "../runtime/sintetizador.ts";
import { conectarTwitch } from "./twitch.ts";
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
    for (const id of [
      "fichero",
      "grabar",
      "sintetizar",
      "sinConvertir",
      "conectar",
      "enviar",
      "simular",
    ])
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

// ---------------------------------------------------------------- chat (ADR 0010)
//
// A stream of messages instead of one sentence: chat normalizer -> queue that
// renders the next message while the current one plays. The voice policy
// decides whether each message goes through the converter, which on wasm costs
// ~3 s per message and therefore decides whether the reader keeps up.

const chatLog = $("chatLog");
const anotar = (usuario: string, texto: string, nota = "") => {
  const linea = document.createElement("div");
  const quien = document.createElement("b");
  quien.textContent = usuario;
  linea.append(quien, ` ${texto}`);
  if (nota) {
    const n = document.createElement("i");
    n.textContent = `  · ${nota}`;
    linea.append(n);
  }
  chatLog.prepend(linea);
  while (chatLog.childElementCount > 60) chatLog.lastElementChild?.remove();
};

function vozParaUsuario(usuario: string): Float32Array | null {
  if (!voces || voces.voces.length === 0) return null;
  let h = 0;
  for (const c of usuario) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return Float32Array.from(voces.voces[h % voces.voces.length].voz);
}

async function sintetizarMensaje(m: Mensaje) {
  if (!tts || !voces) return null;
  const semilla = Number(($("semilla") as HTMLInputElement).value);
  const base = await sintetizar(
    tts,
    contrato,
    m.texto,
    ($("idioma") as HTMLSelectElement).value,
    voces.base.embedding_tts ? Float32Array.from(voces.base.embedding_tts) : null,
    {
      noise_scale: Number(($("noise") as HTMLInputElement).value),
      noise_scale_w: Number(($("noise_w") as HTMLInputElement).value),
      length_scale: Number(($("length") as HTMLInputElement).value),
      semilla,
    },
  );
  if (!m.voz || !grafoConversor) return { onda: base.onda, sampleRate: base.sampleRate };
  const r = await convertir(
    grafoConversor,
    contrato,
    base.onda,
    Float32Array.from(voces.base.voz),
    m.voz,
    {
      tau: Number(($("tau") as HTMLInputElement).value),
      semilla,
    },
  );
  return { onda: r.onda, sampleRate: r.sampleRate };
}

const pintarStats = (s: Estadisticas, evento: string, m?: Mensaje) => {
  $("colaStats").textContent =
    `pendientes ${s.pendientes} · leídos ${s.reproducidos} · descartados ` +
    `${s.descartadosViejos} viejos, ${s.descartadosLlenos} por cola llena, ${s.descartadosRepetidos} repetidos · ` +
    `espera media ${(s.esperaMediaMs / 1000).toFixed(1)} s · síntesis media ${s.sintesisMediaMs.toFixed(0)} ms · ` +
    `velocidad ×${s.velocidad.toFixed(2)}`;
  if (evento === "sonando" && m) anotar(m.usuario ?? "", m.texto, "sonando");
  else if ((evento === "viejo" || evento === "lleno") && m)
    anotar(m.usuario ?? "", m.texto, `descartado: ${evento}`);
  else if (evento.startsWith("error") && m) anotar(m.usuario ?? "", m.texto, evento);
  if (s.pendientes > 0 || evento === "sonando")
    pastilla("chat", `${s.pendientes} en cola`, "trabajando");
  else if (evento === "vacía") pastilla("chat", desconectar ? "escuchando" : "al día", "listo");
};

let cola = new Cola(
  sintetizarMensaje,
  (o) => {
    visor.mostrar(o.onda, o.sampleRate);
    return visor.reproducir(reproducir);
  },
  pintarStats,
);
let desconectar: (() => void) | null = null;

/** What the demo does with one chat line, wherever it came from. */
function recibir(usuario: string, textoBruto: string): void {
  const maxChars = Number(($("maxChars") as HTMLInputElement).value) || 200;
  const texto = normalizarChat(textoBruto, maxChars);
  if (!texto) {
    anotar(usuario, textoBruto, "nada que leer");
    return;
  }
  const politica = ($("politica") as HTMLSelectElement).value;
  const voz =
    politica === "elegida" ? vozDestino : politica === "usuario" ? vozParaUsuario(usuario) : null;
  const nombre = ($("leerNombre") as HTMLInputElement).checked ? `${nombreLegible(usuario)}: ` : "";
  const aceptado = cola.encolar({ texto: nombre + texto, usuario, voz });
  if (aceptado)
    anotar(usuario, texto, texto === textoBruto ? "" : `de: ${textoBruto.slice(0, 60)}`);
}

$("enviar").addEventListener("click", () => {
  const entrada = $("mensaje") as HTMLInputElement;
  if (entrada.value.trim()) recibir("tú", entrada.value);
  entrada.value = "";
});
$("mensaje").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("enviar").click();
});
$("vaciarCola").addEventListener("click", () => cola.vaciar());
$("maxEdad").addEventListener("change", () => {
  cola.parar();
  cola = new Cola(
    sintetizarMensaje,
    (o) => {
      visor.mostrar(o.onda, o.sampleRate);
      return visor.reproducir(reproducir);
    },
    pintarStats,
    { maxEdadMs: Number(($("maxEdad") as HTMLInputElement).value) * 1000 },
  );
});

const SIMULACION: Array<[string, string]> = [
  ["Pepe_Gamer", "holaaaa q tal todos!!!!"],
  // the same line twice in a row: the queue must refuse the second while the first waits
  ["Pepe_Gamer", "holaaaa q tal todos!!!!"],
  ["xXDark_LordXx", "JAJAJAJAJA no me lo creo"],
  ["Luci", "KEKW KEKW KEKW"],
  ["mod_ana", "mira esto https://clips.twitch.tv/abc123 brutal"],
  ["Raúl99", "GG WP EZ"],
  ["Pepe_Gamer", "xq no juegas al lol?"],
  ["nerea", "tb me pasa a mi, ntp"],
  ["Carlos", "pls sube el volumen porfa"],
  ["Luci", "***spoiler*** el final es una locura"],
  ["mod_ana", "el murciélago vuela de nooooche"],
  ["Raúl99", "¿¿¿qué???!!! jajaja"],
  ["nerea", "cómo se llama la canción de fondo"],
  ["Carlos", "CÁLLATE YA 😂😂😂"],
  ["Pepe_Gamer", "siiiiii vamosssss"],
  ["Luci", "Kappa Kappa PogChamp"],
  ["mod_ana", "llego tarde, perdón, tenía que ir al médico"],
  ["Raúl99", "salu2 grax por el stream tqm"],
  ["nerea", "3,5 euros a las 10:30 en el 2º piso"],
  ["Carlos", "no no no no no NO"],
];
$("simular").addEventListener("click", () => {
  let i = 0;
  const paso = () => {
    if (i >= SIMULACION.length) return;
    const [usuario, texto] = SIMULACION[i++];
    recibir(usuario, texto);
    setTimeout(paso, 150);
  };
  paso();
});

$("conectar").addEventListener("click", () => {
  if (desconectar) {
    desconectar();
    desconectar = null;
    return;
  }
  const canal = ($("canal") as HTMLInputElement).value.trim();
  if (!canal) {
    log("chat: escribe el nombre del canal");
    return;
  }
  desconectar = conectarTwitch(
    canal,
    (m) => recibir(m.usuario, m.texto),
    (estado, detalle) => {
      log(`twitch: ${estado}${detalle ? ` (${detalle})` : ""}`);
      $("conectar").textContent =
        estado === "cerrado" || estado === "error" ? "conectar" : "desconectar";
      if (estado === "conectado") pastilla("chat", "escuchando", "listo");
      else if (estado === "conectando") pastilla("chat", "conectando…", "trabajando");
      else pastilla("chat", estado, estado === "error" ? "error" : "");
      if (estado === "cerrado" || estado === "error") desconectar = null;
    },
  );
});

// For the e2e test and for anyone wiring their own source: push a line in.
(window as unknown as { __ttspro_chat: unknown }).__ttspro_chat = {
  recibir,
  estadisticas: () => cola.estadisticas,
};
