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
import { type Hardware, calibrar, describir, detectar, recomendar } from "../runtime/hardware.ts";
import { type Proveedor, type Sesion, crearSesion, hilos, soportaF16 } from "../runtime/ort.ts";
import {
  type MetaPaquete,
  URL_VOCES,
  URL_VOCES_PAGES,
  type VozBase,
  cargarIndice,
  cargarPaquete,
} from "../runtime/paquetes.ts";
import { sintetizar } from "../runtime/sintetizador.ts";
import { conectarSSE, conectarWebSocket, escucharPostMessage } from "./fuentes.ts";
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
// Base voices: "local" is models/tts.onnx; the rest are packs downloaded on demand
// (ADR 0011). Each carries its own contract and base-voice vector.
const bases = new Map<string, VozBase>();
let baseActual = "local";
let indicePaquetes: Record<string, MetaPaquete> = {};
let hardware: Hardware | null = null;
let precisionCargada: "" | ".fp16" = "";
let proveedoresCargados: Proveedor[] = ["wasm"];
const hablanteActual = () => bases.get(baseActual) as VozBase;
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
    precisionCargada = precision as "" | ".fp16";
    proveedoresCargados = proveedores;
    bases.clear();
    const presets = voces as Voces;
    bases.set("local", {
      meta: {
        clave: "local",
        nombre: presets.base.locutor,
        idioma: presets.base.idioma,
        region: "",
        calidad: "",
        licencia: "",
        parametros_M: 0,
        MB: { fp32: 0, fp16: 0 },
        ms_frase_cpu: 0,
        ficheros: { fp32: "tts.onnx", fp16: "tts.fp16.onnx" },
      },
      sesion: t,
      contrato,
      vozBase: Float32Array.from(presets.base.voz),
    });
    baseActual = "local";
    ($("vozBase") as HTMLSelectElement).value = "local";
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
      "calibrar",
      "descargarVozBase",
      "descargarTodas",
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
  const entrada = ev.target as HTMLInputElement;
  const f = entrada.files?.[0];
  if (!f) return;
  // Before this had no try/catch: a file the browser cannot decode (some .m4a,
  // a video, a corrupt download) failed silently and the button looked dead.
  pastilla("voz", "leyendo fichero…", "trabajando");
  try {
    if (!grafoVoz) throw new Error("carga los modelos antes de elegir un fichero");
    let onda = await decodificar(await f.arrayBuffer(), srConversor());
    // A whole podcast is not a reference: 3-10 s is what the scope asks for, and the
    // voice extractor (a GRU) walks every sample. Keep the first 15 s.
    const maximo = srConversor() * 15;
    if (onda.length > maximo) {
      log(
        `${f.name}: ${(onda.length / srConversor()).toFixed(0)} s de audio, se usan los primeros 15`,
      );
      onda = onda.slice(0, maximo);
    }
    await vozDesdeOnda(onda, f.name);
  } catch (err) {
    log(`fichero ${f.name}: ${String(err)}`);
    pastilla("voz", "no se pudo leer", "error");
    $("infoVoz").textContent = `no se pudo leer ${f.name}: ${String(err).slice(0, 90)}`;
  } finally {
    // so choosing the same file again fires `change` again
    entrada.value = "";
  }
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

async function hablar(pedirConversor: boolean): Promise<void> {
  if (!tts || !voces) return;
  // "sintetizar" without a target voice (no preset, nothing cloned) is not an error:
  // the base voice speaks on its own. A downloaded base voice IS a voice; the
  // converter only repaints the timbre onto a preset or a clone.
  const conConversor = pedirConversor && vozDestino !== null;
  if (pedirConversor && !conConversor) {
    log(`sin voz destino (preset o clonada): habla la voz base ${hablanteActual().meta.nombre}`);
  }
  const botones = ["sintetizar", "sinConvertir"].map((id) => $(id) as HTMLButtonElement);
  for (const b of botones) b.disabled = true;
  pastilla("audio", "generando…", "trabajando");
  visor.poner("trabajando");
  try {
    const semilla = Number(($("semilla") as HTMLInputElement).value);
    const hablante = hablanteActual();
    const base = await sintetizar(
      hablante.sesion,
      hablante.contrato,
      ($("texto") as HTMLTextAreaElement).value,
      hablante.contrato.idiomas[0],
      baseActual === "local" && voces.base.embedding_tts
        ? Float32Array.from(voces.base.embedding_tts)
        : null,
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
      const r = await convertir(grafoConversor, contrato, base.onda, hablante.vozBase, vozDestino, {
        tau: Number(($("tau") as HTMLInputElement).value),
        semilla,
      });
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
  const hablante = bases.get(m.vozBase ?? baseActual) ?? hablanteActual();
  const base = await sintetizar(
    hablante.sesion,
    hablante.contrato,
    m.texto,
    hablante.contrato.idiomas[0],
    hablante.meta.clave === "local" && voces.base.embedding_tts
      ? Float32Array.from(voces.base.embedding_tts)
      : null,
    {
      noise_scale: Number(($("noise") as HTMLInputElement).value),
      noise_scale_w: Number(($("noise_w") as HTMLInputElement).value),
      length_scale: Number(($("length") as HTMLInputElement).value),
      semilla,
    },
  );
  if (!m.voz || !grafoConversor) return { onda: base.onda, sampleRate: base.sampleRate };
  const r = await convertir(grafoConversor, contrato, base.onda, hablante.vozBase, m.voz, {
    tau: Number(($("tau") as HTMLInputElement).value),
    semilla,
  });
  return { onda: r.onda, sampleRate: r.sampleRate };
}

function vozBaseParaUsuario(usuario: string): string | undefined {
  const claves = [...bases.keys()];
  if (claves.length < 2) return undefined;
  let h = 0;
  for (const c of usuario) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return claves[h % claves.length];
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
  const vozBase = politica === "baseUsuario" ? vozBaseParaUsuario(usuario) : undefined;
  const nombre = ($("leerNombre") as HTMLInputElement).checked ? `${nombreLegible(usuario)}: ` : "";
  const aceptado = cola.encolar({ texto: nombre + texto, usuario, voz, vozBase });
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

// The source is whatever hands lines to `recibir`: Twitch itself, or another
// application that already reads the chat (streex, Rails) over WebSocket /
// ActionCable, SSE, or postMessage when this page sits in its <iframe>.
const fuente = $("fuente") as HTMLSelectElement;
const mostrarCamposDeFuente = () => {
  for (const el of Array.from(document.querySelectorAll<HTMLElement>("[data-fuente]")))
    el.hidden = !(el.dataset.fuente ?? "").split(" ").includes(fuente.value);
};
fuente.addEventListener("change", mostrarCamposDeFuente);
mostrarCamposDeFuente();

const alEstadoFuente = (estado: string, detalle?: string) => {
  log(`${fuente.value}: ${estado}${detalle ? ` (${detalle})` : ""}`);
  $("conectar").textContent =
    estado === "cerrado" || estado === "error" ? "conectar" : "desconectar";
  if (estado === "conectado") pastilla("chat", "escuchando", "listo");
  else if (estado === "conectando") pastilla("chat", "conectando…", "trabajando");
  else pastilla("chat", estado, estado === "error" ? "error" : "");
  if (estado === "cerrado" || estado === "error") desconectar = null;
};

$("conectar").addEventListener("click", () => {
  if (desconectar) {
    desconectar();
    desconectar = null;
    return;
  }
  const valor = (id: string) => ($(id) as HTMLInputElement).value.trim();
  const alLinea = (l: { usuario: string; texto: string }) => recibir(l.usuario || "chat", l.texto);
  switch (fuente.value) {
    case "twitch": {
      if (!valor("canal")) return log("chat: escribe el nombre del canal");
      desconectar = conectarTwitch(
        valor("canal"),
        (m) => recibir(m.usuario, m.texto),
        alEstadoFuente,
      );
      break;
    }
    case "websocket": {
      if (!valor("url")) return log("chat: escribe la URL del WebSocket");
      desconectar = conectarWebSocket(valor("url"), valor("canalCable"), alLinea, alEstadoFuente);
      break;
    }
    case "sse": {
      if (!valor("url")) return log("chat: escribe la URL del EventSource");
      desconectar = conectarSSE(valor("url"), alLinea, alEstadoFuente);
      break;
    }
    case "postmessage":
      desconectar = escucharPostMessage(valor("origen") || "*", alLinea, alEstadoFuente);
      break;
  }
});

// For the e2e test and for anyone wiring their own source: push a line in.
(window as unknown as { __ttspro_chat: unknown }).__ttspro_chat = {
  recibir,
  estadisticas: () => cola.estadisticas,
};

// ---------------------------------------------------------------- hardware (ADR 0011)
//
// Detect on load and say what the rule recommends; "calibrar" measures instead of
// guessing: one sentence through the TTS and the converter on each candidate.

void (async () => {
  hardware = await detectar();
  const regla = recomendar(hardware);
  $("hardware").textContent =
    `${describir(hardware)} · recomendado: ${regla.proveedor} ${regla.precision || "fp32"} (${regla.motivo})`;
  const guardada = localStorage.getItem("ttspro-calibracion");
  if (guardada) {
    const c = JSON.parse(guardada) as { proveedor: Proveedor; precision: "" | ".fp16" };
    ($("proveedor") as HTMLSelectElement).value = c.proveedor;
    ($("precision") as HTMLSelectElement).value = c.precision;
    log(`calibración guardada: ${c.proveedor} ${c.precision || "fp32"}`);
  } else if (($("proveedor") as HTMLSelectElement).value === "auto") {
    ($("proveedor") as HTMLSelectElement).value = regla.proveedor;
    ($("precision") as HTMLSelectElement).value = regla.precision;
  }
})();

$("calibrar").addEventListener("click", async () => {
  if (!hardware || !voces) return;
  const boton = $("calibrar") as HTMLButtonElement;
  boton.disabled = true;
  const salida = $("calibracion");
  salida.hidden = false;
  salida.textContent = "calibrando…";
  const candidatos: Array<[Proveedor, "" | ".fp16"]> = hardware.webgpu
    ? hardware.f16
      ? [
          ["webgpu", ".fp16"],
          ["webgpu", ""],
          ["wasm", ""],
        ]
      : [
          ["webgpu", ""],
          ["wasm", ""],
        ]
    : [["wasm", ""]];
  const frase = "Hola a todos, bienvenidos al directo de hoy.";
  const filas: string[] = [];
  const pinta = () => {
    salida.textContent = filas.join("\n");
  };
  try {
    const tts = await calibrar(
      (p) => `/models/tts${p}.onnx`,
      async (s) => (await sintetizar(s, contrato, frase, contrato.idiomas[0], null, {})).onda,
      candidatos,
      (m) => {
        filas.push(
          `tts ${m.proveedor} ${m.precision || "fp32"}: ${m.ms === null ? m.error : `${m.ms.toFixed(0)} ms`}`,
        );
        pinta();
      },
    );
    const baseOnda = (
      await sintetizar(
        hablanteActual().sesion,
        hablanteActual().contrato,
        frase,
        contrato.idiomas[0],
        null,
        {},
      )
    ).onda;
    const origen = hablanteActual().vozBase;
    const destino = Float32Array.from(voces.voces[0]?.voz ?? voces.base.voz);
    const conv = await calibrar(
      (p) => `/models/conversor${p}.onnx`,
      async (s) => (await convertir(s, contrato, baseOnda, origen, destino, {})).onda,
      candidatos,
      (m) => {
        filas.push(
          `conversor ${m.proveedor} ${m.precision || "fp32"}: ${m.ms === null ? m.error : `${m.ms.toFixed(0)} ms`}`,
        );
        pinta();
      },
    );
    const mejor = conv.mejor ?? tts.mejor;
    if (mejor) {
      filas.push(
        `→ mejor: ${mejor.proveedor} ${mejor.precision || "fp32"} (recarga los modelos para aplicarlo)`,
      );
      ($("proveedor") as HTMLSelectElement).value = mejor.proveedor;
      ($("precision") as HTMLSelectElement).value = mejor.precision;
      localStorage.setItem(
        "ttspro-calibracion",
        JSON.stringify({ proveedor: mejor.proveedor, precision: mejor.precision }),
      );
    }
    pinta();
    log(`calibración: ${filas.join(" · ")}`);
  } catch (err) {
    salida.textContent = `calibración: ${String(err)}`;
  } finally {
    boton.disabled = false;
  }
});

// ---------------------------------------------------------------- base voice packs (ADR 0011)

const selectorBase = $("vozBase") as HTMLSelectElement;
// Where the packs come from, in order: `?voces=<url>` (the e2e serves a local
// folder), what the visitor typed last time, the project's default. Any static
// host with CORS works: a Hugging Face repo is
// `https://huggingface.co/<usuario>/<repo>/resolve/main/`.
const origenVoces = $("origenVoces") as HTMLInputElement;
let urlVoces =
  new URLSearchParams(location.search).get("voces") ??
  localStorage.getItem("ttspro-voces") ??
  URL_VOCES;
origenVoces.value = urlVoces;

// Only the LAST origin asked for gets to paint: a slow failure from a previous
// origin must not overwrite the index of the current one.
let peticionIndice = 0;
async function pintarIndice(): Promise<void> {
  const mia = ++peticionIndice;
  const elegida = selectorBase.value;
  for (const o of Array.from(selectorBase.options)) if (o.value !== "local") o.remove();
  indicePaquetes = {};
  let indice: Record<string, MetaPaquete>;
  try {
    indice = await cargarIndice(urlVoces);
  } catch (err) {
    if (mia !== peticionIndice) return;
    $("infoVozBase").textContent =
      `sin índice de voces en ${urlVoces} (${String(err).slice(0, 60)})`;
    return;
  }
  if (mia !== peticionIndice) return;
  indicePaquetes = indice;
  for (const m of Object.values(indicePaquetes)) {
    const o = document.createElement("option");
    o.value = m.clave;
    o.textContent = `${m.nombre} · ${m.calidad} · ${m.MB.fp16} MB`;
    selectorBase.appendChild(o);
  }
  if (elegida in indicePaquetes) selectorBase.value = elegida;
  $("infoVozBase").textContent = `${Object.keys(indicePaquetes).length} voces base descargables`;
}
void pintarIndice();
function cambiarOrigen(url: string): void {
  const nueva = url.trim().replace(/\/?$/, "/");
  // the input fires `change` again on blur when its value was set by code: same origin, nothing to do
  if (nueva === urlVoces) return;
  urlVoces = nueva;
  origenVoces.value = urlVoces;
  localStorage.setItem("ttspro-voces", urlVoces);
  log(`origen de voces: ${urlVoces}`);
  void pintarIndice();
}
origenVoces.addEventListener("change", () => cambiarOrigen(origenVoces.value));
// One click instead of a URL: a Hugging Face repo is `<usuario>/<repo>` and the
// files hang from `resolve/main/`; GitHub Pages is the project's default.
$("usarHF").addEventListener("click", () => {
  const repo = ($("repoHF") as HTMLInputElement).value
    .trim()
    .replace(/^https?:\/\/huggingface\.co\//, "")
    .replace(/\/+$/, "");
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    log("repo de Hugging Face: escribe usuario/nombre");
    return;
  }
  cambiarOrigen(`https://huggingface.co/${repo}/resolve/main/`);
});
$("usarPages").addEventListener("click", () => cambiarOrigen(URL_VOCES_PAGES));

async function descargarBase(clave: string): Promise<void> {
  if (bases.has(clave)) return;
  const barra = $("progreso");
  const info = $("infoVozBase");
  const mb = indicePaquetes[clave]?.MB[precisionCargada === ".fp16" ? "fp16" : "fp32"];
  // The person is looking at THIS panel, not at panel 1: say what is happening here.
  // A 114 MB pack takes two minutes on a slow line and looked dead without this.
  info.textContent = `descargando ${clave}${mb ? ` (${mb} MB)` : ""}…`;
  pastilla("modelos", `descargando ${clave}…`, "trabajando");
  const t0 = performance.now();
  const voz = await cargarPaquete(
    urlVoces,
    clave,
    contrato,
    precisionCargada,
    proveedoresCargados,
    (r, t) => {
      if (t) barra.style.width = `${Math.min(100, (100 * r) / t).toFixed(1)}%`;
      const seg = (performance.now() - t0) / 1000;
      const ritmo = seg > 0.5 ? ` · ${(r / 1e6 / seg).toFixed(1)} MB/s` : "";
      info.textContent = `descargando ${clave}: ${(r / 1e6).toFixed(0)}${t ? ` / ${(t / 1e6).toFixed(0)}` : ""} MB${ritmo}`;
      if (t && r >= t) info.textContent = `abriendo ${clave} en ${proveedoresCargados[0]}…`;
    },
  );
  bases.set(clave, voz);
  pastilla("modelos", `${precisionCargada || "fp32"} · ${voz.sesion.proveedor}`, "listo");
  log(
    `voz base ${clave}: ${(voz.sesion.bytes / 1e6).toFixed(1)} MB en ${voz.sesion.ms_carga.toFixed(0)} ms, ${voz.sesion.proveedor}`,
  );
}

selectorBase.addEventListener("change", () => {
  if (bases.has(selectorBase.value)) {
    baseActual = selectorBase.value;
    $("infoVozBase").textContent =
      `voz base: ${hablanteActual().meta.nombre} · pulsa «sintetizar» o «solo voz base»`;
  } else {
    $("infoVozBase").textContent = "no descargada: pulsa «descargar y usar»";
  }
});

$("descargarVozBase").addEventListener("click", async () => {
  if (!tts) {
    $("infoVozBase").textContent = "carga los modelos (panel 1) antes de descargar una voz base";
    return;
  }
  const clave = selectorBase.value;
  const boton = $("descargarVozBase") as HTMLButtonElement;
  boton.disabled = true;
  try {
    await descargarBase(clave);
    baseActual = clave;
    $("infoVozBase").textContent =
      `voz base: ${hablanteActual().meta.nombre} · pulsa «sintetizar» o «solo voz base»`;
  } catch (err) {
    log(`voz base ${clave}: ${String(err)}`);
    $("infoVozBase").textContent = `error con ${clave}: ${String(err).slice(0, 120)}`;
    pastilla("modelos", "error", "error");
  } finally {
    boton.disabled = false;
  }
});

$("descargarTodas").addEventListener("click", async () => {
  if (!tts) return;
  const boton = $("descargarTodas") as HTMLButtonElement;
  boton.disabled = true;
  try {
    for (const clave of Object.keys(indicePaquetes)) await descargarBase(clave);
    $("infoVozBase").textContent = `${bases.size} voces base cargadas`;
  } catch (err) {
    log(`voces base: ${String(err)}`);
  } finally {
    boton.disabled = false;
  }
});
