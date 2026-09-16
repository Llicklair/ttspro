/**
 * La página: dos modos sobre un solo motor (ADR 0012).
 *
 * Lo que cambió respecto de la página de los cuatro pasos numerados no es el
 * aspecto, es el modelo mental. Antes había una cadena que el usuario tenía que
 * entender —una voz base fija, un conversor detrás, y dos conceptos de "voz" que
 * se elegían en sitios distintos— y la página la reflejaba fielmente, que es
 * justo por qué era confusa. Con Supertonic la voz es una entrada del modelo, así
 * que hay UNA lista de voces, eliges una y habla. Lo que quedaba de la cadena
 * (proveedor, precisión, hardware) se va a Ajustes, porque es de la máquina, no
 * de la tarea.
 */

import { nombreLegible, normalizarChat } from "../frontend/chat.ts";
import { normalizar } from "../frontend/normalizar.ts";
import { IDIOMAS } from "../frontend/unicode.ts";
import { aWav, grabarPCM, reproducir } from "../runtime/audio.ts";
import { Cola, type Estadisticas, type Mensaje } from "../runtime/cola.ts";
import { describir, detectar } from "../runtime/hardware.ts";
import type { Proveedor } from "../runtime/ort.ts";
import { Pocket } from "../runtime/pocket.ts";
import {
  type Estilo,
  type EstiloJSON,
  Supertonic,
  descargarEstilo,
  leerEstilo,
} from "../runtime/supertonic.ts";
import { cambiarVelocidad } from "../runtime/velocidad.ts";
import { conectarSSE, conectarWebSocket, escucharPostMessage } from "./fuentes.ts";
import { type MensajeTwitch, conectarTwitch } from "./twitch.ts";
import { Visor, pintarNivel } from "./visor.ts";

const $ = (id: string) => document.getElementById(id) as HTMLElement;
const $$ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const log = (msg: string) => {
  const pre = $("registro");
  pre.textContent += `${new Date().toLocaleTimeString()}  ${msg}\n`;
  pre.scrollTop = pre.scrollHeight;
};

const pastilla = (id: string, texto: string, tono = "") => {
  const el = $(id);
  el.textContent = texto;
  if (tono) el.dataset.tono = tono;
  else delete el.dataset.tono;
};

const dato = (id: string, texto: string) => {
  $(id).textContent = texto;
};

// ------------------------------------------------------------------ estado

interface Ficha {
  nombre: string;
  origen: "fabrica" | "construida" | "clonada";
  detalle: string;
  estilo?: Estilo; // se descarga al elegirla, no antes: 292 KB cada una
}

/** `models/supertonic/contrato.json`, generado leyendo los propios .onnx. */
interface Contrato {
  motor: string;
  version: string;
  frecuencia_salida_hz: number;
  voces: string[];
  grafos: Record<string, { fichero: string; bytes: number }>;
}

/**
 * Los dos motores, y por que hay dos.
 *
 * Pocket clona desde una grabacion en medio segundo (0,413 de similitud, el
 * mejor numero del proyecto) y pesa 177 MB. Supertonic lee algo mejor (WER 0,008
 * contra 0,027), sale a 44,1 kHz y habla 31 idiomas, pero **no sabe clonar**: su
 * encoder de voz no se publico nunca y reconstruirlo no llega — seis intentos en
 * docs/evidencia.md. Ninguno gana en todo, asi que se elige.
 */
type MotorActivo = { tipo: "supertonic"; motor: Supertonic } | { tipo: "pocket"; motor: Pocket };

let activo: MotorActivo | null = null;
let contrato: Contrato | null = null;

const tipoElegido = (): "supertonic" | "pocket" =>
  $$<HTMLSelectElement>("motor").value === "supertonic" ? "supertonic" : "pocket";

const puedeClonar = () => activo?.tipo === "pocket";
const fichas = new Map<string, Ficha>();
let vozActual: string | null = null;
let ultimo: { onda: Float32Array; sampleRate: number } | null = null;

const LOCALES = "ttspro.voces.locales";

/**
 * De donde salen los cuatro grafos. Hugging Face los sirve con las cabeceras CORS
 * correctas (la misma razon por la que los paquetes de voz viven alli, ADR 0011),
 * asi que el navegador puede bajarlos **directamente** y la pagina funciona sin
 * haber instalado nada: ni Python, ni `descargar`, ni copiar 398 MB a public/.
 * La revision va fijada porque el repo esta archivado (ADR 0012).
 */
const ORIGENES = {
  local: "/models/supertonic/",
  huggingface: "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/main/",
} as const;

/**
 * Lo que dice el contrato cuando no hay contrato.
 *
 * `contrato.json` lo genera `ttspro.supertonic.descargar` leyendo los propios
 * .onnx, asi que existe en local y NO existe en Hugging Face, que solo tiene lo
 * que publico Supertone. Estos son los unicos datos que la pagina necesita de el,
 * y valen para la revision fijada; si algun dia dejan de valer, el motor falla al
 * cargar un grafo, que es un fallo ruidoso y no silencioso.
 */
const CONTRATO_MINIMO: Contrato = {
  motor: "supertonic-3",
  version: "v1.7.3",
  frecuencia_salida_hz: 44100,
  voces: ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"],
  grafos: {},
};
const visor = new Visor($$<HTMLCanvasElement>("onda"));

const origen = () => {
  const v = $$<HTMLInputElement>("origenModelos").value.trim();
  return v.endsWith("/") ? v : `${v}/`;
};

// ------------------------------------------------------------------ modos

const botonesModo = Array.from(document.querySelectorAll<HTMLButtonElement>(".modos button"));
for (const boton of botonesModo) {
  boton.addEventListener("click", () => {
    for (const otro of botonesModo) {
      const puesto = otro === boton;
      otro.setAttribute("aria-selected", String(puesto));
      $(`modo-${otro.dataset.modo}`).hidden = !puesto;
    }
  });
}

$("abrir-ajustes").addEventListener("click", () => $$<HTMLDialogElement>("ajustes").showModal());
$("cerrar-ajustes").addEventListener("click", () => $$<HTMLDialogElement>("ajustes").close());

// ------------------------------------------------------------------ idiomas

const selectorIdioma = $$<HTMLSelectElement>("idioma");
for (const codigo of IDIOMAS) {
  if (codigo === "na") continue; // el comodín del modelo: no es un idioma que ofrecer
  const o = document.createElement("option");
  o.value = codigo;
  o.textContent = codigo;
  if (codigo === "es") o.selected = true;
  selectorIdioma.append(o);
}

// ------------------------------------------------------------------- carga

/**
 * ¿Este origen sirve de verdad los grafos?
 *
 * Se pregunta con `tts.json`, que son 8 KB, antes de comprometerse a 398 MB. Y no
 * vale mirar el codigo de estado: el servidor de desarrollo de Vite responde
 * **200 con el index.html** a cualquier ruta que no conoce, asi que hay que abrir
 * la respuesta y comprobar que es lo que dice ser.
 */
async function sirveModelos(raiz: string): Promise<boolean> {
  try {
    const r = await fetch(`${raiz}onnx/tts.json`);
    if (!r.ok) return false;
    const c = (await r.json()) as { ae?: { sample_rate?: number } };
    return typeof c?.ae?.sample_rate === "number";
  } catch {
    return false;
  }
}

/**
 * Carga el motor sin que nadie pulse nada.
 *
 * Con `respaldo`, si el origen local no tiene los grafos se pasa solo a Hugging
 * Face. Esa es la promesa de la pagina: abrirla y que hable, sin instalar nada y
 * sin tener que entender de donde salen 398 MB.
 */
// Cambiar de motor mientras uno esta cargando lanzaba DOS cargas y se quedaba la
// que acabara ultima, que no tiene por que ser la que pediste: medido el
// 2026-09-16, la pagina acababa en Pocket con las tarjetas de los dos motores
// mezcladas. Mismo patron que el turno de sintesis: solo la ultima manda.
let cargaActual = 0;

async function cargar(respaldo = true): Promise<void> {
  const mia = ++cargaActual;
  const boton = $$<HTMLButtonElement>("cargar");
  boton.disabled = true;
  const progreso = $("progreso");
  try {
    let raiz = origen();
    if (respaldo && !raiz.startsWith("http") && !(await sirveModelos(raiz))) {
      log(`${raiz} no tiene los grafos; tirando de Hugging Face`);
      $("estado").textContent = "no estan en este servidor: bajandolos de Hugging Face…";
      $$<HTMLInputElement>("origenModelos").value = ORIGENES.huggingface;
      raiz = ORIGENES.huggingface;
    }
    if (activo?.tipo === "pocket") activo.motor.cerrar();
    activo = null;
    fichas.clear();

    if (tipoElegido() === "pocket") {
      // Pocket se trae sus pesos de Hugging Face y los deja en la Cache API: el
      // `origen` de Supertonic no pinta nada aqui, y por eso no se toca.
      const pocket = await Pocket.cargar({
        idioma: "spanish",
        alCargar: (etapa, recibidos, total) => {
          progreso.style.width = `${total > 0 ? ((recibidos / total) * 100).toFixed(1) : 0}%`;
          const mb =
            total > 0 ? ` — ${(recibidos / 1e6).toFixed(0)} de ${(total / 1e6).toFixed(0)} MB` : "";
          $("estado").textContent = `descargando ${etapa}${mb}`;
        },
      });
      if (mia !== cargaActual) {
        pocket.cerrar();
        log("descartada la carga de pocket: hay otra mas nueva");
        return;
      }
      activo = { tipo: "pocket", motor: pocket };
      pintarFichas(pocket.voces);
      // El modelo espanol trae 1 paso de fabrica y admite 4; la pagina arranca en
      // 4 a proposito, porque es lo mejor que da y quien abre esto quiere oirlo
      // bien. Lo que el modelo usaria por su cuenta se dice en el registro, no se
      // impone: imponerlo seria volver al minimo sin avisar.
      log(
        `pocket: ${pocket.pasosPorDefecto} paso(s) de fabrica, temperatura ${pocket.temperaturaPorDefecto}`,
      );
      dato("e-motor", `pocket · ${pocket.sampleRate / 1000} kHz · clona`);
      dato("e-proveedor", "worker");
      $("estado").textContent = `listo: pocket en español, ${pocket.voces.length} voces`;
      log(`pocket cargado, ${pocket.voces.length} voces`);
      await elegirVoz(pocket.vozPorDefecto, false);
    } else {
      contrato = await leerContrato(raiz);
      pintarFichas(await listaDeVoces(raiz, contrato.voces));

      const preferidos: Proveedor[] =
        $$<HTMLSelectElement>("proveedor").value === "auto"
          ? ["webgpu", "wasm"]
          : [$$<HTMLSelectElement>("proveedor").value as Proveedor];

      // El progreso se cuenta por grafo, no por bytes totales: asi no depende de
      // que haya contrato, y sirve igual viniendo de Hugging Face.
      const st = await Supertonic.cargar(`${raiz}onnx/`, preferidos, (c) => {
        const dentro = c.bytes > 0 ? c.recibidos / c.bytes : 0;
        progreso.style.width = `${(((c.indice + dentro) / c.total) * 100).toFixed(1)}%`;
        const cuanto =
          c.bytes > 0
            ? ` — ${(c.recibidos / 1e6).toFixed(0)} de ${(c.bytes / 1e6).toFixed(0)} MB`
            : "";
        $("estado").textContent = `descargando ${c.grafo} (${c.indice + 1}/${c.total})${cuanto}`;
      });
      if (mia !== cargaActual) {
        log("descartada la carga de supertonic: hay otra mas nueva");
        return;
      }
      activo = { tipo: "supertonic", motor: st };
      dato("e-motor", `${contrato.motor} · ${contrato.frecuencia_salida_hz / 1000} kHz`);
      dato("e-proveedor", st.proveedor);
      $("estado").textContent = `listo: ${(st.bytes / 1e6).toFixed(0)} MB en ${st.proveedor}`;
      log(`supertonic cargado en ${st.proveedor}`);
      await elegirVoz(contrato.voces[0], false);
    }
    progreso.style.width = "100%";
    $$<HTMLButtonElement>("hablar").disabled = false;
    $$<HTMLButtonElement>("conectar").disabled = false;
    $$<HTMLButtonElement>("enviar").disabled = false;
    $$<HTMLButtonElement>("simular").disabled = false;
    await comprobarPuente();
    boton.textContent = "recargar";
  } catch (e) {
    // El servidor de desarrollo responde con el index.html a lo que no encuentra,
    // asi que un modelo que falta llega aqui como "Unexpected token '<'". Decirlo
    // tal cual no ayuda a nadie: lo que hay que decir es que hacer.
    $("estado").textContent = `no se pudo cargar desde ${origen()}: ${String(e).slice(0, 220)}`;
    pastilla("p-voces", "sin modelos", "error");
    log(`ERROR al cargar desde ${origen()}: ${String(e)}`);
  } finally {
    if (mia === cargaActual) boton.disabled = false;
  }
}

/**
 * Que voces hay en este origen.
 *
 * `voice_styles/indice.json` lo escribe `npm run preparar`, y lo reescribe el
 * voice builder al publicar una voz nueva: asi una voz construida aparece sola,
 * sin tocar el contrato ni ninguna lista. Si no existe (Hugging Face no lo
 * tiene), valen las del contrato.
 *
 * Se comprueba la forma, no el codigo de estado: el dev server de Vite responde
 * 200 con el index.html a lo que no encuentra.
 */
async function listaDeVoces(raiz: string, porDefecto: string[]): Promise<string[]> {
  try {
    const r = await fetch(`${raiz}voice_styles/indice.json`);
    if (!r.ok) throw new Error(String(r.status));
    const lista = (await r.json()) as unknown;
    if (Array.isArray(lista) && lista.length && lista.every((v) => typeof v === "string")) {
      log(`indice de voces: ${lista.length}`);
      return lista as string[];
    }
    throw new Error("no es una lista de nombres");
  } catch {
    return porDefecto;
  }
}

/** El contrato si el origen lo tiene; si no, lo minimo que la pagina necesita. */
async function leerContrato(raiz: string): Promise<Contrato> {
  try {
    const r = await fetch(`${raiz}contrato.json`);
    if (!r.ok) throw new Error(String(r.status));
    const c = (await r.json()) as Contrato;
    const mb = Object.values(c.grafos).reduce((n, g) => n + g.bytes, 0) / 1e6;
    log(
      `contrato ${c.motor} ${c.version}, ${Object.keys(c.grafos).length} grafos, ${mb.toFixed(1)} MB`,
    );
    return c;
  } catch {
    log(`sin contrato.json en ${raiz}: usando el minimo para ${CONTRATO_MINIMO.version}`);
    return CONTRATO_MINIMO;
  }
}

async function cambiarOrigen(url: string): Promise<void> {
  $$<HTMLInputElement>("origenModelos").value = url;
  if (activo?.tipo === "pocket") activo.motor.cerrar();
  activo = null;
  fichas.clear();
  $$<HTMLButtonElement>("hablar").disabled = true;
  log(`origen de los modelos: ${url}`);
  // Sin respaldo: si alguien pide expresamente un origen, se le dice si falla en
  // vez de llevarle a otro por detras.
  await cargar(false);
}

$("cargar").addEventListener("click", () => void cargar());
$("motor").addEventListener("change", () => {
  log(`motor: ${tipoElegido()}`);
  void cargar();
});
// Cambiar de origen recarga solo: dejarlo a medias, con el campo cambiado y el
// motor viejo en memoria, es un estado que nadie quiere y que miente.
$("descargarHF").addEventListener("click", () => void cambiarOrigen(ORIGENES.huggingface));
$("usarLocal").addEventListener("click", () => void cambiarOrigen(ORIGENES.local));
$("usarHF").addEventListener("click", () => void cambiarOrigen(ORIGENES.huggingface));
$("origenModelos").addEventListener(
  "change",
  () => void cambiarOrigen($$<HTMLInputElement>("origenModelos").value.trim()),
);

// ------------------------------------------------------------------- voces

/**
 * Que se escribe debajo del nombre de una voz de fabrica.
 *
 * Supertonic **documenta** su convencion — M1 a M5 masculinas, F1 a F5
 * femeninas —, asi que ahi se puede decir. Pocket no: su manifiesto trae el
 * idioma de cada voz y nada mas. Deducir el sexo del nombre daba «lola:
 * masculina», que es lo que pasa por inventarse un dato que no se tiene.
 */
function detalleDeFabrica(nombre: string): string {
  if (activo?.tipo === "pocket") return activo.motor.idiomaDeVoz(nombre) ?? "de fábrica";
  return /^F\d$/.test(nombre) ? "femenina" : /^M\d$/.test(nombre) ? "masculina" : "de fábrica";
}

function pintarFichas(deFabrica: string[]): void {
  for (const nombre of deFabrica) {
    if (!fichas.has(nombre)) {
      fichas.set(nombre, { nombre, origen: "fabrica", detalle: detalleDeFabrica(nombre) });
    }
  }
  for (const [nombre, crudo] of Object.entries(locales())) {
    if (!fichas.has(nombre)) {
      const meta = crudo.metadata ?? {};
      fichas.set(nombre, {
        nombre,
        origen: meta.construida_por ? "construida" : "clonada",
        detalle: (meta.source_file as string) ?? "tuya",
        estilo: leerEstilo(crudo, nombre),
      });
    }
  }
  render();
}

function locales(): Record<string, EstiloJSON> {
  try {
    return JSON.parse(localStorage.getItem(LOCALES) ?? "{}");
  } catch {
    return {};
  }
}

function render(): void {
  const filtro = $$<HTMLInputElement>("buscar-voz").value.trim().toLowerCase();
  const caja = $("voces");
  caja.textContent = "";
  let visibles = 0;
  for (const ficha of fichas.values()) {
    if (filtro && !ficha.nombre.toLowerCase().includes(filtro)) continue;
    visibles++;
    const tarjeta = document.createElement("button");
    tarjeta.className = "voz";
    tarjeta.setAttribute("role", "radio");
    tarjeta.setAttribute("aria-checked", String(ficha.nombre === vozActual));
    // Con createElement y textContent, no con innerHTML: el nombre de una voz
    // clonada lo escribe el usuario, y el de un fichero lo escribe su sistema.
    const trozo = (clase: string, texto: string) => {
      const el = document.createElement("span");
      el.className = clase;
      el.textContent = texto;
      return el;
    };
    tarjeta.append(trozo("nombre", ficha.nombre), trozo("detalle", ficha.detalle));
    if (ficha.origen !== "fabrica") {
      const marcas = document.createElement("span");
      marcas.className = "marcas";
      const marca = trozo("marca", ficha.origen);
      marca.dataset.tipo = ficha.origen;
      marcas.append(marca);
      tarjeta.append(marcas);
    }
    tarjeta.addEventListener("click", () => void elegirVoz(ficha.nombre));
    caja.append(tarjeta);
  }
  const propias = [...fichas.values()].filter((f) => f.origen !== "fabrica").length;
  pastilla(
    "p-voces",
    propias
      ? `${fichas.size - propias} de fábrica · ${propias} tuyas`
      : `${fichas.size} de fábrica`,
  );
  if (!visibles) caja.innerHTML = `<p class="nota">ninguna voz se llama así</p>`;
}

$("buscar-voz").addEventListener("input", render);

/**
 * Elegir una voz la descarga si hace falta y, salvo que se diga lo contrario, la
 * **dice al momento** con el texto que haya.
 *
 * Cambiar de voz sin oirla obliga a un segundo clic para saber si acertaste, y
 * elegir voz es justo la tarea en la que hay que probar varias. No hace falta
 * recargar nada del motor: con este modelo la voz es una entrada, asi que basta
 * con volver a sintetizar.
 */
async function elegirVoz(nombre: string, escuchar = true): Promise<void> {
  const ficha = fichas.get(nombre);
  if (!ficha) return;
  // En Pocket una voz es un nombre que el worker ya tiene; en Supertonic es un
  // fichero de 292 KB que hay que bajar la primera vez.
  if (activo?.tipo === "supertonic" && !ficha.estilo) {
    $("estado").textContent = `descargando la voz ${nombre}…`;
    try {
      ficha.estilo = await descargarEstilo(`${origen()}voice_styles/${nombre}.json`, nombre);
    } catch (e) {
      $("estado").textContent = `no se pudo traer la voz ${nombre}: ${String(e).slice(0, 160)}`;
      log(`ERROR con la voz ${nombre}: ${String(e)}`);
      return;
    }
    $("estado").textContent = "listo";
  }
  vozActual = nombre;
  dato("e-voz", nombre);
  render();
  if (escuchar && activo && $$<HTMLInputElement>("escucharAlElegir").checked) {
    await hablar();
  }
}

// ---------------------------------------------------------------- sintetizar

/** La semilla del campo, donde **0 es una semilla**, no "ninguna".
 *
 * `Number(v) || undefined` convertia el 0 en aleatorio, que es justo lo contrario
 * de lo que dice el campo ("fija el ruido para poder repetir la misma toma") y de
 * la regla 5. Vacio si significa aleatorio. */
/** El deslizador de temperatura, donde **0 significa «la del modelo»**.
 *
 * Es el unico rango en el que 0 no es un valor util: una temperatura de cero es
 * un muestreador determinista y el habla sale plana y rota (medido: WER 0,246 a
 * 0,1 contra 0,027 a 0,5). Asi que el extremo izquierdo se usa para «no tocar»,
 * y la etiqueta lo dice.
 */
function temperaturaActual(): number | undefined {
  const v = Number($$<HTMLInputElement>("temperatura").value);
  return v > 0 ? v : undefined;
}

function semillaActual(): number | undefined {
  const v = $$<HTMLInputElement>("semilla").value.trim();
  if (v === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

/** Texto -> onda con el motor que este puesto. Lo unico que sabe de los dos. */
async function decir(
  texto: string,
  ficha: Ficha,
  opciones: {
    pasos: number;
    velocidad: number;
    semilla?: number;
    idioma: string;
    temperatura?: number;
  },
): Promise<{ onda: Float32Array; sampleRate: number; ms: number; detalle: string }> {
  if (!activo) throw new Error("no hay motor cargado");
  if (activo.tipo === "pocket") {
    const r = await activo.motor.sintetizar(texto, ficha.nombre, {
      pasos: opciones.pasos,
      temperatura: opciones.temperatura,
      semilla: opciones.semilla,
    });
    // Pocket no tiene velocidad: se estira el tiempo despues, sin tocar el tono,
    // que en una voz clonada es la mitad de lo que la hace reconocible.
    const onda = cambiarVelocidad(r.onda, opciones.velocidad, r.sampleRate);
    const rtf = r.ms / 1000 / (onda.length / r.sampleRate);
    const temp =
      opciones.temperatura === undefined ? "" : ` · t ${opciones.temperatura.toFixed(2)}`;
    return {
      onda,
      sampleRate: r.sampleRate,
      ms: r.ms,
      detalle: `RTF ${rtf.toFixed(2)} · ${opciones.pasos} pasos${temp} · pocket · ${ficha.nombre}`,
    };
  }
  if (!ficha.estilo) throw new Error(`la voz ${ficha.nombre} no esta cargada`);
  const r = await activo.motor.sintetizar(texto, ficha.estilo, opciones);
  return {
    onda: r.onda,
    sampleRate: r.sampleRate,
    ms: r.ms,
    detalle: `RTF ${r.rtf.toFixed(2)} · ${r.pasos} pasos · ${ficha.nombre}`,
  };
}

// Pulsar tres voces seguidas lanza tres sintesis; solo importa la ultima. ORT no
// se puede cancelar a mitad, asi que lo que se hace es tirar lo que llega tarde.
let turno = 0;

async function hablar(): Promise<void> {
  const ficha = vozActual ? fichas.get(vozActual) : null;
  if (!activo || !ficha) return;
  const mio = ++turno;
  const boton = $$<HTMLButtonElement>("hablar");
  boton.disabled = true;
  pastilla("p-audio", "sintetizando…", "aviso");
  visor.poner("trabajando");
  try {
    // El normalizador español va DELANTE del frontend Unicode: Supertonic lee
    // "3.522" como dígitos y "10:30 h" como puntuación (medido, evidencia
    // 2026-09-15), y esta pieza ya existía y ya está probada.
    const crudo = $$<HTMLTextAreaElement>("texto").value;
    const idioma = selectorIdioma.value;
    const texto = idioma === "es" ? normalizar(crudo) : crudo;
    const r = await decir(texto, ficha, {
      idioma,
      pasos: Number($$<HTMLInputElement>("pasos").value),
      velocidad: Number($$<HTMLInputElement>("velocidad").value),
      temperatura: temperaturaActual(),
      semilla: semillaActual(),
    });
    if (mio !== turno) {
      log(`descartada la toma de ${ficha.nombre}: ya hay otra voz elegida`);
      return;
    }
    ultimo = { onda: r.onda, sampleRate: r.sampleRate };
    visor.mostrar(r.onda, r.sampleRate);
    visor.poner("listo");
    const segundos = r.onda.length / r.sampleRate;
    $("medidas").textContent = `${segundos.toFixed(2)} s · ${r.ms.toFixed(0)} ms · ${r.detalle}`;
    dato("e-ms", `${r.ms.toFixed(0)} ms`);
    pastilla("p-audio", "sonando", "ok");
    const enlace = $$<HTMLAnchorElement>("descargar");
    enlace.href = URL.createObjectURL(aWav(r.onda, r.sampleRate));
    enlace.download = `${ficha.nombre}.wav`;
    enlace.hidden = false;
    $$<HTMLButtonElement>("repetir").disabled = false;
    reproducir(r.onda, r.sampleRate);
    log(`"${texto.slice(0, 40)}…" en ${r.ms.toFixed(0)} ms · ${r.detalle}`);
  } catch (e) {
    pastilla("p-audio", "falló", "error");
    log(`ERROR al sintetizar: ${String(e)}`);
  } finally {
    if (mio === turno) boton.disabled = false;
  }
}

$("hablar").addEventListener("click", () => void hablar());
$("repetir").addEventListener("click", () => {
  if (ultimo) reproducir(ultimo.onda, ultimo.sampleRate);
});

const etiquetaTemperatura = () => {
  const v = Number($$<HTMLInputElement>("temperatura").value);
  $("temperatura_v").textContent = v > 0 ? v.toFixed(2) : "por defecto";
};
$("temperatura").addEventListener("input", etiquetaTemperatura);
etiquetaTemperatura();

for (const id of ["pasos", "velocidad"]) {
  const entrada = $$<HTMLInputElement>(id);
  const salida = $(`${id}_v`);
  const pintar = () => {
    salida.textContent = entrada.value;
  };
  entrada.addEventListener("input", pintar);
  pintar();
}

// --------------------------------------------------------------- crear voz

/**
 * Clonar desde una grabación necesita el puente (ADR 0012, decisión 6), y hoy no
 * hay ninguno que funcione: los cuatro intentos están medidos en
 * docs/evidencia.md y el mejor llega a 0,227 sobre un objetivo de 0,55. Así que
 * este camino sale **apagado** y lo dice, en vez de ofrecer un botón que devuelve
 * la voz media. Lo que sí funciona es importar un `.json` construido fuera.
 */
async function comprobarPuente(): Promise<boolean> {
  const puede = puedeClonar();
  pastilla("p-clonar", puede ? "clona desde audio" : "solo .json", puede ? "ok" : "aviso");
  $$<HTMLButtonElement>("grabar").disabled = !puede;
  return puede;
}

/**
 * Clonar desde una grabación. Con Pocket es medio segundo; con Supertonic no se
 * puede y se dice por qué, en vez de dejar un botón muerto.
 */
async function clonar(fichero: File | Blob, nombre: string): Promise<void> {
  if (!activo || activo.tipo !== "pocket") {
    pastilla("p-clonar", "no con este motor", "error");
    $("estado").textContent =
      "Supertonic no sabe clonar desde una grabación: su encoder de voz nunca se publicó." +
      " Cambia el motor a Pocket arriba y vuelve a soltar el fichero.";
    log(`rechazado ${nombre}: el motor activo no clona`);
    return;
  }
  try {
    pastilla("p-clonar", "clonando…", "aviso");
    const t0 = performance.now();
    const segundos = await activo.motor.clonarFichero(fichero as File, nombre);
    fichas.set(nombre, {
      nombre,
      origen: "clonada",
      detalle: `${segundos.toFixed(0)} s de voz`,
      estilo: undefined,
    });
    pastilla("p-clonar", "clona desde audio", "ok");
    log(
      `voz ${nombre} clonada en ${(performance.now() - t0).toFixed(0)} ms con ${segundos.toFixed(1)} s`,
    );
    // Decir cuanto se ha usado: con 2 s la similitud es 0,396 y con 20 es 0,527,
    // asi que quien da poco tiene que enterarse sin tener que leer la evidencia.
    if (segundos < 15) {
      $("estado").textContent =
        `clonada con ${segundos.toFixed(0)} s. El encoder lee hasta 20 y se nota mucho: con 20 s la similitud sube de 0,40 a 0,53.`;
    }
    await elegirVoz(nombre);
  } catch (e) {
    pastilla("p-clonar", "falló", "error");
    $("estado").textContent = `no se pudo clonar de «${nombre}»: ${String(e).slice(0, 180)}`;
    log(`ERROR al clonar: ${String(e)}`);
  }
}

/** Importar una voz `.json`: la de fábrica de cualquier runtime de Supertonic, o
 * una construida aquí con `constructor` o `ajustador`. Esto sí funciona hoy. */
async function importar(fichero: File): Promise<void> {
  const nombre = fichero.name.replace(/\.json$/i, "");
  try {
    const estilo = leerEstilo(JSON.parse(await fichero.text()) as EstiloJSON, nombre);
    const meta = (estilo.metadatos ?? {}) as Record<string, unknown>;
    fichas.set(nombre, {
      nombre,
      origen: meta.construida_por ? "construida" : "clonada",
      detalle: (meta.source_file as string) ?? "importada",
      estilo,
    });
    guardarLocal(nombre, estilo);
    await elegirVoz(nombre);
    log(`voz ${nombre} importada`);
  } catch (e) {
    $("estado").textContent = `ese .json no es una voz: ${String(e).slice(0, 200)}`;
    log(`ERROR al importar ${nombre}: ${String(e)}`);
  }
}

function guardarLocal(nombre: string, estilo: Estilo): void {
  try {
    const todas = locales();
    todas[nombre] = {
      style_ttl: { data: [Array.from(estilo.ttl)], dims: [1, 50, 256], type: "float32" },
      style_dp: { data: [Array.from(estilo.dp)], dims: [1, 8, 16], type: "float32" },
      metadata: estilo.metadatos ?? {},
    };
    localStorage.setItem(LOCALES, JSON.stringify(todas));
  } catch (e) {
    // 292 KB por voz: el cupo de localStorage se llena sobre la quincena.
    log(`no se pudo guardar la voz (${String(e).slice(0, 80)}); sigue disponible hasta recargar`);
  }
}

/**
 * Un solo sitio donde soltar cosas, y sin `accept`.
 *
 * Habia dos entradas de fichero, una filtrada a `.json` y otra a `audio/*`, y en
 * Windows ese filtro **escondia los audios en el explorador** (depende del mapeo
 * de tipos MIME del sistema, y con .m4a, .opus o .ogg falla a menudo). Peor: la
 * de audio salia desactivada, asi que quien queria clonar su voz veia un cuadro
 * de dialogo vacio o un control muerto y ninguna explicacion.
 *
 * Ahora entra cualquier fichero y es el codigo el que decide, y el que dice en
 * voz alta lo que no puede hacer.
 */
async function anadir(f: File): Promise<void> {
  if (/\.json$/i.test(f.name) || f.type === "application/json") {
    if (activo?.tipo === "pocket") {
      $("estado").textContent = `«${f.name}» es una voz de Supertonic y el motor activo es
        Pocket: sus estilos no se cruzan. Cambia el motor arriba, o suelta una grabación para
        clonarla aquí.`.replace(/\s+/g, " ");
      return;
    }
    await importar(f);
    return;
  }
  await clonar(f, f.name.replace(/\.[^.]+$/, ""));
}

$("importar").addEventListener("change", async (e) => {
  const entrada = e.target as HTMLInputElement;
  const f = entrada.files?.[0];
  if (f) await anadir(f);
  // Vaciarlo permite volver a elegir el MISMO fichero: sin esto, el segundo
  // intento no dispara `change` y parece que la pagina se ha colgado.
  entrada.value = "";
});

// Soltar el fichero encima, que es lo que la gente intenta primero.
const zona = $("soltar");
for (const evento of ["dragenter", "dragover"]) {
  zona.addEventListener(evento, (e) => {
    e.preventDefault();
    zona.classList.add("encima");
  });
}
for (const evento of ["dragleave", "drop"]) {
  zona.addEventListener(evento, () => zona.classList.remove("encima"));
}
zona.addEventListener("drop", async (e) => {
  e.preventDefault();
  const f = (e as DragEvent).dataTransfer?.files?.[0];
  if (f) await anadir(f);
});

$("grabar").addEventListener("click", async () => {
  const boton = $$<HTMLButtonElement>("grabar");
  boton.disabled = true;
  try {
    // 20 s, que es lo que el encoder lee: el boton decia 6 y grababa 8, o sea que
    // se le daba menos de la mitad de lo que puede usar. 16 kHz basta porque el
    // encoder remuestrea igual.
    const onda = await grabarPCM(20, 16000, (s_, nivel) => {
      pintarNivel($$<HTMLCanvasElement>("nivel"), nivel);
      pastilla("p-clonar", `grabando… ${s_.toFixed(0)} s`, "aviso");
    });
    const wav = aWav(onda, 16000);
    await clonar(wav, `mi voz ${new Date().toLocaleTimeString()}`);
  } catch (e) {
    log(`ERROR al grabar: ${String(e)}`);
  } finally {
    boton.disabled = !puedeClonar();
  }
});

// -------------------------------------------------------------------- tema

const aplicarTema = () => {
  const v = $$<HTMLSelectElement>("tema").value;
  if (v === "auto") delete document.documentElement.dataset.tema;
  else document.documentElement.dataset.tema = v;
  localStorage.setItem("ttspro.tema", v);
};
$("tema").addEventListener("change", aplicarTema);
$$<HTMLSelectElement>("tema").value = localStorage.getItem("ttspro.tema") ?? "auto";
aplicarTema();

// -------------------------------------------------------------- hardware

void (async () => {
  const h = await detectar();
  $("hardware").textContent = describir(h);
})();

// ------------------------------------------------------------------- chat

const anotar = (usuario: string, texto: string, nota = "") => {
  const pre = $("chatLog");
  pre.textContent += `${usuario ? `${usuario}: ` : ""}${texto}${nota ? `   ${nota}` : ""}\n`;
  pre.scrollTop = pre.scrollHeight;
};

/** Reparte los usuarios entre las voces que haya, de forma estable. */
function vozParaUsuario(usuario: string): Ficha | undefined {
  const lista = [...fichas.values()].filter((f) => f.estilo || f.origen === "fabrica");
  if (!lista.length) return undefined;
  let h = 0;
  for (const c of usuario) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return lista[h % lista.length];
}

const cola = new Cola(
  async (m: Mensaje) => {
    if (!activo) return null;
    const ficha =
      $$<HTMLSelectElement>("politica").value === "usuario"
        ? vozParaUsuario(m.usuario ?? "")
        : vozActual
          ? fichas.get(vozActual)
          : undefined;
    if (!ficha) return null;
    if (activo.tipo === "supertonic" && !ficha.estilo) {
      ficha.estilo = await descargarEstilo(
        `${origen()}voice_styles/${ficha.nombre}.json`,
        ficha.nombre,
      );
    }
    $("quien-habla").textContent = `${m.usuario ?? ""} con ${ficha.nombre}`;
    const r = await decir(m.texto, ficha, {
      idioma: selectorIdioma.value,
      pasos: Number($$<HTMLSelectElement>("pasosChat").value),
      velocidad: Number($$<HTMLInputElement>("velocidad").value),
      temperatura: temperaturaActual(),
    });
    return { onda: r.onda, sampleRate: r.sampleRate };
  },
  async (o) => {
    const fuente = reproducir(o.onda, o.sampleRate);
    await new Promise<void>((listo) => {
      fuente.onended = () => listo();
    });
  },
  (s: Estadisticas, evento: string) => pintarCola(s, evento),
  { maxEdadMs: 30_000 },
);

function pintarCola(s: Estadisticas, evento: string): void {
  $("cola-texto").textContent =
    `${s.pendientes} en cola · ${s.reproducidos} leídos · espera ${(s.esperaMediaMs / 1000).toFixed(1)} s · x${s.velocidad.toFixed(2)}`;
  ($("cola-barra") as HTMLElement).style.width = `${Math.min(100, s.pendientes * 5)}%`;
  pastilla("p-chat", s.pendientes ? "leyendo" : "al día", s.pendientes > 12 ? "aviso" : "ok");
  if (evento === "lleno" || evento === "viejo") log(`cola: ${evento}`);
}

function recibir(usuario: string, crudo: string): void {
  const maxChars = Number($$<HTMLInputElement>("maxChars").value) || 200;
  const texto = normalizarChat(crudo, maxChars);
  if (!texto) {
    anotar(usuario, crudo, "(nada que leer)");
    return;
  }
  const nombre = $$<HTMLInputElement>("leerNombre").checked ? `${nombreLegible(usuario)}, ` : "";
  anotar(usuario, texto);
  cola.encolar({ texto: nombre + texto, usuario });
}

(window as unknown as { __ttspro_chat: unknown }).__ttspro_chat = { recibir };

$("enviar").addEventListener("click", () => {
  const caja = $$<HTMLInputElement>("mensaje");
  if (caja.value.trim()) recibir("tú", caja.value.trim());
  caja.value = "";
});
$("mensaje").addEventListener("keydown", (e) => {
  if ((e as KeyboardEvent).key === "Enter") $("enviar").click();
});
$("vaciarCola").addEventListener("click", () => cola.vaciar());
$("simular").addEventListener("click", () => {
  const ejemplos = [
    "hola a todos!!",
    "jajajajaja que bueno",
    "q tal el stream?",
    "primera vez aqui",
    "GG",
    "saludos desde Chile",
    "pon la otra cancion",
    "me encanta esta voz",
  ];
  for (let i = 0; i < 20; i++) {
    recibir(`usuario${i % 6}`, `${ejemplos[i % ejemplos.length]} ${i}`);
  }
});

const mostrarCamposDeFuente = () => {
  const fuente = $$<HTMLSelectElement>("fuente").value;
  for (const el of Array.from(document.querySelectorAll<HTMLElement>("[data-fuente]"))) {
    el.hidden = !(el.dataset.fuente as string).split(" ").includes(fuente);
  }
};
$("fuente").addEventListener("change", mostrarCamposDeFuente);
mostrarCamposDeFuente();

let cerrar: (() => void) | null = null;

$("conectar").addEventListener("click", () => {
  if (cerrar) {
    cerrar();
    cerrar = null;
    $("conectar").textContent = "conectar";
    pastilla("p-chat", "apagado");
    return;
  }
  const alEstado = (estado: string, detalle?: string) => {
    pastilla("p-chat", estado, estado === "conectado" ? "ok" : "aviso");
    log(`chat: ${estado}${detalle ? ` — ${detalle}` : ""}`);
  };
  const fuente = $$<HTMLSelectElement>("fuente").value;
  const url = $$<HTMLInputElement>("url").value.trim();
  if (fuente === "twitch") {
    const canal = $$<HTMLInputElement>("canal").value.trim().toLowerCase();
    if (!canal) return;
    cerrar = conectarTwitch(canal, desdeTwitch, alEstado);
  } else if (fuente === "websocket") {
    cerrar = conectarWebSocket(
      url,
      $$<HTMLInputElement>("canalCable").value.trim(),
      (l) => recibir(l.usuario, l.texto),
      alEstado,
    );
  } else if (fuente === "sse") {
    cerrar = conectarSSE(url, (l) => recibir(l.usuario, l.texto), alEstado);
  } else {
    cerrar = escucharPostMessage(
      $$<HTMLInputElement>("origen").value.trim(),
      (l) => recibir(l.usuario, l.texto),
      alEstado,
    );
  }
  $("conectar").textContent = "desconectar";
});

function desdeTwitch(m: MensajeTwitch): void {
  const filtro = $$<HTMLSelectElement>("filtroTwitch").value;
  const destacado = m.destacado || (filtro === "recompensas" && m.recompensa);
  if (filtro !== "todos" && !destacado) {
    $("info-filtro").textContent = "filtrando los no destacados";
    return;
  }
  recibir(m.usuario, m.texto);
}

// ---------------------------------------------------------------- arranque

pintarFichas([]);
log("página lista; cargando el motor sola");
// Sin botones: la página se abre y habla. Esa es toda la instalación.
void cargar();
