/**
 * Message sources other than Twitch: how another application (streex, a Ruby on
 * Rails app that already reads the chat) hands lines to the reader.
 *
 * Three doors, all delivering the same `{ usuario, texto }`:
 *
 *   - WebSocket, plain JSON or ActionCable. ActionCable is what Rails speaks: the
 *     page sends a `subscribe` command for a channel and unwraps the `message`
 *     envelope, answering nothing to pings (the server does not need it).
 *   - Server-Sent Events: an `EventSource` on a URL that streams JSON lines.
 *   - postMessage: when the page lives in an <iframe> inside the other app, the
 *     parent window posts `{ tipo: "ttspro", usuario, texto }`.
 *
 * Field names are forgiving on purpose (`usuario` / `user` / `username` /
 * `display_name` / `nombre`, `texto` / `text` / `message` / `mensaje` / `body`):
 * the Rails side should not have to rename its columns to talk to this.
 */

export interface Linea {
  usuario: string;
  texto: string;
}

export type AlEstado = (
  estado: "conectando" | "conectado" | "cerrado" | "error",
  detalle?: string,
) => void;

/** Pull a chat line out of whatever JSON shape the other side sends. */
export function extraerLinea(dato: unknown): Linea | null {
  if (typeof dato === "string") {
    try {
      return extraerLinea(JSON.parse(dato));
    } catch {
      return dato.trim() ? { usuario: "", texto: dato } : null;
    }
  }
  if (!dato || typeof dato !== "object") return null;
  const o = dato as Record<string, unknown>;
  // ActionCable envelope: { identifier, message: {...} }; also plain { data: {...} }
  if (o.message && typeof o.message === "object") return extraerLinea(o.message);
  if (o.data && typeof o.data === "object") return extraerLinea(o.data);
  const usuario = [o.usuario, o.user, o.username, o.display_name, o.nombre, o.name].find(
    (v) => typeof v === "string",
  ) as string | undefined;
  const texto = [o.texto, o.text, o.message, o.mensaje, o.body, o.content].find(
    (v) => typeof v === "string",
  ) as string | undefined;
  if (!texto || !texto.trim()) return null;
  return { usuario: usuario ?? "", texto };
}

/**
 * WebSocket source. With `canal` set it speaks ActionCable: subscribes to that
 * channel on `welcome` and ignores pings and confirmations. Without it, every
 * frame is a JSON line.
 */
export function conectarWebSocket(
  url: string,
  canal: string,
  alLinea: (l: Linea) => void,
  alEstado: AlEstado,
): () => void {
  const ws = new WebSocket(url);
  alEstado("conectando", url);
  const identificador = canal ? JSON.stringify({ channel: canal }) : "";
  ws.onopen = () => {
    if (!canal) alEstado("conectado", url);
  };
  ws.onmessage = (ev) => {
    let dato: unknown = ev.data;
    if (typeof dato === "string") {
      try {
        dato = JSON.parse(dato);
      } catch {
        // not JSON: treated as a bare text line below
      }
    }
    const sobre = dato && typeof dato === "object" ? (dato as Record<string, unknown>) : null;
    if (canal && sobre) {
      if (sobre.type === "welcome") {
        ws.send(JSON.stringify({ command: "subscribe", identifier: identificador }));
        return;
      }
      if (sobre.type === "confirm_subscription") {
        alEstado("conectado", `${url} · ${canal}`);
        return;
      }
      if (sobre.type === "ping" || sobre.type === "disconnect") {
        if (sobre.type === "disconnect")
          alEstado("error", `el servidor cerró: ${String(sobre.reason ?? "")}`);
        return;
      }
      if (sobre.type === "reject_subscription") {
        alEstado("error", `suscripción rechazada a ${canal}`);
        return;
      }
    }
    const linea = extraerLinea(dato);
    if (linea) alLinea(linea);
  };
  ws.onerror = () => alEstado("error", "websocket");
  ws.onclose = () => alEstado("cerrado", url);
  return () => ws.close();
}

/** Server-Sent Events source: each event's data is one JSON line. */
export function conectarSSE(
  url: string,
  alLinea: (l: Linea) => void,
  alEstado: AlEstado,
): () => void {
  const es = new EventSource(url);
  alEstado("conectando", url);
  es.onopen = () => alEstado("conectado", url);
  es.onmessage = (ev) => {
    const linea = extraerLinea(ev.data);
    if (linea) alLinea(linea);
  };
  es.onerror = () => alEstado("error", "sse");
  return () => {
    es.close();
    alEstado("cerrado", url);
  };
}

/**
 * postMessage source, for the page embedded in another app's <iframe>:
 *   iframe.contentWindow.postMessage({ tipo: "ttspro", usuario, texto }, "*")
 * `origen` restricts who is listened to; "*" accepts any parent, which is fine
 * on a machine you control and not fine on a public page.
 */
export function escucharPostMessage(
  origen: string,
  alLinea: (l: Linea) => void,
  alEstado: AlEstado,
): () => void {
  const oyente = (ev: MessageEvent) => {
    if (origen !== "*" && ev.origin !== origen) return;
    const dato = ev.data as Record<string, unknown> | null;
    if (!dato || typeof dato !== "object" || dato.tipo !== "ttspro") return;
    const linea = extraerLinea(dato);
    if (linea) alLinea(linea);
  };
  window.addEventListener("message", oyente);
  alEstado("conectado", origen === "*" ? "cualquier ventana" : origen);
  return () => {
    window.removeEventListener("message", oyente);
    alEstado("cerrado", "postMessage");
  };
}
