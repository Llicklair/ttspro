/**
 * Read a Twitch channel's chat, anonymously, straight from the browser.
 *
 * Twitch chat is IRC over WebSocket and reading needs no account: the nick
 * `justinfan<number>` is the documented anonymous login. With the `tags`
 * capability each message carries `display-name`, which is what a reader
 * should say instead of the lowercase login.
 *
 * This is the demo's only network use after load, and it is opt-in: nothing
 * connects until the visitor types a channel and presses the button.
 */

export interface MensajeTwitch {
  usuario: string;
  texto: string;
  canal: string;
}

export type AlEstado = (
  estado: "conectando" | "conectado" | "cerrado" | "error",
  detalle?: string,
) => void;

const SERVIDOR = "wss://irc-ws.chat.twitch.tv:443";

interface Linea {
  tags: Record<string, string>;
  comando: string;
  canal: string;
  texto: string;
  login: string;
}

/** Parse `@tags :prefix COMMAND params :trailing`. Enough of IRC for PRIVMSG. */
function parsear(linea: string): Linea | null {
  let resto = linea;
  const tags: Record<string, string> = {};
  if (resto.startsWith("@")) {
    const fin = resto.indexOf(" ");
    for (const par of resto.slice(1, fin).split(";")) {
      const igual = par.indexOf("=");
      tags[par.slice(0, igual)] = par.slice(igual + 1);
    }
    resto = resto.slice(fin + 1);
  }
  let login = "";
  if (resto.startsWith(":")) {
    const fin = resto.indexOf(" ");
    const prefijo = resto.slice(1, fin);
    login = prefijo.slice(0, prefijo.indexOf("!") >= 0 ? prefijo.indexOf("!") : undefined);
    resto = resto.slice(fin + 1);
  }
  const trailing = resto.indexOf(" :");
  const texto = trailing >= 0 ? resto.slice(trailing + 2) : "";
  const partes = (trailing >= 0 ? resto.slice(0, trailing) : resto).split(" ");
  return { tags, comando: partes[0], canal: partes[1] ?? "", texto, login };
}

/** Unescape IRCv3 tag values (\s -> space, \: -> ;, \\ -> \). */
function destag(v: string): string {
  return v.replace(
    /\\(.)/g,
    (_, c: string) => ({ s: " ", ":": ";", "\\": "\\", r: "\r", n: "\n" })[c] ?? c,
  );
}

/**
 * Connect and deliver every chat line to `alMensaje`. Returns a function that
 * disconnects. Reconnects are NOT automatic: a reader that silently reconnects
 * hides the fact that it lost messages.
 */
export function conectarTwitch(
  canal: string,
  alMensaje: (m: MensajeTwitch) => void,
  alEstado: AlEstado,
): () => void {
  const nombre = canal.trim().toLowerCase().replace(/^#/, "");
  const ws = new WebSocket(SERVIDOR);
  alEstado("conectando", nombre);
  ws.onopen = () => {
    ws.send("CAP REQ :twitch.tv/tags");
    ws.send(`NICK justinfan${Math.floor(10_000 + Math.random() * 89_999)}`);
    ws.send(`JOIN #${nombre}`);
  };
  ws.onmessage = (ev) => {
    for (const linea of String(ev.data).split("\r\n")) {
      if (!linea) continue;
      if (linea.startsWith("PING")) {
        ws.send("PONG :tmi.twitch.tv");
        continue;
      }
      const m = parsear(linea);
      if (!m) continue;
      if (m.comando === "JOIN") alEstado("conectado", nombre);
      else if (m.comando === "NOTICE") alEstado("error", m.texto);
      else if (m.comando === "PRIVMSG") {
        alMensaje({
          usuario: destag(m.tags["display-name"] || m.login),
          texto: m.texto,
          canal: nombre,
        });
      }
    }
  };
  ws.onerror = () => alEstado("error", "websocket");
  ws.onclose = () => alEstado("cerrado", nombre);
  return () => ws.close();
}
