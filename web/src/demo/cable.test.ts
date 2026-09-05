/**
 * The ActionCable handshake, without a Rails server: a fake WebSocket that
 * replays what Rails sends (welcome, confirm_subscription, ping, message) and
 * records what the page sends back. What matters is the order: subscribe only
 * after `welcome`, "conectado" only after the confirmation, and a broadcast
 * envelope unwrapped into { usuario, texto }.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { conectarWebSocket } from "./fuentes.ts";

class WebSocketFalso {
  static ultima: WebSocketFalso;
  enviados: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  cerrado = false;
  constructor(public url: string) {
    WebSocketFalso.ultima = this;
  }
  send(d: string) {
    this.enviados.push(d);
  }
  close() {
    this.cerrado = true;
    this.onclose?.();
  }
  /** What the server would say. */
  llega(objeto: unknown) {
    this.onmessage?.({ data: JSON.stringify(objeto) });
  }
}

const original = globalThis.WebSocket;
beforeEach(() => {
  (globalThis as unknown as { WebSocket: unknown }).WebSocket = WebSocketFalso;
});
afterEach(() => {
  (globalThis as unknown as { WebSocket: unknown }).WebSocket = original;
});

describe("conectarWebSocket con canal habla ActionCable", () => {
  it("se suscribe tras welcome, confirma, ignora pings y desenvuelve el mensaje", () => {
    const lineas: Array<{ usuario: string; texto: string }> = [];
    const estados: string[] = [];
    const cerrar = conectarWebSocket(
      "ws://localhost:3000/cable",
      "ChatChannel",
      (l) => lineas.push(l),
      (e) => estados.push(e),
    );
    const ws = WebSocketFalso.ultima;
    ws.onopen?.();
    expect(ws.enviados).toEqual([]); // nothing before welcome
    ws.llega({ type: "welcome" });
    expect(JSON.parse(ws.enviados[0])).toEqual({
      command: "subscribe",
      identifier: JSON.stringify({ channel: "ChatChannel" }),
    });
    expect(estados).toEqual(["conectando"]);
    ws.llega({ identifier: '{"channel":"ChatChannel"}', type: "confirm_subscription" });
    expect(estados).toEqual(["conectando", "conectado"]);
    ws.llega({ type: "ping", message: 1725570000 });
    expect(lineas).toEqual([]);
    ws.llega({
      identifier: '{"channel":"ChatChannel"}',
      message: { usuario: "Pepe", texto: "hola a todos" },
    });
    expect(lineas).toEqual([{ usuario: "Pepe", texto: "hola a todos" }]);
    cerrar();
    expect(ws.cerrado).toBe(true);
    expect(estados.at(-1)).toBe("cerrado");
  });

  it("sin canal es un WebSocket plano: conectado al abrir y una línea por trama", () => {
    const lineas: Array<{ usuario: string; texto: string }> = [];
    const estados: string[] = [];
    conectarWebSocket(
      "ws://localhost:8080",
      "",
      (l) => lineas.push(l),
      (e) => estados.push(e),
    );
    const ws = WebSocketFalso.ultima;
    ws.onopen?.();
    expect(estados).toEqual(["conectando", "conectado"]);
    expect(ws.enviados).toEqual([]);
    ws.llega({ user: "Ana", text: "buenas" });
    expect(lineas).toEqual([{ usuario: "Ana", texto: "buenas" }]);
  });

  it("una suscripción rechazada es un error, no un silencio", () => {
    const estados: string[] = [];
    conectarWebSocket(
      "ws://x/cable",
      "Nada",
      () => {},
      (e) => estados.push(e),
    );
    const ws = WebSocketFalso.ultima;
    ws.llega({ type: "welcome" });
    ws.llega({ type: "reject_subscription" });
    expect(estados.at(-1)).toBe("error");
  });
});
