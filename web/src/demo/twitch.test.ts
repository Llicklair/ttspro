import { describe, expect, it } from "vitest";
import { mensajeDesdeLinea } from "./twitch.ts";

const base =
  "@badge-info=;badges=;color=#FF0000;display-name=Pepe_Gamer;emotes=;id=abc;mod=0;room-id=1;subscriber=0;tmi-sent-ts=1;turbo=0;user-id=2;user-type=";
const prefijo = ":pepe_gamer!pepe_gamer@pepe_gamer.tmi.twitch.tv PRIVMSG #talk2play :";

describe("mensajeDesdeLinea: lo que Twitch etiqueta en cada PRIVMSG", () => {
  it("un mensaje normal: usuario, texto, no destacado, sin recompensa", () => {
    const m = mensajeDesdeLinea(`${base} ${prefijo}hola a todos`);
    expect(m).toEqual({
      usuario: "Pepe_Gamer",
      texto: "hola a todos",
      canal: "",
      destacado: false,
      recompensa: undefined,
    });
  });

  it("«Highlight My Message» llega como msg-id=highlighted-message", () => {
    const m = mensajeDesdeLinea(`${base};msg-id=highlighted-message ${prefijo}leedme esto`);
    expect(m?.destacado).toBe(true);
    expect(m?.texto).toBe("leedme esto");
  });

  it("una recompensa con texto llega con custom-reward-id", () => {
    const m = mensajeDesdeLinea(`${base};custom-reward-id=9f6a-1234 ${prefijo}canción por favor`);
    expect(m?.destacado).toBe(false);
    expect(m?.recompensa).toBe("9f6a-1234");
  });

  it("lo que no es PRIVMSG no es un mensaje", () => {
    expect(mensajeDesdeLinea("PING :tmi.twitch.tv")).toBeNull();
    expect(mensajeDesdeLinea(":justinfan123!x@x.tmi.twitch.tv JOIN #talk2play")).toBeNull();
  });
});
