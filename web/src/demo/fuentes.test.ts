import { describe, expect, it } from "vitest";
import { extraerLinea } from "./fuentes.ts";

describe("extraerLinea: lo que manda la otra aplicación llega como { usuario, texto }", () => {
  it("acepta los nombres de campo habituales", () => {
    expect(extraerLinea({ usuario: "Pepe", texto: "hola" })).toEqual({
      usuario: "Pepe",
      texto: "hola",
    });
    expect(extraerLinea({ user: "Pepe", text: "hola" })).toEqual({
      usuario: "Pepe",
      texto: "hola",
    });
    expect(extraerLinea({ display_name: "Pepe", message: "hola" })).toEqual({
      usuario: "Pepe",
      texto: "hola",
    });
    expect(extraerLinea({ username: "Pepe", body: "hola" })).toEqual({
      usuario: "Pepe",
      texto: "hola",
    });
  });

  it("desenvuelve el sobre de ActionCable y el JSON en texto", () => {
    expect(
      extraerLinea({
        identifier: '{"channel":"ChatChannel"}',
        message: { user: "Pepe", text: "hola" },
      }),
    ).toEqual({ usuario: "Pepe", texto: "hola" });
    expect(extraerLinea('{"usuario":"Pepe","texto":"hola"}')).toEqual({
      usuario: "Pepe",
      texto: "hola",
    });
    expect(extraerLinea({ data: { nombre: "Pepe", mensaje: "hola" } })).toEqual({
      usuario: "Pepe",
      texto: "hola",
    });
  });

  it("sin texto no hay línea; sin usuario sí", () => {
    expect(extraerLinea({ usuario: "Pepe" })).toBeNull();
    expect(extraerLinea({ texto: "   " })).toBeNull();
    expect(extraerLinea(null)).toBeNull();
    expect(extraerLinea(42)).toBeNull();
    expect(extraerLinea({ texto: "hola" })).toEqual({ usuario: "", texto: "hola" });
    expect(extraerLinea("hola a secas")).toEqual({ usuario: "", texto: "hola a secas" });
  });
});
