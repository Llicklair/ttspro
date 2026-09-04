import { describe, expect, it } from "vitest";
import { normalizar } from "./normalizar.ts";

describe("normalizar", () => {
  it("collapses whitespace and strips", () => {
    expect(normalizar("  a   b\tc  ")).toBe("a b c");
  });
  it("maps typographic punctuation to ASCII", () => {
    expect(normalizar("“hola” — ‘ok’")).toBe("\"hola\" - 'ok'");
  });
  it("normalizes to NFC", () => {
    expect(normalizar("café")).toBe("café");
  });
});
