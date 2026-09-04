import { describe, expect, it } from "vitest";
import { trocear } from "./trocear.ts";

const t = (valor: string) => ({ tipo: "texto", valor });
const p = (valor: string) => ({ tipo: "puntuacion", valor });

describe("trocear", () => {
  it("keeps punctuation as tokens", () => {
    expect(trocear("Hola, ¿cómo estás?")).toEqual([
      t("Hola"),
      p(","),
      p("¿"),
      t("cómo estás"),
      p("?"),
    ]);
  });
  it("does not split . , : between digits", () => {
    expect(trocear("a las 10:30 son 3,5 o 3.5.")).toEqual([t("a las 10:30 son 3,5 o 3.5"), p(".")]);
  });
  it("keeps apostrophes and hyphens inside words, splits spaced dashes", () => {
    expect(trocear("Don't stop twenty-two - now")).toEqual([
      t("Don't stop twenty-two"),
      p("-"),
      t("now"),
    ]);
  });
  it("drops empty chunks", () => {
    expect(trocear("...")).toEqual([p("."), p("."), p(".")]);
  });
});
