/**
 * Chat text -> readable text. Mirror of `src/ttspro/frontend/chat.py`.
 *
 * A Twitch chat is not prose: links, @mentions, emote names, emoji, "jajajaja",
 * "holaaaa", "q tal", "KEKW KEKW KEKW". Fed raw to the synthesizer, every one of
 * those becomes either garbage or a five-second spelling exercise. This turns a
 * message into what a person would say out loud, BEFORE `normalizar`.
 *
 * Keep the two files rule-for-rule identical, in the same order. The Python
 * parity test runs both over `tests/fixtures/frontend/chat.txt`. Deliberate
 * parity traps avoided here: no `\b` and no `\w` (ASCII-only in JavaScript,
 * Unicode in Python) — word edges use the explicit letter class below.
 */

const LETRA = "A-Za-zÁÉÍÓÚÜÑáéíóúüñ";
const SIN_LETRA_ANTES = `(?<![${LETRA}0-9])`;
const SIN_LETRA_DESPUES = `(?![${LETRA}0-9])`;

// 1. Links: read as one word, not spelled out character by character.
const ENLACE = /(?:https?:\/\/|www\.)\S+/gi;
// 2. Emoji and pictographs (plus the joiners and skin tones that ride with them).
//    The variation selector and the joiner are alternations, not class members:
//    they combine with neighbours and a class of combining marks is a lint error.
const EMOJI = /<3|\u{FE0F}|\u{200D}|[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}]/gu;
// 3. Symbols the model cannot say: become spaces. Sentence punctuation survives.
const SIMBOLO = /[*#~^|<>[\]{}/\\+=_`]/g;
// 4. Laughter in all its spellings -> "jajaja". `xd`, `lol`, LUL, KEKW included.
const RISA = new RegExp(
  `${SIN_LETRA_ANTES}(?:(?:j+[aeiou]+){2,}[jaeiou]*|jsj[sj]*|x+d+|lo+l|lmao|kekw?|omegalul|lul)${SIN_LETRA_DESPUES}`,
  "gi",
);
// 5. Emote names are not words: dropped. Case-sensitive on purpose (Twitch is).
export const EMOTES = new Set(
  `Kappa KappaPride Keepo PogChamp Pog PogU POGGERS Poggers PauseChamp monkaS monkaW
    Pepega PepeLaugh PepeHands FeelsBadMan FeelsGoodMan FeelsStrongMan Sadge Madge Bedge
    Prayge Okayge Copium Hopium Aware Clueless EZ TriHard 4Head BibleThump ResidentSleeper
    Jebaited NotLikeThis CoolStoryBob DansGame HeyGuys VoHiYo SeemsGood Kreygasm BabyRage
    WutFace catJAM widepeepoHappy peepoClap peepoHappy HYPERS AYAYA OMEGALUL LUL KEKW
    KEKWait PepoG Clap gachiHYPER forsenE forsenCD ppL YEP NOPERS PepegaAim HandsUp
    Stare D: :) :( ;) :D xD XD :P :p <3 o7 F`.split(/\s+/),
);
// 6. Abbreviations that are read as words, expanded. Whole tokens, lowercase.
export const ABREVIATURAS: Record<string, string> = {
  q: "que",
  k: "que",
  xq: "porque",
  pq: "porque",
  xk: "porque",
  pk: "porque",
  porq: "porque",
  tb: "también",
  tmb: "también",
  tmbn: "también",
  bn: "bien",
  x: "por",
  dnd: "dónde",
  tqm: "te quiero mucho",
  tkm: "te quiero mucho",
  ntp: "no te preocupes",
  msj: "mensaje",
  pls: "por favor",
  plis: "por favor",
  porfa: "por favor",
  grax: "gracias",
  grx: "gracias",
  salu2: "saludos",
  bss: "besos",
  gg: "buena partida",
  wp: "bien jugado",
  ez: "fácil",
};
// 6b. English loanwords that espeak's Spanish rules mangle ("streamer" came out as
//     "estréamer", "follow" as "follogo"): respelled the way a Spanish streamer says them.
export const PRESTAMOS: Record<string, string> = {
  stream: "estrim",
  streams: "estrims",
  streamer: "estrímer",
  streamers: "estrímers",
  streaming: "estrímin",
  follow: "fólou",
  follows: "fólous",
  hype: "jaip",
  nice: "nais",
  ok: "okey",
  link: "linc",
  spoiler: "espóiler",
  spoilers: "espóilers",
  gameplay: "guéimplei",
  speedrun: "espídran",
  host: "jost",
  raid: "reid",
  prime: "praim",
  random: "rándom",
  fail: "feil",
  noob: "nub",
  loot: "lut",
  boss: "bos",
  cringe: "crinch",
  chill: "chil",
  team: "tim",
};
// 7. Letter runs: "holaaaa" -> "hola". Three or more of the same letter never occur
//    in Spanish (nor in English), so 3+ collapse to ONE; a double is left alone.
const ALARGAMIENTO = new RegExp(`([${LETRA}])\\1{2,}`, "gi");
// 8. Punctuation runs: "!!!!" -> "!", "???" -> "?". "..." survives as an ellipsis.
const PUNTUACION_REPETIDA = /([!?¡¿,;:])\1+/g;
// Used by rule 9: only real letters, so "C3PO" and "NASA1" stay as typed.
const GRITO = new RegExp(`^[${LETRA}]{3,}$`);
const ESPACIOS = /\s+/g;
const ESPACIO_ANTES_DE_SIGNO = /\s+([,.;:!?])/g;

function token(palabra: string): string {
  if (EMOTES.has(palabra)) return "";
  // Mentions: "@Dark_Lord" -> "Dark Lord" (the underscore already became a space in 3).
  let p = palabra;
  if (p.startsWith("@")) p = p.slice(1);
  const minuscula = p.toLowerCase();
  if (Object.hasOwn(ABREVIATURAS, minuscula)) return ABREVIATURAS[minuscula];
  if (Object.hasOwn(PRESTAMOS, minuscula)) return PRESTAMOS[minuscula];
  // 9. Shouting: an all-caps word of three or more letters is a word, not initials.
  if (GRITO.test(p) && p === p.toUpperCase()) return minuscula;
  return p;
}

/**
 * Rule order matters and must match the Python mirror exactly.
 *
 * Returns "" when nothing readable is left (an emote-only message), so the
 * caller can skip it instead of playing silence.
 */
export function normalizarChat(texto: string, maxCaracteres = 200): string {
  let salida = texto.replace(ENLACE, ", enlace, ");
  salida = salida.replace(EMOJI, " ");
  salida = salida.replace(SIMBOLO, " ");
  salida = salida.replace(RISA, "jajaja");
  salida = salida
    .split(/\s+/)
    .filter((p) => p !== "")
    .map(token)
    .filter((t) => t !== "")
    .join(" ");
  salida = salida.replace(ALARGAMIENTO, "$1");
  salida = salida.replace(PUNTUACION_REPETIDA, "$1");
  // 10. Repeated words ("no no no", "jajaja jajaja") -> once. Case-insensitive.
  const unicas: string[] = [];
  for (const p of salida.split(/\s+/)) {
    if (p === "") continue;
    if (unicas.length > 0 && p.toLowerCase() === unicas[unicas.length - 1].toLowerCase()) continue;
    unicas.push(p);
  }
  salida = unicas.join(" ");
  // 11. Length cap at a word edge: a wall of text must not hold the queue for a minute.
  if (salida.length > maxCaracteres) {
    const corte = salida.lastIndexOf(" ", maxCaracteres - 1);
    salida = salida.slice(0, corte > 0 ? corte : maxCaracteres);
  }
  // A comma or a full stop never follows a space ("mira , enlace" from rule 1).
  salida = salida.replace(ESPACIO_ANTES_DE_SIGNO, "$1");
  return salida.replace(ESPACIOS, " ").trim();
}

// 12. User names, for "Nombre: mensaje". "xXDark_Lord99Xx" is typed, never said:
//     decoration off ("xX…Xx"), trailing digits off, separators to spaces, camelCase
//     split, emoji and symbols out. No abbreviation table here: a user called "x" is
//     not "por". Falls back to the raw name when less than two characters are left,
//     because a message with no author is worse than an odd name.
const DECORACION = /^(?:xx|x)(?=[A-Za-z])|(?<=[A-Za-z0-9])(?:xx|x)$/gi;
const DIGITOS_FINALES = /[0-9]+$/;
const CAMELLO = /(?<=[a-záéíóúüñ])(?=[A-ZÁÉÍÓÚÜÑ])/g;

export function nombreLegible(usuario: string): string {
  let nombre = usuario
    .trim()
    .replace(/^@/, "")
    .replace(/[_\-.]/g, " ");
  nombre = nombre
    .split(/\s+/)
    .filter((p) => p !== "")
    .map((p) => p.replace(DECORACION, "").replace(DIGITOS_FINALES, ""))
    .join(" ");
  nombre = nombre.replace(CAMELLO, " ");
  nombre = nombre.replace(EMOJI, " ").replace(SIMBOLO, " ");
  nombre = nombre.replace(ESPACIOS, " ").trim();
  return nombre.length >= 2 ? nombre : usuario;
}
