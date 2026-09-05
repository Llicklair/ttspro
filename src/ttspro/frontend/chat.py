"""Chat text -> readable text. Mirror of ``web/src/frontend/chat.ts``.

A Twitch chat is not prose: links, @mentions, emote names, emoji, "jajajaja",
"holaaaa", "q tal", "KEKW KEKW KEKW". Fed raw to the synthesizer, every one of
those becomes either garbage or a five-second spelling exercise. This turns a
message into what a person would say out loud, BEFORE ``normalizar``.

Keep the two files rule-for-rule identical, in the same order. The parity test
runs both over ``tests/fixtures/frontend/chat.txt``. Deliberate parity traps
avoided here: no ``\\b`` and no ``\\w`` (ASCII-only in JavaScript, Unicode in
Python) — word edges use the explicit letter class below.
"""

from __future__ import annotations

import re

LETRA = "A-Za-zÁÉÍÓÚÜÑáéíóúüñ"
_SIN_LETRA_ANTES = f"(?<![{LETRA}0-9])"
_SIN_LETRA_DESPUES = f"(?![{LETRA}0-9])"

# 1. Links: read as one word, not spelled out character by character.
_ENLACE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# 2. Emoji and pictographs (plus the joiners and skin tones that ride with them).
#    The variation selector and the joiner are alternations, not class members:
#    they combine with neighbours and a class of combining marks is a lint error.
_EMOJI = re.compile("<3|️|‍|[🀀-🫿☀-➿⬀-⯿]")
# 3. Symbols the model cannot say: become spaces. Sentence punctuation survives.
_SIMBOLO = re.compile(r"[*#~^|<>\[\]{}/\\+=_`]")
# 4. Laughter in all its spellings -> "jajaja". `xd`, `lol`, LUL, KEKW included.
_RISA = re.compile(
    _SIN_LETRA_ANTES
    + r"(?:(?:j+[aeiou]+){2,}[jaeiou]*|jsj[sj]*|x+d+|lo+l|lmao|kekw?|omegalul|lul)"
    + _SIN_LETRA_DESPUES,
    re.IGNORECASE,
)
# 5. Emote names are not words: dropped. Case-sensitive on purpose (Twitch is).
EMOTES = frozenset(
    """Kappa KappaPride Keepo PogChamp Pog PogU POGGERS Poggers PauseChamp monkaS monkaW
    Pepega PepeLaugh PepeHands FeelsBadMan FeelsGoodMan FeelsStrongMan Sadge Madge Bedge
    Prayge Okayge Copium Hopium Aware Clueless EZ TriHard 4Head BibleThump ResidentSleeper
    Jebaited NotLikeThis CoolStoryBob DansGame HeyGuys VoHiYo SeemsGood Kreygasm BabyRage
    WutFace catJAM widepeepoHappy peepoClap peepoHappy HYPERS AYAYA OMEGALUL LUL KEKW
    KEKWait PepoG Clap gachiHYPER forsenE forsenCD ppL YEP NOPERS PepegaAim HandsUp
    Stare D: :) :( ;) :D xD XD :P :p <3 o7 F""".split()
)
# 6. Abbreviations that are read as words, expanded. Whole tokens, lowercase.
ABREVIATURAS = {
    "q": "que",
    "k": "que",
    "xq": "porque",
    "pq": "porque",
    "xk": "porque",
    "pk": "porque",
    "porq": "porque",
    "tb": "también",
    "tmb": "también",
    "tmbn": "también",
    "bn": "bien",
    "x": "por",
    "dnd": "dónde",
    "tqm": "te quiero mucho",
    "tkm": "te quiero mucho",
    "ntp": "no te preocupes",
    "msj": "mensaje",
    "pls": "por favor",
    "plis": "por favor",
    "porfa": "por favor",
    "grax": "gracias",
    "grx": "gracias",
    "salu2": "saludos",
    "bss": "besos",
    "gg": "buena partida",
    "wp": "bien jugado",
    "ez": "fácil",
}
# 6b. English loanwords that espeak's Spanish rules mangle ("streamer" came out as
#     "estréamer", "follow" as "follogo"): respelled the way a Spanish streamer says them.
PRESTAMOS = {
    "stream": "estrim",
    "streams": "estrims",
    "streamer": "estrímer",
    "streamers": "estrímers",
    "streaming": "estrímin",
    "follow": "fólou",
    "follows": "fólous",
    "hype": "jaip",
    "nice": "nais",
    "ok": "okey",
    "link": "linc",
    "spoiler": "espóiler",
    "spoilers": "espóilers",
    "gameplay": "guéimplei",
    "speedrun": "espídran",
    "host": "jost",
    "raid": "reid",
    "prime": "praim",
    "random": "rándom",
    "fail": "feil",
    "noob": "nub",
    "loot": "lut",
    "boss": "bos",
    "cringe": "crinch",
    "chill": "chil",
    "team": "tim",
}
# 7. Letter runs: "holaaaa" -> "hola". Three or more of the same letter never occur
#    in Spanish (nor in English), so 3+ collapse to ONE; a double is left alone.
_ALARGAMIENTO = re.compile(f"([{LETRA}])\\1{{2,}}", re.IGNORECASE)
# 8. Punctuation runs: "!!!!" -> "!", "???" -> "?". "..." survives as an ellipsis.
_PUNTUACION_REPETIDA = re.compile(r"([!?¡¿,;:])\1+")
# Used by rule 9: only real letters, so "C3PO" and "NASA1" stay as typed.
_GRITO = re.compile(f"[{LETRA}]{{3,}}")
_ESPACIOS = re.compile(r"\s+")
_ESPACIO_ANTES_DE_SIGNO = re.compile(r"\s+([,.;:!?])")


def _token(palabra: str) -> str:
    if palabra in EMOTES:
        return ""
    # Mentions: "@Dark_Lord" -> "Dark Lord" (the underscore already became a space in 3).
    if palabra.startswith("@"):
        palabra = palabra[1:]
    minuscula = palabra.lower()
    if minuscula in ABREVIATURAS:
        return ABREVIATURAS[minuscula]
    if minuscula in PRESTAMOS:
        return PRESTAMOS[minuscula]
    # 9. Shouting: an all-caps word of three or more letters is a word, not initials.
    if _GRITO.fullmatch(palabra) and palabra == palabra.upper():
        return minuscula
    return palabra


def normalizar_chat(texto: str, max_caracteres: int = 200) -> str:
    """Rule order matters and must match the TypeScript mirror exactly.

    Returns "" when nothing readable is left (an emote-only message), so the
    caller can skip it instead of playing silence.
    """
    texto = _ENLACE.sub(", enlace, ", texto)
    texto = _EMOJI.sub(" ", texto)
    texto = _SIMBOLO.sub(" ", texto)
    texto = _RISA.sub("jajaja", texto)
    texto = " ".join(t for t in (_token(p) for p in texto.split()) if t)
    texto = _ALARGAMIENTO.sub(r"\1", texto)
    texto = _PUNTUACION_REPETIDA.sub(r"\1", texto)
    # 10. Repeated words ("no no no", "jajaja jajaja") -> once. Case-insensitive.
    palabras = texto.split()
    unicas: list[str] = []
    for p in palabras:
        if unicas and p.lower() == unicas[-1].lower():
            continue
        unicas.append(p)
    texto = " ".join(unicas)
    # 11. Length cap at a word edge: a wall of text must not hold the queue for a minute.
    if len(texto) > max_caracteres:
        corte = texto.rfind(" ", 0, max_caracteres)
        texto = texto[: corte if corte > 0 else max_caracteres]
    # A comma or a full stop never follows a space ("mira , enlace" from rule 1).
    texto = _ESPACIO_ANTES_DE_SIGNO.sub(r"\1", texto)
    return _ESPACIOS.sub(" ", texto).strip()


# 12. User names, for "Nombre: mensaje". "xXDark_Lord99Xx" is typed, never said:
#     decoration off ("xX…Xx"), trailing digits off, separators to spaces, camelCase
#     split, emoji and symbols out. No abbreviation table here: a user called "x" is
#     not "por". Falls back to the raw name when less than two characters are left,
#     because a message with no author is worse than an odd name.
# Two letters, never one: "Alex", "Max" and "streex" end in a real x.
_DECORACION = re.compile(r"^xx(?=[A-Za-z])|(?<=[A-Za-z0-9])xx$", re.IGNORECASE)
_DIGITOS_FINALES = re.compile(r"[0-9]+$")
_CAMELLO = re.compile(r"(?<=[a-záéíóúüñ])(?=[A-ZÁÉÍÓÚÜÑ])")


def nombre_legible(usuario: str) -> str:
    nombre = usuario.strip().removeprefix("@").replace("_", " ").replace("-", " ").replace(".", " ")
    nombre = " ".join(_DIGITOS_FINALES.sub("", _DECORACION.sub("", p)) for p in nombre.split())
    nombre = _CAMELLO.sub(" ", nombre)
    nombre = _SIMBOLO.sub(" ", _EMOJI.sub(" ", nombre))
    nombre = _ESPACIOS.sub(" ", nombre).strip()
    return nombre if len(nombre) >= 2 else usuario
