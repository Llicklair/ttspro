# TTS pro

Contexto ejecutable para agentes. Formato [AGENTS.md](https://agents.md), que leen Claude Code,
Codex, Cursor, Copilot, Gemini CLI y Aider. La ley está en [ARCHITECTURE.md](ARCHITECTURE.md), lo
que no entra en [SCOPE.md](SCOPE.md), el porqué en [docs/adr/](docs/adr/) y los números en
[docs/evidencia.md](docs/evidencia.md).

En una frase: TTS multilingüe con clonación de voz a partir de un audio, **en el navegador** con
ONNX Runtime Web. Desde el [ADR 0012](docs/adr/0012-supertonic-como-motor.md) aquí no se entrena ni
se exporta nada: los pesos se descargan ya en ONNX. Leen las voces predeterminadas (Supertonic 3) y
clona un segundo modelo (Pocket TTS) que solo hace eso.

## Comandos

Lo de esta sección se EJECUTA, así que no puede pudrirse en silencio: si miente, falla.

```bash
uv sync --extra dev --extra export --extra voz     # torch cu126 viene del índice fijado en pyproject
uv sync ... --extra eval          # AÑADE esto para medir: sin él `uv sync` PODA Whisper y jiwer
winget install eSpeak-NG.eSpeak-NG                # SOLO para tests/test_frontend_paridad.py; el motor no tiene fonemizador (ADR 0012)
uv run pytest tests -q              # suite rápida: suelo, frontend y paridad Python/JS (necesita node)
uv run pytest tests/terminado -q    # criterio de terminado del MVP — hoy FALLA: mide la cadena que retiro el ADR 0012
cd web && npm ci && npm test        # runtime JS; npm run typecheck y npm run lint tambien
gb who --html mapa.html             # el mapa navegable del repo (derivado, no se commitea)

./instalar.sh | instalar.bat                      # arranque completo; ./actualizar.sh trae la ultima version y reconstruye modelos solo si hace falta
arrancar.bat [--lib]                             # Windows: levanta la pagina (o la libreria con su ejemplo) y abre el navegador

# el motor (ADR 0012): cuatro grafos ONNX que se DESCARGAN, no se exportan
uv run python -m ttspro.supertonic.descargar          # 398 MB en models/supertonic/ + contrato.json
uv run pytest tests/test_supertonic_paridad.py -q     # regla 3 del frontend Unicode, sin espeak

# construir una voz SIN grabacion: mezcla de las diez predeterminadas
uv run python -m ttspro.supertonic.constructor --referencia voz.wav --nombre marcos
uv run python -m ttspro.supertonic.pack --n 8         # N voces mutuamente distintas de una tirada
                                                      # clonar DESDE una grabacion se hace en la pagina,
                                                      # no aqui: lo hace Pocket TTS en el navegador

# los seis intentos de reconstruir el voice builder de Supertonic. NO funcionan, y estan
# aqui porque son el codigo detras de las medidas (ADR 0012, enmienda; evidencia 2026-09-16)
uv run python -m ttspro.supertonic.inversor --audio voz.wav --reconstruida recon.wav
uv run python -m ttspro.supertonic.puente muestrear --n 4000
uv run python -m ttspro.supertonic.puente entrenar --desde ambos

# el ciclo del modelo pieza a pieza (cadena ANTERIOR al ADR 0012; ver data/README.md)
uv run python -m ttspro.export.speaker_encoder                 # models/speaker_encoder.onnx (+ .fp16)
uv run python -m ttspro.data.preparar --manifiesto data/manifests/X.tsv --salida cache/X
uv run python -m ttspro.train.entrenar --cache cache/X --salida runs/X --batch 16 --fp16
uv run python -m ttspro.export.tts --checkpoint runs/X/G_NNNN.pt  # models/tts.onnx (+ .fp16)
uv run python -m ttspro.train.inicializar --coqui <carpeta> --salida runs/init/G_0.pt  # partir del VITS de coqui
uv run python -m ttspro.export.piper --onnx <voz.onnx> --salida runs/piper_es/G_0.pt  # portar una voz de Piper (ADR 0008)
uv run python -m ttspro.export.conversor                        # models/voz.onnx y conversor.onnx (ADR 0007)
uv run python -m ttspro.train.evaluar --checkpoint runs/X/G_NNNN.pt --cache cache/*  # WER y SECS
uv run python -m ttspro.export.voces --cache cache/openslr_es cache/vctk  # presets del demo
uv run python -m ttspro.export.paquete --todas es   # paquetes de voz base (ADR 0011) en paquetes/; se publican en la rama `voces` (GitHub Pages)
uv run python -m ttspro.export.publicar --repo <usuario>/ttspro-voces   # sube paquetes/ a Hugging Face (antes: uv run hf auth login)
```

El afinado real lleva la consistencia de locutor (`--scl 9`), que es lo que convierte "una voz"
en "esta voz"; sin ella el modelo usa el embedding como una pista de estilo floja:

```bash
uv run python -m ttspro.train.entrenar --cache cache/openslr_es cache/vctk   --salida runs/X --batch 16 --fp16 --scl 9 --reanudar runs/anterior/G_NNNN.pt

# el demo en el navegador (copia modelos y espeak a web/public y sirve con COOP/COEP)
# PowerShell 5.1 no acepta `&&`: dos lineas, `cd web` y luego `npm run dev`
cd web && npm run dev                # http://localhost:5173
cd web && npx playwright test e2e/pagina.spec.ts   # la pagina entera: arranque, voces, importar, ajustes
cd web && npx playwright test e2e/pocket.spec.ts   # clonar desde un wav soltado y hablar con esa voz (baja 216 MB la 1a vez)
cd web && npm run build:lib && npm run servir   # la libreria (dist/lib/ttspro.js) y la pagina ejemplo/ en http://127.0.0.1:8080
cd web && npx playwright test e2e/libreria.spec.ts   # el BUNDLE construido, no las fuentes (necesita build:lib)
cd web && npx playwright test -g "CDN"   # que espeak, que ya no se empaqueta, cargue estando aislados
```

## Gates

Ratchet: la deuda heredada pasa, la nueva falla. Las corre el pre-commit
([.githooks/pre-commit](.githooks/pre-commit)); engánchalo una vez con
`git config core.hooksPath .githooks`.

```bash
uv run ruff check src tests && uv run ruff format --check src tests
cd web && npx biome check . && npx tsc --noEmit
gb graph . --gate                   # ciclos de import nuevos y fronteras de .gb-boundaries
```

## Cuando algo pete (contrato con gb)

- Si muere un script, CLI o servidor: lee el estado YA capturado — `gb show <id>` (el aviso trae
  el id) o `gb last` — antes de re-ejecutar con prints. La ficha llega con su nodo del grafo y
  quién le llama.
- Para saber quién llama a un símbolo o qué rompes al tocarlo: `gb calls <simbolo> [--depth 2]`
  antes de grepear o abrir ficheros a mano.
- De vez en cuando, `gb list`: el embudo capturada→leída→intervenida→en-silencio es el termómetro
  del proyecto. No usar gb también es dato: se investiga, no se esconde.

## Arquitectura

`src/ttspro` (Python) tiene el frontend de texto, el modelo (VITS multilingüe con embedding de
locutor externo), el entrenamiento y el export; produce dos grafos y un contrato en `models/`.
`web/` (TypeScript) los consume con ONNX Runtime Web: el mismo frontend de texto, el mismo
contrato, cero Python. Lo único que cruza entre los dos mundos es `models/`.

Contrato específico de este repo para un agente:

- Tocas el frontend en un lado → tocas el otro y corres la paridad
  (`uv run pytest tests/test_frontend_paridad.py`). Regla 3. Eso incluye la etapa `chat`
  (ADR 0010): una abreviatura o un emote nuevo va en `ttspro.frontend.chat` **y** en
  `web/src/frontend/chat.ts`, y su caso en `tests/fixtures/frontend/chat.txt`.
- Añades una op o cambias una firma del modelo → compruebas la lista de ops de ORT Web y actualizas
  `models/contrato.json`. Reglas 2 y 4.
- Mides algo (latencia, tamaño, WER, similitud) → entrada en `docs/evidencia.md` con fecha y
  montaje, también si sale mal. Sobre todo si sale mal.
- Vas a cambiar una regla de ARCHITECTURE.md → ADR primero, código después.

## Convenciones de commit y PR

- Formato: `type: descripción corta` con `feat`, `fix`, `refactor`, `docs`, `chore`, `exp`.
  `exp:` es un experimento de entrenamiento o evaluación y su commit enlaza la entrada de evidencia.
- Un cambio lógico por commit. Comportamiento y documentación, por separado.
- Un PR entra con la suite verde, los gates verdes, ADR si toca ARCHITECTURE.md o el contrato, y
  entrada de evidencia si cambia un número.
- Idioma: **español** en los documentos de decisión (SCOPE, ARCHITECTURE, ADR, evidencia, este
  fichero). **Inglés** en lo publicable: README, código, comentarios, model card.
