#!/usr/bin/env bash
# ttspro - one-shot install on Linux (and macOS): tools, Python env, model weights, browser demo.
# Safe to run again: every step skips what is already there.
#   ./instalar.sh            install everything, then offer to start the demo
#   ./instalar.sh --sin-demo install only
#
# Tools are installed without sudo where possible: uv through its official installer
# into ~/.local/bin, Node.js 22 from nodejs.org into ~/.local/node when the system one
# is missing or too old. espeak-ng comes from the distro's package manager (needs sudo)
# and is only required for the Python tests: the browser demo uses its own WASM build.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
NODE_MIN=22            # package.json says >=22.6; distros ship far older versions
NODE_VERSION=22.14.0   # what gets downloaded when the system has none that fits

echo
echo "== ttspro: install =="
echo

# ---------------------------------------------------------------- helpers
tiene() { command -v "$1" >/dev/null 2>&1; }

version_node_ok() {
  tiene node || return 1
  local mayor
  mayor="$(node -p 'process.versions.node.split(".")[0]')"
  [ "$mayor" -ge "$NODE_MIN" ]
}

instalar_uv() {
  echo "   uv: installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

instalar_node() {
  local so arq destino
  case "$(uname -s)" in
    Linux)  so=linux ;;
    Darwin) so=darwin ;;
    *) echo "   node: unsupported OS $(uname -s); install Node.js >= $NODE_MIN yourself"; return 1 ;;
  esac
  case "$(uname -m)" in
    x86_64)         arq=x64 ;;
    aarch64|arm64)  arq=arm64 ;;
    *) echo "   node: unsupported arch $(uname -m); install Node.js >= $NODE_MIN yourself"; return 1 ;;
  esac
  destino="$HOME/.local/node"
  echo "   node: installing v$NODE_VERSION to $destino (no sudo)"
  mkdir -p "$destino"
  curl -fsSL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-$so-$arq.tar.gz" \
    | tar -xz --strip-components=1 -C "$destino"
  export PATH="$destino/bin:$PATH"
  echo "   node: add this to your shell profile to keep it:  export PATH=\"$destino/bin:\$PATH\""
}

instalar_espeak() {
  echo "   espeak-ng: installing with the system package manager (sudo)"
  if tiene apt-get; then
    sudo apt-get update -qq && sudo apt-get install -y -qq espeak-ng
  elif tiene dnf; then
    sudo dnf install -y espeak-ng
  elif tiene pacman; then
    sudo pacman -S --noconfirm espeak-ng
  elif tiene zypper; then
    sudo zypper install -y espeak-ng
  elif tiene brew; then
    brew install espeak-ng
  else
    echo "   espeak-ng: no known package manager; install it yourself (only the OLD frontend tests need it)"
    return 1
  fi
}

# ---------------------------------------------------------------- tools
export PATH="$HOME/.local/bin:$HOME/.local/node/bin:$PATH"

if tiene uv; then echo "   uv: found"; else instalar_uv; fi
if version_node_ok; then
  echo "   node: found ($(node -v))"
else
  if tiene node; then echo "   node: $(node -v) is older than v$NODE_MIN"; fi
  instalar_node
fi
if tiene espeak-ng; then
  echo "   espeak-ng: found ($(espeak-ng --version 2>/dev/null | head -n1))"
  echo "   (the tests hold Python and the browser WASM to identical output, which needs 1.52;"
  echo "    an older distro build still runs, but the parity test may report differences)"
else
  instalar_espeak || echo "   espeak-ng: skipped. The engine has no phonemizer since ADR 0012, so this only affects tests/test_frontend_paridad.py"
fi

tiene uv   || { echo "uv is not on PATH: open a new shell and run this script again"; exit 1; }
tiene node || { echo "node is not on PATH: open a new shell and run this script again"; exit 1; }

# ---------------------------------------------------------------- python side
echo
echo "== Python environment (uv sync) =="
echo "   (torch comes from the PyTorch cu126 index pinned in pyproject: it is a large download"
echo "    and runs on CPU too, no GPU needed)"
uv sync --extra dev --extra export --extra voz

echo
echo "== Model weights =="
echo "   Nothing is required here any more (ADR 0012). The page loads its engine by itself:"
echo "   Pocket TTS (177 MB) comes straight from Hugging Face into the browser cache, and it is"
echo "   the one that can clone a voice from a recording."
echo
echo "   Supertonic is the other engine — it reads a touch better, at 44.1 kHz and in 31"
echo "   languages, but it cannot clone. It is 398 MB and only worth keeping locally if you"
echo "   are going to use it: the page can also pull it from Hugging Face on demand."
if [ -f models/supertonic/onnx/vector_estimator.onnx ]; then
  echo "   models/supertonic is already here, skipping."
else
  read -r -p "   Download Supertonic now (398 MB)? [s/N] " quiere
  case "$quiere" in
    s|S|y|Y) uv run python -m ttspro.supertonic.descargar ;;
    *) echo "   skipped; the page pulls it from Hugging Face when you pick that engine." ;;
  esac
fi

# ---------------------------------------------------------------- browser side
echo
echo "== Browser runtime (npm) =="
(
  cd web
  if [ -d node_modules ]; then
    # npm ci wipes node_modules first, which fails while a dev server holds files in it
    npm install --no-audit --no-fund
  else
    npm ci --no-audit --no-fund
  fi
  npm run preparar
)

echo
echo "== Done =="
echo "   demo:   cd web && npm run dev     then open http://localhost:5173"
echo "           the page loads its engine on its own; drop a recording on it to clone a voice"
echo "   tests:  uv run pytest tests -q     and, in web/, npm test"
echo "   measuring: add --extra eval (Whisper, jiwer) before running tests/terminado or"
echo "              anything that computes WER; uv sync prunes it if you leave it out"
echo

if [ "${1:-}" = "--sin-demo" ]; then exit 0; fi
read -r -p "Start the demo now? [s/N] " respuesta
case "$respuesta" in
  s|S|y|Y)
    if tiene xdg-open; then (sleep 2 && xdg-open http://localhost:5173) >/dev/null 2>&1 &
    elif tiene open; then (sleep 2 && open http://localhost:5173) >/dev/null 2>&1 &
    fi
    cd web && npm run dev
    ;;
esac
