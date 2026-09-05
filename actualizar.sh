#!/usr/bin/env bash
# ttspro - update an existing install to the latest version on GitHub.
#   ./actualizar.sh            pull, sync the Python env, rebuild models only if
#                              the model pipeline changed, refresh the browser side
#   ./actualizar.sh --modelos  force the model rebuild
#
# The expensive step is the model rebuild, so it only runs when the commits that
# came in touched what produces models/: the ports, the exporter, the model code
# or the contract. `git diff --name-only` between the old and the new HEAD decides,
# not a guess.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
export PATH="$HOME/.local/bin:$HOME/.local/node/bin:$PATH"

echo
echo "== ttspro: update =="
echo

for herramienta in git uv node npm; do
  command -v "$herramienta" >/dev/null 2>&1 || {
    echo "$herramienta is not on PATH: run ./instalar.sh first"
    exit 1
  }
done

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "You have local changes; commit or stash them before updating:"
  git status --short --untracked-files=no
  exit 1
fi

antes="$(git rev-parse HEAD)"
git pull --ff-only
despues="$(git rev-parse HEAD)"

if [ "$antes" = "$despues" ]; then
  echo "   already up to date ($(git rev-parse --short HEAD))"
else
  echo "   $(git rev-list --count "$antes..$despues") new commit(s):"
  git log --oneline "$antes..$despues" | sed 's/^/     /'
fi

echo
echo "== Python environment (uv sync) =="
uv sync --extra dev --extra export

# ---------------------------------------------------------------- models
reconstruir=0
if [ "${1:-}" = "--modelos" ]; then
  reconstruir=1
  echo "   models: rebuild forced"
elif [ ! -f models/tts.onnx ] || [ ! -f models/conversor.onnx ] || [ ! -f models/voz.onnx ]; then
  reconstruir=1
  echo "   models: some graphs are missing"
elif [ "$antes" != "$despues" ] && git diff --name-only "$antes" "$despues" -- \
      src/ttspro/export src/ttspro/model models/contrato.json pyproject.toml | grep -q .; then
  reconstruir=1
  echo "   models: the model pipeline changed in this update:"
  git diff --name-only "$antes" "$despues" -- src/ttspro/export src/ttspro/model models/contrato.json pyproject.toml | sed 's/^/     /'
fi

if [ "$reconstruir" = 1 ]; then
  echo
  echo "== Model weights: rebuild from public weights (about a minute on CPU) =="
  uv run python -m ttspro.export.modelos --sin-encoder
else
  echo "   models: unchanged, kept"
fi

# ---------------------------------------------------------------- browser side
echo
echo "== Browser runtime (npm) =="
(
  cd web
  npm install --no-audit --no-fund
  npm run preparar
)

echo
echo "== Updated to $(git rev-parse --short HEAD) =="
echo "   demo:   cd web && npm run dev     then open http://localhost:5173"
echo
