#!/usr/bin/env bash
# ttspro - update an existing install to the latest version on GitHub.
#   ./actualizar.sh   pull, sync the Python env, refresh the browser side
#
# There is no model rebuild any more (ADR 0012): the weights are DOWNLOADED at a
# pinned revision, so a code change cannot leave them stale. The only thing that
# can make them wrong is the pinned revision itself changing, and that is one
# `git diff` on one file.
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
uv sync --extra dev --extra export --extra voz

# ---------------------------------------------------------------- models
# Ya no se reconstruye nada: desde ADR 0012 los pesos se DESCARGAN con la revision
# fijada, asi que no pueden quedarse viejos por un cambio de codigo. Lo unico que
# puede pasar es que cambie la revision fijada, y entonces hay que volver a bajarlos.
if [ ! -f models/supertonic/onnx/vector_estimator.onnx ]; then
  echo "   models: Supertonic is not downloaded; the page pulls it from Hugging Face on demand"
  echo "           (or run: uv run python -m ttspro.supertonic.descargar)"
elif [ "$antes" != "$despues" ] && git diff --name-only "$antes" "$despues" -- src/ttspro/supertonic/descargar.py | grep -q .; then
  echo "   models: the pinned revision may have changed in this update; re-downloading"
  uv run python -m ttspro.supertonic.descargar
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
