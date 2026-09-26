#!/usr/bin/env bash
# Start OmniLinker with one command.
#
# The default path needs no Docker, no MongoDB and no Neo4j: the embedded store
# backend is a JSON document store plus an in-memory adjacency graph, so this
# script installs the backend, builds the frontend if needed, and serves both
# from a single uvicorn process.
#
#   ./run.sh              # install if needed, build the UI, serve on :8900
#   ./run.sh --reinstall  # force a fresh venv
#   ./run.sh --no-ui      # skip the frontend build (API only)
#   ./run.sh --docker     # the real Mongo + Neo4j profile
#
# Port: 8900 by default, not 8000. 8000 is conventionally an
# OpenAI-compatible server, and a collision there surfaces as a 502 from
# whatever fronts the model API rather than as an obvious "port in use".
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
VENV="$BACKEND/.venv"
PORT="${OMNI_PORT:-8900}"
REINSTALL=0
BUILD_UI=1
USE_DOCKER=0

for arg in "$@"; do
  case "$arg" in
    --reinstall) REINSTALL=1 ;;
    --no-ui)     BUILD_UI=0 ;;
    --docker)    USE_DOCKER=1 ;;
    -h|--help)   sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $arg (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

# --- docker profile --------------------------------------------------------
if [ "$USE_DOCKER" = 1 ]; then
  command -v docker >/dev/null || die "--docker needs Docker, which is not installed.
  The default path (no flags) runs with no external services at all."
  say "building the frontend image assets"
  [ -d "$FRONTEND/node_modules" ] || (cd "$FRONTEND" && npm install)
  (cd "$FRONTEND" && npm run build)
  say "docker compose up (Mongo + Neo4j + API)"
  exec docker compose up --build
fi

# --- python ----------------------------------------------------------------
command -v python3 >/dev/null || die "python3 not found"
PY=python3

if [ "$REINSTALL" = 1 ]; then rm -rf "$VENV"; fi
if [ ! -d "$VENV" ]; then
  say "creating the virtualenv"
  "$PY" -m venv "$VENV"
fi
PY="$VENV/bin/python"

# --only-binary matters: pydantic-core has no source build path without a Rust
# toolchain, and the resulting error is very long and says nothing useful.
if ! "$PY" -c "import fastapi, uvicorn, cryptography, pytest" >/dev/null 2>&1; then
  say "installing backend dependencies"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet --only-binary :all: -r "$BACKEND/requirements.txt"
fi

# --- frontend --------------------------------------------------------------
if [ "$BUILD_UI" = 1 ]; then
  if command -v npm >/dev/null; then
    if [ ! -d "$FRONTEND/node_modules" ]; then
      say "installing frontend dependencies"
      (cd "$FRONTEND" && npm install --silent)
    fi
    if [ ! -f "$FRONTEND/dist/index.html" ] \
       || [ -n "$(find "$FRONTEND/src" -newer "$FRONTEND/dist/index.html" 2>/dev/null | head -1)" ]; then
      say "building the frontend"
      (cd "$FRONTEND" && npm run build)
    else
      say "frontend build is current"
    fi
  else
    say "npm not found - serving the API only (see / for instructions)"
    BUILD_UI=0
  fi
fi

if [ "$BUILD_UI" = 1 ]; then
  export OMNI_FRONTEND_DIST="$FRONTEND/dist"
  say "frontend: $FRONTEND/dist"
fi

# --- run -------------------------------------------------------------------
export OMNI_DATA_DIR="${OMNI_DATA_DIR:-$BACKEND/omni-data}"
export OMNI_VECTOR="${OMNI_VECTOR:-1}"
mkdir -p "$OMNI_DATA_DIR"

say "serving on http://127.0.0.1:$PORT"
say "data dir: $OMNI_DATA_DIR"
say "seed the demo workspace:  curl -XPOST http://127.0.0.1:$PORT/api/sync \\"
say "                           -H 'content-type: application/json' -d '{\"connector_id\":\"demo\"}'"
exec "$PY" -m uvicorn omnilinker.api.app:app \
  --app-dir "$BACKEND" --host 127.0.0.1 --port "$PORT" "$@"
