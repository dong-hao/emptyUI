#!/usr/bin/env bash
# Serve emptyUI from one port: the API plus the built web frontend
# (same-origin — no node and no second server at runtime).
#
#   ./scripts/serve.sh [--host H] [--port P] [--data-root DIR] [--rebuild]
#
# Builds packages/web/dist if missing (or with --rebuild); building needs
# node/npm ONCE — serving afterwards is pure Python (.venv). The built
# frontend is mounted via EMPTYUI_SERVE_DIST (docs/deployment.md): /api/*
# routes win, index.html is served at the root.
#
# Examples:
#   ./scripts/serve.sh                              # http://127.0.0.1:8000
#   ./scripts/serve.sh --host 0.0.0.0 --port 8080   # expose to the network
#   ./scripts/serve.sh --data-root /srv/mt/data     # confine file access
#   ./scripts/serve.sh --rebuild                    # rebuild dist first
set -euo pipefail
cd "$(dirname "$0")/.."

host=127.0.0.1
port=8000
data_root=""
rebuild=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --rebuild) rebuild=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ ! -x .venv/bin/uvicorn ]; then
  echo "error: .venv/bin/uvicorn not found — run ./scripts/install.sh first" >&2
  exit 1
fi

dist=packages/web/dist
if [ "$rebuild" = "1" ] || [ ! -f "$dist/index.html" ]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "error: node/npm not found and $dist is missing/builder requested." >&2
    echo "       Install node >= 20 once (the build result is plain static" >&2
    echo "       files), or obtain a prebuilt $dist." >&2
    exit 1
  fi
  echo "== Building web frontend ($dist) =="
  if [ ! -d packages/web/node_modules ]; then
    npm --prefix packages/web install
  fi
  npm --prefix packages/web run build
fi

export EMPTYUI_SERVE_DIST="$dist"
if [ -n "$data_root" ]; then
  export EMPTYUI_DATA_ROOT="$data_root"
fi

echo
echo "== emptyUI on http://$host:$port (frontend + API, same origin) =="
if [ -n "$data_root" ]; then
  echo "== data confined to $data_root =="
fi
exec .venv/bin/uvicorn emptyui.server.app:create_app --factory \
    --host "$host" --port "$port"
