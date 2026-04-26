#!/usr/bin/env bash
# Launch the emptyUI backend AND frontend together for local development.
#
#   ./scripts/dev.sh
#
# Starts the API server (scripts/dev-backend.sh: uvicorn on :8000) and the
# Vite dev server (packages/web on :7860, /api proxied to the backend).
# Ctrl+C stops both cleanly. Useful when you want one command instead of
# two terminals; the Playwright e2e suite still starts its own servers.
set -euo pipefail
cd "$(dirname "$0")/.."

backend_pid=""
frontend_pid=""

kill_tree() {
  local pid=$1
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping dev servers…"
  [ -n "$backend_pid" ] && kill_tree "$backend_pid"
  [ -n "$frontend_pid" ] && kill_tree "$frontend_pid"
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait_for() {
  local url=$1 name=$2 tries=${3:-30}
  for _ in $(seq 1 "$tries"); do
    if curl -sf -o /dev/null "$url"; then
      echo "$name ready: $url"
      return 0
    fi
    sleep 0.5
  done
  echo "WARNING: $name did not become ready at $url (still starting?)" >&2
}

./scripts/dev-backend.sh &
backend_pid=$!

(cd packages/web && npm run dev) &
frontend_pid=$!

echo "Starting emptyUI dev servers (Ctrl+C to stop both)…"
wait_for http://127.0.0.1:8000/api/health "backend" 30
wait_for http://localhost:7860 "frontend (Vite)" 40

echo
echo "  emptyUI:      http://localhost:7860"
echo "  backend API:  http://127.0.0.1:8000/api"
echo

wait
