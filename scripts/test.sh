#!/usr/bin/env bash
# One-command gate runner, mirroring CI (.github/workflows/tests.yml).
#
#   ./scripts/test.sh                 # backend (both pytest groups) + ruff
#                                     #   + frontend (tsc, eslint, vitest)
#   ./scripts/test.sh --fast          # happy-path pytest + ruff (quick loop)
#   ./scripts/test.sh --backend       # full pytest + ruff
#   ./scripts/test.sh --frontend      # tsc -b, eslint, vitest
#   ./scripts/test.sh --e2e           # Playwright suite (starts both servers)
#   ./scripts/test.sh --fast --frontend   # flags combine
#
# Exit code is nonzero on the first failing gate.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/pytest ]; then
  echo "error: .venv/bin/pytest not found — run ./scripts/install.sh first" >&2
  exit 1
fi

want_backend=0 want_frontend=0 want_e2e=0 fast=0 any=0
for arg in "$@"; do
  case "$arg" in
    --fast) want_backend=1; fast=1; any=1 ;;
    --backend) want_backend=1; any=1 ;;
    --frontend) want_frontend=1; any=1 ;;
    --e2e) want_e2e=1; any=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$any" = "0" ]; then
  want_backend=1; want_frontend=1   # no flags: the full local gate
fi

if [ "$want_backend" = "1" ]; then
  if [ "$fast" = "1" ]; then
    echo "== pytest happy path =="
    .venv/bin/python -m pytest packages -q -m "not errors"
  else
    echo "== pytest happy path =="
    .venv/bin/python -m pytest packages -q -m "not errors"
    echo "== pytest edge cases =="
    .venv/bin/python -m pytest packages -q -m errors
  fi
  echo "== ruff =="
  .venv/bin/ruff check .
fi

if [ "$want_frontend" = "1" ]; then
  echo "== tsc -b =="
  npm --prefix packages/web run typecheck
  echo "== eslint =="
  npm --prefix packages/web run lint
  echo "== vitest =="
  npm --prefix packages/web run test:unit
fi

if [ "$want_e2e" = "1" ]; then
  echo "== Playwright e2e =="
  npm --prefix packages/web test
fi

echo "all gates passed"
