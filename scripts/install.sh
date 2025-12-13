#!/usr/bin/env bash
# One-shot dependency installer for emptyUI development.
#
#   ./scripts/install.sh [--with-browsers]
#
# Sets up the repo-root .venv (Python) and packages/web/node_modules (npm).
# The Python side is installed EDITABLE from the workspace pyproject files —
# no separate requirements.txt: each package's [dev] extra is the single
# source of truth for its test/lint dependencies (uv is preferred; falls
# back to python3 -m venv + pip when uv is not installed).
#
# Flags:
#   --with-browsers   also download the Playwright Chromium build (needed
#                     for the e2e suite: cd packages/web && npm test)
set -euo pipefail
cd "$(dirname "$0")/.."

with_browsers=0
for arg in "$@"; do
  case "$arg" in
    --with-browsers) with_browsers=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v uv >/dev/null 2>&1; then
    PY="uv"
  else
    PY="python3"
  fi
fi

echo "== Python environment =="
if [ ! -d .venv ]; then
  echo "creating .venv …"
  if [ "$PY" = "uv" ]; then
    uv venv --python 3.12 .venv
  else
    "$PY" -m venv .venv
  fi
else
  echo "reusing existing .venv"
fi

if [ "$PY" = "uv" ]; then
  uv pip install \
    -e "packages/emptyui[dev]" \
    -e "packages/emptyui-core-nodes[dev]" \
    -e "packages/emptyui-mth5[dev]" \
    -e "packages/emptyui-aurora[dev]" \
    -e "examples/plugin-template[dev]"
else
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install \
    -e "packages/emptyui[dev]" \
    -e "packages/emptyui-core-nodes[dev]" \
    -e "packages/emptyui-mth5[dev]" \
    -e "packages/emptyui-aurora[dev]" \
    -e "examples/plugin-template[dev]"
fi

echo "== Node dependencies =="
npm --prefix packages/web install

if [ "$with_browsers" = "1" ]; then
  echo "== Playwright Chromium =="
  npx --prefix packages/web playwright install chromium
fi

echo
echo "Install complete."
echo "  run the backend + frontend:  ./scripts/dev.sh"
echo "  run the e2e suite:           cd packages/web && npm test"
echo "  backend tests:               .venv/bin/python -m pytest packages -q"
