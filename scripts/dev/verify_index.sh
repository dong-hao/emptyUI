#!/bin/zsh
# Verify the staged index in isolation (dev tool, not part of the app).
#
# Exports the git index to a temporary directory, points PYTHONPATH at the
# exported aurora sources (overriding the editable install that points at the
# working tree), then runs the given pytest targets and ruff on the export.
#
# usage (repository root):  scripts/dev/verify_index.sh <pytest targets...>
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(mktemp -d /tmp/emptyui-index.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

cd "$ROOT" && git checkout-index -a --prefix="$OUT/"
cd "$OUT"
export PYTHONPATH="$OUT/packages/emptyui-aurora/src"
"$ROOT/.venv/bin/python" -c "import emptyui_aurora._processing as p; print('using', p.__file__)"
"$ROOT/.venv/bin/python" -m pytest "$@" -q -p no:cacheprovider
"$ROOT/.venv/bin/ruff" check "$OUT/packages/emptyui-aurora"
