#!/usr/bin/env bash
# Start the emptyUI backend API server from the repo-root venv.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn emptyui.server.app:create_app --factory \
    --host 127.0.0.1 --port 8000 --reload "$@"
