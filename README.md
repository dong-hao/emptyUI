# emptyUI

A node-based, ComfyUI-inspired visual workflow interface for magnetotelluric
(MT) data processing and visualization. See [DESIGN.md](DESIGN.md) for the
architecture and design decisions.

## Layout

```
packages/
├── emptyui/              # Python core: node API, registry, DAG executor, FastAPI server
├── emptyui-core-nodes/   # bundled minimal node plugin
└── web/                  # React + React Flow frontend (Vite)
```

## Development

Requires Python ≥3.10 with [uv](https://docs.astral.sh/uv/) and Node ≥20.

```bash
# Backend — repo-root venv, both packages installed editable
uv venv --python 3.12 .venv
uv pip install -e "packages/emptyui[dev]" -e "packages/emptyui-core-nodes[dev]"

# Run tests
.venv/bin/python -m pytest packages -q

# Start API server on http://127.0.0.1:8000
./scripts/dev-backend.sh

# Frontend — dev server on http://localhost:5173 (proxies /api to :8000)
cd packages/web && npm install && npm run dev
```

## Status

- **M0 — complete**:
  - Python core (`packages/emptyui`): Node declaration API (typed ports +
    params), entry-point plugin registry, workflow JSON validation
    (type checking, cycle detection, actionable errors), sequential DAG
    executor with handle-based data store, FastAPI server
    (`/api/health`, `/api/nodes`, `/api/runs`, `/api/runs/{id}`).
  - Bundled plugin (`packages/emptyui-core-nodes`): `core.util.echo` node,
    discovered via entry points.
  - Web shell (`packages/web`): React + @xyflow/react canvas with custom
    typed nodes (ComfyUI-style port dots aligned to labels inside node edges),
    type-colored thick connection wires, resizable/deletable nodes,
    category-grouped palette and schema-driven properties panel fed by the
    backend catalog; property edits persist in the workflow store.
  - Repo-root uv venv; backend tests green, ruff clean, frontend builds.
- **Next: M1** — real pipeline nodes (`load_csv → bandpass → plot`),
  WebSocket run events, artifact streaming, workflow save/load.
