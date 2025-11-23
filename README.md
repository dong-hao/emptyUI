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

- **M0 — complete**: monorepo scaffold; core Node API/registry/validator;
  sequential executor; FastAPI server; React Flow web shell with custom typed
  nodes, palette, schema-driven properties panel.
- **M1 — complete**:
  - Real pipeline nodes: `core.io.load_csv` (auto channel/time-column
    detection) → `core.filter.bandpass` / `core.filter.detrend` (scipy
    filtfilt) → `core.plot.timeseries` / `core.plot.spectrogram` (matplotlib
    PNG artifacts). Verified end-to-end on a 4-channel 150 Hz synthetic CSV.
  - Async run execution on worker threads; events streamed to the browser
    over WebSocket (`/api/ws`); per-node status/progress overlaid on nodes.
  - Artifact store: `GET /api/artifacts/{id}/preview` serves plot PNGs;
    docked result panel in the UI shows the latest plot + thumbnails.
  - Workflow save/load as `emptyui/workflow` v1 JSON (port names and
    positions preserved; edge colors re-derived from port types on load).
- **Next: M2** — MT ecosystem integration (`emptyui-mth5`, `emptyui-aurora`
  plugins, ComputeFC/EstimateTF, EDI export, table viewer).

### UI conventions (stable)

- **ComfyUI paradigm**: node parameters render inline on the node body; the
  right-hand panel is metadata-only (node id, version, ports). Path widgets
  include a server-side file browser ("…" button).
- **Plots as data**: Display nodes emit a typed `plot` port; `Preview Plot`
  nodes render figures inline (drag-resizable) and `Save Plot` exports them —
  ComfyUI's Preview/Save Image pattern. The Results panel is a run-level
  gallery, not the primary viewer.
