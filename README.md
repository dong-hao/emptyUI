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
- **M2 — in progress**: mth5/aurora plugins scaffolded; port-type naming
  overhaul; LEMI-424 loader (`mth5.io.lemi`), Extract Run, Detrend
  (mean/linear/none), ComputeFC (TS→FC windowed STFT), EstimateTF (FC→TF
  cross-power LSQ), TF export to EDI, Plot AppRes & Phase (diagonal/
  off-diagonal). Verified end-to-end through the executor on real 1 Hz
  LEMI-424 and BGS long-period data, plus the official synthetic halfspace.
  Processing-correctness fixes:
  - **Unit normalization**: B→H (nT→T→H) and mV/km→V/m; the missing nT→T
    factor was the root cause of non-physical apparent resistivity (now
    ~10–10³ Ω·m).
  - **Impedance phase convention** verified against aurora on the synthetic
    halfspace (`test12rr`): aligned with aurora/EDI (Zxy≈+45°, Zyx≈−135°).
  - **Robust estimator**: `EstimateTF` offers `Least Squares` or `Robust`
    (aurora-backed) with a separate remote-reference checkbox (Robust → RME/RME_RR).
    No least-squares+remote-reference path by design. Verified on the halfspace.
  - **Table viewer**: `mth5.table.selection` emits a JSON table artifact and
    `core.display.view_table` renders it inline in the browser.
  - `scripts/convert_bgs_lemi.py` converts BGS `.xyz` long-period files to
    LEMI-424 `.TXT`.
  - **Calibrate**: `mth5.calibrate` removes the instrument response from a
    `ts.run` — using the run's own mth5 channel responses, or an explicit
    `calibration` filter via frequency-domain response removal.
  M2 is now complete (all §11 M2 deliverables delivered).
- **M2 — Phase 4 polish**:
  - **Auto field change**: switching `Compute FC` magnetic input `h ⇄ b`
    auto-updates the channel names (`hx,hy/ex,ey` ⇄ `bx,by/e1,e2`); ticking
    `Remote Reference` on `Estimate TF` auto-switches the estimator to Robust.
    Driven by manifest `Param.cascade` rules (see §4).
  - **Auto field fill**: `Compute FC`'s survey/station/run become dropdowns
    populated from the connected upstream time-series node, via manifest
    `Node.inherit` (survey/station/run auto-filled once connected). Designed as a
    frontend-only, recursive **chain-walking** resolver with per-node name
    mapping (see DESIGN.md §4 "Chain-walking inheritance"), so identities flow
    through intermediate nodes and future processing nodes without losing info.
  - **Bug fix**: `Load LEMI-424` now overwrites a stale/corrupt MTH5 left by a
    force-stopped run instead of failing on it.
  - **CI fixture**: `test12rr` (synthetic halfspace) is generated on the fly by
    test fixtures, so the MT tests run in CI without gitignored LEMI/BGS data.

### UI conventions (stable)

- **ComfyUI paradigm**: node parameters render inline on the node body; the
  right-hand panel is metadata-only (node id, version, ports). Path widgets
  include a server-side file browser ("…" button).
- **Plots as data**: Display nodes emit a typed `plot` port; `Preview Plot`
  nodes render figures inline (drag-resizable) and `Save Plot` exports them —
  ComfyUI's Preview/Save Image pattern. The Results panel is a run-level
  gallery, not the primary viewer.
