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

## Browser end-to-end tests (Playwright)

Headless full-stack regression tests live in `packages/web/e2e/` (Playwright
Test runner — separate from the interactive Playwright MCP setup). The suite
starts both servers itself (Vite + the FastAPI backend, see
`playwright.config.ts`) and drives the real UI:

```bash
cd packages/web
npm install
npx playwright install chromium   # one-time browser download

npm test                    # headless (default)
npm run test:headed         # watch the browser while it runs
```

- Requires the Python packages installed per the Development section (the
  backend is launched from the repo-root `.venv`).
- CI: `.github/workflows/e2e.yml` runs the same suite headless on
  ubuntu-latest (backend via `uv`, frontend via `npm ci`).
- Artifacts (`test-results/`, `playwright-report/`, `.playwright-mcp/`) are
  gitignored.

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
  - Artifact store: `GET /api/artifacts/{id}/preview` serves plot PNGs and
    JSON tables; `Preview Plot` / `View Table` nodes render them inline in the
    browser (no run-level gallery — node-embedded viewers are the display, see
    DESIGN.md §9).
  - Workflow save/load as `emptyui/workflow` v1 JSON (port names and
    positions preserved; edge colors re-derived from port types on load).
- **M2 — complete**: mth5/aurora plugins; port-type naming
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
  - **Channel nomenclature**: `Compute FC` (and `Split Channels`, `Load
    LEMI-424`) expose an mt_metadata `channel_nomenclature` select
    (`auto`/`default`/`lemi12`/`lemi34`/`phoenix123`/`musgraves`), driven by
    `mt_metadata`'s canonical `CHANNEL_MAPS`. Selecting `lemi12`/`lemi34`
    auto-sets the channel names (`bx,by`/`e1,e2` or `e3,e4`) via manifest
    `cascade`; the **magnetic B/H mode is now derived automatically** from the
    nomenclature/channel names (bx/by → fluxgate B→H conversion; hx/hy → coil H),
    replacing the old manual `magnetic_unit` radio (kept as an `auto`/`h`/`b`
    select for legacy saved workflows). `Load LEMI-424` now selects the
    electric pair via nomenclature (lemi12 = e1/e2, lemi34 = e3/e4 — LEMI
    instruments can carry four E channels). `EstimateTF` builds the TF the
    way aurora does: with the **actual channel names as coordinates** plus the
    `channel_nomenclature` map attached to the mt_metadata TF (so aurora
    tooling can consume it and the instrument naming is preserved as
    provenance); exporters that require standard names (EDI/XML/AVG)
    standardize on write, and `ExportTF` relabels for the ZMM writer, which
    mt_metadata 1.0.10 assumes standard names.
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
    Inherited fields are **editable comboboxes** (datalist): they stay typeable,
    and only become dropdowns when an upstream node actually provides a value
    (a declared param connected to a source that yields nothing stays editable).
  - **Bug fix**: `Load LEMI-424` now overwrites a stale/corrupt MTH5 left by a
    force-stopped run instead of failing on it.
  - **Bug fix**: `Extract Run` with empty survey/station/run now falls back to
    the first available (instead of erroring on the removed `surveys_list`
    attribute) and supports both modern (`surveys_group`) and legacy
    single-survey (`survey_group`) archive layouts.
  - **Tippers**: `Compute FC`/`Estimate TF` accept a vertical magnetic output
    (e.g. `e1,e2,bz`) to estimate the geomagnetic transfer function (tipper) in
    addition to the 2×2 impedance; the unit normalization now treats a magnetic
    output as magnetic (not electric) so the tipper stays dimensionless.
  - **CI fixture**: `test12rr` (synthetic halfspace) is generated on the fly by
    test fixtures, so the MT tests run in CI without gitignored LEMI/BGS data.
  - **Cleanup**: removed superseded legacy aurora config/selection nodes
    (`BuildConfig`/`FrequencyBands`/`SelectRuns`) and the unused `bands`/
    `selection` input ports on `Compute FC`/`Estimate TF`; the default pipeline
    is `ExtractRun → ComputeFC → EstimateTF (+ MergeTFLevels)`.
- **M3 — in progress**:
  - **ComputeFC channel auto-configuration**: the tipper is on by default —
    `Compute FC`'s output channels resolve to the electric pair **plus** the
    vertical magnetic channel (bz/h3/hz) when the run contains one
    (impedance-only or tipper-only FCs are derived from what the data actually
    has; a requested-but-missing output is dropped with a warning). Empty
    input/output channel fields auto-detect from the data + nomenclature
    (case-insensitive, e.g. CSV `Ex,Ey,Hx,Hy`), and a nomenclature keyword
    inherited from the upstream chain (Load LEMI-424 → Extract Run → Compute
    FC) auto-fills the fields — explicit user choices are never overridden.
  - **Display nodes accept standard names / auto-select**: `Plot Time
    Series`'s channel list and `Plot Spectrogram`'s channel field now resolve
    through the inferred channel nomenclature — standard names (ex/ey/hx/hy/hz)
    map to the instrument names (e.g. `ex` → `e1` on LEMI data), matching is
    case-insensitive, and `Plot Spectrogram` defaults to the first electric
    channel when left empty (previously hardcoded `"Ex"`, which failed on
    LEMI/phoenix runs).
  - **Run cancellation**: `POST /api/runs/{id}/cancel` sets a per-run cancel
    token that the executor polls between nodes; the toolbar **✕ Cancel**
    button is enabled while a run is queued/running and stops the pipeline at
    the next node boundary (a cancelled run reports `cancelled`, and the Run
    button is freed for a retry). Nodes already executing finish first.
  - **Artifact download**: `GET /api/artifacts/{id}` serves the full artifact
    with `Content-Disposition: attachment`; `Preview Plot` / `View Table` nodes
    expose a **⤓ download** link next to the inline display (the `/preview`
    endpoint stays inline-only).
  - **E2E regression suite**: headless Playwright tests in `packages/web/e2e/`
    (full-stack — see the Browser end-to-end tests section) cover the app
    boot, nomenclature cascade, the Load CSV → Plot → Preview pipeline,
    artifact download, and run cancellation.

### UI conventions (stable)

- **ComfyUI paradigm**: node parameters render inline on the node body; the
  right-hand panel shows metadata (node id, version, ports, description, deps)
  and a Delete action. Path widgets include a server-side file browser ("…"
  button).
- **Plots as data**: Display nodes emit a typed `plot` port; `Preview Plot`
  nodes render figures inline (drag-resizable) and `Save Plot` exports them —
  ComfyUI's Preview/Save Image pattern. There is no run-level gallery; the
  node-embedded viewers are the primary (and only) display (DESIGN.md §9).
