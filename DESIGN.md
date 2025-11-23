# emptyUI — Design Document

A node-based, ComfyUI-inspired visual workflow interface for magnetotelluric (MT)
data processing and visualization.

Status: v0.2 — design frozen for M0/M1 (2026-08-25).
Changes after v0.1: aurora-informed port type system, node categories,
TS→FC→TF stage split, MTH5-backed file storage, backend-decided port colors
and ComfyUI-style port rendering conventions (§3).
Implementation status: **M0 complete; M1 complete** — real pipeline nodes
(`load_csv → bandpass → plot.timeseries/spectrogram`) verified end-to-end on
a 4-channel 150 Hz synthetic CSV; async runs with WebSocket event streaming,
artifact store + preview endpoint, workflow save/load (format v1).
Next milestone: M2 (mth5/aurora plugins, ComputeFC/EstimateTF, EDI export).

---

## 1. Goals & Non-Goals

### Goals
- Lower the barrier to MT processing: replace scattered CLI tools with a
  drag-and-drop, composable workflow builder.
- **Detached front-end / back-end**: browser UI talks to a Python FastAPI server
  over HTTP + WebSocket; the server may run locally or remotely.
- **Plugin-first extensibility**: new nodes are added as plugins with zero
  modifications to the core codebase — and ideally zero frontend changes.
- Minimal core dependencies; heavy scientific libraries (mth5, aurora,
  mt_metadata, scipy…) are pulled in only by the plugins that need them.
- Workflows are portable JSON documents — save, share, re-run, batch.

### Non-Goals (for MVP)
- Multi-user auth, quotas, or cloud orchestration (design leaves room for it).
- Real-time collaborative editing.
- A general-purpose dataflow language; we target DAG workflows only.

---

## 2. Architecture Overview

```
┌───────────────────────────── Browser ─────────────────────────────┐
│  React + TypeScript (Vite)                                        │
│  ┌───────────────┐ ┌──────────────────┐ ┌──────────────────────┐  │
│  │ React Flow    │ │ Properties panel │ │ Result viewers       │  │
│  │ graph canvas  │ │ (schema-driven)  │ │ (plots/tables/text)  │  │
│  └───────────────┘ └──────────────────┘ └──────────────────────┘  │
│        │                  │                       ▲               │
│        └────────┬─────────┴───────────────────────┘               │
│           REST (workflow CRUD, node catalog)                      │
│           WebSocket (run events, progress, artifacts)             │
└──────────────────────────────┬────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│  FastAPI backend                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Node registry│  │ DAG executor │  │ Artifact store           │  │
│  │ (+ discovery)│→ │ (topological │→ │ (in-memory refs + files  │  │
│  └──────────────┘  │  scheduler)  │  │  on disk)                │  │
│                    └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Core nodes plugin (import / filter / plot — minimal deps)    │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ External plugins: mth5, aurora, mt_metadata, ML, …           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

Key principle: **the frontend never hardcodes node behavior**. The backend is
the single source of truth for what nodes exist and what parameters they take;
the frontend renders everything from that metadata.

---

## 3. Core Concepts

| Concept      | Description |
|--------------|-------------|
| **Node type**   | A registered processing/visualization step (e.g. `core.import.csv`). Declares inputs, outputs, parameters, and a Python `execute()` function. |
| **Node instance** | A placed copy of a node type in a workflow, with concrete parameter values and connections. |
| **Workflow**    | A DAG of node instances + their parameter settings. Serialized as one JSON document. |
| **Plugin**      | A pip-installable Python package that registers one or more node types. |
| **Artifact**    | Output produced by a node run (plot image, table, file). Identified by ID, streamed to the frontend over WebSocket or fetched via REST. |
| **Run**         | One execution of a workflow; produces per-node status, logs, artifacts. |

### Node categories

Nodes fall into three broad categories (used for palette grouping and icons;
categories are open-ended strings so plugins can add their own):

| Category | Role | Examples |
|----------|------|----------|
| **Data I/O** | Bring data in/out of the graph; convert formats | LoadMTH5, LoadNIMS, LoadCSV, SaveMTH5, SaveEDI, ExportXML, MetadataEditor |
| **Processing** | Transform data; produce new typed outputs | Decimate, Detrend, BandpassFilter, NotchFilter, Calibrate(RemoveResponse), STFT, RobustEstimate, MergeTFLevels |
| **Display** | Render data to `plot` outputs (no data deps downstream) | PlotTimeSeries, PlotSpectrogram, PlotAppResPhase, PlotCoherency, PlotPhaseTensor |

Display nodes emit a **`plot` output port** (rendered-figure reference), not
typed data — so they can't feed processing nodes, but their figures can branch
to `Preview Plot` / `Save Plot` nodes. This mirrors ComfyUI's IMAGE port +
Preview/Save Image pattern: multiple results are each handled by their own
viewer node (drag-resizable, positioned anywhere on the canvas) instead of a
single fixed results pane. The run-level Results panel remains as a
convenience gallery.

### Port types (the ComfyUI-equivalent)

Like ComfyUI's colored sockets (`MODEL`, `VAE`, `CLIP`…), every node port has a
**type tag**, and connections are allowed only between compatible tags. Types
are chosen to mirror the real containers of the mth5/aurora ecosystem, so a
node boundary ≈ an aurora pipeline stage:

| Tag | Backed by | Stage / notes |
|-----|-----------|---------------|
| `file` | path string | Generic file reference (importer/exporter params) |
| `mt.archive` | open `mth5.MTH5` | HDF5 archive; stateful source of runs/FCs/TFs |
| `ts.run` | `mth5.RunTS` (xr.Dataset) | Multi-channel time series — the main signal-processing currency |
| `ts.channel` | `mth5.ChannelTS` (xr.DataArray) | Single channel (ex/ey/hx/hy/hz); split/merge nodes convert ⇄ `ts.run` |
| `mt.response` | `mt_metadata.filters.*` | Instrument/calibration response; `Calibrate` takes `ts.run` + `mt.response` → calibrated `ts.run` |
| `mt.selection` | RunSummary/KernelDataset (DataFrame) | Which stations/runs/intervals process; local+remote role assignment |
| `mt.decimation` | aurora `DecimationLevel` spec | *Plugin-tier*: per-level recipe for advanced multi-config assembly |
| `mt.config` | aurora `Processing` config | *Plugin-tier*: full-config passthrough for advanced workflows; core splits stage params across the two pipeline nodes instead |
| `mt.bands` | FrequencyBands | Band definitions feeding TF estimation |
| `spectra` | xr.Dataset `(time, frequency)` complex | Fourier coefficients / spectrogram. TS→FC is the natural direction; FC→TS is not offered in core (a lossy inverse can be added later as a plugin coercion node, e.g. for demonstrating frequency-domain denoising) |
| `tf.level` | aurora `TTFZ` | Per-decimation-level TF estimate |
| `mt.tf` | `mt_metadata.transfer_functions.TF` (xr.Dataset) | Final merged transfer function; exporters & analysis plots consume this |
| `table` | pandas DataFrame | Generic tabular output |
| `plot` | rendered-figure reference (PNG/SVG/PDF path + metadata) | Produced by Display nodes; consumed by `Preview Plot` (inline, resizable) and `Save Plot` (export). Distinct from a future generic `image` type for non-plot bitmaps |
| `any` | — | Wildcard; connects to anything, executor trusts the node |

Rules:
- Tags are **open strings**, namespaced (`<domain>.<name>`). Plugins may define
  new tags; the frontend renders unknown types with a neutral color.
- **Port colors are decided by the backend**: each type tag carries UI metadata
  (`color`, label) served via `GET /api/nodes`; this overrides the frontend
  fallback palette, so plugins define colors for their own tags without any
  frontend change. Connection lines are rendered thick and in the source
  port's color (ComfyUI-style), making data domains readable at a glance.
- Suggested color families (frontend fallback follows this guide):
  - **blue** — time series: `ts.run` medium blue, `ts.channel` light blue
  - **dark green** — frequency domain: `spectra` / Fourier coefficients
  - **red** — transfer functions: `mt.tf`, `tf.level`
  - **purple** — metadata/config: `mt.selection`, `mt.config`,
    `mt.response`, `mt.bands`, `mt.decimation`
  - gray/yellow — generic (`any`, `file`, `table`, `image`)
- Connection rule: source and target tags must be equal, or the target port
  declares `accepts: [ ... ]` listing extra admissible tags. `any` matches all.
- Each tag carries UI metadata (color, label, description) served by
  `GET /api/nodes` alongside manifests → frontend stays schema-driven.
- Subtyping is deliberately **not** implemented; explicit coercion nodes
  (split/merge/configure) are preferred — they're visible in the graph and
  match how aurora itself stages conversions.

### Port rendering conventions

Following ComfyUI's visual grammar: port dots sit **just inside** the node
edges (not exactly on them), vertically aligned with their name/description
text row — inputs near the left edge with names to their right, outputs near
the right edge with names to their left. Node bodies therefore reserve rows
for ports so dot and label always align.

### Reference MT workflow (type flow)

```
LoadMTH5 ─mt.archive─▶ SelectRuns ──┬─mt.selection────────────────────────┐
                                    │                                     ▼
LoadMTH5 ─mt.archive─▶ ExtractRun ─ts.run─▶ ComputeFC ─spectra─▶ EstimateTF ─mt.tf─┬▶ SaveEDI
                                              │                                     └▶ PlotAppResPhase
                                              └─spectra─▶ Denoise ─spectra─▶ EstimateTF

LoadMTH5 ─mt.archive─▶ ExtractRun ─ts.run─▶ Detrend ─ts.run─▶ STFT ─spectra─▶ PlotSpectrogram
```

Coercion points that require dedicated nodes (mirroring aurora internals):
1. `mt.selection` + roles → kernel dataset ("configure" node).
2. `ts.run` ⇄ `ts.channel` via explicit SplitChannels / MergeChannels nodes.
3. uncalibrated `ts.run` + `mt.response` → calibrated `ts.run` (units change).
4. `tf.level[]` → `mt.tf` (merge; resolves duplicate periods across sampling
   rates / decimation levels — the multi-level merge case).
5. Format coercions for popular types ship in core (`SaveEDI`, `LoadEDI`,
   …); exotic import/export converters are plugins.

### Pipeline stage split: TS → FC → TF

The aurora processing run is **not** one monolithic node. It is split at the
Fourier-coefficient boundary, mirroring both the conceptual MT pipeline
(decimation → windowing → time/frequency transform → power spectra → TF
estimation) and aurora v0.3.x's own ability to persist and reuse FCs:

- **`ComputeFC` node (TS → FC)** — input `ts.run`; params: number of levels,
  per-level decimation factor, window type/length/overlap. Output `spectra`.
  Internally expands uniform decimation levels sequentially (level sample
  rates derived from factor × initial rate). Persists computed FCs into the
  workspace MTH5's Fourier-CoEfficients layer.
- **`EstimateTF` node (FC → TF)** — inputs `spectra` + `mt.selection`;
  params: estimator engine (`OLS`/`RME`/`RME_RR`), remote-reference on/off,
  input/output/reference channels; optional `mt.bands` input for custom band
  edges. Output `mt.tf`.

Why this boundary:
1. **Frequency-domain work**: a first-class `spectra` port lets users branch
   into denoising, spectral analysis, or custom estimators before TF
   estimation.
2. **Cheap re-runs**: STFT/decimation is the expensive stage. Estimator
   parameters live *only* on `EstimateTF`, so with content-addressed caching
   changing the estimator invalidates just that node — saved/cached FCs are
   reused untouched.
3. **No parameter wall**: each node carries only its stage's parameters;
   aurora's monolithic `Processing` object is assembled internally by the two
   nodes sharing context, never shown to the user as one form.

The full-config passthrough (`mt.config` port) remains available at plugin
tier for advanced cases (e.g. per-level customization via an `mt.decimation`
fan-in assembler).

### Data passing between nodes
Nodes exchange typed *handles* (tagged above), not raw payloads, in the API layer:
- All array-like containers → `xarray` objects held in an in-process store keyed by handle ID.
- Tabular objects → DataFrames in the same store.
- Files → paths within a per-run workspace directory.
- Plots → PNG/SVG artifact files.

This keeps WebSocket messages small and lets the same executor serve batch/CLI use later.

### Storage: file-based, backed by MTH5

No internal database or bespoke data format. **MTH5 is the server-side
container** — it already stores time series, Fourier coefficients, and
transfer functions with full metadata:

- The server maintains a workspace directory (e.g. `~/.emptyui/workspace/`).
  Every intermediate result a node chooses to persist is written as an
  HDF5/Zarr group inside a workspace MTH5-compatible tree; handles reference
  these paths.
- Nothing reaches the user's disk outside the workspace until they
  explicitly export/save-as (`SaveEDI`, `SaveMTH5`, download artifact…).
- Benefits: free interoperability with the existing CLI ecosystem (any mth5/
  aurora script can read what emptyUI produced), no migration burden, and
  remote deployments are just "a directory on the server".

---

## 4. Node Manifest Schema

Each node type exposes a JSON manifest via the registry. The properties panel
and sidebar are generated entirely from it.

```jsonc
{
  "type": "core.filter.bandpass",          // namespaced, unique
  "version": "1.0.0",                       // semver of the node definition
  "title": "Bandpass Filter",
  "description": "Butterworth bandpass on time series.",
  "category": "Processing",
  "plugin": "emptyui-core-nodes",
  "inputs": [
    { "name": "data", "type": "ts.run", "required": true }
  ],
  "outputs": [
    { "name": "filtered", "type": "ts.run" }
  ],
  "params": [
    {
      "name": "low_freq",
      "label": "Low cutoff (Hz)",
      "widget": "number",
      "default": 0.001,
      "min": 0, "step": 0.0001,
      "optional": true
    },
    {
      "name": "high_freq",
      "label": "High cutoff (Hz)",
      "widget": "number",
      "default": null,
      "optional": true
    },
    {
      "name": "order",
      "label": "Order",
      "widget": "integer",
      "default": 4, "min": 1, "max": 10
    },
    {
      "name": "detrend",
      "label": "Detrend first",
      "widget": "boolean",
      "default": true
    },
    {
      "name": "method",
      "widget": "select",
      "options": ["butter", "cheby1"],
      "default": "butter"
    }
  ],
  "deps": ["scipy>=1.11"]                   // informational; enforced by plugin extras
}
```

Widget vocabulary (`number`, `integer`, `string`, `text`, `boolean`, `select`,
`file`, `directory`, `code`, `list`) is fixed in core; plugins choose from it so
the UI stays consistent without frontend changes.

### Server-side definition

```python
# emptyui_core/api.py (illustrative)
from emptyui import Node, Input, Output, Param

class Bandpass(Node):
    type = "core.filter.bandpass"
    title = "Bandpass Filter"
    inputs  = [Input("data", "ts.run")]
    outputs = [Output("filtered", "ts.run")]
    params = [
        Param("low_freq",  "number",  default=None, optional=True),
        Param("high_freq", "number",  default=None, optional=True),
        Param("order",     "integer", default=4, min=1, max=10),
        Param("detrend",   "boolean", default=True),
    ]

    def execute(self, ctx, data, low_freq=None, high_freq=None,
                order=4, detrend=True):
        ...
        return {"filtered": out}
```

`ctx` gives nodes access to: workspace dir, logger, artifact publishing,
progress reporting, and the shared data store.

---

## 5. Plugin System

Discovery uses Python **entry points** (importlib.metadata), the standard
mechanism — no config files, no core edits:

```toml
# pyproject.toml of a plugin package
[project.entry-points."emptyui.nodes"]
my_nodes = "emptyui_mth5.plugin:register"
```

```python
def register(registry):
    registry.add(LoadMTH5)
    registry.add(ComputeFC)
    registry.add(EstimateTF)
```

Rules that keep the ecosystem healthy:
- Core ships with **only** the framework + one `emptyui-core-nodes` plugin
  (numpy/matplotlib-level deps).
- Each plugin declares its own heavy deps as pip extras
  (`pip install emptyui-mth5[full]`), so users install only what they need.
- Node types are namespaced by convention: `core.<domain>.<name>` for the
  bundled plugin, `<org>.<domain>.<name>` for external plugins
  (e.g. `simpeg.aurora.compute_fc`).
- Manifests include `version`; workflows record which plugin versions produced
  results for reproducibility.
- A future `emptyui plugin publish/index` flow can distribute community
  plugins as ordinary PyPI packages — no custom registry required for MVP.

---

## 6. Workflow JSON Schema

One file = one reproducible pipeline.

```jsonc
{
  "format": "emptyui/workflow",
  "version": 1,
  "meta": { "name": "kilauea-demo", "created": "...", "author": "..." },
  "nodes": [
    {
      "id": "n1",
      "type": "core.io.load_csv",
      "params": { "path": "/data/station01.csv" },
      "position": { "x": 80, "y": 120 }         // UI-only, ignored by executor
    },
    {
      "id": "n2",
      "type": "core.filter.bandpass",
      "params": { "order": 4 }
    },
    {
      "id": "n3",
      "type": "core.plot.spectrogram",
      "params": {}
    }
  ],
  "edges": [
    { "id": "e1", "source": "n1", "source_port": "data",
      "target": "n2", "target_port": "data" },
    { "id": "e2", "source": "n2", "source_port": "filtered",
      "target": "n3", "target_port": "data" }
  ]
}
```

Design notes:
- Edges reference ports by name → resilient to port reordering.
- `position` is UI-only → headless/batch execution ignores it cleanly.
- Executor validates against current manifests at load time and reports
  missing/incompatible node types with actionable messages.

---

## 7. Execution Model

- **Scheduler**: Kahn topological sort; independent branches run concurrently
  in a thread pool (nodes declare `cpu_bound=True/False` to hint threads vs
  process pool later).
- **Caching**: each node result is content-addressed by `(node_type, version,
  params, input_handles)`; unchanged upstream ⇒ cached downstream skip.
  MVP: in-memory LRU + optional disk cache.
- **Progress**: nodes call `ctx.progress(frac, message)`; executor forwards to
  all connected WebSocket clients.
- **Failure**: a failed node halts its dependents only; other branches continue.
  Errors carry structured info (message, traceback excerpt, node id).
- **Partial runs**: user can run a single node ("execute to here"), enabled by
  the caching scheme.
- **Cancellation**: client sends a cancel message; executor checks a token
  between node steps.

### Run lifecycle events (WebSocket)

```
{ "event": "run_started",   "run_id": "r1" }
{ "event": "node_status",   "node": "n2", "status": "running" }
{ "event": "node_progress", "node": "n2", "frac": 0.42, "msg": "filtering…" }
{ "event": "node_log",      "node": "n2", "level": "info", "line": "…" }
{ "event": "artifact",      "node": "n3", "id": "a7", "kind": "image",
  "handle": "h12",
  "preview_url": "/api/artifacts/a7/preview" }
{ "event": "node_status",   "node": "n3", "status": "done" }
{ "event": "run_finished",  "run_id": "r1", "status": "completed" }
```

The `artifact` event carries the output `handle` id so viewer nodes
(e.g. Preview Plot) can resolve exactly which figure arrives on which input
edge, even when one node publishes several figures.

---

## 8. Backend API (REST)

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/nodes` | Catalog of node manifests (drives sidebar + panels) |
| POST | `/api/runs` | Validate + start a workflow run |
| POST | `/api/runs/{id}/cancel` | Cancel a run |
| GET  | `/api/runs/{id}` | Run status/history |
| GET  | `/api/artifacts/{id}` | Download artifact |
| GET  | `/api/artifacts/{id}/preview` | Inline preview (PNG/JSON table) |
| WS   | `/api/ws` | Run events stream |

MVP keeps state on the file-based workspace (§3 Storage); a DB is added only
if run history/UI state demands it.

---

## 9. Frontend Design

Stack: **Vite · React · TypeScript · @xyflow/react · Zustand · TanStack Query**

Components:
- **GraphCanvas** — React Flow; maps workflow JSON ⇄ RF nodes/edges bidirectionally.
- **NodePalette** — grouped by `category`, populated from `GET /api/nodes`.
- **Inline node params (default behavior, ComfyUI paradigm)** — a node's
  parameters render as widgets directly on the node body; the right-hand
  PropertiesPanel keeps only metadata (node id, type, version, ports,
  description) and node-level actions. Path-like widgets get a browse button
  backed by the server-side file browser (`GET /api/files`).
- **Preview Plot node** — renders an incoming `plot` inside the node body;
  sized by the standard node drag-resize. Multiple viewers can attach to the
  same figure.
- **ResultPanel** — run-level gallery of all artifacts (convenience; the
  Preview Plot nodes are the primary per-figure viewers).
- **Toolbar** — new/open/save workflow, run, cancel, export.

State split:
- Zustand store: workflow document (single source), selection, connection drafts.
- TanStack Query: server resources (node catalog, artifacts).
- WebSocket client: ephemeral run state overlaid onto node statuses.

Connection rules (port types, cycle prevention) enforced client-side for UX,
re-validated server-side before every run.

---

## 10. Repository Layout (monorepo)

```
emptyui/
├── README.md
├── DESIGN.md
├── packages/
│   ├── emptyui/                 # Python core (framework only)
│   │   ├── src/emptyui/
│   │   │   ├── api.py           # Node/Input/Output/Param declarations
│   │   │   ├── registry.py      # entry-point discovery + manifest export
│   │   │   ├── executor.py      # DAG scheduler, caching, cancellation
│   │   │   ├── context.py       # ctx: workspace, logging, progress, artifacts
│   │   │   ├── store.py         # handle-based data store
│   │   │   ├── schema.py        # workflow JSON validation (pydantic)
│   │   │   └── server/          # FastAPI app, routes, WS hub
│   │   └── tests/
│   ├── emptyui-core-nodes/      # bundled minimal plugin (numpy/matplotlib)
│   │   └── src/emptyui_core_nodes/
│   └── web/                     # React frontend
│       └── src/
│           ├── graph/           # canvas, node rendering, edges
│           ├── panels/          # palette, properties, results
│           ├── api/             # REST/WS clients, generated types
│           └── stores/
├── examples/                    # sample workflows + tiny sample datasets
├── docs/
└── scripts/                     # dev up/down, codegen
```

Dependency policy:
- `emptyui` core deps: `fastapi`, `uvicorn`, `pydantic`, `numpy` only.
- `emptyui-core-nodes`: adds `matplotlib` (+ optional `pandas`, `scipy`).
- Everything else (mth5, aurora, mt_metadata, scikit-learn, …): plugin extras.
- Frontend pinned via lockfile; API types generated from OpenAPI spec
  (`scripts/codegen`) to keep the contract honest.

---

## 11. MVP Roadmap

**M0 — Skeleton (week 1)**
- Monorepo scaffold, CI (lint + tests both sides).
- Core: Node API, registry, workflow schema validation, trivial echo node.
- Web: Vite+React+React Flow shell, palette fed by mock then real `/api/nodes`.

**M1 — First real pipeline (weeks 2–3)** ✅ MVP target
- Nodes: `core.io.load_csv` (timeseries) → `core.filter.bandpass` →
  `core.plot.timeseries` / `core.plot.spectrogram`.
- Executor: topo run, WS progress, artifacts, error surfaces.
- Properties panel generation from manifests; workflow save/load JSON.
- Demo workflow running end-to-end on a sample dataset.

Note: §7's caching/partial-execution/cancellation are *designed in* from M0
(handle IDs, cache keys) but fully exercised in M3; M1 ships basic run + WS
progress.

**M2 — MT ecosystem integration**
- `emptyui-mth5` plugin: Load/SaveMTH5, ExtractRun/SplitChannels/MergeChannels,
  transfer-function I/O.
- `emptyui-aurora` plugin: `SelectRuns` / `ComputeFC` / `EstimateTF` /
  `FrequencyBands` nodes; per-level `tf.level` merge node;
  TF plots (apparent resistivity/phase, phase tensor).
- Table viewer component; export coercion nodes for popular formats
  (EDI/ZMM via mt_metadata).

**M3 — Power features**
- Caching + partial execution; sub-workflows; remote-server mode docs
  (same API, deployed elsewhere); plugin packaging guide + example template.

**M4 — Community**
- Plugin index/publish flow, docs site, more visualization nodes,
  batch/queue runner.

---

## 12. Open Questions

- FC→TS inverse node: deferred; plugin coercion node when needed
  (e.g. frequency-domain denoising demos).
- How do ML plugins request GPU availability? (`ctx.resources.gpu_count`?)
- Sub-workflow semantics (nested DAGs) — defer until after M1.
- Authentication model for remote deployments (defer; design API so middleware
  can be inserted).
