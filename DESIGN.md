# emptyUI — Design Document

A node-based, ComfyUI-inspired visual workflow interface for magnetotelluric (MT)
data processing and visualization.

Status: v0.2 — design frozen for M0/M1 (2026-08-25).
Changes after v0.1: aurora-informed port type system, node categories,
TS→FC→TF stage split, MTH5-backed file storage, backend-decided port colors
and ComfyUI-style port rendering conventions (§3).
Implementation status: **M0 complete; M1 complete; M2 in progress** — mth5/
aurora plugins scaffolded; ComputeFC (TS→FC windowed STFT) and EstimateTF
(FC→TF cross-power LSQ) implemented and verified end-to-end through the
executor on real 1 Hz LEMI-424 and BGS long-period data (Load LEMI → Detrend
→ ComputeFC → EstimateTF → Export EDI / Plot AppRes&Phase).
Processing correctness milestones:
- **Unit normalization**: magnetic channels converted nT→T→H (B→H) and
  electric mV/km→V/m; missing nT→T factor was the root cause of non-physical
  apparent resistivity (now ~10–10³ Ω·m).
- **Impedance phase convention**: verified against aurora on the official
  synthetic halfspace (`test12rr`); all three estimators (OLS/RME/RME_RR) are
  aurora-backed and share aurora's convention (Zxy≈+45°, Zyx≈−135°).
- Detrending moved to the dedicated `core.filter.detrend` node (mean/linear/
  none); `e_dipole_length_m` optional (off by default — LEMI output is mV/km).
Estimator: **robust RME/RME_RR implemented** (aurora-backed) with remote
reference support; multi-level decimation/FC persistence and Calibrate are
follow-ups. See §13 for the current plan, priorities and divide-and-conquer
tracks.

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

| Tag | Display name | Real library class | Stage / notes |
|-----|--------------|--------------------|---------------|
| `mth5` | MTH5 Archive | `mth5.MTH5` | HDF5 archive; stateful source of runs/FCs/TFs. *(was `mt.archive`)* |
| `ts.run` | Run Time Series | `mth5.timeseries.RunTS` (= `mt_timeseries.RunTS`) | Multi-channel time series — the main signal-processing currency; wrap the `RunTS` object directly (emptyUI is mth5-native) |
| `ts.channel` | Channel Time Series | `mth5.timeseries.ChannelTS` | Single channel (ex/ey/hx/hy/hz); split/merge nodes convert ⇄ `ts.run` |
| `calibration` | Calibration / Instrument Response | `mt_metadata.filters.*` (Coefficient/PoleZero/FIR/TimeDelay/FAP) | `Calibrate` takes `ts.run` + `calibration` → calibrated `ts.run`. Named for end-user familiarity *(was `mt.response`)* |
| `selection` | Run Selection | `mth5.processing.KernelDataset` (DataFrame) | Which stations/runs/intervals process; local+remote roles; supports multiple run rows → FC-level merge. *Plugin-tier / M4-deferred — no producer in the default pipeline (superseded by ExtractRun → ComputeFC)* |
| `config` | Processing Config | `mt_metadata.processing.aurora.Processing` | The aurora "recipe". *Plugin-tier / M4-deferred — the TS→FC→TF pipeline splits at the FC boundary and assembles config internally (no live producer)* |
| `bands` | Frequency Bands | `aurora.time_series.frequency_band.FrequencyBands` | Band edges. *Plugin-tier / M4-deferred — band edges are derived inside EstimateTF (`_period_bands`); EMTF band-setup files are an M4 item* |
| `decimation` | Decimation Level | aurora `DecimationLevel` | *Plugin-tier*: per-level recipe for advanced multi-config assembly *(was `mt.decimation`)* |
| `fc` | Fourier Coefficients | mth5 `FCGroup`/`FCDecimationGroup`; aurora STFT | The frequency-domain currency. TS→FC natural; FC→TS plugin-tier (lossy inverse). *(was `spectra` — no formal "spectra" type exists; libraries call this FC)* |
| `tf.level` | TF (decimation level) | aurora `TTFZ` | Per-decimation-level TF estimate |
| `tf` | Transfer Function | `mt_metadata.transfer_functions.TF` (also mth5 `TransferFunctionGroup`) | Final merged TF; exporters & analysis plots consume this *(was `mt.tf`)* |
| `table` | Table | pandas DataFrame | Generic tabular output (run summaries, coherency, etc.) |
| `plot` | Plot | rendered-figure ref (PNG/SVG/PDF + metadata) | Produced by Display nodes; consumed by `Preview Plot`/`Save Plot` |
| `any` | Any | — | Wildcard; connects to anything, executor trusts the node |

> `file` was dropped as a port type (paths are parameter widgets, not data
> flows). `table` covers generic tabular data. `config` / `bands` / `decimation`
> / `selection` are plugin-tier and M4-deferred: the default pipeline splits at
> the FC boundary (ComputeFC/EstimateTF) and derives band edges internally, so
> these tags currently have no live core producer.
>
> The right-hand panel always shows both the **display name** and the real
> **type tag** for every port (e.g. `data: ts.run`), so users can trace the
> underlying library container.

### Port-type audit notes (kept in sync with real MT processing)

- **Missing / considered**: derived quantities (apparent resistivity, phase,
  impedance) are *computed* from `tf` by Display nodes, not separate ports —
  keeping them as plot outputs avoids type sprawl. Remote-reference selection
  lives in `selection` (remote column), not its own type. Calibrated vs.
  uncalibrated `ts.run` is currently one type distinguished by metadata; a
  `ts.run.calibrated` subtype is a possible future refinement.
- **Consistency rule**: every port tag maps 1:1 to a real library container
  (class name above); we avoid inventing new names where the ecosystem already
  has one (`fc` not `spectra`, `mth5` not `archive`).

### M2 ecosystem pins (stable releases, audited 2026-08)
- `mth5==0.6.8`, `aurora==0.6.2`, `mt_metadata==1.0.10`, `mt_timeseries==0.0.2`
  (the `mt_timeseries` package now hosts `RunTS`/`ChannelTS`; mth5 re-exports them
  from `mth5.timeseries`).
- `KernelDataset`/`RunSummary` live in **`mth5.processing`** (legacy aurora
  `read_run_summaries`/`get_selected_runs`/`make_kernel_dataset` are removed).
- Decimation levels are **independent by default** (fresh `Window`/`Estimator` per
  level, no shared reference) → `ComputeFC`/`EstimateTF` keep a constant window
  in samples across levels and share estimator settings (they split the old
  monolithic `BuildConfig` assembly at the FC boundary).
- `channel_nomenclature` lives on `Processing` config, not `RunTS`.
- These pins are plugin-level extras (`emptyui-mth5`, `emptyui-aurora`), never
  core dependencies; workflows record plugin versions for reproducibility.

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
  - **dark green** — frequency domain: `fc` / Fourier coefficients
  - **red** — transfer functions: `tf`, `tf.level`
  - **purple** — metadata/config: `selection`, `config`,
    `calibration`, `bands`, `decimation`
  - gray/yellow — generic (`any`, `table`, `plot`)
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
LoadMTH5 ─mth5─▶ ExtractRun ─ts.run─▶ ComputeFC ─fc─▶ EstimateTF ─tf─┬▶ ExportEDI
(LoadLemi / LoadCsv)                        │                        └▶ PlotAppResPhase
                                             └─fc─▶ Denoise ─fc─▶ EstimateTF
                                                                   └─tf.level─▶ MergeTFLevels

LoadMTH5 ─mth5─▶ ExtractRun ─ts.run─▶ Detrend ─ts.run─▶ ComputeFC ─fc─▶ PlotSpectrogram
```

Coercion points that require dedicated nodes (mirroring aurora internals):
1. `ts.run` ⇄ `ts.channel` via explicit SplitChannels / MergeChannels nodes.
2. uncalibrated `ts.run` + `calibration` → calibrated `ts.run` (units change).
3. `tf.level[]` → `tf` (merge; resolves duplicate periods across sampling
   rates / decimation levels — the multi-level merge case).
4. Format coercions for popular types ship in core (`SaveEDI`, `LoadEDI`,
   …); exotic import/export converters are plugins.

### Pipeline stage split: TS → FC → TF

The aurora processing run is **not** one monolithic node. It is split at the
Fourier-coefficient boundary, mirroring both the conceptual MT pipeline
(decimation → windowing → time/frequency transform → power spectra → TF
estimation) and aurora v0.3.x's own ability to persist and reuse FCs:

- **`ComputeFC` node (TS → FC)** — input `ts.run`; params: input/output
  channel names, magnetic unit (h/b), decimation levels + factor, window
  length/overlap/type. Output `fc`. Internally expands uniform decimation
  levels sequentially (level sample rates derived from factor × initial rate).
  Persists computed FCs into the workspace MTH5's FC layer.
- **`EstimateTF` node (FC → TF)** — inputs `fc` + optional `ref_fc` (remote
  reference); params: estimator (`Robust`/`Least Squares`), remote-reference
  on/off, periods-per-octave. Band edges are derived internally
  (`_period_bands`). Output `tf` + per-level `tf.level`.

Why this boundary:
1. **Frequency-domain work**: a first-class `fc` port lets users branch
   into denoising, spectral analysis, or custom estimators before TF
   estimation.
2. **Cheap re-runs**: STFT/decimation is the expensive stage. Estimator
   parameters live *only* on `EstimateTF`, so with content-addressed caching
   changing the estimator invalidates just that node — saved/cached FCs are
   reused untouched.
3. **No parameter wall**: each node carries only its stage's parameters;
   aurora's monolithic `Processing` object is assembled internally by the two
   nodes sharing context, never shown to the user as one form.

The full-config passthrough (`config` port) remains available at plugin
tier for advanced cases (e.g. per-level customization via a `decimation`
fan-in assembler).

### Merging multiple runs into one TF estimate

A single TF estimate is often built from **several acquisition runs** (e.g. a
Phoenix MTU-5 writes many segments). aurora supports this natively by allowing
multiple rows in the `KernelDataset` and **concatenating their STFTs at the FC
level** (`merge_stfts` → `xr.concat` over time); it does *not* merge in the
time domain (source TODO: `ANY MERGING OF RUNS IN TIME DOMAIN WOULD GO HERE`).

emptyUI therefore offers a **`Merge Runs` node** with two output modes:
1. **`selection` (FC-level, default)** — keeps multiple run rows in a
   `KernelDataset`; aurora concatenates the FCs. No fabricated samples; gaps
   handled at the STFT/band level. This is the idiomatic path.
2. **`ts.run` (physical)** — combines raw series via `mt_timeseries.RunTS.merge()`
   and persists into a new mth5 run group (`RunGroup.from_runts`). Explicitly
   user-opt-in because it **interpolates across gaps** (gap policy is a param:
   interpolate vs. leave NaN).

Constraints enforced by the node: all runs must share the same native sample
rate (aurora hard-errors on mixed rates), and channel components must be
consistent across runs. Remote-reference multi-row intersection is an open
aurora TODO — the node should warn when a local run spans multiple remote runs.

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
`radio`, `file`, `directory`, `code`, `list`) is fixed in core; plugins choose
from it so the UI stays consistent without frontend changes. The frontend has
dedicated controls for `boolean`/`select`/`radio`/`number`/`integer`/
`file`/`directory`; the remaining string-like widgets (`string`/`text`/`code`/
`list`) fall back to a single-line text input.

### Param auto-change (`cascade`) & upstream auto-fill (`inherit`)

Two manifest-driven UI conveniences keep the frontend schema-agnostic:

- **`cascade`** (per param): when the user changes this param to a given value
  (key), the UI sets the listed `(other_param, value)` pairs. Keys serialize as
  strings for JSON (`bool` → `"true"`/`"false"`). Used, e.g., for `Compute FC`'s
  magnetic input `h ⇄ b` to swap channel names, and for `Estimate TF`'s
  remote-reference to force the Robust estimator:

  ```python
  Param("magnetic_unit", "radio", default="h", options=("h", "b"),
        cascade={
            "h": (("input_channels", "hx,hy"), ("output_channels", "ex,ey")),
            "b": (("input_channels", "bx,by"), ("output_channels", "e1,e2")),
        }),
  Param("remote_reference", "boolean", default=False,
        cascade={True: (("estimator_engine", "Robust"),)}),
  ```

- **`inherit`** (per node): `{input_port: {target_param: upstream_param}}`. When
  an upstream node is connected to that port, those params render as dropdowns
  populated from the upstream node's params and are auto-filled if empty.
  Same-name inheritance is just an identity mapping. Used for `Compute FC` to
  pull `survey`/`station`/`run` from an `Extract Run`, and for mth5-sourced
  nodes to map e.g. `survey ← survey_id` from `Load LEMI-424`:

  ```python
  inherit: ClassVar[dict[str, dict[str, str]]] = {
      "data": {"survey": "survey", "station": "station", "run": "run"},
  }
  # Extract Run pulling from Load LEMI-424 (name mapping):
  inherit: ClassVar[dict[str, dict[str, str]]] = {
      "mth5": {"survey": "survey_id", "station": "station_id"},
  }
  ```

Both serialize into the node manifest; the frontend applies them generically.

#### Chain-walking inheritance (guideline)

Inheritance is resolved by a **single recursive, frontend-only resolver**
(`resolve(nodeId, paramName) -> candidate values`), walking the workflow
document (nodes + edges + params) — never the backend. The backend derives the
same identity at run time from the actual payload metadata
(`runts_context`/`coerce_run`), so this is purely a **pre-execution UI** feature
(dropdown population + auto-fill) and correctness never depends on it.

Rules:

1. **Per-node declaration.** Each node declares only its own
   `inherit: {input_port: {target_param: upstream_param}}`. There is no global
   registry of identities.
2. **Only declared params auto-fill.** A param is auto-filled / rendered as a
   dropdown **only if** it appears as a target in the node's own `inherit`
   mapping. Everything else — root-node defaults, radio/select config (e.g.
   `Detrend` method, `Compute FC` magnetic unit, `Estimate TF` estimator) —
   stays fully manual. This keeps the "set once at the source, inherit
   downstream" intent: a user types the identity at the first node, and every
   declared downstream param is then filled from it.
3. **Recursive chain-walk.** `resolve` returns a node's param value if it is
   non-empty; otherwise it follows the node's own `inherit` mapping for that
   port, follows the edges into the port, and recurses on each source. The walk
   is transitive across arbitrary processing nodes even when an intermediate
   node declares nothing for the param (pass-through fallback) — e.g.
   `Compute FC → Detrend → Extract Run → Load LEMI-424` still resolves
   `survey`, so inserting a future time-domain node (de-noising, filtering)
   between the source and `Compute FC` never loses the identity.
4. **Name mapping.** Each hop uses the *current* node's mapping to decide which
   upstream param to look for, so heterogeneous sources (e.g. `survey_id`) are
   bridged without changing the downstream node.
5. **Cycles & fan-in.** Use a visited set (the graph is a DAG but be defensive)
   and union candidates across parallel sources feeding a port.
6. **Same-name default.** Prefer the identity mapping; only declare a mapping
   when names actually differ.

Adding a new node type therefore needs **zero changes to the resolver** — just a
one-line `inherit` declaration on the node itself, mirroring how `cascade`
auto-change and the widget vocabulary keep the UI schema-agnostic.

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

- **Scheduler**: Kahn topological sort. The current executor runs branches
  **sequentially** (M0); concurrent execution in a thread pool is future work
  (nodes would declare `cpu_bound=True/False` to hint threads vs process pool).
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
{ "event": "artifact",      "node": "n3", "id": "a7", "kind": "image",
  "handle": "h12",
  "url": "/api/artifacts/a7/preview" }
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
| GET  | `/api/runs/{id}` | Run status/outputs |
| GET  | `/api/artifacts/{id}/preview` | Inline preview (PNG/JSON table) |
| WS   | `/api/ws` | Run events stream |

> `POST /api/runs/{id}/cancel` and `GET /api/artifacts/{id}` are not yet
> implemented (cancellation + artifact download are future work).

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
  PropertiesPanel shows metadata (node id, type, version, ports, description,
  deps) plus a **Delete node** action. Path-like widgets get a browse button
  backed by the server-side file browser (`GET /api/files`).
- **Preview Plot / View Table nodes** — render an incoming `plot`/`table`
  inside the node body, sized by the standard node drag-resize. Multiple
  viewers can attach to the same figure/table.
- **Toolbar** — run; new/open/save workflow. (Cancel and export are future
  work; run-level artifact gallery is not implemented — viewers on nodes are
  the per-figure/table display.)

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
- `emptyui-mth5` plugin (`mth5==0.6.8`): Load/SaveMTH5, ExtractRun /
  SplitChannels / MergeChannels, transfer-function I/O, `Merge Runs`
  (FC-level + physical modes), FC read/write.
- `emptyui-aurora` plugin (`aurora==0.6.2`): `ComputeFC` / `EstimateTF` /
  `MergeTFLevels` nodes (FC-boundary split; estimators OLS / robust RME /
  remote-reference RME_RR); TF plots (apparent resistivity/phase).
- Table viewer component; export coercion nodes for popular formats
  (EDI/ZMM via mt_metadata).

**M3 — Power features**
- Caching + partial execution; sub-workflows; remote-server mode docs
  (same API, deployed elsewhere); plugin packaging guide + example template.
- **Dynamic "select from" options** (deferred from M2): selection-style nodes
  (`Select Runs`, `Extract Run`, `Split Channels`) auto-populate dropdowns by
  inspecting the connected upstream source (e.g. open the mth5 read-only and
  list surveys/stations/runs). Design: manifest params gain an
  `options_from: { input, attr }` field; new `POST /api/options` resolves the
  upstream source and returns candidates; frontend renders a populated
  `<select>` refreshed on connection changes. Nice-to-have UX, not critical.

**M4 — Community**
- Plugin index/publish flow, docs site, more visualization nodes,
  batch/queue runner.
- **Band config files**: load an aurora-style band config (exact centre
  frequencies + bandwidths) or auto-default when none given, replacing/augmenting
  the auto-derived Fs/4→window/4 period table (§14 deferred task).
- **WORTH EXPLORING — Paradigm B: cross-decimation-level FC pooling**
  (§14.5). At long periods the deepest decimation levels have few STFT windows,
  so per-level TF estimates are statistically weak. Instead of estimating a TF
  per level and then merging overlapping periods (current Paradigm A), pool the
  **raw Fourier coefficients across all decimation levels** whose frequencies
  fall inside a fixed band, and run one robust regression per band. Deep levels
  have finer raw frequency resolution (more harmonics per Hz), so pooling
  recovers more samples per band where they're scarce (long periods) while
  remaining cheap at high frequencies where samples already abound. Existing
  codes (EMTF, BIRRP, aurora) define bands this way but process each level's
  bands *separately* — cross-level pooling is a novel extension.

---

## 12. Open Questions

- FC→TS inverse node: deferred; plugin coercion node when needed
  (e.g. frequency-domain denoising demos).
- How do ML plugins request GPU availability? (`ctx.resources.gpu_count`?)
- Sub-workflow semantics (nested DAGs) — defer until after M1.
- Authentication model for remote deployments (defer; design API so middleware
  can be inserted).

---

## 13. Upcoming Plan — Remaining M2 (2026-08-29)

Status: M2 is in progress; ComputeFC/EstimateTF and the EDI/XML export coercion
are verified end-to-end. This section records the agreed priority order and the
divide-and-conquer tracks for finishing M2. It supersedes the one-line "next
milestone" note above.

### Remaining M2 deliverables

Inventory against §11. ~~Struck~~ items are delivered; the gaps:

| # | Deliverable | Where | Status |
|---|-------------|-------|--------|
| 1 | Robust **RME/RME_RR** estimator | `_processing.estimate_tf` + `process.py` | ✅ **done** — OLS/RME/RME_RR all aurora-backed; `remote_reference` supported (ref `fc` or self-reference/dummy) |
| 2 | **Multi-level decimation + FC persistence** | `compute_fc_levels` + `persist.py` | ✅ **done** — cascading decimation, per-level FC persisted to workspace MTH5 (§14) |
| 3 | **MergeTFLevels** (`tf.level` → `tf`) | `merge_tf_levels` + `process.py` | ✅ **done** — per-level estimate + deepest-wins merge of overlapping octave periods (§14) |
| 4 | **Table viewer** | `mth5.table.selection`/`mth5.table.tf` + `core.display.view_table` + frontend | ✅ **done** — JSON table artifacts rendered inline |
| 5 | **Calibrate** node | `mth5.calibrate` + `calibrate.py` | ✅ **done** — `ts.run` (+ optional `calibration`) → calibrated `ts.run`; uses mth5 `RunTS.calibrate()` channel responses, or frequency-domain response removal for an explicit filter |

(Export coercion EDI/ZMM/XML/AVG/J is already delivered via `ImportTF`/`ExportTF`.)

### Priority

- **P1 — Robust RME/RME_RR estimator.** ✅ delivered (2026-08-29). Highest
  scientific value: real MT data is outlier-contaminated and LSQ is unreliable.
  **Wraps aurora estimators**: `OLS` via `RegressionEstimator.estimate_ols`,
  robust `RME`/`RME_RR` via their aurora classes (note the regression module does
  *not* re-export at package level, import the submodules directly). `remote_reference` is supported: a second station's
  `fc` on the new `ref_fc` input supplies the reference magnetic channels; when
  no `ref_fc` is connected the local channels are reused as a self-reference
  (what a duplicated/dummy remote station reduces to). All three estimators
  verified to match aurora's phase convention on `test12rr` (Zxy≈+45°, Zyx≈−135°).

  **Estimator UI (2026-08-29)**: the `EstimateTF` node exposes **two** estimator
  options — **`Least Squares`** and **`Robust`** — plus the separate
  `Remote Reference` checkbox. `Remote Reference` only applies to `Robust`
  (Robust+off → `RME`; Robust+on → `RME_RR`); it is disabled in the UI when
  `Least Squares` is selected. Legacy saved-workflow values (`OLS`/`RME`/`RME_RR`)
  are still accepted on load.

  **No LS+RR, by design**: there is intentionally **no least-squares +
  remote-reference** estimator. aurora offers remote reference only through the
  robust `RME_RR` (a plain-LS RR variant would add little statistical value —
  remote reference exists to suppress incoherent noise, where robust weighting
  matters — and would multiply estimator variants). Requesting `Least Squares`
  with `Remote Reference` raises a clear error rather than silently degrading.
  Should LS+RR ever be wanted, the `_solve_band` dispatch makes it a small,
  local addition — but we keep the two-option surface to match aurora's design.
- **P1 — Multi-level decimation + FC persistence + MergeTFLevels.** Second value:
  without it we process a single sampling rate and cannot produce broadband
  apparent-resistivity/phase curves spanning decades. **Largest, riskiest** chunk
  (data model + mth5 persistence + per-level estimator + merge node). Sequenced
  *after* the estimator; it reuses the per-level robust estimator.
- **P2 — Table viewer.** ✅ delivered (2026-08-29). Low effort, low risk, high UX:
  selections (`KernelDataset`) and run summaries are now visible. `mth5.table.selection`
  emits a JSON table artifact; `core.display.view_table` renders it inline.
- **P2 — Calibrate node.** Medium value (coil-sensor / instrument response),
  medium effort (mth5/mt_metadata filter parsing, response removal). Independent.

### Divide-and-conquer tracks

```
Track A (backend, sci)   Robust RME/RME_RR  →  EstimateTF honors estimator_engine (+ remote_reference)   ✅
Track B (backend, arch)  Multi-level ComputeFC + FC persistence + MergeTFLevels   (largest)
Track C (frontend)       Table viewer + "View Selection" Display node            ✅
Track D (backend, I/O)   Calibrate node + calibration port wiring
```

- **A and C are fully independent** → run concurrently immediately.
- **A lands before B**: B's per-level `tf.level` estimation reuses A's robust
  estimator; B can be specced in parallel while A is underway.
- **D is independent** of A/B/C.
- **Cross-cutting enabler:** make the MT integration tests runnable in CI by
  *generating* the synthetic `test12rr` fixture (as `test_phase_convention.py`
  already scripts) instead of relying on gitignored raw LEMI/BGS data — so every
  track has a real green test target regardless of fixtures on disk.

### Sequencing & landing

1. **A → C** (fast, low-risk wins, verifiable immediately).
2. **B** (biggest, builds on A's per-level estimator).
3. **D** (independent; slot in when A/C land).
4. Land each track as its own PR/branch so the work can be divided in parallel.

---

## 14. Current Plan — M2: Multi-level Decimation + FC Persistence (2026-08-29)

Status: the biggest remaining M2 chunk (Track B). This is the plan for
broadband MT — processing more than one decimation level so the apparent
resistivity/phase curves span decades. It is grounded in reading the pinned
aurora 0.6.2 / mth5 0.6.8 source (§14.1).

### 14.1 What aurora actually does (grounding)

- **Cascading decimation** (`aurora/pipelines/time_series_helpers.py::prototype_decimate`):
  each decimation level *i* takes the *previous* level's time series and decimates it by
  that level's integer `factor` with `scipy.signal.decimate`. Level 0 is the native
  sample rate; level *i*'s rate = native / ∏(factors 1..i).
- **Per-level STFT** (`aurora/time_series/windowing_scheme.py`, `spectrogram_helpers.py::run_ts_to_stft`):
  from the level's `Window` (`num_samples`, `overlap` in samples, taper family) and `sample_rate`,
  aurora does: prewhiten → truncate to clock-zero → sliding window → linear detrend →
  taper → FFT → recolor. Window `num_samples` is conventionally kept **constant across levels**
  (so window *duration* in seconds grows at deeper levels → better long-period resolution).
- **FC generation vs. use** (`spectrogram_helpers.py::get_spectrograms`, `make_stft_objects`,
  `save_fourier_coefficients`): per (run, decimation-level) pair aurora either *loads* an
  already-persisted FC from mth5 (`load_stft_obj_from_mth5`, when the row has `fc`) or *computes*
  it, then optionally persists it (`save_fourier_coefficients`) into the archive FC layer.
- **Merging runs at the FC level** (`merge_stfts`): concatenates per-run STFTs along time
  (`xr.concat(stfts, "time")`) — this is the *across-runs* FC merge, not the across-levels one.
- **Per-level TF + level merge** (`process_mth5.py`, `transfer_function_kernel.py`):
  each decimation level is estimated independently to a per-level `TTFZ`; a
  `TransferFunctionCollection` + `export_tf_collection` merges the per-level TFs into one
  mt_metadata TF, resolving duplicate periods across levels. This is what emptyUI calls
  `MergeTFLevels`.
- **FC persistence format** (`mth5/groups/fourier_coefficients.py`): per station,
  `fourier_coefficients_group → add_fc_group(run_id) → add_decimation_level(f"{i}") →
  from_xarray(fc, sample_rate_decimation_level)`. **Verified: `from_xarray` requires a
  `datetime64` time coordinate** (float seconds fails) — so persisted FCs must carry the run's
  absolute start time.

### 14.2 Design decisions (with recommendations)

**D1 — FC time axis.** To persist to mth5, the fc `time` coordinate must be `datetime64`.
The `ts.run` payload must therefore carry an absolute start time (mth5 `RunTS.starttime`;
core `TimeSeries` currently has a float `start` seconds — needs a start timestamp too).
Recommendation: normalize a `start_datetime` onto `ts.run`/fc so persistence round-trips.

**D2 — Window-length semantics.** Follow aurora: express the window as **`num_samples`
constant across levels** (window duration in seconds grows at deeper levels), rather than
the current seconds-based `window_length_s` (which would *shrink* spectral resolution at deep
levels). Recommendation: switch ComputeFC's window param to samples (or support both), matching
aurora for correct broadband behaviour.

**D3 — Multi-level FC container.** ComputeFC must carry *N* per-level FCs on one `fc` handle.
Recommendation: a lightweight container (e.g. `dict[int, xr.Dataset]` or an xr.Dataset with a
`decimation_level` dim) so EstimateTF can iterate levels. Port type stays `fc`.

**D4 — FC persistence trigger.** The DESIGN §3 goal is "ComputeFC persists FCs into the workspace
MTH5 FC layer" so the expensive STFT/decimation stage is reusable. Recommendation: ComputeFC
auto-persists each level to a workspace mth5 when given the archive/run context (mirroring
aurora `save_fcs`), and returns the in-memory multi-level `fc` handle regardless. The existing
`mth5.fc.write` node remains for explicit single-level control.

**D5 — Per-level TF + merge (node-graph shape).** The main flow per DESIGN is
`ComputeFC → EstimateTF → tf`. Recommendation: EstimateTF estimates each decimation level and
returns a **single merged `tf`**, with the merge logic factored into a reusable
`merge_tf_levels(per_level_tfs)` helper. The `MergeTFLevels` node exposes the same helper for
advanced use (per-level / separately-loaded `tf.level` inputs). This keeps the common graph a
single straight line while keeping the lower-level building block available.

### 14.3 Phased implementation (FC persistence first, per the decision)

**Phase 1 — FC persistence groundwork** (foundation; enables merge of multi-level results)
- Carry an absolute `start_datetime` through `ts.run` (mth5 `RunTS`) and into the fc `time`
  coordinate (`datetime64`).
- Wire a workspace mth5 that ComputeFC/EstimateTF can write to (DESIGN §3 storage).
- Extend the FC write path to persist per decimation level (reuse/refactor `fc.py`).
- Tests: fc → mth5 → fc round-trip (datetime), synthetic + real run.

**Phase 2 — Multi-level FC generation (ComputeFC)**
- Expand ComputeFC to N levels: cascade-decimate (`scipy.signal.decimate` by factor) then STFT
  each level per D2/D3.
- Params: `num_levels`, `factor`, window (`num_samples`, overlap, type), channels, magnetic
  unit, dipole length; per-level sample rates = native / factor^level.
- Persist each level (Phase 1).
- Tests: multi-level fc on synthetic + real LEMI/BGS; per-level sample rate + frequency range;
  EstimateTF consumes it.

**Phase 3 — Per-level TF estimation + MergeTFLevels**
- EstimateTF loops levels, reuses `_solve_band` at each level's sample rate → per-level TFs.
- `merge_tf_levels()` resolves duplicate periods (aurora-style: prefer the highest-rate level
  covering a period) → merged `tf`. EstimateTF returns merged `tf`; `MergeTFLevels` node exposes
  the helper for `tf.level` inputs.
- Tests: multi-level → merged TF on LEMI/BGS; broadband coverage; per-level phase convention.

**Phase 4 — Integration & verification**
- End-to-end reference workflow (Load → ExtractRun → ComputeFC(multi) → EstimateTF → Export EDI /
  Plot AppRes&Phase) on real data; compare vs single-level; update DESIGN/README.
- Cross-cutting: keep the `test12rr`-generated fixture in CI so these tests run without gitignored
  raw data.

### 14.4 Sequencing / parallelization

- **Phase 1** is the critical path (unblocks D4 persistence + the datetime requirement) — start here.
- **Phases 2 and 3** build sequentially on Phase 1; within Phase 2 the decimation core and the
  persistence are separable.
- **Phase 4** is integration across all of the above.
- Landing order: P1 → P2 → P3 → P4, each its own PR/branch.

**Decisions confirmed (2026-08-29):** D2 → constant `num_samples` window across
levels (aurora-style); D4 → ComputeFC auto-persists to the workspace mth5 (and
also returns the in-memory multi-level fc); D5 → EstimateTF estimates all levels
and returns a single merged `tf`, with `merge_tf_levels()` shared by the
`MergeTFLevels` node for advanced per-level use.

**Phase 1 delivered (2026-08-29):** absolute `start_datetime` flows through
`ts.run` (mth5 `RunTS` and core `TimeSeries`), fc `time` is `datetime64` when a
start is known; a specially named `workspace.mth5` in `ctx.workspace` holds
persisted FCs per (survey, station, run, decimation level) via `persist_fc_level`
/ `load_fc_level` (in `emptyui-mth5/persist.py`); `ComputeFC` auto-persists
level 0 when an absolute start is available and skips with a warning otherwise.
Round-trip fc→mth5→fc verified (datetime required).

**Phase 2 delivered (2026-08-29):** `ComputeFC` now generates N cascading
decimation levels via `compute_fc_levels` (`_processing.py`): level 0 is the
native-rate STFT, each deeper level decimates the previous (already-normalized)
series by `factor` (`scipy.signal.decimate`) and re-STFTs it with the **same
constant `num_samples_window`** (D2). Returns a multi-level `fc` container
(`{level: xr.Dataset}`); each level's sample rate = native/factor^level. It
persists every level into the workspace mth5 (D4). Window param renamed from
seconds (`window_length`) to `window_samples` (D2). EstimateTF unwraps a
single-level container for backward compatibility; multi-level TF estimation
+ merge remains Phase 3 (raises a clear NotImplementedError).

**Phase 3 delivered (2026-08-29):** per-level TF estimation + merge.
`estimate_tf_levels` (`_processing.py`) estimates a per-level TF for every
decimation level (reusing the aurora-backed `_solve_band`); `merge_tf_levels`
concatenates levels, sorts by period, drops degenerate bands, and resolves
near-duplicate periods by keeping the highest-sample-rate level → one broadband
`tf`. `EstimateTF` now loops levels and returns the merged `tf` (D5) plus a
`tf_levels` (`tf.level`) output; `MergeTFLevels` is functional (merges a
`tf.level` container). Verified: multi-level merged TF keeps aurora's phase
convention on the halfspace (Zxy≈+45°, Zyx≈−135°); end-to-end on LEMI
(4 levels → 64 merged periods → EDI).

**Period banding rework (2026-08-29, feedback-driven).**
- **Periods per octave** (`periods_per_octave`, default 4) replaces `periods_per_decade`;
  bands are placed on a log2 (power-of-2) grid, matching FFT/decimation windows (128/256).
- **Octave-aligned, shared boundaries (`_period_bands`)**: each decimation level spans from
  its Nyquist period (2/fs) to the next level's Nyquist period (2·factor/fs); band centers sit
  on a global log2 grid including both end points, so level *i*'s last period **exactly equals**
  level *i+1*'s first period. Levels therefore tile continuously (no gaps), and the merge simply
  de-duplicates the single shared boundary period — keeping the higher-sample-rate level (whose
  boundary estimate is well within its window, vs the deeper level's noisier Nyquist edge). This
  achieves aurora's "consistent frequency points between levels" without external band files.
- **Realistic period range**: the longest period is clamped to the window reliability cap
  (N/(2·fs), ≥2 cycles) and the octave boundary — no more "longest period = window length"
  (DC-contaminated). Shortest period is the true Nyquist (2/fs).
- **TF table node** (`mth5.table.tf` "View TF"): renders a transfer function as a table
  (period, rho_xy/yx, phase_xy/yx), giving the table viewer a producer in the TF workflow
  (previously only `mth5.table.selection` existed, which needed a KernelDataset).

**Period banding — final spec (2026-08-29, feedback #4):**
- Shortest period = **Fs/4** per level (highest trusted frequency Fs/4, avoiding
  noisy Nyquist-edge bins).
- Longest period = **window_length/4** (N/(4·Fs)).
- Band centers sit on a global log2 grid (``periods_per_octave``/octave) anchored
  at 4/Fs; because decimation factors are powers of 2, consecutive levels land on
  the **same absolute periods** wherever they overlap → consistent frequency
  points across levels.
- Levels therefore **overlap** (each covers Fs/4→window/4); the merge keeps the
  **deepest** level that covers each period (a deeper level has a longer window →
  more cycles → better long-period resolution), dropping degenerate bands.

**Deferred task → M4 (band config files):** load an aurora-style band config file
(exact centre frequencies + bandwidths) or auto-generate a sensible default when
none is given, rather than always deriving the period table from Fs/4 → window/4.
This gives power users exact control over the frequency table and matches aurora's
preset band files. (Not required for the current auto-derived table.)

**Decimation convention (2026-08-29):** decimation level *i* = the series has
been decimated *i* times. `ComputeFC`'s `decimation_levels` param = number of
cascading decimations (default **0** = none, native rate only); it produces
levels `0..decimation_levels`, each level's sample rate = native/factor^level.

### 14.5 Worth exploring — Paradigm B (cross-decimation-level FC pooling)

**Motivation.** At long periods the deepest decimation levels have few STFT
windows, so a per-level TF estimate is noisy. Deeper levels also have **finer raw
frequency resolution** (bin spacing Fsᵢ/N, i.e. more harmonics per Hz), so for a
fixed band (central frequency + bandwidth) the deeper levels supply more raw FC
samples. Pooling all levels' FCs that fall in a band into **one** robust
regression therefore recovers more samples precisely where they are scarce.

**Band design — how "central frequency + bandwidth" is modelled in existing code
(researched 2026-08-29).**
- **aurora `Band`** (mt_metadata): `frequency_min`, `frequency_max`,
  `center_averaging_type="geometric"` (center = geometric mean of the edges),
  `fractional_bandwidth` (= width/center) and `Q` (= center/width), plus the
  STFT `harmonic_indices`/`index_min`/`index_max` that fall in the band. Bands are
  log-spaced (≈ constant fractional bandwidth / constant Q). `extract_band()`
  pulls every harmonic in the band → exactly the per-band pooling primitive.
- **EMTF band setup file** (`aurora/config/emtf_band_setup/*.cfg`, e.g. `bs_test.cfg`):
  line 1 = band count; each subsequent line = `decimation_level index_min index_max`
  (a band = one level + a contiguous STFT harmonic-index range). Center
  frequency/bandwidth are derived from the index range and the level's
  fs/N geometry. aurora loads these via `ConfigCreator` (`band_specification_style="EMTF"`).
- **Key finding:** EMTF, BIRRP and aurora define bands this way but process each
  decimation level's bands **separately** (Paradigm A), and their band tables are
  **non-overlapping across levels**. **No existing open-source code pools FCs
  across decimation levels into one band** — Paradigm B is a novel extension
  enabled by our FC layer (mth5 stores each level's raw FCs; emptyUI already
  computes per-level raw-frequency fc).

**Implementation sketch (if pursued).** Build a global band table (center +
bandwidth; reuse our log-octave `periods_per_octave` grid or a fixed-Hz scheme);
for each band, gather FC samples from *every* level whose raw bins fall inside
the band (weighted by level sample count); run the existing aurora-backed
`_solve_band` once per band. Value is highest at long periods; at high
frequencies the shallow levels already provide abundant samples.

**References.** Egbert (1997), *Robust multiple-station magnetotelluric data
processing* (EMTF); Chave & Thomson (2004), *Bounded influence magnetotelluric
response function estimation* (BIRRP); aurora (simpeg, Kappler et al.) — `Band`
metadata + EMTF band setup files; mth5 FC layer.
