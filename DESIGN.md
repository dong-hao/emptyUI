# emptyUI — Design Document

A node-based, ComfyUI-inspired visual workflow interface for magnetotelluric (MT)
data processing and visualization.

Status: v0.2 — design frozen for M0/M1 (2026-08-25).
Implementation status: **M0–M3 complete (2026-09-09)**; post-M3 deliveries:
automatic l2u frequency grid (2026-09-11, §14.9) and the TF rotation node
(2026-09-11, §11/§16.3). Every remaining item is M4 and ranked by gain/effort in
§11. Sections §1–§12 are the current contract; §13–§17 are condensed records of
delivered plans (kept for the decisions they encode, not as work logs).

Processing-correctness milestones (all verified):
- **Unit normalization**: magnetic channels converted nT→A/m (B→H; both fluxgate
  and coil archives are nT — verified against the USArray EDI, §16.3) and
  electric mV/km→V/m; the missing nT→T factor was the root cause of
  non-physical apparent resistivity (now ~10–10³ Ω·m).
- **Impedance phase convention**: verified against aurora on the official
  synthetic halfspace (`test12rr`); all three estimators (OLS/RME/RME_RR) are
  aurora-backed and share aurora's convention (Zxy≈+45°, Zyx≈−135°).
- Detrending is the dedicated `core.filter.detrend` node (mean/linear/none);
  `e_dipole_length_m` optional (off by default — LEMI output is mV/km).
- **Channel nomenclature** (2026-09-05): instrument naming handled through
  mt_metadata's canonical `CHANNEL_MAPS` (§3) instead of hardcoded tables; the
  magnetic B/H mode is derived automatically and TFs are built aurora-style
  (actual channel names + nomenclature map).

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

Nodes fall into three broad categories (palette grouping and icons; categories
are open-ended strings so plugins can add their own):

| Category | Role | Examples |
|----------|------|----------|
| **Data I/O** | Bring data in/out of the graph; convert formats | LoadMTH5, LoadNIMS, LoadCSV, SaveMTH5, SaveEDI, ExportXML, MetadataEditor |
| **Processing** | Transform data; produce new typed outputs | Decimate, Detrend, BandpassFilter, NotchFilter, Calibrate(RemoveResponse), STFT, RobustEstimate, MergeTFLevels |
| **Display** | Render data to `plot` outputs (no data deps downstream) | PlotTimeSeries, PlotSpectrogram, PlotAppResPhase, PlotTipper, PlotCoherency, PlotPhaseTensor |

Display nodes emit a **`plot` output port** (rendered-figure reference), not
typed data — they can't feed processing nodes, but their figures can branch to
`Preview Plot` / `Save Plot` nodes (ComfyUI's IMAGE port + Preview/Save Image
pattern). There is **no run-level gallery**; the node-embedded viewers are the
primary (and only) display.

### Port types (the ComfyUI-equivalent)

Every node port has a **type tag**; connections are allowed only between
compatible tags. Types mirror the real containers of the mth5/aurora ecosystem,
so a node boundary ≈ an aurora pipeline stage:

| Tag | Display name | Real library class | Stage / notes |
|-----|--------------|--------------------|---------------|
| `mth5` | MTH5 Archive | `mth5.MTH5` | HDF5 archive; stateful source of runs/FCs/TFs. *(was `mt.archive`)* |
| `ts.run` | Run Time Series | `mth5.timeseries.RunTS` (= `mt_timeseries.RunTS`) | Multi-channel time series — the main signal-processing currency; wrap the `RunTS` object directly (emptyUI is mth5-native) |
| `ts.channel` | Channel Time Series | `mth5.timeseries.ChannelTS` | Single channel (ex/ey/hx/hy/hz); split/merge nodes convert ⇄ `ts.run` |
| `calibration` | Calibration / Instrument Response | `mt_metadata.filters.*` (Coefficient/PoleZero/FIR/TimeDelay/FAP) | `Calibrate` takes `ts.run` + `calibration` → calibrated `ts.run`. *(was `mt.response`)* |
| `selection` | Run Selection | `mth5.processing.KernelDataset` (DataFrame) | Which stations/runs/intervals process; local+remote roles. *Plugin-tier / M4-deferred — superseded by ExtractRun → ComputeFC* |
| `config` | Processing Config | `mt_metadata.processing.aurora.Processing` | The aurora "recipe". *Plugin-tier / M4-deferred — pipeline splits at the FC boundary and assembles config internally* |
| `bands` | Frequency Bands | `aurora.time_series.frequency_band.FrequencyBands` | Band edges. *Plugin-tier / M4-deferred — derived inside EstimateTF; EMTF band-setup files are M4 #9* |
| `decimation` | Decimation Level | aurora `DecimationLevel` | *Plugin-tier*: per-level recipe for advanced multi-config assembly *(was `mt.decimation`)* |
| `fc` | Fourier Coefficients | mth5 `FCGroup`/`FCDecimationGroup`; aurora STFT | The frequency-domain currency. TS→FC natural; FC→TS plugin-tier (lossy inverse). *(was `spectra`)* |
| `tf.level` | TF (decimation level) | aurora `TTFZ` | Per-decimation-level TF estimate |
| `tf` | Transfer Function | `mt_metadata.transfer_functions.TF` (also mth5 `TransferFunctionGroup`) | Final merged TF; exporters & analysis plots consume this *(was `mt.tf`)* |
| `table` | Table | pandas DataFrame | Generic tabular output (run summaries, coherency, etc.) |
| `plot` | Plot | rendered-figure ref (PNG/SVG/PDF + metadata) | Produced by Display nodes; consumed by `Preview Plot`/`Save Plot` |
| `any` | Any | — | Wildcard; connects to anything, executor trusts the node |

> `file` was dropped as a port type (paths are parameter widgets, not data
> flows). `config`/`bands`/`decimation`/`selection` are plugin-tier and
> M4-deferred: the default pipeline splits at the FC boundary and derives band
> edges internally. The right-hand panel always shows both the **display name**
> and the real **type tag** for every port.

### Port-type audit notes

- Derived quantities (apparent resistivity, phase, impedance) are *computed*
  from `tf` by Display nodes, not separate ports — avoids type sprawl.
  Remote-reference selection lives in `selection` (remote column).
- **Consistency rule**: every port tag maps 1:1 to a real library container; we
  avoid inventing new names where the ecosystem already has one (`fc` not
  `spectra`, `mth5` not `archive`).

### M2 ecosystem pins (stable releases, audited 2026-08)
- `mth5==0.6.8`, `aurora==0.6.2`, `mt_metadata==1.0.10`, `mt_timeseries==0.0.2`
  (`mt_timeseries` now hosts `RunTS`/`ChannelTS`; mth5 re-exports them).
- `KernelDataset`/`RunSummary` live in **`mth5.processing`**.
- Decimation levels are **independent by default** (fresh `Window`/`Estimator`
  per level) → `ComputeFC`/`EstimateTF` keep a constant window in samples across
  levels and share estimator settings.
- `channel_nomenclature` lives on `Processing` config, not `RunTS`.
- These pins are plugin-level extras (`emptyui-mth5`, `emptyui-aurora`), never
  core dependencies; workflows record plugin versions for reproducibility.

Rules:
- Tags are **open strings**, namespaced (`<domain>.<name>`); the frontend
  renders unknown types with a neutral color.
- **Port colors are decided by the backend** (UI metadata per tag via
  `GET /api/nodes`); connection lines render thick and in the source port's
  color. Fallback families: blue = time series, dark green = frequency domain
  (`fc`), red = transfer functions (`tf`), purple = metadata/config,
  gray/yellow = generic (`any`, `table`, `plot`).
- Connection rule: source and target tags must be equal, or the target port
  declares `accepts: [...]`. `any` matches all.
- Subtyping is deliberately **not** implemented; explicit coercion nodes
  (split/merge/configure) are preferred.

### Port rendering conventions

Port dots sit **just inside** the node edges, vertically aligned with their
name/description text row — inputs near the left edge with names to their right,
outputs near the right edge with names to their left (ComfyUI grammar).

### Reference MT workflow (type flow)

```
LoadMTH5 ─mth5─▶ ExtractRun ─ts.run─▶ ComputeFC ─fc─▶ EstimateTF ─tf─┬▶ ExportEDI
(LoadLemi / LoadCsv)                        │                        ├▶ PlotAppResPhase
                                              └─fc─▶ Denoise ─fc─▶ EstimateTF
                                                                    ├─tf.level─▶ MergeTFLevels
                                                                    └─tf─▶ PlotTipper

LoadMTH5 ─mth5─▶ ExtractRun ─ts.run─▶ Detrend ─ts.run─▶ ComputeFC ─fc─▶ PlotSpectrogram

Remote reference:  stationA ─▶ ExtractRun ─▶ Align Runs ─▶ ComputeFC ─fc─▶ EstimateTF (RME_RR)
                   stationB ─▶ ExtractRun ─▶ Align Runs ─▶ ComputeFC ─ref_fc─▶ (magnetic-only OK)
```

Coercion points that require dedicated nodes (mirroring aurora internals):
1. `ts.run` ⇄ `ts.channel` via explicit SplitChannels / MergeChannels nodes.
2. uncalibrated `ts.run` + `calibration` → calibrated `ts.run` (units change).
3. `tf.level[]` → `tf` (merge; resolves duplicate periods across levels).
4. Format coercions for popular types ship in core; exotic converters are plugins.

### Pipeline stage split: TS → FC → TF

The aurora processing run is split at the Fourier-coefficient boundary:

- **`ComputeFC` node (TS → FC)** — input `ts.run`; params: channel nomenclature
  (presets via `cascade`), input/output channel names, decimation levels +
  factor, window samples/overlap/type. Output `fc`. Internally expands uniform
  decimation levels sequentially (constant window in samples, aurora-style D2)
  and persists computed FCs into the workspace MTH5's FC layer. Outputs may be
  the electric pair and/or a vertical magnetic channel (e.g. `e1,e2,bz`) to also
  produce the tipper. The magnetic B/H mode is **derived automatically** from
  the nomenclature/channel names (bx/by → fluxgate B→H; hx/hy → coil H, archived
  in nT like fluxgate B, so also converted to A/m; §16.3) — the `magnetic_unit`
  select defaults to `auto` and exists only for legacy saved workflows.
  **Channel auto-configuration**: empty input/output channel fields resolve
  from the data and its nomenclature — inputs = horizontal magnetic pair,
  outputs = electric pair **plus** the vertical magnetic channel when present
  (tipper by default); a run with no electric channels yields a tipper-only FC;
  requested-but-missing outputs are dropped with a warning (missing inputs are a
  hard error). Matching is case-insensitive. When the nomenclature keyword is
  known upstream, the frontend auto-fills the fields from the inherited keyword
  (explicit user choices never overridden).
- **`EstimateTF` node (FC → TF)** — inputs `fc` + optional `ref_fc` (remote
  reference); params: estimator (`Robust`/`Least Squares`), remote-reference
  on/off, band style (`l2u` default / `uniform`, §14.9), `min_bins_per_band`,
  `t_short`, periods-per-octave. Band edges are derived internally. Output `tf`
  + per-level `tf.level`. The TF is built aurora-style: rows/cols carry the
  **actual** channel names and the `channel_nomenclature` map is attached — a
  2×2 impedance, or a 3×2 impedance + tipper. The remote reference (**RME_RR**,
  aurora-backed) pairs local/remote FC samples by row, so the ref fc must share
  the local fc's **frequency AND time grids** (both validated with clear errors);
  the runs must be aligned first (see `Align Runs`).
- **`Align Runs` node (`mth5.runs.align`)** — two `ts.run` inputs → two `ts.run`
  outputs, both normalized to an mth5 `RunTS` cropped to their **common
  acquisition overlap**. Required for remote reference: same sample rate
  (aurora hard-errors on mixed rates) and identical time grids. Overlap is
  computed from absolute start/end times (GPS) when available, with a warning
  fallback to relative offsets. The remote station may be **magnetic-only**
  (bx,by). The node also computes inter-station horizontal-magnetic coherence
  (64–1024 s band) and **warns when it is weak** — a clock offset or un-filled
  data gap silently degrades remote reference (LEMI readers index by row, not
  timestamp; gaps must be filled at conversion, see `scripts/convert_bgs_lemi.py`).

Why this boundary: (1) a first-class `fc` port lets users branch into denoising,
spectral analysis, or custom estimators; (2) cheap re-runs — the expensive
STFT/decimation stage is cached and estimator changes invalidate only
`EstimateTF`; (3) no parameter wall — each node carries only its stage's
parameters, aurora's monolithic `Processing` object is assembled internally.

### Channel nomenclature (mt_metadata CHANNEL_MAPS)

Instrument channel naming is **not** re-invented: `emptyui-mth5/nomenclature.py`
builds on mt_metadata's canonical `CHANNEL_MAPS` as the single source of truth.
Channels are **never renamed in the data** — a RunTS keeps `e1`/`bx`/... names;
the nomenclature keyword is carried as metadata (fc attrs, TF map) and used only
to interpret names:

- **Unit normalization** (B vs H) is derived from the names, not a manual switch
  (§16.3: coil archives are nT too, so hx/hy → A/m as well).
- **Load LEMI-424** exposes a `channel_nomenclature` select (lemi12/lemi34):
  LEMI instruments can carry four E channels; the keyword selects the electric
  pair read into the MTH5.
- **Split/Merge Channels** resolve standard ex/ey/hx/hy/hz ports onto actual
  names via the nomenclature (exact-name lookup, never `getattr` on an absent
  component).
- **ComputeFC** records the resolved keyword on each fc's attrs
  (`channel_nomenclature`, `magnetic_unit`) so EstimateTF/MergeTFLevels can map
  back without re-entry; FCs read from an mth5 archive are enriched by
  `mth5.fc.read`.
- **EstimateTF** builds the TF aurora-style (`TF(channel_nomenclature=...)`):
  coordinates are the actual instrument names and `tf.channel_nomenclature`
  carries the standard→actual map, so aurora's `TTFZ`/`TransferFunctionCollection`
  can consume emptyUI TFs. All TF accessors are nomenclature-aware; display
  nodes select components via `tf.ex/ey/hx/hy`.
- **Display nodes** accept standard names / auto-select (`core.plot.*`): channel
  specs resolve through the inferred nomenclature, case-insensitive, with
  sensible defaults.
- **Exporters**: EDI/XML/AVG standardize on write; the ZMM writer in mt_metadata
  1.0.10 assumes standard names, so `ExportTF` relabels to a standard-coords
  copy for `zmm` output.

### Merging multiple runs into one TF estimate

A single TF estimate is often built from **several acquisition runs** (e.g.
Phoenix MTU-5 segments). aurora supports this by allowing multiple
`KernelDataset` rows and **concatenating their STFTs at the FC level** — it does
*not* merge in the time domain. emptyUI offers a **`Merge Runs` node** with two
modes:
1. **`selection` (FC-level, default)** — multiple run rows in a `KernelDataset`;
   aurora concatenates the FCs. No fabricated samples. The idiomatic path.
2. **`ts.run` (physical)** — combines raw series via `RunTS.merge()` into a new
   mth5 run group; explicitly opt-in because it **interpolates across gaps**
   (gap policy is a param).

Constraints: all runs must share the same native sample rate, and channel
components must be consistent. Remote-reference multi-row intersection is an
open aurora TODO — the node warns when a local run spans multiple remote runs
(M4 #2).

### Data passing between nodes

Nodes exchange typed *handles*, not raw payloads: array-like containers →
xarray objects in an in-process store keyed by handle ID; tables → DataFrames;
files → paths in a per-run workspace; plots → artifact files. This keeps
WebSocket messages small and lets the same executor serve batch/CLI use later.

### Storage: file-based, backed by MTH5

No internal database. **MTH5 is the server-side container**:
- A workspace directory **per run** (`ctx.workspace`); intermediates a node
  chooses to persist are written into `<workspace>/workspace.mth5` (see
  `persist.py`). A persistent `~/.emptyui/workspace/` is a future item.
- Nothing reaches the user's disk until they explicitly export/save-as.
- Benefits: interoperability with the existing CLI ecosystem (any mth5/aurora
  script can read what emptyUI produced) and remote deployments are just "a
  directory on the server".

---

## 4. Node Manifest Schema

Each node type exposes a JSON manifest via the registry; the properties panel
and sidebar are generated entirely from it.

```jsonc
{
  "type": "core.filter.bandpass",          // namespaced, unique
  "version": "1.0.0",                       // semver of the node definition
  "title": "Bandpass Filter",
  "description": "Butterworth bandpass on time series.",
  "category": "Processing",
  "inputs": [{ "name": "data", "type": "ts.run", "required": true }],
  "outputs": [{ "name": "filtered", "type": "ts.run" }],
  "params": [
    { "name": "low_freq", "label": "Low cutoff (Hz)", "widget": "number",
      "default": 0.001, "min": 0, "step": 0.0001, "optional": true },
    { "name": "order", "label": "Order", "widget": "integer",
      "default": 4, "min": 1, "max": 10 }
  ],
  "deps": ["scipy>=1.11"]                   // informational; enforced by plugin extras
}
```

Widget vocabulary (`number`, `integer`, `string`, `text`, `boolean`, `select`,
`radio`, `file`, `directory`, `code`, `list`) is fixed in core; the frontend has
dedicated controls for `boolean`/`select`/`radio`/`number`/`integer`/
`file`/`directory`, and the remaining string-like widgets fall back to a
single-line text input.

#### Tooltip / description / palette protocol

- **`tooltip`** — the hover tip; exactly **one sentence** saying what the node
  *does* (imperative, ≤ ~100 chars, no parameter lists).
- **`description`** — full details shown in the properties panel on click.
- **Fallback**: empty `tooltip` resolves to the first sentence of `description`
  (`emptyui.api.first_sentence`) — write every description so its first sentence
  works as a standalone tip.
- **`hidden`** — utility/test nodes are excluded from the palette by default but
  remain registered (saved workflows still load and execute).
- **Palette ordering**: fixed category order (`Data I/O`, `Processing`,
  `Display`, then custom categories alphabetically); nodes alphabetical within a
  category. Plot-export nodes are **Display**, not Data I/O.

Plugins following these rules need zero frontend changes.

### Param auto-change (`cascade`) & upstream auto-fill (`inherit`)

Two manifest-driven UI conveniences keep the frontend schema-agnostic:

- **`cascade`** (per param): when the user changes this param to a given value
  (key), the UI sets the listed `(other_param, value)` pairs. Keys serialize as
  strings (`bool` → `"true"`/`"false"`). Used for `Compute FC`'s nomenclature
  select to preset channel names per mt_metadata, and for `Estimate TF`'s
  remote-reference to force the Robust estimator:
  ```python
  Param("channel_nomenclature", "select", default="auto",
        options=("auto", "lemi12", "lemi34", "phoenix123", "musgraves", "default"),
        cascade={"lemi12": (("input_channels", "bx,by"), ("output_channels", "e1,e2")), ...}),
  Param("remote_reference", "boolean", default=False,
        cascade={True: (("estimator_engine", "Robust"),)}),
  ```
  The magnetic B/H mode is **not** a manual choice (derived from nomenclature;
  §3).
- **`inherit`** (per node): `{input_port: {target_param: upstream_param}}`. When
  an upstream node is connected, those params render as dropdowns populated from
  the upstream node's params and auto-fill if empty. Same-name inheritance is an
  identity mapping. `channel_nomenclature` is inheritable end-to-end
  (`Load LEMI-424` → `Extract Run` → `Compute FC`); the run-time fallback infers
  it from actual channel names when no upstream value exists.

Both serialize into the manifest; the frontend applies them generically.

#### Chain-walking inheritance (guideline)

Inheritance is resolved by a **single recursive, frontend-only resolver**
(`resolve(nodeId, paramName)`), walking the workflow document — never the
backend (the backend derives the same identity at run time from payload
metadata, so correctness never depends on it). Rules:

1. **Per-node declaration** — each node declares only its own `inherit` map; no
   global identity registry.
2. **Only declared params auto-fill** — everything else stays fully manual.
3. **Editable until a source provides a value** — a declared param stays a plain
   text field unless an upstream node actually yields a value; the inherited
   dropdown is an editable combobox.
4. **Recursive chain-walk** — transitive across arbitrary processing nodes, with
   pass-through fallback when an intermediate node declares nothing (inserting a
   node between source and `Compute FC` never loses the identity).
5. **Name mapping** — each hop uses the *current* node's mapping (bridges e.g.
   `survey_id` → `survey` without changing downstream nodes).
6. **Cycles & fan-in** — visited set (DAG but be defensive); union candidates
   across parallel sources.
7. **Same-name default** — prefer identity mapping; declare only when names
   differ.

Adding a new node type needs **zero changes to the resolver**.

### Server-side definition

```python
from emptyui import Node, Input, Output, Param

class Bandpass(Node):
    type = "core.filter.bandpass"
    inputs  = [Input("data", "ts.run")]
    outputs = [Output("filtered", "ts.run")]
    params  = [Param("order", "integer", default=4, min=1, max=10)]

    def execute(self, ctx, data, order=4):
        ...
        return {"filtered": out}
```

`ctx` gives nodes access to: workspace dir, logger, artifact publishing,
progress reporting, and the shared data store.

---

## 5. Plugin System

Discovery uses Python **entry points** (importlib.metadata) — no config files,
no core edits:

```toml
[project.entry-points."emptyui.nodes"]
my_nodes = "emptyui_mth5.plugin:register"
```

Rules that keep the ecosystem healthy:
- Core ships with **only** the framework + one `emptyui-core-nodes` plugin
  (numpy/matplotlib-level deps).
- Each plugin declares its heavy deps as pip extras.
- Node types are namespaced by convention: `core.<domain>.<name>` for the
  bundled plugin, `<org>.<domain>.<name>` for external plugins.
- **Open, namespaced port types** (2026-09-08): plugins may introduce new types
  but must namespace them and declare UI metadata via the node's `port_types`
  ClassVar; reusing core tags is strongly preferred, and consumers duck-type
  documented payload attributes instead of importing plugin containers.
- **Self-tests required**: every plugin ships a pytest suite (see template).
- **Models declared, not bundled**: `models` in the manifest
  (id/filename/sha256/url/license); binaries live in a configured model
  directory (platform-provisioned in remote mode, §17) — never inside the wheel.
- Manifests include `version`; workflows record plugin versions for
  reproducibility. The full authoring contract (immutable-input execute
  contract, packaging layout, publishing checklist) is **`docs/plugin-guide.md`**
  with a working template in `examples/plugin-template/`.
- A future `emptyui plugin publish/index` flow can distribute community plugins
  as ordinary PyPI packages.

### 5.1 Plugin enable/disable — implemented (2026-09-09)

Users switch third-party plugins on/off from the web UI; bundled plugins
(`core`, `mth5`, `aurora` — marked via `register.bundled = True`) are always on
and shown read-only.

**Contract:**
1. A toolbar ⚙ Settings dialog lists every installed third-party plugin (name,
   version, node count) with an enabled tick box.
2. The palette is catalog-driven — toggling refetches `GET /api/nodes`.
3. **In-process hot reload** (no process respawn): disabling means "don't
   register the plugin's nodes"; the Registry rebuilds in place, the cross-run
   cache is cleared, and the UI refetches. A true uvicorn restart is avoided.
4. **Silent when nothing to switch**: empty state with bundled plugins as
   read-only rows; config entries naming unknown plugins are ignored without
   warning (config files travel between machines).

**Backend API**

| Method | Path | Purpose |
|---|---|---|
| GET    | `/api/plugins` | installed plugins: `[{name, version, node_count, bundled, enabled}]` |
| POST   | `/api/plugins/{name}/enable` | hot-reload the registry with the plugin active |
| POST   | `/api/plugins/{name}/disable` | hot-reload without it; clear the cross-run cache |

**Config & persistence**
- `EMPTYUI_DISABLED_PLUGINS` sets the baseline at startup (the §17 platform knob).
- Runtime toggles are **in-memory only** (2026-09-09); optional JSON-file
  persistence activates only when `EMPTYUI_SETTINGS_PATH` (or
  `create_app(settings_path=...)`) is configured; the default user-config-dir
  settings file is deferred. Unknown names are silently ignored.
- `create_app(disabled_plugins=[...])` mirrors the env for module embedding.

**Robustness**
- `find_plugins` records which node types each entry point contributed; the
  registry exposes `status_of(type)` → `active | disabled | unknown`, so
  validation distinguishes *"plugin 'mth5' is disabled — enable it in Settings"*
  from *"plugin not installed?"*.
- The executor snapshots manifests at run start, so toggling mid-run cannot yank
  a node out of an executing workflow.
- Concurrent toggles are serialized by the app lock; discovery is re-run per
  toggle (module imports are cached by Python).

---

## 6. Workflow JSON Schema

One file = one reproducible pipeline.

```jsonc
{
  "format": "emptyui/workflow",
  "version": 1,
  "meta": { "name": "kilauea-demo", "created": "...", "author": "..." },
  "nodes": [
    { "id": "n1", "type": "core.io.load_csv",
      "params": { "path": "/data/station01.csv" },
      "position": { "x": 80, "y": 120 } },        // UI-only, ignored by executor
    { "id": "n2", "type": "core.filter.bandpass", "params": { "order": 4 } },
    { "id": "n3", "type": "core.plot.spectrogram", "params": {} }
  ],
  "edges": [
    { "id": "e1", "source": "n1", "source_port": "data",
      "target": "n2", "target_port": "data" },
    { "id": "e2", "source": "n2", "source_port": "filtered",
      "target": "n3", "target_port": "data" }
  ]
}
```

- Edges reference ports by name → resilient to port reordering.
- `position` is UI-only → headless/batch execution ignores it cleanly.
- Executor validates against current manifests at load time with actionable
  messages.

---

## 7. Execution Model

- **Scheduler**: Kahn topological sort; branches run **sequentially** (M0).
  Concurrent thread-pool execution is M4 #12 (`cpu_bound` hints).
- **Caching** (M3, 2026-09-08): server-lifetime in-memory LRU (`NodeCache`,
  256 entries). Cache key = **structural signature** computed in topological
  order: `sha256(type, manifest version, effective params, per-input upstream
  signatures)` — node ids excluded (copy/paste duplicates share entries);
  signatures recurse through the DAG so invalidation propagates. Data payloads
  are **never hashed** (xarray/RunTS impractical to digest); string params
  pointing at existing files hash `(path, size, mtime_ns)` (Nextflow-style), so
  externally modified input files invalidate. On a cache hit the executor
  re-registers outputs and **copies artifact files into the new run's
  workspace** (renamed for the restoring node); vanished artifact files
  re-execute the node. Cached nodes emit `node_status` with `"cached": true`.
  **Immutability contract** (ComfyUI model): nodes must be functional — never
  mutate inputs or previously returned outputs. Disk persistence across
  restarts is M4 #5. `DELETE /api/cache` drops all entries.
- **Partial runs** (M3, 2026-09-08): `POST /api/runs` accepts
  `targets: [node_id, …]` — "execute to here" runs the targets plus their
  ancestral closure; other nodes emit no events. Unknown targets rejected with
  422. The web UI exposes "Run to here" in the properties panel.
- **Progress**: nodes call `ctx.progress(frac, message)`; forwarded to all
  connected WebSocket clients.
- **Failure**: *current executor* aborts the whole run (error event +
  `run_finished: failed`); per-branch continuation is future work.
- **Cancellation**: implemented — `POST /api/runs/{id}/cancel` sets a per-run
  `threading.Event` polled between nodes (including after the final node, so
  cancelling during the last node reports `cancelled`); toolbar **✕ Cancel**
  enabled while queued/running. A node cannot be interrupted mid-execution —
  cancellation takes effect at the next node boundary.

### Run lifecycle events (WebSocket)

```
{ "event": "run_started",   "run_id": "r1" }
{ "event": "node_status",   "node": "n2", "status": "running" }
{ "event": "node_progress", "node": "n2", "frac": 0.42, "msg": "filtering…" }
{ "event": "artifact",      "node": "n3", "id": "a7", "kind": "image",
  "handle": "h12", "url": "/api/artifacts/a7/preview" }
{ "event": "node_status",   "node": "n3", "status": "done" }
{ "event": "run_finished",  "run_id": "r1", "status": "completed" }
```

The `artifact` event carries the output `handle` id so viewer nodes resolve
exactly which figure arrives on which input edge, even when one node publishes
several figures.

---

## 8. Backend API (REST)

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/nodes` | Catalog of node manifests (drives sidebar + panels) |
| POST | `/api/runs` | Validate + start a workflow run (`targets` restricts to "execute to here") |
| GET  | `/api/runs/{id}` | Run status/outputs |
| GET  | `/api/artifacts/{id}/preview` | Inline preview (PNG/JSON table) |
| GET  | `/api/artifacts/{id}` | Full artifact download (`Content-Disposition: attachment`) |
| POST | `/api/runs/{id}/cancel` | Request cancellation; executor checks the token between nodes |
| DELETE | `/api/cache` | Drop all cached node results |
| GET/POST | `/api/plugins[/{name}/enable|disable]` | Plugin listing + enable/disable hot reload (§5.1) |
| WS   | `/api/ws` | Run events stream |

A run-level artifact gallery is not implemented — viewers on nodes are the
per-figure/table display, each with its own download link. MVP keeps state on
the file-based workspace (§3); a DB is added only if run history demands it.

---

## 9. Frontend Design

Stack: **Vite · React · TypeScript · @xyflow/react · Zustand · TanStack Query**

Components:
- **GraphCanvas** — React Flow; maps workflow JSON ⇄ RF nodes/edges bidirectionally.
- **NodePalette** — grouped by `category`, populated from `GET /api/nodes`.
- **Inline node params (default, ComfyUI paradigm)** — parameters render as
  widgets directly on the node body; the right-hand PropertiesPanel shows
  metadata (id, type, version, ports, description, deps) plus Delete. Path
  widgets have a server-side file browser ("…" button).
- **Preview Plot / View Table nodes** — render an incoming `plot`/`table` inside
  the node body (drag-resizable), each with a **⤓ download** link for the full
  artifact.
- **Toolbar** — run, cancel (✕), new/open/save workflow.

State split: Zustand for the workflow document/selection/connection drafts;
TanStack Query for server resources; WebSocket client for ephemeral run state.

End-to-end testing: **Playwright Test runner** (`packages/web/e2e/`, headless,
full-stack — starts Vite dev server + FastAPI backend via `webServer`; run with
`npm test`, CI on ubuntu-latest). **Playwright MCP** is the interactive
counterpart for agent-driven debugging.

Connection rules (port types, one edge per input port, no self-loops, no
cycles) are enforced client-side during drag (`isValidConnection`) and
re-checked in `onConnect`, then re-validated server-side before every run.

### Canvas interaction protocol (ComfyUI parity)

- **Disconnect gestures**: drag a connected handle to the empty pane → unlink;
  drag either endpoint anchor of an edge → drop on a compatible handle re-routes,
  drop on the pane deletes; click an edge + `Delete`/`Backspace`.
- **Copy/paste**: `Ctrl/Cmd+C` snapshots the selected subgraph (nodes + internal
  edges); `Ctrl/Cmd+V` pastes with fresh ids, deep-copied params, remapped
  internal edges and a 24 px fan-out offset — external edges are *not* rewired.
  `Ctrl/Cmd+D` duplicates in place. Keystrokes in form controls are ignored.
- **Selection**: click = exclusive, `Ctrl/Cmd+click` = additive, `Shift+drag` =
  box selection (`selectionMode: Full`); `Delete`/`Backspace` deletes the full
  selection.
- **Initial view**: `fitView` capped at `maxZoom: 1` with 40 px padding.

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
- Everything else (mth5, aurora, mt_metadata, …): plugin extras.
- Frontend pinned via lockfile; API types generated from OpenAPI spec
  (`scripts/codegen`).

---

## 11. MVP Roadmap

**M0 — Skeleton** ✅: monorepo scaffold, CI, core Node API + registry + schema
validation + echo node, web shell with palette from `/api/nodes`.

**M1 — First real pipeline** ✅: `load_csv → bandpass → timeseries/spectrogram`
nodes, topo executor with WS progress/artifacts/errors, manifest-driven
properties panel, workflow save/load.

**M2 — MT ecosystem integration** ✅ (2026-09-05/06):
`emptyui-mth5` plugin (mth5 0.6.8: Load/SaveMTH5, ExtractRun/SplitChannels/
MergeChannels, TF I/O, Merge Runs, FC read/write), `emptyui-aurora` plugin
(aurora 0.6.2: ComputeFC/EstimateTF/MergeTFLevels; OLS/RME/RME_RR), table
viewer, export coercion nodes, and publication-quality plot standardization
(§15).

**M3 — Power features** ✅ (2026-09-09). Delivered: nomenclature-aware automatic
channel configuration; remote reference end-to-end (Align Runs); run
cancellation + artifact download; headless Playwright regression suite;
robustness batches A & B (fan-in duplicate rejection, legacy mth5 API
resolvers, NaN/Inf guards, tipper ylim auto-expand, remote-reference
partial-ref-container error, calibration unmatched-channel warnings,
node-instance-unique artifact filenames, `Calibrate` explicit-filter
delegation); content-addressed caching + partial execution (§7); instrument
calibration framework + LEMI support (§16.1); EMTF-style coherence sorting
(§16.2); plugin packaging guide + template; plugin enable/disable (§5.1);
remote-server contract (§17.6 items 1–6). Sub-workflows deferred to M4 (§12).

**Post-M3 deliveries:**
- **Improved auto frequency grid** — **delivered 2026-09-11** (M4 #17): the
  `Estimate TF` node default is the `l2u` scheme (fixed reference template +
  log-uniform window-starvation coarsening + optional `t_short` head clip),
  reproducing the official hand-crafted USArray grid strategy for any window
  (§14.9).
- **TF rotation node** (`mth5.tf.rotate`) — **delivered 2026-09-11**: rotate a
  TF to true north before comparison/inversion. Standard convention
  `Z' = R(angle) Z R(angle)ᵀ`, `R(angle) = [[cos α, sin α], [−sin α, cos α]]`,
  positive = clockwise from true north; magnetic→true north ⇒
  `angle = −declination` (tipper rotates as a vector; per-element error stds
  rotate with |R| — the §16.3-validated conservative bound, no covariance is
  stored). `Import TF` parses the EDI declination (explicit
  Declination/Decl field, else the Hx measurement azimuth; EDI formats are
  diverse — when neither is present the node falls back to the `angle`
  parameter with a warning). Station metadata is carried over unchanged; a
  one-line "rotated X deg" processing note is appended.

**M4 — Community & advanced backlog (2026-09-09, ranked by gain/effort)**

All remaining tasks (M3 leftovers, M4 items, documented open questions) are
consolidated here, ranked by expected **gain per unit effort** (descending).

| # | Task | Gain | Effort | Ratio | Notes |
|---|------|------|--------|-------|-------|
| 1 | **Plot Coherency + Plot Phase Tensor nodes** | High | Low | ★★★★★ | Standard MT QC/interpretation products; data already flows through `fc`/coherence diagnostics/`tf` — plot code patterns exist (mth5/plotting.py). Designed, not implemented. |
| 2 | **Merge Runs: remote multi-row warning** | Medium | Trivial | ★★★★★ | §3 open item (aurora TODO): warn when a local run spans multiple remote runs instead of silently mis-pairing; one `ctx.warn`/log guard + test. |
| 3 | **Batch/queue runner** | High | Medium | ★★★★ | Multi-station processing is the real repetition pattern; headless `POST /api/runs` + a small queue UI/CLI reuses the executor as-is. |
| 4 | **In-band smoothing taper (Parzen/Hann)** | Medium | Low | ★★★★ | §14.8/M4: `√weight` FC rows by band-center proximity before regression — the `_coherence_weight` mechanism already exists; composes with OLS/RME/RME_RR. Genuine enhancement (neither EMTF nor aurora does this). |
| 5 | **Disk-persisted result cache** | Medium | Medium | ★★★ | Cache keys/restore logic already exist (§7); FC/STFT recomputation across restarts is the expensive part. Needs a payload serialization strategy for xarray/RunTS. |
| 6 | **Calibration absolute-gain verification** | Medium | Low | ★★★ | §16.1 open item: process one site independently (mtpy/other) and compare ρ/phase — validation, not a feature. |
| 7 | **True EMTF per-frequency coherence sort** | Medium | Medium | ★★★ | §16.2: window-ensemble per-frequency variant (current sort is a broadband heuristic); parity with EMTF conventions. |
| 8 | **mtpy golden-image plot comparison** | Low-Med | Low | ★★ | §15 open item: test-only dev extra; guards plot regressions against future style changes. |
| 9 | **Hand-authored band-config files** | Low-Med | Medium | ★★ | §11/M4: load exact EMTF band-setup files (exact-table node option — the `bands_override` of §16.3). The l2u scheme (§14.9) already reproduces the official grid strategy, so this is an optional preset layer. |
| 10 | **Paradigm B: cross-decimation FC pooling** | High | High | ★★ | §14.5 novel extension: pool raw FCs across levels per band for statistically stronger long-period estimates; research-y, unproven, but the highest scientific upside. |
| 11 | **FC→TS inverse node** | Low | Low-Med | ★★ | §12 open question: inverse STFT/overlap-add for frequency-domain denoising demos; plugin-tier coercion node. |
| 12 | **Executor thread-pool concurrency** | Low | Medium | ★★ | §7 future work: `cpu_bound` hints + branch parallelism; MT pipelines are 5–10 mostly-linear nodes, so the practical gain is small. |
| 13 | **Sub-workflows (nested DAGs)** | Medium | High | ★ | §12: defer unless workflows outgrow copy/paste + templates + the batch runner (>20-node workflows or un-maintainable repeated blocks). |
| 14 | **Docs site** | Low-Med | Medium-High | ★ | Plugin guide + deployment guide exist; a hosted site is marketing/onboarding, not functionality. |
| 15 | **Plugin index/publish flow** | Low | High | ★ | PyPI + the §11 checklist already cover distribution; a marketplace adds discovery, not capability. |
| 16 | **ML plugin GPU querying** (`ctx.resources.gpu_count`) | Low | Trivial | ★ | §12 open question; only matters once ML noise-suppression plugins exist. |
| 17 | **Improved auto frequency grid (window-count adaptive)** | High | Medium | ★★★★ | §14.9: **delivered 2026-09-11** — `Estimate TF` node default is the `l2u` scheme, reproducing the official grid strategy for any window. |
| 18 | **MMT (multi-station MT) estimator** | Low-Med | Medium-High | ★★ | **Long-term nice-to-have** (2026-09-10, Kelbert et al. 2026, Rev. Geophys. 10.1029/2024RG000850): Egbert (1997) / Smirnov & Egbert (2012) extension of RR using **multiple sites** to define reference fields; the official short-period products were made this way. aurora 0.6.2 has only OLS/RME/RME_RR (**single** remote) — needs a new emptyUI solver (K reference stations → 2K reference channels, robust multivariate regression). Expected benefit: the <15 s end, where a single 1 Hz NIMS remote (SDH28) is itself noise-limited (§16.3). Do **after** the basic processing chain + frequency grid are pinned. |

**Priority note (2026-09-11):** TF rotation is **delivered**; next in line is
the exact-table node option (M4 #9 — still needed for apples-to-apples
comparisons with the official product). MMT (#18) stays a long-term target;
basic processing is pinned first.

Deferred-by-design items with rationale: sub-workflows (§12), disk-persisted
cache (restart survival only), band-config presets (the l2u scheme reproduces
the official grid strategy automatically, so hand-authored files are optional),
and the in-band smoothing taper (a genuine enhancement neither classic EMTF nor
current aurora implement).

---

## 12. Open Questions

All open questions are M4 backlog items, ranked in §11:

- **FC→TS inverse node** (M4 #11): inverse STFT/overlap-add for frequency-domain
  denoising demos; plugin-tier coercion node when needed.
- **ML plugin GPU availability** (M4 #16): `ctx.resources.gpu_count`?
- **Sub-workflows / nested DAGs** (M4 #13) — **deferred** (2026-09-08): after
  delivering caching + partial execution (§7), completing M3, and a survey of
  comparable systems (ComfyUI subgraphs, n8n, Node-RED, KNIME), sub-workflows
  were judged not worth the complexity: typical MT pipelines are 5–10
  mostly-linear nodes, the one genuine repetition pattern (multi-station
  blocks) is covered by copy/paste (§9) and, more directly, an M4 batch runner
  (#3). Lighter alternatives — example-workflow templates and visual grouping —
  are preferred. Revisit only if real usage produces >20-node workflows or an
  un-maintainable repeated block.
- **Authentication model** — resolved by design (not implementation): emptyUI
  stays identity-blind behind the platform gateway; see §17.

---

## 13. Remaining M2 — delivered (historical plan record, 2026-08-29)

Status: **M2 delivered (2026-09-05)**; every item below is done. Key delivered
decisions (kept for their content, not the plan):

- **Robust RME/RME_RR estimator** (2026-08-29): wraps aurora estimators (`OLS`
  via `estimate_ols`; `RME`/`RME_RR` via their aurora classes — the regression
  module does not re-export at package level, import submodules directly).
  `remote_reference` supported via the `ref_fc` input; when unconnected, local
  channels are reused as a self-reference. All three verified to match aurora's
  phase convention on `test12rr` (Zxy≈+45°, Zyx≈−135°).
- **Estimator UI**: `EstimateTF` exposes **two** estimator options —
  **`Least Squares`** and **`Robust`** — plus a separate `Remote Reference`
  checkbox that only applies to `Robust` (Robust+off → `RME`; Robust+on →
  `RME_RR`; disabled in the UI for LSQ). Legacy saved-workflow values
  (`OLS`/`RME`/`RME_RR`) still load. **No LS+RR by design**: aurora offers
  remote reference only through robust `RME_RR` (remote reference exists to
  suppress incoherent noise, where robust weighting matters); requesting LSQ
  with Remote Reference raises a clear error.
- **Multi-level decimation + FC persistence + MergeTFLevels** (2026-08-29):
  cascading decimation (`compute_fc_levels`), per-level FC persisted to the
  workspace MTH5 (`persist.py`), per-level TF estimation + deepest-wins merge.
- **Table viewer, Calibrate node** — delivered (see §16.1 for calibration).
- **Cross-cutting enabler**: MT integration tests run in CI by *generating* the
  synthetic `test12rr` fixture (as `test_phase_convention.py` scripts) instead
  of relying on gitignored raw LEMI/BGS data.

---

## 14. Current Plan — M2: Multi-level Decimation + FC Persistence (delivered 2026-08-29)

Status: **delivered**. This section is the historical plan for broadband MT;
§14.6–14.9 are the band-construction history and the current delivered default.

### 14.1 What aurora actually does (grounding)

- **Cascading decimation** (`prototype_decimate`): level *i* decimates level
  *i−1* by that level's integer `factor` (`scipy.signal.decimate`); level *i*'s
  rate = native / ∏(factors 1..i).
- **Per-level STFT**: prewhiten → truncate to clock-zero → sliding window →
  linear detrend → taper → FFT → recolor. Window `num_samples` is kept
  **constant across levels** (window duration grows at deeper levels → better
  long-period resolution).
- **FC generation vs use**: aurora either *loads* a persisted FC from mth5 or
  *computes* it, then optionally persists (`save_fourier_coefficients`).
- **Merging runs at the FC level** (`merge_stfts`): `xr.concat(stfts, "time")` —
  the across-runs merge, not the across-levels one.
- **Per-level TF + level merge** (`process_mth5.py`): each level estimated
  independently to a per-level `TTFZ`; a `TransferFunctionCollection` merges
  them, resolving duplicate periods (emptyUI's `MergeTFLevels`).
- **FC persistence format**: `from_xarray` requires a **`datetime64` time
  coordinate** (float seconds fails) — persisted FCs must carry the run's
  absolute start time.

### 14.2 Design decisions (all delivered)
- **D1** — fc `time` must be `datetime64`; `ts.run` carries an absolute
  `start_datetime` (normalized end-to-end so persistence round-trips).
- **D2** — window expressed as **`num_samples` constant across levels**
  (aurora-style), not seconds.
- **D3** — multi-level FC container `{level: xr.Dataset}` on one `fc` handle.
- **D4** — ComputeFC auto-persists each level to the workspace mth5 and returns
  the in-memory multi-level `fc` regardless; `mth5.fc.write` remains for
  explicit single-level control.
- **D5** — EstimateTF estimates each level and returns a **single merged `tf`**,
  with `merge_tf_levels()` shared by the `MergeTFLevels` node for advanced use.

### 14.3 Phased implementation (delivered 2026-08-29)
- **Phase 1** — FC persistence groundwork: absolute `start_datetime` through
  `ts.run`/fc, `workspace.mth5` with `persist_fc_level`/`load_fc_level`,
  ComputeFC auto-persists every level when an absolute start is available
  (warning otherwise). Round-trip fc→mth5→fc verified. (2026-09-05: the mth5
  read-back path is complete — `mth5.fc.read` enriches read-back fc metadata
  and `estimate_tf` normalizes mth5's `(time, frequency)` layout.)
- **Phase 2** — multi-level FC generation (`compute_fc_levels`): N cascading
  levels, constant `num_samples_window`; window param renamed to
  `window_samples`.
- **Phase 3** — per-level TF + merge (`estimate_tf_levels`, `merge_tf_levels`):
  deepest-wins duplicate resolution; EstimateTF returns merged `tf` + `tf_levels`.
- **Phase 4** — integration: end-to-end on LEMI (4 levels → 64 periods → EDI);
  fixture generated in CI.

### 14.4 Sequencing / banding spec (final, still current)

- **Periods per octave** (`periods_per_octave`, default 4) replaces
  periods-per-decade; bands sit on a log2 grid matching FFT/decimation windows.
- **Final banding spec** (2026-08-29): shortest period = **Fs/4** per level;
  longest period = **window_length/4** (N/(4·Fs)); band centers on a global log2
  grid anchored at 4/Fs, so consecutive levels land on the **same absolute
  periods** wherever they overlap. Levels therefore **overlap** (Fs/4 →
  window/4); the merge keeps the **deepest** level covering each period
  (deeper level = longer window = more cycles = better long-period resolution),
  dropping degenerate bands. This achieved aurora's "consistent frequency
  points between levels" without external band files.
- **TF table node** (`mth5.table.tf` "View TF") renders a TF as a table
  (period, rho_xy/yx, phase_xy/yx).
- **Decimation convention**: level *i* = decimated *i* times; `decimation_levels`
  default 0 = native rate only.
- **Deferred → M4 #9**: load an aurora-style band config file (exact center
  frequencies + bandwidths) rather than always deriving the table — superseded
  in practice by the l2u scheme (§14.9).

### 14.5 Worth exploring — Paradigm B (cross-decimation-level FC pooling) — M4 #10

At long periods the deepest decimation levels have few STFT windows, so per-level
TF estimates are statistically weak. Deeper levels also have **finer raw
frequency resolution** (more harmonics per Hz). Pooling all levels' FCs that
fall in a band into **one** robust regression recovers more samples precisely
where they are scarce. Existing codes (EMTF, BIRRP, aurora) define bands this
way but process each level's bands *separately* (Paradigm A); their band tables
are non-overlapping across levels. **No existing open-source code pools FCs
across decimation levels into one band** — Paradigm B is a novel extension
enabled by emptyUI's raw FC layer. Implementation sketch (if pursued): a global
band table (reuse the log-octave grid); per band, gather FC samples from *every*
level whose raw bins fall inside (weighted by level sample count); run the
existing aurora-backed `_solve_band` once per band. Value is highest at long
periods. References: Egbert (1997, EMTF); Chave & Thomson (2004, BIRRP); aurora
`Band` metadata + EMTF band setup files; mth5 FC layer.

### 14.6 Discrete, EMTF-style band construction (`band_style` switch — history; `"index"` retired 2026-09-09)

> **Superseded (2026-09-09, §14.8; further superseded 2026-09-11, §14.9):** the
> `band_style="index"` mode was retired after the comparative assessment (its
> per-level fallback role lives on inside the `"l2u"` scheme); `"log"` was
> renamed `"uniform"` (legacy value still accepted as an alias). Retained as
> history; reproducible from commit e2046df/bd75d14.

Key findings from the research (2026-09-08/09): aurora's default production band
table (`bs_256_26.cfg`) is a hand-authored, non-overlapping, gapless integer-bin
partition tuned for one (window, factor) configuration — no band-averaging taper
is applied by either classic EMTF or current aurora (aurora's own source carries
an unresolved TODO for it). emptyUI's `_period_bands()` instead partitions a
continuous log2 grid and afterwards tests which raw bins fall in each band —
near the long-period end several nominal bands can silently collect **zero** raw
bins in a way that depends on floating-point boundary coincidences.

The `_period_bands_indexed()` function auto-derived the EMTF-equivalent table
for any configuration by walking the integer FFT-bin index grid directly —
guaranteeing every raw bin in a level's reliable range is assigned to exactly one
band. A **zigzag defect** was found and fixed (2026-09-09): walking the bin-index
grid yields the same index sequence at every level, but identical indices sit at
frequencies differing by `factor**level`, so overlapping bands got different
period labels per level and the merge dedup never fired (interleaved
near-duplicates). **Fix: one GLOBAL boundary grid for the whole container**
(`_container_band_edges` → `_index_band_edges`), shared by all levels — band
labels identical across levels, so the exact merge dedup fires again and the
deepest level wins each overlap. The per-level walk survives only for standalone
single-level calls. Verified on BGS NO46+NX37 (window 256): merged curve smooth
again (50 interleaved points → 36 = one per global band).

### 14.7 Disjoint EMTF/aurora-style band table (`band_style="emtf"` — history)

> **Superseded (2026-09-11, §14.9):** replaced by the l2u scheme as the default
> frequency-grid method; `"emtf"` is still accepted as a legacy alias of
> `"l2u"`. Retained as history; reproducible from pre-2026-09-11 commits.

The merge-based modes deliberately let levels overlap and resolve post-hoc
(deepest wins) — smooth boundaries at the cost of redundant robust regressions.
Classic EMTF instead assigns every band to **exactly one** level — no overlap,
each frequency evaluated once — at the cost of small handover discontinuities.
Decoding the classic table (`bs_256_26.cfg`, N=256 → 128 bins/level, factor 4):
per-level usable bins ≈ 5..55; level 1 dense (~3.2 bands/octave); levels 2–3
repeat an **identical 6-band bin template** with frequencies ÷4 per level;
bands shrink to a **single bin** at each level's bottom; nearly disjoint across
levels (one small real overlap even in the hand-tuned file). The single rule
"walk centers down a ~3/octave geometric ladder snapped to whole bins; hand over
to the next level when the width would drop below ~1 bin" reproduces the entire
shape automatically for any (window, factor, ppo).

**Comparative assessment (2026-09-09, BGS NO46+NX37, window 256/512):** all four
configurations recover the same physical structure. Key findings: (a)
long-period tail smoothness is **dominated by ladder PHASE, not the band
method** (shifting a discrete ladder by half a band onto `log`'s phase made the
tail ~2.7× smoother; at window 512 the ranking reshuffled) — do not rank band
methods by tail smoothness; (b) the genuine replicable effect is **ownership
policy** (emtf-shallowest wins the window-poor tail clearly); (c) boundary nit:
the grids bottom exactly at the window/4 cutoff and the `(lo, hi]` convention
leaves the exact cutoff bin unassigned; (d) open question: isolated narrowband
dips (~4700/4100/3100/16384 s at window 256) — cultural noise vs sampling
artifact, needs a longer record. Provisional verdict superseded 2026-09-11:
neither scheme reproduces the official USArray grid, and the l2u scheme
(§14.9) was delivered as the single automatic method.

### 14.8 Consolidation: two band schemes (`"uniform"` + `"l2u"`, 2026-09-11)

Following the assessments above, the band-configuration surface was consolidated
to the two schemes that matter:

- **`band_style="l2u"`** — the **default** (§14.9): fixed reference template +
  log-uniform window-starvation coarsening + optional head clip. Replaced the
  auto-derived EMTF-style table on 2026-09-11 after the USArray verification
  (§16.3) showed it does NOT reproduce the official grid (42 vs 30 periods, no
  tail coarsening) while l2u reproduces the official strategy (counts
  6/6/6/6/4/2 at window 256, metrics matching the exact-table reference).
- **`band_style="uniform"`** — the uniform log2-grid partition (formerly
  `"log"`): merge-based across overlapping levels, deepest wins.
- Legacy values `"log"` (alias of `"uniform"`) and `"emtf"` (alias of `"l2u"`)
  are accepted so saved workflows keep running; the retired `"index"` mode
  raises a clear error.
- Removed: the `index` mode, the deepest-owner variant, `band_owner`, and the
  emtf table machinery (the per-level bin walk `_period_bands_indexed` survives
  as the `"l2u"` fallback for metadata-less fcs).

### 14.9 Frequency grid determination — the delivered default method (`band_style="l2u"`, 2026-09-11)

**Motivation.** §16.3's SDG31 verification exposed that the then-default
`band_style="emtf"` table does **not** reproduce the official USArray grid: 42
vs 30 periods (window 256, ppo 3 — 12/6/6/6/6/6 vs 6/6/6/6/4/2 per level),
uniform 1/3-octave widths everywhere (no tail coarsening), and a long-period
tail running to the reliability floor (52.4 ks) where the official stops at
18.7 ks. The coarsening prototype (merge-by-closest-center) was also worse than
the exact table (pass_rho 76% vs 83%) — it changed the ladder phase. Goal: an
**automatic** algorithm that mimics the official hand-crafted band list's
*strategy* for any (window, factor), so no hand-authored config is needed.
Setting the grid **exactly** as the official EDI stays an option
(`bands_override` / M4 #9) but is not flexible to different window lengths.

**Reference grid decoded (official `USArray.SDG31.2017.edi`, ppo ≈ 3).**

| level | fs (Hz) | bands | period range (s) | mean spacing (oct/band) | ≈ bands/octave |
|-------|---------|-------|------------------|--------------------------|----------------|
| 0 | 1 | 6 | 7.31 – 25.6 | 0.36 | 2.8 |
| 1 | 0.25 | 6 | 33.0 – 102.4 | 0.33 | 3.1 |
| 2 | 0.0625 | 6 | 132.1 – 409.6 | 0.33 | 3.1 |
| 3 | 0.015625 | 6 | 528.5 – 1638.4 | 0.33 | 3.1 |
| 4 | 3.9e-3 | 4 | 2259.9 – 7281.8 | 0.56 | 1.8 |
| 5 | 9.8e-4 | 2 | 11915.6 – 18724.6 | 0.65 | 1.5 |

Decoded findings (the "hand-crafted" structure the method mimics):
- **Head (L0–L3) = ~3 bands/octave** (ppo 3); **tail (L4–L5) ≈ half as dense —
  bandwidth doubled** (0.56–0.65 oct/band ≈ 2/ppo). The doubling starts where
  levels become window-starved, not at a fixed period.
- **The L1–L3 template repeats VERBATIM at every level** (window-128 native bins
  `[15-16],[12],[9-10],[7-8],[6],[5]`; at window-256 k = 31/24/19/15/12/10 —
  exactly 2× the classic `bs_256_26` level-2/3 template) — the official grid is
  NOT a per-level walk, it is a FIXED template applied to all levels. The single
  most important decoded insight.
- **Head clip at 7.31 s** (not Fs/4 = 4 s): data-driven (Kelbert et al. 2026:
  NIMS magnetic noise exceeds signal below 10–20 s; RR reliable only to ~7 s).
- **Tail clip at 18.7 ks** (not the ~65 ks L5 floor): window-starvation-driven —
  window counts (10-day, 1 Hz, window 256): 6673/1667/416/103/25/**5** → the
  official tail coarsening keeps ≥ ~2.5–6 windows/band.

**The delivered method — `_l2u_band_table` (`emptyui-aurora/_processing.py`),
the Estimate TF node default.** Two parts:

1. **Fixed reference template.** `_L2_TEMPLATE_128` = the official L1–L3
   template family, scaled to the container's window (×window/128, overlaps
   merged). The SAME template is applied to EVERY decimation level — periods
   scale by the cascade factor per level, exactly the official grid's structure.
   Window-rich levels (0–3) keep the full 6-band template untouched.
2. **Log-uniform window-starvation coarsening.** Per level, if any band's total
   samples (n_windows × band width) fall below `min_bins_per_band` (default 40),
   the level's grid is REBUILT as a coarser uniform table: `n − ⌊n/3⌋` bands,
   each widened to the minimal width whose total samples exceed the threshold,
   re-spaced uniformly in log period across the template's span (endpoints
   included), overlapping bands dropped until no overlap or ≤2 bands remain
   (extreme starvation: width shrunk to tile the span — bands stay disjoint).
   This reproduces the official tail's strategy while keeping the grid uniform
   and log-even.

`t_short` (default 0 = template head) drops bands shorter than the given period;
the official ~7.3 s clip is data-driven, so it stays a parameter.
`min_bins_per_band=0` disables coarsening. Levels without window metadata are
left unchanged; a container with insufficient metadata falls back to the
per-level bin walk (`_period_bands_indexed`).

**Why this over the alternatives (experiment history, condensed).** Reproducible
via `verify_sdg31.py`; rotation-corrected −5.7° metrics, T ≥ 15 s, 27 matched
periods:

| grid | rotated med/p90/pass | outcome |
|------|----------------------|---------|
| exact official table | 0.019 / 0.045 / 98% | reference (best possible) |
| **l2u, window 256 (delivered default)** | **0.020 / 0.071 / 93%** | best automatic; official counts 6/6/6/6/4/2 exactly; window-invariant grid (see note) |
| l2u, window 128 | 0.019 / 0.089 / 93% | same scheme, coarser bins |
| fixed template + MIN=3 coarsening (`l2`) | 0.022 / 0.070 / 94% | best non-uniform scheme; residual from hand-crafted L0/L4/L5 |
| emtf auto table (legacy) | 0.051 / 0.153 / 76% | superseded — did not reproduce the official grid |
| l2u bin-uniform placement | 0.024 / 0.131 / 85% | log placement wins |
| l2u ppo=2 / ppo=4 (128 w) | 0.031/0.095/91% · 0.024/0.092/93% | official density is ~3/octave; both sparser and denser lose |
| close-gap (L4→L5 handover) | degrades all ppo variants | the official's own L4→L5 handover is 1.64 — the coarsened 1.65/1.78 matches it; closing to the template's 1.29 hurts |

Key conclusions: (a) the window length matching the official's own (256/512)
matters more than the coarsening details — at window 256 the template bins ARE
the official's positions; (b) the L4→L5 "gap" is NOT a defect (the official has
the same sparse tail); (c) the official grid is **not a single uniform
template** — its L0 and L4/L5 ladders are hand-crafted variants, which l2u
reproduces by construction (fixed template head + window-starvation coarsened
tail). Note: the l2u grid is **window-invariant** (the template scales
×window/128, so windows 128 and 256 evaluate the SAME periods — the window only
changes FC bin resolution; the window-256 metric advantage comes from the finer
bins, not the grid). The earlier 0.018/0.058 window-256 result was produced by
the verify-script prototype accepting OVERLAPPING L5 bands at the 2-band floor;
the package fixes that (bands must stay disjoint), hence 0.020/0.071.

**Validation protocol status.** Unit tests (synthetic containers): template
scaling (window 128/256/512/64 — disjoint, non-degenerate); window-rich levels
keep the template; starved levels rebuilt uniformly; extreme starvation shrinks
to a tile; metadata-less levels untouched; `t_short` clips the head; node params
reach the grid (`test_band_style.py`). Robustness across window 128/256/512,
ppo 3/4 and short records is covered by the same mechanics.

**Non-goals.** Replacing `bands_override` (M4 #9); Paradigm B cross-level
pooling (M4 #10); auto-deriving the head clip from data (follow-up).

**Effort.** Medium — `_l2u_band_table`/`_l2u_template`/`_clip_head_bands` +
`estimate_tf_levels` pass-through + node params + tests; no estimator changes.

---

## 15. Publication-quality plot standardization (delivered 2026-09-06)

**Motivation.** The Display nodes were proof-of-concept level: no error bars
(TF errors never stored), no MT display conventions (raw phase, no y-limits
policy), components distinguished by color only, no site/processing metadata in
titles, low DPI (110), inconsistent style.

**Status: delivered** (P1–P4). Conventions implemented as decided below.
Reference figures: `SDG31b_H28coh.png` (USArray convention — primary style
reference) and `202511.PNG` (MTpy-style combined response — tipper Re/Im panel
layout). We do **not** build a combined plot — `Plot AppRes & Phase` and
`Plot Tipper` stay separate nodes.

### Conventions implemented (the spec)
- **TF error propagation (enabler)**: `_solve_band`/`build_tf` propagate
  `transfer_function_error` + `tipper_error` (std of each complex element — what
  mt_metadata's EDI writer squares into `.VAR`). RME/RME_RR populate
  `cov_ss_inv`/`cov_nn` (default `return_covariance=True`); OLS uses the
  classical `(XᴴX)⁻¹·σ²` covariance from residuals.
- **Error-bar math** (aurora/mtpy formulas): `σ_|Z| = σ_Z/√2`;
  `σ_ρ/ρ = 2·σ_|Z|/|Z|`; `σ_φ(rad) = σ_|Z|/|Z|`; tipper error used directly
  (Re/Im panels).
- **Phase display**: off-diagonal components (Zxy, Zyx) folded into (0°, 90°) by
  ±180° (aurora `RhoPlot`/USArray convention), ylim 0–90°, ticks every 15°;
  **diagonal components (Zxx, Zyy) keep signed phase in [-180°, 180°]**.
- **Style** (shared module `emptyui-mth5/plotting.py`; mirrored minimal
  `emptyui-core-nodes/_plotstyle.py` — core stays mth5-free): font sizes ~9–10
  pt, dpi ≥ 150, `bbox_inches="tight"`, PNG + optional PDF, boxed legend
  (marker+label, never color-only).
- **Axes (USArray convention)**: x = Period (secs) log10 (decade ticks, shared
  across panels); ρ_a = log10 (Ω·m); φ = linear (Degrees); tipper = Re/Im,
  ylim [-1, 1]; **equal per-decade scale** (one decade of period = same physical
  length as one decade of ρ_a — MT convention).
- **Markers & colors**: **XY = ○ (blue), YX = ✕ (red)** + distinct linestyles
  (colorblind-safe); XX = square, YY = diamond (proposal). Tipper: Re = ○ /
  Im = ✕ per component panel.
- **TF station metadata (prerequisite)**: ComputeFC attaches
  survey/station/run + lat/lon/elev to fc attrs; EstimateTF/MergeTFLevels pass
  them to `build_tf`, which sets `tf.survey_metadata`/`tf.station_metadata`/
  `tf.run_metadata` — the **single point of truth** for plot titles AND
  EDI/XML/ZMM export headers (verified: EDI carries `DATAID`/`LAT`/`LON`/
  `SURVEY` + `*.VAR` lines). `AlignRuns._crop_to_runts` carries the same
  metadata onto aligned runs.
- **Title & metadata**: EMTF/USArray-style `<station id><run id> [
  LON = …; LAT = … ]` plus a processing-provenance subtitle (estimator, remote
  reference, decimation levels, periods per octave, nomenclature); figure
  metadata via `fig.savefig(metadata=…)`.
- **Core plots** (`PlotTimeSeries`, `PlotSpectrogram`): datetime x-axis
  (ConciseDateFormatter; relative-seconds fallback), channel unit hints
  (mV/km electric, nT magnetic), station/run in title, unit-aware colorbar.

### Verification & references
Tests assert the conventions (off-diagonal ylim (0, 90), diagonal [-180, 180],
tipper [-1, 1], errorbar containers, ○/✕ markers, equal per-decade scale < 5%
tolerance, station in title, dpi ≥ 150); visual review against the USArray
reference. References: aurora 0.6.2 `TTFZ` (error-bar math + phase-fold
reference); mtpy `PlotMTResponse` (tipper layout; **not a dependency** — optional
dev extra for golden-image comparison, M4 #8); mt_metadata 1.0.10
`TF.impedance_error`/`TF.tipper_error`/`TF.transfer_function_error`.

### Decisions (2026-09-06)
- x-axis: **Period (secs), log10** — decided.
- Tipper: **Re/Im panels, ylim [-1, 1]** — decided; replaces |T|/phase panels.
- Diagonal phase: **signed, [-180°, 180°]** — decided.
- Markers/colors: **USArray convention** (○ XY, ✕ YX, blue/red); XX = square /
  YY = diamond (proposal).
- Open: add `mtpy` as a test-only dev extra for golden-image comparison (M4 #8).

---

## 16. M3 tasks — instrument calibration & coherence sorting (delivered 2026-09-07)

Two independent follow-ups identified while investigating the BGS NO46/NX37
short-period results. §16.3 is the subsequent USArray ground-truth verification
(2026-09-10).

### 16.1 General calibration framework + LEMI-417/424 calibration support

**Grounding (2026-09-07).** The LEMI logger outputs physical units (nT, µV/m)
assuming unity gain and zero phase; the deviation grows near Nyquist. The
manufacturer's `.chn` (LEMI-417-era) and `.def` (KMS-200 wrapper) files are FAP
files: header `'#FP' <NTF> <NCH>`, rows = 12 frequencies × 5 channels, complex
response as modulus + argument (degrees), first column = frequency (Hz). **No
public parser exists** for either format; mt_io/mth5 accept only per-channel
JSON, and `LEMICollection.get_calibrations` globs only `*.json` — manufacturer
files were silently ignored. Empirical check on the BGS pair: the short-period
ρ_a collapse is primarily the **instrument noise floor** (LEMI-424 is a
long-period system, designed T=20–100,000 s); applying the `.chn` file (flat
clamp below 0.25 Hz) scales ρ by ~×6.25 and shifts phase ~±30° — a real but
modest correction that cannot recover sub-noise-floor periods.

**Implemented (2026-09-07).** `emptyui-mth5/calibration.py`:
`parse_fap` (`.chn`/`.def`), `parse_json_calibration` (mt_io format),
`build_fap_filters` (per-channel `FrequencyResponseTableFilter`, units
nanoTesla / mV·km by channel prefix), `attach_filters` (mt_timeseries style —
`run.filters` + per-channel `attrs["filters"]` in the AppliedFilter format that
survives an MTH5 round trip; channels without a filter get a unity coefficient
filter so `RunTS.calibrate()` passes temperature channels warning-free),
`find_calibration_file` (auto-pickup, station-pair files preferred),
`load_calibration` (dispatch by extension). `Load LEMI-424`: `calibration_path`
param + `extend_flat`; filters attached **before** `from_runts`, so the
calibration is archived with the run. New `mth5.calibration.load` node feeds the
explicit path of `mth5.calibrate`. Below-table policy: **flat clamp** (first-row
modulus, zero phase, extended to 0.001·f_min) — log-extrapolation was worse.
Verified on BGS NO46 LEMI-417: ρ_a ≈ ×6.25, phase ≈ 5–15° — consistent with the
empirical check. Open: absolute-gain verification (M4 #6); `.def` parsing is
heuristic (no sample in repo — clear error if the FAP block is not found).

### 16.2 EMTF-style coherence sorting (filter low-coherence FCs)

**Grounding (2026-09-07).** aurora 0.6.2 has **no sample-level coherence
sorting**: the robust M-estimator computes squared coherence but only **logs**
it; its coherence helpers are band-widening only. Measured on the BGS pair:
E–B coh² ≈ 0.01–0.1 at T=4–16 s (signal below the noise floor — no recovery),
0.03–0.3 at T=11–30 s (transition zone — sorting can help), 0.6–0.9 at T>30 s.
Grounding for the method: Egbert (1997, EMTF) and Chave & Thomson (2004, BIRRP)
use coherence-based identification/downweighting of contaminated samples; not
implemented in the pinned aurora, so added in emptyUI (pre-regression, external
to aurora's residual-based M-estimator).

**Design (2026-09-07, as delivered).** A dedicated node
**`aurora.process.coherence_sort`** ("Coherence Sort"), placed between
`Compute FC` and `Estimate TF`, in `emptyui-aurora` (reuses the same band
geometry as `Estimate TF`; `periods_per_octave` exposed, default 4).

- **Two-FC design (user decision)**: accepts **both** `fc` + `ref_fc` inputs and
  outputs (mirroring `Estimate TF`) because remote reference *replaces* the local
  magnetic channels: single-FC mode uses local outputs' multiple coherence on
  the **local** magnetic inputs; two-FC mode uses the local outputs' coherence
  on the **remote** magnetic inputs **combined** (minimum, fixed — no
  `coherence_source` param) with the local↔remote magnetic transfer coherence,
  since the reference can only clean the local H to the extent the two magnetic
  fields are coherent. Both criteria are reported in the diagnostic table.
  The two FCs must share identical time/frequency grids (validated, same errors
  as `Estimate TF`). The shared per-row weight is baked as √w into **both**
  output FCs (weighted-LS semantics — one weight per observation row), making
  remote-reference row weighting exactly correct.
- **Params**: `coherence_min` (0.25), `coherence_max` (0.98 — perfect multiple
  coherence can be a noise signature: few DOF/overfit or coherent noise, so
  coh ≈ 1 must not automatically mean "good data"), `mode` (`weight` default /
  `drop` / `off`), `coherence_trim` (False — band-edge trimming: set bins whose
  band median coherence is below `coherence_min` to NaN with a warning),
  `periods_per_octave` (4).
- **Statistic**: per-window **multiple coherence** over the frequency-bin
  ensemble inside one STFT window, per level; sparse bands fall back to the
  minimum pairwise coherence; degenerate windows score 0 (conservative); per
  output channel the minimum across outputs is used. Degenerate bands
  (n_bins < n_in + 1) fit exactly so their coherence is trivially 1.0 — the
  `coherence_max` rule is **not** applied there (implemented 2026-09-07),
  otherwise the longest-period bands would be dropped unconditionally.
- **Semantics note (Batch B)**: this is a per-window broadband (bin-ensemble)
  QC heuristic — **not** EMTF/BIRRP's window-ensemble coherence per frequency.
  The two agree qualitatively but not numerically; a true EMTF-style
  per-frequency variant is M4 #7. The node description states this.
- **`Estimate TF` change (small, generic)**: skip non-finite FC samples when
  gathering band rows — required by `drop`/`coherence_trim`, and gives correct
  RR semantics for free (a row is used only if both local and remote FCs are
  finite; row pairing stays aligned because the time grid is never dropped).
- **Diagnostics**: `table` output with per-period median coherence for each
  criterion, retained/total window counts, applied weight range. A
  coherence-vs-period plot is a later add-on (M4 #1).
- **Implementation**: `_simple_coherence`, `_multiple_coherence`,
  `_coherence_weight`, `coherence_sort_fc(s)` in `emptyui-aurora/_processing.py`;
  `CoherenceSort` node + `_fc_io_channels` in `process.py`.
- **Verification**: synthetic FCs with injected low-coherence windows (weights
  exclude/down-weight and recover the true impedance; weight vs drop compared;
  upper bound tested with overfit bands); two-FC mode synthetic stations (a
  noisy local H with high E–H_remote coherence must be **retained** for RR; low
  E–H_remote or low H_local–H_remote must be dropped); real BGS RR workflow
  (T=11–30 s improves, long periods unchanged; `coherence_trim` removes
  noise-floor bands with a warning). Defaults are loose — BGS long-period
  results must remain stable.

**Addendum (2026-09-11, review §18.2):** `min_bins_per_band` and `t_short` are
now exposed on the node (previously the sort hard-coded 40 / had no head clip),
so its banding can match a customized Estimate TF exactly — a test asserts
set-identical band centers under matching params. The statistic itself remains
the per-window broadband heuristic (M4 #7); Estimate TF should inherit the
banding parameters from an upstream Coherence Sort when one is connected
(§18.2 design note).

### 16.3 USArray verification: SDG31 vs the official EMTF (SPUD) TF (2026-09-10)

**Goal.** Pin the emptyUI MT processing chain (TS → FC → TF) against a
ground-truth transfer function before relying on it: process USArray station
**4P.SDG31** (Stratford Slough, SD) with emptyUI and compare with the official
EMTF result for the same site, using **4P.SDH28** as the remote reference (the
official processing used `REMOTESITE=SDH28`). Reproducible via
`examples/data/lims-usarray-test/verify_sdg31.py`.

**Data** (aurora operate-aurora recipe, `build_mth5.py`): 1 Hz channels
LQE/LQN (100 m dipoles) + LFE/LFN/LFZ (NIMS), 2017-10-16→10-26 (SDG31) and
2017-10-12→10-31 (SDH28); stored as standard-named ex/ey/hx/hy/hz in raw
counts; archived responses turn them into mV/km and nT via `Calibrate`.
Processing inferred from the official EDI grid: **factor 4, 6 decimation levels,
~3 bands/octave head, coarser tail**; band centers are exact dyadic FFT-bin
values and the L1–L3 template is exactly **2× the classic `bs_256_26`
level-2/3 index template**. Window 256 is **wrong** for this site (dilutes the
intermittent short-period signal); window 128 with 2-bin bands reproduces the
official centers to ≤0.05%, which is why the exact-grid comparison works. For
apples-to-apples the band table is reconstructed **exactly** from the official
band-center frequencies (`exact_official_bands`; 6,6,6,6,4,2 per level); the
§14.9 default `l2u` scheme reproduces the official strategy automatically.

**Result (verified 2026-09-10).** For the reliable band **T ≥ 15 s** (27 matched
periods; exact official grid, robust RR):
- Un-rotated (geomagnetic frame): median |Δlog₁₀ρ| = 0.06 (ρ within 15%),
  p90 = 0.11, 83% within the 26% gate; median |Δφ| = 1.6°, p90 = 3.6°, 98%
  within 10°; coherence sorting (drop, trim) improves pass_rho to 91%.
- **Coordinate-frame rotation (the xy-down / yx-up offset)**: MT acquisition is
  in the **geomagnetic frame**; TFs must be **rotated to true north before
  inversion**. The USArray data are geomagnetic; the official EMTF product was
  **rotated −5.7° to true north** (EarthScope EMTF page: "Declination: 5.7
  degree", "Angle To GG North: 0.00"). The IGRF model value (4.26°, in the mth5
  archive) is **not** the relevant declination here — both the EDI and the
  product page record 5.7°. Comparing un-rotated results shows exactly the
  expected signature (ρ_xy ~10–20% low, ρ_yx ~5–15% high from ~100 s on) with a
  consistent best-fit frame rotation of **−5.85°** (median; std 1.7°; opposite
  sign 1.9× worse; 90° swap 100× worse; diagonal ratios ≈ 1 → no gain error) —
  matching the documented −5.7° (residual 0.15° is banding noise).
- **Rotated by −5.7°** with the standard convention
  (`Z' = R(angle)·Z·R(angle)ᵀ`, `R(angle) = [[cos α, sin α], [−sin α, cos α]]`,
  positive = clockwise from true north; magnetic→true north ⇒
  `angle = −declination`): median |Δlog₁₀ρ| = **0.019** (ρ within ~4.5%),
  p90 = 0.045, 98% within gate; median |Δφ| = 0.69°, p90 = 2.95°, 100% within
  10°. (2026-09-10 correction: a script bug had mislabeled a later variant as
  this "rotation-corrected" comparison; the numbers above are the genuine
  exact-grid results.)
- **TF rotation node** (`mth5.tf.rotate`) — **delivered 2026-09-11** (see §11):
  rotates a TF by an arbitrary angle or to true north via the station
  declination; `Import TF` parses the EDI declination (explicit field, else Hx
  azimuth) with a fallback to the `angle` parameter; the verify script's
  `render_appres_figure` now uses the package rotation.
- **T < 15 s is noise-limited in the 1 Hz archive**: E/H multiple coherence²
  drops to 0.001–0.05 (telluric signal at 7–12 s is intermittent — only ~2% of
  windows coherent) and ρ collapses (median |Δlog₁₀ρ| ≈ 1.5 with the 128 s
  window). A **64-sample window + coherence sorting recovers 9–15 s to within
  ~40%** (median |Δlog₁₀ρ| = 0.15); 7.3 s stays degenerate. The official's own
  variance is 40× larger there, and the short-period (40 Hz) data are not in
  the EarthScope FDSN archive — the 1 Hz E noise floor is the physical limit.
  Verdict: pin the 15 s–18.7 ks band (or the 64 s-window short-period recovery).
  **Paper confirmation (Kelbert et al. 2026, Rev. Geophys.,
  10.1029/2024RG000850)**: the SS low-amplitude bias below 10–20 s is documented,
  matching our finding that RR is unbiased where SS is biased low. Their
  short-period products additionally used **MMT (multi-station MT)** — which
  explains why the official grid is clean at 7–15 s while our **single** remote
  (SDH28, itself a 1 Hz NIMS site) cannot be; MMT is a **long-term M4
  nice-to-have** (§11 #18). Pin the basic processing chain first.

**Bugs found & fixed by this verification.**
1. **Magnetic nT→A/m conversion was skipped for coil channels**
   (`magnetic_unit='h'`): mth5 coil archives are in nT (like fluxgate B), but
   only the fluxgate path (`'b'`) divided by μ0 — Z off by μ0·1e9 ≈ 1.26e3, ρ
   off by ~1.6e6. Fixed via units-aware conversion (`_magnetic_to_h`): nT units
   → convert; explicit A/m (or unknown, legacy synthetic) → leave. Regression
   tests in `test_magnetic_units.py`.
2. **`Align Runs` dropped channel units**: the aligned RunTS rebuilt ChannelTS
   pieces without units metadata, so downstream Compute FC skipped the
   conversion (same 1.6e6 error, now silently through the RR path). Fixed:
   `_crop_to_runts` preserves per-channel units; the inter-station coherence
   guard now checks the horizontal magnetic channels (the vertical field is
   naturally incoherent between stations — the old check false-alarmed on `hz`).

**Follow-up (M4).** Expose the exact band table as a node-level option
(§11 #9) instead of the script-level `bands_override`; short-period verification
needs the 40 Hz data (outside EarthScope FDSN); MMT multi-station estimator
(§11 #18).

---

## 17. Remote deployment — embedded mode (designed 2026-09-08; contract items 1–6 delivered 2026-09-08)

emptyUI is **not** a standalone multi-user app. The target remote scenario is
emptyUI's FastAPI backend (+ built web frontend) embedded **as a module** into a
platform that already provides authentication — one backend instance **per user
or team**, mounted under a sub-path of the platform's URL space. The design goal
is a small **compatibility contract**, not an auth subsystem.

### 17.1 Trust model & division of responsibility

| Concern | Owner |
|---|---|
| Authentication (certs, SSO, session cookies), TLS termination | platform gateway |
| User/team management, authorization, routing, rate limiting | platform |
| Data provisioning (original time series, archives — system of record) | platform |
| emptyUI process lifecycle (one per user/team) | platform |
| Identity-blind execution, confined file access, proxy-compatible API/WS, retention & limits | emptyUI |

**Identity-blindness contract** — emptyUI contains no auth code and never
enforces identity. The gateway authenticates every request (including WebSocket
upgrades — browsers cannot set `Authorization` headers on WS, so gateway-side
cookie/cert auth is the natural fit; same-origin serving makes this seamless).
Identity headers (`X-Forwarded-User`) may be logged but are not interpreted.
Per-user isolation is achieved by construction (one process per user/team);
process-global state — runs, artifacts, the cross-run cache — is therefore
per-user without any ownership metadata.

### 17.2 Storage realms (read-only inputs vs writable outputs)

- **Data root (read-only)**: the platform provisions original data (CSVs, MTH5
  archives, calibration FAP files) into a configured directory. Input path
  params and the `/api/files` picker resolve **only inside** the data root;
  nodes never write here.
- **Output workspace (writable)**: per-run workspaces hold artifacts, plots and
  TF/export files — served through the artifact endpoints; nothing outside the
  output workspace is writable.

Local development keeps today's behavior (data root defaults to
unrestricted/cwd); confinement activates only in remote mode. Manifests mark
path-kind params as `in`/`out` so a single resolver confines inputs to the data
root and forces outputs into the workspace.

### 17.3 Serving, mounting, proxies

- **Single origin**: `create_app` optionally serves the built `dist/` bundle —
  no CORS in remote mode (the dev `*` middleware is replaced by a locked-down /
  disabled configuration); session cookies flow naturally to REST and WS.
- **Sub-path mount**: the app must be mountable under e.g. `/tools/emptyui/` —
  FastAPI `root_path` plus a configurable frontend API base; the WS URL derives
  from that base.
- **Proxy compatibility**: standard `X-Forwarded-*` handling; WS upgrade
  proxying with generous idle timeouts (runs are long-lived); the existing
  polling fallback is a deliberate asset — the UI keeps working when WS
  connections are dropped by intermediate proxies.

### 17.4 Operational policy (config knobs)

- **Workspace retention**: optional TTL + periodic cleanup coupled with artifact
  GC; expired workspaces invalidate their artifact entries; cleanup only removes
  worker-complete terminal runs. Disabled TTL preserves local-dev behavior.
- **Max concurrent runs** per instance (queued or HTTP 429), bounding memory/CPU
  in a shared platform; default unlimited for local dev.
- **Config surface**: env-first knobs (`EMPTYUI_DATA_ROOT`, `EMPTYUI_SERVE_DIST`,
  `EMPTYUI_BASE_PATH`, `EMPTYUI_WORKSPACE_TTL`, `EMPTYUI_MAX_CONCURRENT_RUNS`)
  mirrored as `create_app(...)` keyword arguments for module embedding without
  env coupling.

### 17.5 Explicit non-goals
- Authentication/user management inside emptyUI (the gateway's job).
- Multi-user shared process / horizontal scaling — scale out is "more
  instances", not "a bigger instance".
- File-upload API — data is provisioned by the platform; uploads are a possible
  follow-up.
- Database persistence for runs/artifacts — the file-based workspace (§3) stands
  until run history demands more.

### 17.6 Compatibility checklist (2026-09-08 — items 1–6 delivered)

1. ✅ Data-root confinement: `EMPTYUI_DATA_ROOT` + `create_app(data_root=…)`;
   `/api/files` clamped to the root; `path_kind: "in"` params validated inside
   the root; `path_kind: "out"` params resolve into the per-run workspace.
   Confinement activates only when a data root is configured.
2. ✅ Same-origin serving + CORS: `EMPTYUI_SERVE_DIST` / `create_app(serve_dist=…)`;
   `EMPTYUI_CORS_ORIGINS` / `create_app(cors_origins=…)` replaces the dev `*`
   default (empty tuple disables the middleware).
3. ✅ Sub-path support: the frontend derives its API base at runtime from the
   document location (one segment below the app root) — works at `/`, under a
   sub-path mount and in dev with no rebuild; `root_path` plumbing for module
   embedding.
4. ✅ Workspace TTL cleanup + artifact GC — `EMPTYUI_WORKSPACE_TTL` /
   `create_app(workspace_ttl_seconds=...)`; disabled when unset or ≤ 0.
5. ✅ Max-concurrent-runs guard — `EMPTYUI_MAX_CONCURRENT_RUNS` /
   `create_app(max_concurrent_runs=...)`; unlimited when unset or ≤ 0; excess
   submissions receive HTTP 429.
6. ✅ Config plumbing + deployment guide: `emptyui/config.py` (env-first, kwargs
   win) + `docs/deployment.md` (gateway contract, nginx sub-path/WS example,
   data provisioning, module embedding).

---

## 18. Code review record (2026-09-11) — findings, fixes, and follow-ups

A full codebase review (framework, MT plugins, frontend) verified the
delivered milestones and surfaced a set of defects, all fixed with
regression tests the same day unless noted. Status of record:

### 18.1 Delivered fixes (batches 1–4)

1. **Rotate TF registration** — `mth5.tf.rotate` was implemented and tested
   but never registered in `emptyui-mth5/plugin.py` (unreachable from the
   UI); the "delivered 2026-09-11" claims were premature by one wiring line.
   Registered + registration test; the dead duplicate `register` in `tf.py`
   removed.
2. **Output-path confinement escape** — `PathPolicy._resolve_output` joined
   relative `path_kind="out"` values without an under-workspace check, so
   `../…` escaped the run workspace (§17.2). Inputs were already correct;
   outputs now normalize + validate identically.
3. **Registry robustness** — `rebuild()` swaps the node table atomically
   (scratch registry → swap), so a run starting during a plugin toggle sees
   the old or new table, never a partial one (the "serialized by the app
   lock" docstring claim was wrong); disabled plugins register into a
   scratch registry during discovery, so a node-type collision with an
   enabled plugin no longer crashes order-dependently (re-enabling still
   surfaces the conflict loudly); unknown node types fail the run with a
   clean `ExecutionError` instead of a bare `AttributeError`.
4. **Run lifecycle** — terminal run status is monotonic (`_merge_summary`):
   a cancel racing a fast-completing run can no longer flap
   cancelled→completed; WS clients disconnect in a `finally` on any exit
   path.
5. **Processing guards** — `merge_tf_levels` tolerates level payloads
   without `Z_err` (zeros) and broadcasts scalar `Z_err`; the ComputeFC NaN
   guard is scoped to the actually-used in/out channels (a NaN in an unused
   auxiliary channel — e.g. LEMI temperature — no longer hard-fails the
   node; NaN in a used channel still errors).
6. **Align Runs coherence guard** — the long-period (64–1024 s) diagnostic
   resolved magnetic channels by literal `hx/hy` names and therefore never
   fired on LEMI data (`bx/by`) — exactly the data it was built for. It now
   resolves both stations' channels through the nomenclature (per-station
   keyword inference), including mixed standard/LEMI pairs.
7. **Coherence Sort banding consistency** — the sort hard-coded
   `min_bins_per_band=40` and had no `t_short`, so it could band
   differently from a customized Estimate TF (see §16.2/§18.2). Both
   params are now exposed on the node with the estimator's defaults, and a
   test asserts set-identical band centers between the sort's diagnostics
   and `estimate_tf_levels` under matching params.
8. **Absolute-ohm anchor test** — the prior known-Z regressions used
   abs=0.5 tolerances (a tens-of-percent global FC-normalization error
   would pass). New analytic test: with E = Z·H per sample, every STFT bin
   satisfies the relation individually and common-mode window scaling
   cancels in Z, so recovery is exact up to noise (~1e-4 relative measured);
   the test asserts ~1% — any FC-normalization regression now fails loudly.
9. **Frontend** — radio/datalist DOM ids scoped per node instance;
   workflow load validates the §6 envelope (format/version) and prunes
   stale edges (missing nodes/ports); `meta.name` round-trips (no more
   permanent `untitled.emptyui.json`); events arriving in the
   reset→setRunId window are buffered and flushed instead of dropped (poll
   dedupe key includes `cached`); the inherit auto-fill is idempotent (an
   inherited value equal to the manifest default no longer re-triggers the
   effect — the update loop is guarded by an actual-change filter);
   "Run to here" disabled while a run is active.

### 18.2 Coherence Sort vs. EMTF — two distinct inconsistencies

The review separated two issues that the "not EMTF" note (§16.2) conflated:

- **Banding mismatch (fixed, item 7 above)** — a parameter-plumbing bug,
  not a statistics question.
- **Statistic mismatch (open, M4 #7)** — emptyUI scores a *per-window*
  multiple coherence over the band's frequency-bin ensemble; EMTF/BIRRP
  compute coherence *per frequency* over the window ensemble. The two
  statistics are transposed axes of the same bins×windows cube: they agree
  qualitatively (BGS verification: T=11–30 s improves, long periods
  stable) but cannot be reconciled by parameter tuning — a per-frequency
  variant needs a different algorithm shape (per-frequency masks instead
  of per-window √w weights) and a decision on how per-frequency masks
  interact with aurora's RME_RR row pairing. aurora 0.6.2 itself has no
  sample-level coherence sorting at all (it only logs coherence), so the
  node is additive, not a deviation.

**Design note (follow-up):** Estimate TF should **inherit its banding
parameters from an upstream Coherence Sort when one is connected**
(`band_style`, `periods_per_octave`, `min_bins_per_band`, `t_short`) via
the §4 chain-walking `inherit` mechanism (downstream auto-fill from the
connected sort node; explicit user choices on Estimate TF are never
overridden). Today the two nodes are consistent by convention (identical
defaults + matching params); the inherit wiring makes the consistency
structural. Deferred as a small M4 item — do it together with (or before)
M4 #7, since the per-frequency variant changes the parameter surface.

### 18.3 Open items (deferred — need a design decision)

- Default retention policy (TTL / cache size bounds) — currently unbounded
  in local mode by design.
- Electric-unit scaling is unconditional (×1e-6 mV/km→V/m); payloads
  already in V/m are mis-scaled, and `e_dipole_length_m` on mV/km data
  double-scales — needs a unit-aware convention (the magnetic path checks
  payload units and is the model).
- Plugin-embedded node viewers (the frontend special-cases
  `core.display.preview_plot`/`view_table` bodies) — needs an extension
  mechanism in the manifest schema.
- `remote_reference` disabling is string-sniffed in `MTNode.tsx`
  (duplicating the backend cascade) — remove once the cascade covers the
  disabled UX.
- RunTS start == 1980-01-01 epoch sentinel ambiguity
  (`runs._run_time_info`); `SpectraToFCGroup` writes decimation level "0"
  with sample-rate default 1.0.
