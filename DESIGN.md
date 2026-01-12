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
- **`enabled_when`** (per param, added 2026-09-12): conditional-UX rule — the
  widget is editable only when the named sibling param currently holds one of
  the listed values:
  ```python
  Param("remote_reference", "boolean", default=False,
        enabled_when=("estimator_engine", ("Robust", "RME", "RME_RR")),
        cascade={True: (("estimator_engine", "Robust"),)}),
  ```
  Complements `cascade` (which *sets* values; `enabled_when` *gates* editing).
  Rendering-only — the executor never enforces it (nodes validate their own
  params). Replaced the frontend's string-sniffing of `estimator_engine`
  (review §18.3), which was node-specific logic in generic code.
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
| 1 | **Plot Coherency + Plot Phase Tensor nodes** | High | Low | ★★★★★ | Standard MT QC/interpretation products; data already flows through `fc`/coherence diagnostics/`tf` — plot code patterns exist (mth5/plotting.py). **Delivered 2026-09-12** (§20): `aurora.plot.coherency` (fc-direct, shared statistic with Coherence Sort) and `mth5.plot.phase_tensor` (Caldwell ellipses, skew-colored). |
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

**emptyUI deviations from the aurora STFT chain (documented 2026-09-13).**
`ComputeFC` implements **taper → FFT only** (Hamming, 50% overlap,
`boundary=None`, `detrend=False`); the remaining aurora steps are
deliberately absent:

- **No prewhitening / recoloring** — aurora 0.6.2 defaults to
  `stft.prewhitening_type: "first difference"` (a time-domain first
  difference, inverted in the frequency domain by `÷ j2πf`;
  `mth5.processing.spectre.prewhitening`). Its EMTF-style ARMA
  prewhitening (`pwset.cfg`, aurora Issue #60) is only a placeholder
  upstream — nothing beyond the first difference exists in the pinned
  aurora either. emptyUI relies on the Hamming taper's sidelobe
  suppression instead; leakage from strongly colored (≈1/f) natural fields
  is therefore higher than aurora's at short windows. Ratio-based TF
  products target the same estimator, but persisted FC amplitudes are not
  directly comparable to aurora-persisted FCs.
- **No per-window detrend** — aurora applies
  `scipy.signal.detrend("linear")` per window, keeping window-level trends
  out of the low-frequency bins. emptyUI detrends globally via the
  upstream `Detrend` node only; windowed trends leak into low-frequency
  bins via window sidelobes.
- **No clock-zero truncation** — emptyUI decimates the (aligned) runs
  directly; aurora truncates to a run-config clock zero first.
- **FC normalization** — scipy 'spectrum' scaling (÷ win.sum()); aurora
  additionally applies a linear spectral-density calibration factor. All
  ratio-based products (Z, tipper, coherence, OLS σ) are scale-equivariant
  and unaffected; persisted FC amplitudes and spectrogram dB are in
  emptyUI's own convention.
- **Window family** — Hamming with 50% overlap vs aurora's default boxcar
  with 25% overlap (128/32).

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
  points between levels" without external band files. Implementation note
  (2026-09-13): the `uniform` scheme's midpoint-edge membership extends the
  first/last band half a bandwidth past these anchors — bins up to
  ≈2^(1/ppo)·Fs/4 enter the shortest band (the `l2u` template table is scaled
  from the same anchors).
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
  `σ_ρ/ρ = 2·σ_|Z|/|Z|`; `σ_φ(rad) = σ_|Z|/|Z|`; tipper per Re/Im panel:
  `σ_component = σ_T/√2` (component std under the isotropic complex-error
  assumption — the same √2 convention as `σ_|Z|`; doc corrected 2026-09-13
  to match the delivered code, which divides by √2).
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
  band median coherence is below `coherence_min` to NaN with a warning;
  trimming applies only in `weight`/`drop` modes — `mode=off` means no
  coherence-based modification at all, warning added 2026-09-13),
  `periods_per_octave` (4).
- **Statistic**: per-window **multiple coherence** over the frequency-bin
  ensemble inside one STFT window, per level; sparse bands fall back to the
  minimum pairwise coherence; degenerate windows score 0 (conservative); per
  output channel the minimum across outputs is used.   statistic falls back to the minimum pairwise coherence there — which is
  **not** trivially 1, so the lower bound still applies (conservative);
  only the upper rule is capped (`coh ≤ coherence_max − ε`,
  implemented 2026-09-07; rationale corrected 2026-09-13 — the original
  "fits exactly, so coherence is trivially 1.0" wording described the
  regression, not the computed statistic).
- **Semantics note (Batch B)**: this is a per-window broadband (bin-ensemble)
  QC heuristic — **not** EMTF/BIRRP's window-ensemble coherence per frequency.
  The two agree qualitatively but not numerically; a true EMTF-style
  per-frequency variant is M4 #7. The node description states this.
- **Known statistical bias (documented 2026-09-13, uncorrected)**: the sample
  multiple coherence treats the band's n_bins × n_windows rows as independent,
  but windows overlap 50% and bins within a window share that window's noise
  realization, so the effective DOF is far smaller and the sample coherence is
  biased high by ≈ n_in/(N_eff − 1) (~5% at the default `min_bins_per_band=40`;
  no Egbert-style DOF correction is applied). Consequences: long-period bands
  pass `coherence_min` more easily than the nominal threshold implies, the
  "coh ≈ 1 is suspicious" rule fires inconsistently, and OLS σ² (dof =
  n_obs − n_in, §15) is underestimated by the same DOF inflation. Correcting
  or replacing the statistic is part of M4 #7.
- **`Estimate TF` change (small, generic)**: skip non-finite FC samples when
  gathering band rows — required by `drop`/`coherence_trim`, and gives correct
  RR semantics for free (a row is used only if both local and remote FCs are
  finite; row pairing stays aligned because the time grid is never dropped).
- **Diagnostics**: `table` output with per-period median coherence for each
  criterion (E–H and, two-FC, the magnetic transfer coherence, plus their
  min), retained/total window counts, and a `trimmed` flag (the
  "applied weight range" wording from the original spec was never
  delivered — weight range is not in the rows). A
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
- **Project directory (stable, designed §21)**: a user-designated writable
  directory holding long-lived, relocatable state — the `.emptyui-cache/`
  disk cache. Unlike the ephemeral per-run workspaces it survives restarts
  and travels with the data folder; `EMPTYUI_PROJECT_DIR` designates it
  (unset → disk cache disabled, in-memory tier unaffected).

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
9. **Frontend** — radio/datalist DOM ids scoped per node instance; workflow
   load validates the §6 envelope (format/version) and prunes stale edges
   (missing nodes/ports); `meta.name` round-trips (no more permanent
   `untitled.emptyui.json`); events arriving in the reset→setRunId window
   are buffered and flushed instead of dropped (poll dedupe key includes
   `cached`); the inherit auto-fill is idempotent (an inherited value equal
   to the manifest default no longer re-triggers the effect — the update
   loop is guarded by an actual-change filter); "Run to here" disabled
   while a run is active.
10. **Post-review browser-testing fixes (2026-09-12)** — (a) **string
   number params**: inheritable number params render as free-text
   comboboxes whose raw string entries ('10') reached node code as str
   and broke arithmetic (`'10' <= 0` TypeError on `t_short`); the
   executor now coerces number/integer (+ numeric select) params to the
   manifest's declared type (empty string → None, clean error naming the
   param) — healing old saved documents — and the combobox coerces
   client-side too; (b) **phase-tensor ellipses rendered flattened**:
   data-space ellipses on the log-period axis are log-warped and their
   rotation mixes the huge data-space half-width into the y-extent;
   ellipses are now constructed in *display* space and mapped back
   (`_ellipse_outline`), preserving aspect/orientation exactly.

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
overridden). — **Delivered 2026-09-12**: `EstimateTF.inherit` declares the
mapping on the `fc` input; e2e-verified (sort's band style flows into the
estimator's unset params). The statistic itself remains the per-window
broadband heuristic (M4 #7); do the inherit-aware parameter surface for
the per-frequency variant together with M4 #7.

### 18.3 Open items (deferred — need a design decision)

- Default retention policy (TTL / cache size bounds) — currently unbounded
  in local mode by design (user decision 2026-09-12: TTL only for now; a
  portable local disk cache is a later design).
- ~~Electric-unit scaling~~ — **decided & delivered 2026-09-12, see §19**
  (attr-aware electric path, modelled on the magnetic path).
- Plugin-embedded node viewers — **decided 2026-09-12 (user)**: keep the
  ComfyUI paradigm literal. Producing Display nodes stay compact (port
  rows) and emit a `plot` port; previewing is a user choice via the
  `Preview Plot`/`Save Plot`/`View Table` utility nodes, whose viewer
  bodies remain deliberately special-cased (frontend literals accepted by
  design — the two utility nodes are the framework's own viewer widgets).
  Generalizing viewers (by output port type or shipped components) is
  parked until table/viewer workflows mature and the inline-settings
  interaction (axis ranges, time windows on plot nodes) is thought
  through.
- ~~`remote_reference` string-sniffing~~ — **delivered 2026-09-12**: the
  manifest-declared `enabled_when` rule (§4) replaced it.
- ~~RunTS start == 1980-01-01 epoch sentinel ambiguity;
  `SpectraToFCGroup` writes decimation level "0" with sample-rate default
  1.0~~ — **delivered 2026-09-12 (metadata-hygiene bundle)**: a shared
  placeholder-epoch helper (`UNKNOWN_EPOCH_STARTS` = 1980 mt_timeseries
  default + 1970 Unix default, common in digitized paper records) is used
  by `runs._run_time_info` AND aurora's `_start_datetime` (previously
  inconsistent — fc time axes could carry fake 1980 stamps). **Old records
  are never rejected**: a placeholder-epoch start is warned about and
  degraded (no absolute-time alignment/persistence), and genuinely old
  stamps (1950s digitized data) are preserved as-is. `Write FC` gains a
  `decimation_level` param and derives the level's sample rate from the
  fc's metadata, falling back to 1.0 with a warning — never silently
  mislabeled.

---

## 19. Unit-aware electric scaling (decided 2026-09-12, delivered)

The review (§18.3) found the electric path scaled **unconditionally**
×1e-6 (mV/km assumed) while the magnetic path checks the payload's unit
attrs — payloads already in V/m were silently mis-scaled by 10⁶, and
`e_dipole_length_m` on mV/km data double-scaled by dipole/1000. The
Calibrate chain made this a live hazard: `RunTS.calibrate` rewrites each
channel's units attr to `channel_response.units_in`
(`get_calibrated_units`), so a calibration declaring a real unit
transformation (e.g. counts → V/m) delivers V/m data that Compute FC then
mis-scaled. Decision (user, 2026-09-12): **check attrs like the magnetic
path**; unitless payloads keep the legacy behavior.

Policy — `Compute FC` (`_electric_to_vm`, applied in `compute_fc` and
`_normalize_channels`):

- **`e_dipole_length_m` set** — the explicit raw-potential path (user
  choice wins): `x · 1e-3 / dipole`, with a **warning** when the channel's
  attrs canonicalize to mV/km (contradiction: the dipole scaling assumes a
  raw mV potential — clear the param for field data).
- **No dipole — scale by the channel's canonical unit attrs** (canonicalized
  via mt_metadata's unit registry, `_canonical_unit`: `'mV/km'` →
  "milliVolt per kilometer", `'V/m'` → "Volt per meter"):

  | canonical attrs | action |
  |---|---|
  | milliVolt per kilometer | ×1e-6 (LEMI archives; legacy path) |
  | Volt per meter | passthrough (already physical) |
  | milliVolt per meter | ×1e-3 |
  | milliVolt (raw potential) | **hard error**: set `e_dipole_length_m` (was a silent wrong ×1e-6) |
  | unrecognized | warning + passthrough (never silently scale what cannot be interpreted) |
  | no attrs (unitless) | ×1e-6 — legacy behavior (synthetic fixtures, core payloads; emptyui's LEMI reader writes mV/km attrs) |

- The fc records `electric_scaling` in attrs ("mV/km -> V/m", "passthrough
  (V/m)", "mV potential -> V/m (dipole)", "legacy (unitless, mV/km
  assumed)", ...; deeper decimation levels: "level 0 (passed down)").

Calibrate side: **no change needed** — `attach_filters` fills missing
units with the archival default, so gain-correction FAPs (LEMI,
`units_in == units_out`) keep the mV/km label after calibration (verified
numerically by the SDG31 chain), and declared unit transformations are
relabelled by `RunTS.calibrate` to `units_in`, which the attr-aware path
honors. Backward compatibility: unitless payloads behave exactly as
before — all fixtures and the USArray verification are unaffected
(regression tests in `test_electric_units.py`).

---

## 20. Plot Coherency + Plot Phase Tensor nodes (designed 2026-09-12, M4 #1)

Standard MT QC/interpretation products; both render to the `plot` port (§3
grammar — producers stay compact, preview is a user choice). **No new
runtime dependencies**: numpy + matplotlib only (both already pinned);
mtpy is not needed for computation (its PhaseTensor math is elementary);
it would only enter as a test-only golden-image reference (M4 #8).

### 20.1 Plot Coherency (`aurora.plot.coherency`, emptyui-aurora, Display)

- **Inputs**: `fc` + optional `ref_fc` (Coherence Sort's grammar); output
  `plot`.
- **Source decision (user, 2026-09-12): fc-direct** — computes per-band
  multiple coherence from the fc itself, not the sort's diagnostics
  table. The per-band coherence loop is refactored out of
  `coherence_sort_fc` into a shared `_band_coherence(local, ref, ins,
  outs, band_bins)` helper, so the plot uses **numerically the same
  per-criterion statistics** as Coherence Sort (honest about the
  broadband-heuristic caveat, §18.2) and works when no sort node is in
  the graph. One nuance (noted 2026-09-13): the sort's criterion is the
  **minimum** of the two curves in two-FC mode, while the plot renders
  the criteria separately without taking the min.
- **Curves**: median coh² vs period per criterion — E–H signal coherence
  (remote H when a `ref_fc` is connected) and, two-FC, the local↔remote
  magnetic transfer coherence; shaded guidance band at
  [`coherence_min`, `coherence_max`] (defaults 0.25/0.98 = the sort's
  defaults) — a one-glance QC view of "what will the sort do".
- **Params**: the banding set (`band_style`, `periods_per_octave`,
  `min_bins_per_band`, `t_short`) with the **same inherit-from-sort
  declaration as Estimate TF** (§18.2), plus `coherence_min`/
  `coherence_max` references. §15 conventions (log period axis, dpi 150,
  station title, framed legend).

### 20.2 Plot Phase Tensor (`mth5.plot.phase_tensor`, emptyui-mth5, Display)

- **Input**: `tf` → `plot`. Computation per period: `Φ = X⁻¹·Y` with
  `Z = X + iY` (2×2 real; `pinv` + condition guard for singular Re(Z)).
- **Presentation decision (user, 2026-09-12): Caldwell-style PT ellipses
  vs period, colored by skew** — ellipses drawn from the SVD/polar form
  (semi-axes = singular values σ1 ≥ σ2 of Φ, principal azimuth θ, skew
  β = (α1+α2)/2 from the polar decomposition Φ = R(α1)·diag(σ1,σ2)·R(α2)),
  colorbar for skew. §15 conventions.
- **Errors (MVP)**: no PT error bars — Z_err → PT covariance propagation
  is nontrivial (linearized through the inverse); noted in the plot label
  and left as a follow-up, consistent with §15's "classical math only"
  stance for error bars.
- **Invariants under test**: 1D earth → Φ scalar × I (circular ellipses,
  θ ≡ 0, β ≡ 0, ellipticity ≡ 0); 2D earth rotated by θ → PT azimuth
  recovers the strike, skew ≡ 0; 3D synthetic → β ≠ 0.

### 20.3 Placement

Both nodes follow §15 via the shared plotting helpers; zero core changes;
manifests carry the usual tooltip/description/palette protocol. The
coherency plot node lives in `emptyui-aurora` (imports the band machinery
and the shared coherence helper); the PT node in `emptyui-mth5` (a
tf-consuming Display node like Plot AppRes & Phase).

---

## 21. Portable local disk cache (designed 2026-09-12; NOT yet implemented)

User decisions (2026-09-12): **workspace-embedded** (the cache travels with
the data folder), **type-native serialization** (no pickle), **opt-in per
node**, **GB + TTL combined** eviction. Two-tier design — the existing
in-memory `NodeCache` stays the fast path; the disk tier adds restart
persistence and relocation. Consequence of "workspace-embedded": the cache
lives in a **user-designated project directory**, NOT the ephemeral per-run
workspaces (TTL-cleaned) — a third storage realm (§17.2).

### 21.1 Project directory

`EMPTYUI_PROJECT_DIR` (env) / `create_app(project_dir=…)` — a stable,
writable directory the user designates, typically where the survey data
lives. Cache root: `<project_dir>/.emptyui-cache/`; copying/moving the
project folder moves the cache with it. Unset → disk cache disabled (no
silent `~` writes); the in-memory tier always works. A Settings-dialog
field is a natural follow-up (MVP: env + kwarg).

### 21.2 Key design (relocation-friendly)

The executor's structural sha256 (§7) plus: payload-format version, node
plugin version, emptyui version — version bumps invalidate cleanly.
**File params become relocatable**: `_value_signature` uses the path
relative to the data root/project dir (plus size, mtime) when the file
lives under them; absolute stat signature otherwise (graceful degradation —
entries involving outside files simply do not relocate).

### 21.3 Layout & codecs (type-native, no pickle)

```
<project_dir>/.emptyui-cache/
  entries/<key[:2]>/<key>/
    meta.json          # node type/version, key inputs, created, atime,
                       #   format versions
    payload/…          # type-native: fc / ts.run → netCDF (per decimation
                       #   level), tf → XML, tables/arrays → JSON / npz
    artifacts/<stem><suffix>   # published files, centrally stored (also
                               #   fixes the stale-artifact-path issue)
```

- Codec registry `{payload-kind → (save, load, suffix)}`; payload kinds not
  in the registry → the node skips disk caching with a log note (graceful,
  plugin-extensible later).
- Atomic writes (temp dir + rename). Deserialize failures → invalidate +
  re-execute (never half-serve — same contract as the in-memory artifact
  existence check).

### 21.4 Opt-in per node

`disk_cache: ClassVar[bool] = False` on `Node`; `True` for ComputeFC and
EstimateTF (the real recomputation cost: STFT/decimation + robust solves).
Plots/tables stay memory-only. The manifest records the flag (properties
panel: "disk-cached").

### 21.5 Executor integration (two-tier)

```
mem.get(key) → hit: as today (by-reference payloads)
disk.get(key) → hit: materialize a CacheEntry (deserialize payloads, touch
              atime) → the same restore path as memory hits
execute → mem.put(key, entry); if manifest.disk_cache: disk.put(key, entry)
```

Disk I/O failures log and continue — **the cache must never break a run**.
Plugin disable clears the memory tier (as today); stale disk entries age
out (keys embed the manifest version, so stale reuse is impossible).

### 21.6 Eviction (GB + TTL combined)

`EMPTYUI_CACHE_MAX_GB` (default 10) — LRU by atime until under budget;
`EMPTYUI_CACHE_TTL_DAYS` (default 30). Cleanup pass on server start +
piggybacked on the existing workspace-TTL cleanup loop; `DELETE
/api/cache?tier=disk|memory|all`. Per-entry size accounting from file
sizes (structural keys stay cheap — no content hashing).

### 21.7 Tests (planned)

Round-trip per codec (fc multi-level, tf, table); restart persistence (new
Executor/server instance, same project dir → cached rerun); relocation
(move project dir + data root to a renamed path → still hits); eviction
(GB + TTL); corruption → re-execute; opt-in flag respected; injected I/O
error → the cache never breaks a run.

---

## 22. Runtime settings mechanism (delivered 2026-09-12)

The ⚙ Settings dialog is the single settings surface; the backend is the
source of truth (`GET/PUT /api/settings`).

- **Precedence (§5.1 rule, generalized)**: env/platform baseline wins →
  UI edits recorded on top → defaults. Knobs whose start value arrived
  from env/kwarg are **platform-baselined and not UI-editable** (PUT →
  409 naming the knob) — confinement/serving semantics stay
  restart-level decisions.
- **Live-editable whitelist** (`LIVE_KNOBS` in `emptyui.settings`):
  `workspace_ttl_seconds` (null/0 = off), `max_concurrent_runs`
  (null/0 = unlimited), `cache_max_entries` (default 256). Applied
  immediately (TTL loop re-reads per pass, admission check reads the
  live holder, `NodeCache.resize()` trims). `RuntimeSettings` is the
  mutable holder the app closures read.
- **Deployment rows (read-only display)**: `data_root`, `serve_dist`,
  `cors_origins`, `root_path` — changing confinement/serving mid-flight
  is an architecture change; env + restart only.
- **Planned (reserved, inert)**: `project_dir`, `cache_max_gb`,
  `cache_ttl_days` — the §21 disk cache lands into this scaffolding.
- **Persistence**: one sectioned JSON file via `EMPTYUI_SETTINGS_PATH`
  (`disabled_plugins` + `settings` sections; atomic write; unknown keys
  ignored on load — files may travel). Opt-in: unset = memory-only, no
  file is created. Settings PUT validates against the whitelist
  (numbers ≥ 0 or null-for-off; null not allowed for knobs with a
  non-null default).
- **Frontend**: Settings dialog sections — Server & retention (live
  numbers, empty = off, explicit Save), Deployment (read-only code
  values), Planned (disabled rows, "planned" tag), Plugins (§5.1,
  unchanged).

---

## 23. Plot Tipper restyle + fc-based Plot Spectrogram (delivered 2026-09-12)

### 23.1 Plot Tipper — AppRes grammar alignment + y-limit param
- **Layout** (user decision: the MT standard): panels = **Re / Im** (top /
  bottom), series = components — the exact mirror of AppRes & Phase
  (panels = parts, series = components): **Tzx = ○ open circle (blue)**,
  **Tzy = ✕ cross (red)** via the shared `MARKERS`/`COLORS`.
- **Open-circle correction (user)**: the USArray reference figure draws XY
  as an OPEN circle — `MARKER_FACES = {"xy": "white"}` added to the shared
  style; **AppRes & Phase XY is now open too** (was filled).
- Symbols only (no connecting lines), shared log-period x-limits
  (`_decade_limits`), same width (6.5), markersize 5, capsize 2.5,
  elinewidth 0.8, boxed legend in the Re panel only.
- **New param** `tipper_ylim` ("auto" | "min,max"), mirroring `rho_ylim`;
  auto keeps the smart default ([-1, 1], auto-expand when |T| > 1 near
  2D/3D contrasts — correct behavior, kept).

### 23.2 Plot Spectrogram — fc input (no in-line FFT)
- **New node `aurora.plot.spectrogram`** (emptyui-aurora, Display): input
  `fc` from Compute FC; power = 10·log10|FC|² per (frequency, time) — the
  coefficients are reused, no second FFT. pcolormesh, **log y
  (frequency)**, x = time (datetime64 ConciseDateFormatter when persisted,
  relative seconds otherwise), viridis + dB colorbar, §15 conventions.
- **Params**: `channel` (string, empty = first electric channel; standard
  names map via the fc's nomenclature) — **`split_inherited` Param flag
  (new, generic §4 extension)**: inherited comma-separated candidates
  (Compute FC's `output_channels = "e1,e2,bz"`) are split into individual
  combobox choices; `decimation_level` (integer, −1 = deepest available,
  invalid → actionable error listing levels — a payload-dependent dropdown
  is impossible pre-run, the frontend never sees the payload);
  `window_samples`/`decimation_levels` are **inherited from Compute FC**
  (§4 map on the `fc` input) for annotation; the authoritative window
  length shown comes from the fc's `num_samples_window` attr.
- **Old node**: `core.plot.spectrogram` (ts.run + in-line FFT) →
  `hidden = True`, re-titled "Plot Spectrogram (TS)" — out of the palette,
  still executable by saved workflows.

### 23.3 Provenance line on the TF display figures (delivered 2026-09-13)
The processing-provenance line (the TF's run comment, drawn under AppRes &
Phase, Tipper and Phase Tensor) got three fixes:
- **Toggle**: new `print_params` boolean param ("Print parameters on plot",
  default **on** — preserves the comparison workflow; untick for production
  images) on all three plot nodes.
- **No overlap**: the line is anchored to the *measured* x-label bottom
  (5 px gap), instead of a fixed `y=0.005` that collided with
  "Period (secs)".
- **Word-wrap**: greedy renderer-measured wrapping (`_wrap_notes`) so no
  line is wider than the axes — the old single-line note rendered ~1100 px
  wide on a ~310 px figure, inflating the saved canvas via
  `bbox_inches="tight"`. When the wrapped block needs more vertical room
  than the layout leaves, the tight bbox extends the saved image downward
  (`impedance_layout` stays single-pass; reserving inside it compounds its
  margin accounting — verified and rejected).

### 23.4 Time selector UI (delivered 2026-09-13)
Selecting a time slot (e.g. to avoid noisy intervals) in records spanning
hours to days:
- **Payload-summary run events (§23.4 mechanism)**: when the executor
  materializes a `ts.run`/`ts.channel` output handle (fresh or restored
  from cache), it emits `{"event": "payload", node, port, type, summary}`
  with a duck-typed extent summary — absolute ISO bounds (UTC) when the
  record carries a real start, relative seconds otherwise; placeholder
  epochs (1980/1970) degrade to relative (start unknown, hygiene #10).
  Poll/WS replay carries them like artifact events. Honest constraint:
  the bar's extent is known only after the upstream has run once.
- **`Param.time_bar`** (rendering-only): the input port name whose summary
  drives the frontend time bar for this param pair.
- **`TimeBar` widget** (custom, zero deps — vis-timeline/d3-brush/rc-slider
  evaluated and rejected as heavyweight or generic): horizontal acquisition
  bar with two pointer-driven handles sandwiching the slot, shaded
  selection, extent tick labels, synced bidirectionally with the paired
  Start/End edit boxes (which remain the node's own string params — the
  bar only writes through their onChange). Guards: handles stay in the
  record, cannot cross, slot kept > 1 s (server rejects ≤ 1 s; widget
  clamps ≥ 2 s at second resolution); out-of-extent box values draw at the
  track edge (server clamps with a warning — degrade, never reject).
- **Plot Time Series** hosts it: new `start`/`end` params (ISO UTC or bare
  seconds; empty = record boundary; ISO bounds measure from the record
  start, so `data.start` offsets are subtracted). The legacy `seconds`
  duration caps the slot from its start and keeps its old no-minimum
  behavior (a 1 s waveform view of a 500 Hz run is legitimate).
- **Node sizing**: standard node width 250 → 280 px so the bar fits with
  consistent look across the palette.
- Extract Run / Align Runs / Merge Runs hosting is deferred: their extent
  source is an `mth5` archive payload (per-run extents inside one object),
  not a streamed `ts.run` summary — needs its own summary mechanism.

#### 23.4.1 Phase 2 hosts (delivered 2026-09-13)
The time selector extended to the mth5-backed hosts:
- **Payload summaries extended**: the executor's summarizer now duck-types
  a third shape — an **open MTH5 archive** (`.run_summary`, populated right
  after open): `kind: "mth5"` with per-run extents
  (survey/station/run/start/end/sample_rate); single-channel payloads
  (ChannelTS, the `ts.channel` ports) summarize as one-channel `ts`. A
  hard-won detail: mt_timeseries objects raise **NameError** (not
  AttributeError) for missing attrs, so `getattr(x, "attr", None)` is not
  safe on them — explicit try/except.
- **`Param.time_bar` multi-port binding**: comma-separated port list;
  the frontend *intersects* the connected ports' extents (Align Runs shows
  the overlap of both inputs; identical extents intersect to themselves).
- **Extract Run**: start/end get the bar bound to `mth5` — the extent is
  the run matching the node's survey/station/run params (client-side
  mirror of the server's first-run fallbacks), absolute GMT. Server-side:
  bounds clamp into the run extent (metadata time_period, falling back to
  the station's channel-derived run summary) — the extent axis is used
  as-is, so even a 1980/1970 placeholder record is a valid *within-record*
  selection axis (placeholders matter for cross-record alignment, not for
  in-record selection). Extent-unknown → passthrough with warning.
- **Align Runs**: one `start`/`end` pair (NOT per-input bars — the node's
  contract is identical-interval outputs, so a single window within the
  aligned overlap is the correct semantics; recorded over the earlier
  two-bar sketch). ISO bounds anchor at the overlap start; without GPS
  time only relative seconds are accepted. Bounds applied after the
  overlap crop (clamped; slot > 1 s).
- **Merge Channels**: `start`/`end` crop the merged run
  (`_crop_to_runts`), bar bound to the first connected channel port.
- **Bug fixes found on the way**: `is_unknown_start` failed on tz-aware
  stamps (aware-vs-naive comparison is False) — placeholders are now
  compared in their own zone; `_warn_if_uncorrelated` crashed when the
  overlap was shorter than one coherence segment (skipped below 256
  samples).
- **Merge Runs deliberately skipped**: its per-run windows already live in
  the selection table rows (edited via View Selection) — a bar per row is
  a table-UI feature, not a node-param widget.

### 23.5 Estimate TF shortest-period (t_short) honored for the uniform scheme (fixed 2026-09-13)
**Bug**: `t_short` (head clip) was applied only inside `_l2u_band_table`
→ `_clip_head_bands`, i.e. only on the primary l2u container-level table.
For `band_style="uniform"` the clip never ran — the per-level
`_resolve_period_bands` → `_period_bands` chain has no `t_short` at all —
so the shortest-period parameter was silently ignored. The same gap hit
Coherence Sort and Plot Coherency (both advertise the param) and the l2u
**fallback** path (`_period_bands_indexed`, metadata-less fcs).

**Fix (single choke point)**: `_resolve_period_bands(..., t_short=None)`
post-filters BOTH schemes with the identical geometric-mean center-period
criterion (`period >= t_short`, same as `_clip_head_bands`); `estimate_tf`,
`coherence_sort_fc` and Plot Coherency thread the param through; the l2u
primary path is untouched (mutually exclusive with this branch — no
double-clip). Fully-clipped levels are skipped in `estimate_tf_levels`
(mirroring the l2u `if not bands: continue`), and Plot Coherency raises
its existing "no usable frequency bands" error when everything is clipped
(uniform now behaves like l2u). Verified: `t_short=0`/`None` remains a
no-op for both schemes (legacy behavior).

## 24. Physics parity with classical EMTF — discrepancy inventory & research plan (started 2026-09-13)

**Motivation.** The SDG31 experiment case is a controlled physics experiment,
not just a regression test: identical input data (4P.SDG31 + remote 4P.SDH28
MTH5 archives, 1 Hz NIMS/100 m dipoles) and a ground-truth output (the
official EMTF/SPUD `USArray.SDG31.2017.edi`) allow *ablation* — change one
processing factor at a time and measure which classical-EMTF ingredient
actually closes the gap. Classical EMTF (Kelbert et al.) is the traditional
gold standard; emptyUI delegates its estimators to aurora 0.6.2 but computes
its own Fourier coefficients, so the FC stage is where the schemes diverge.

### 24.1 Fixed comparison protocol (the "exact" approach)

Every parity experiment keeps these FIXED and changes exactly one factor:

- **Band table**: reconstructed exactly from the official EDI band-center
  frequencies (`exact_official_bands` — 6,6,6,6,4,2 bands per level, edges at
  geometric midpoints; `bands_override`). NOT the l2u default — the residual
  grid difference (§14.9: 93% vs 98% pass) must not confound physics deltas.
- **Frame**: fixed −5.7° true-north rotation (documented coordinate-frame
  rotation, §16.3) — never a per-variant rotation fit.
- **Estimator**: Robust (RME_RR) with the remote reference; aurora defaults.
- **STFT base**: window 128 samples, factor 4, 5 decimations, 50% overlap.
- **Metrics** (matched nearest-period rows, off-diagonal xy/yx,
  reliable band [15 s, 18 725 s] plus a **short-period split [15, 100) s**
  — the low-hanging-fruit target):
  median / p90 / max |Δlog₁₀ρ| and |Δφ| (folded convention), pass rates at
  the §16.3 gates (0.1 decades / 10°), and — added for statistics — **error
  -bar agreement** (our σ vs official `.VAR`, ratio statistics), which the
  original metrics never tested (§16.2/C2 bias makes our OLS dof inflated;
  RME errors are the production path).
- **Statistics discipline**: paired deltas on the identical matched grid;
  one factor per variant; every variant's full kwargs recorded in the
  metrics JSON (fc attrs already carry `prewhitening`/`per_window_detrend`
  provenance, §19-style).

Harness: `examples/data/lims-usarray-test/verify_sdg31.py` (importable
helpers) and `parity_sdg31.py` (the ablation driver).

### 24.2 Discrepancy inventory (2026-09-13 full audit)

**FC/STFT stage** (emptyUI = taper→FFT only; aurora default = first-
difference PW + per-window linear detrend + boxcar 25% overlap + LSDL
factor; original EMTF = AR(3) PW + Slepian-family taper + 128/32):

| # | Discrepancy | Priority | SDG31 signal |
|---|---|---|---|
| A1 | No prewhitening/recoloring | **E1 (first)** | T < 30 s leakage-limited end |
| A2 | No per-window detrend (global node only) | **E2** | long-period tail |
| A3 | Taper family: Hamming 50% vs boxcar 25% vs EMTF Slepian | **E1** (with A1) | in-band p90 gap |
| A4 | No clock-zero truncation | low | negligible (aurora default also ignores) |
| A5 | FC normalization scipy 'spectrum' vs LSDL factor | none for TF (ratios scale-equivariant); blocks FC round-trip comparison with aurora | — |
| A6 | DC bin retained (NaN'd under PW) | cosmetic | spectrogram log-axis row |

**Band grid**: B1 l2u ≈ official (0.020 median, 93% vs exact 0.019/98%);
B2 uniform anchors extend half a bandwidth past Fs/4 · window/4 (documented);
B3 boundary-bin drift via fs-recovery (low probability); B4 hand-authored
band files = M4 #9 (E4).

**Estimation**: C1 estimator parity verified (delegated to aurora, incl.
`RME_RR.qr_input="Z"`); C2 row-pooling independence assumption → OLS dof
inflation, σ² underestimated (uncorrected; robust engines less sensitive);
C3 no error floors anywhere (also absent in pinned aurora); C4 coherence
statistic = per-window broadband heuristic, biased high ≈ n_in/(N_eff−1), no
DOF correction (§16.2, M4 #7); C5 √w weighting exact for OLS, approximate for
robust engines; C6 degenerate-band min-pairwise fallback (conservative,
documented); C7 ref_fc silently ignored by non-RR engines (behavior bug, not
physics); C8 merge deepest-wins parity OK, 1-D `Z_err` fallback latent crash.

**Frame/units** (resolved): −5.7° frame rotation (documented, not a bug);
EDI declination 0.0-override bug (`tf.py`, parsing); units verified
analytically incl. the `_default_unit` hx-mislabel fix (2026-09-13).

### 24.3 Research plan (staged ablations, gain-ordered)

- **E1 — Prewhitening + taper at short periods** (enablers delivered
  2026-09-13; **ablated 2026-09-13**, `parity_sdg31.py`):
  opt-in `prewhitening` ("first difference") and taper family on ComputeFC.
  Ablation grid: taper ∈ {hamming, boxcar, hann} × PW ∈ {off, first
  difference}, all else fixed per §24.1 (exact bands, −5.7° rotation,
  Robust RME_RR). Results (paired medians |Δlog₁₀ρ| / p90, pass = ρ/φ
  gates; short = [15, 100) s, tail = [100, 18 725] s):

  | variant | short med/p90 | tail med/p90 | pass ρ/φ |
  |---|---|---|---|
  | hamming, no PW (**current default**) | 0.015 / 0.065 | 0.019 / 0.044 | 94% / 88% |
  | hamming + PW | 0.010 / 0.016 | 0.009 / 0.030 | 100% / 100% |
  | boxcar, no PW | **0.98 / 1.53** | 0.85 / 1.44 | 0% / 75% |
  | boxcar + PW | 0.008 / 0.023 | 0.009 / 0.031 | 100% / 100% |
  | hann, no PW | 0.008 / 0.016 | 0.008 / 0.032 | 100% / 100% |
  | hann + PW | 0.009 / 0.016 | 0.008 / 0.030 | 100% / 100% |

  **Findings** (statistics on matched official periods):
  1. The current default's entire deficit vs the official product is
     concentrated at T = 7.3–15.1 s (worst: Δlog₁₀ρ +9.7, Δφ +152° at
     11.6 s — a noise-floor band); above 19.7 s all variants agree to
     ≤0.03 decades / 4°.
  2. The controlling factor is **DC/edge leakage suppression**, not PW per
     se: the calibrated SDG31 series carry significant DC offsets
     (|mean|/std up to 0.38 on ey). Hamming's nonzero (0.08) endpoints let
     that offset leak; Hann's zero endpoints kill it — which is why
     hann+noPW already matches hamming+PW. Boxcar without PW is
     catastrophic (~1 decade, all periods) — a clean quantification of why
     classical pipelines (aurora boxcar+PW default, EMTF AR-PW) never
     taper with a nonzero-DC window unwhitened.
  3. First-difference PW with the **exact** inverse recoloring preserves
     clean-signal FCs to ~1e-4 relative (unit-tested) and drops red-noise
     leakage hard (synthetic: >2× out-of-band mean amplitude reduction).
  4. **Error-bar signature (E5 preview)**: our σ_Z vs official σ tracks the
     taper, not PW (median ratio: boxcar 1.7 > hamming 1.1 > hann 0.66) —
     the official EMTF σ convention embeds its taper/DOF choices; the PW
     shrinkage (0.66→0.68 on hamming) needs the DOF study before any
     interpretation. Do not tune error floors against this yet.
  5. Cross-check: hamming+noPW tail median 0.019 reproduces the §16.3
     record (0.019) — the review fixes since 2026-09-10 did not shift the
     reliable band.
  **Decision (pending E2)**: leakage-suppressed variants are pairwise
  statistically indistinguishable; per the §24.3 rule PW qualifies for
  default adoption, but the default flip is deferred until E2 (per-window
  detrend) is ablated and the persisted-FC convention (A5/E6) is settled —
  PW changes persisted FC amplitudes.
  Comparison figures (boxcar+PW vs official, §24.1 protocol):
  `examples/data/lims-usarray-test/sdg31_parity_boxcar_first_difference_vs_official.png`
  (raw overlay, each dataset at its own periods, ours rotated −5.7°) and
  `sdg31_parity_boxcar_first_difference_appres_phase.png` (§15
  EMTF-convention figure via the plot node); rendered with
  `parity_sdg31.py --plot <taper> <pw>`. Working-recipe figures (full
  remote b+c + weight sort 0.7):
  `sdg31_parity_boxcar_first_difference_b-c_sort0.7_vs_official.png`
  (the head now sits on the official points, 7.3–11.6 s included) and
  `sdg31_parity_boxcar_first_difference_b-c_sort0.7_appres_phase.png`;
  `--plot <taper> <pw> <remote-runs e.g. b,c> [coh_min]`.
- **E1b — Data selection / remote coverage: the short-period gap closed**
  (2026-09-13, `short_period_sdg31.py`). The residual deficit at the first
  three official periods (7.3/9.1/11.6 s) was **not** an estimator or FC
  factor — it was **remote-data coverage**. The official product's tag is
  `SDG31b_H28coh` (RUNLIST=SDG31a SDG31b, REMOTESITE=SDH28): an EMTF-FCU
  coherence-processed product using all SDH28 data (runs b+c, ~14.7 d),
  while our alignment against SDH28-b only cropped away the last ~29% of
  SDG31-b (SDH28-b ends 2017-10-23, SDG31-b 2017-10-26). Findings:
  - **Full remote (b+c) alone fixes 9.1 s and 11.6 s with no sorting**
    (Δlog₁₀ρ ≤ 0.07, Δφ ≤ 3.2°; head [7.3, 15) s median 0.534 → 0.050);
    window-64 also becomes viable (it was harmful under b-only coverage —
    the earlier "window 64 remedy" was confounded by the missing remote
    data). EMTF's decset.cfg uses 128-sample windows at ALL levels;
    window reduction is confirmed NOT the classical lever.
  - **7.3 s closes with mild coherence selection** (weight,
    coherence_min=0.7: Δlog₁₀ρ −0.046/−0.050, Δφ +3.7/−1.3° — 119/13348
    windows retained; fragile — the estimate rides on 0.9% of windows).
    The official `…H28coh` tag supports coherence processing as the FCU's
    actual lever; the principled form is the per-frequency window-ensemble
    statistic (M4 #7/E7), not our broadband per-window heuristic.
  - **Self-reference quantifies the fluxgate noise floor**: RR-off
    underestimates ρ by up to 5.8 decades below 15 s (smooth SNR
    degradation with period) — the remote reference does the heavy
    lifting; below ~50 s the local-H noise floor ≈ signal (user hypothesis
    confirmed quantitatively).
  - **σ-honesty discrepancy (E5 input)**: the un-cleaned estimate reports
    σ/|Z| ≈ 0.3–0.9 at 7.3–11.6 s (honest) while the official .VAR claims
    0.053/0.030/0.008 — the FCU error convention at noise-floor periods is
    far more optimistic than the raw data support; reconcile before any
    error-floor tuning.
  - Level-0 sort diagnostics: E–remote-H multiple coherence
    0.32/0.49/0.52 at 7.3/9.1/11.6 s — most of E is noise there, yet the
    cleaned estimates match the official to ≤5%: coherence selection at
    the noise floor works because the *clean subset* is consistent.
  - Trade-off: weight-0.7 selection slightly regresses the deepest tail
    (tail p90 0.039 → 0.069, pass 97 → 95) — selection belongs at the
    noise-floor end. A period-dependent selection schedule is the natural
    refinement (E7).
  **Working recipe (SDG31)**: boxcar + first-difference PW, window 128,
  exact band table, −5.7° rotation, Robust RME_RR, **full remote coverage
  (SDH28 b+c)**, optional weight-mode coherence selection at the head.
- **E2 + tail probes — long-period mismatches** (2026-09-13,
  `long_period_sdg31.py`). With the recipe baseline (full remote), the tail
  picture changed completely: 18 724.6 s was a remote-coverage artifact
  (now Δlog₁₀ρ ≤ 0.044, within the official's own σ/|Z| = 0.098 there).
  Tail level geometry: level 4 (window 9.1 h, ~52 windows) owns
  2 260–7 282 s; level 5 (window 36.4 h, ~13 windows ≈ 6 independent)
  owns 11 916–18 725 s. Findings:
  - **The dominant tail factor is the taper** — the diurnal (Sq) magnetic
    variation (~24 h + harmonics) lies inside every level-4/5 window, and
    boxcar's −13 dB sidelobes leak that huge line into the tail bands;
    zero-endpoint tapers suppress it (4681 s: Δ −0.075 → −0.025; 11 916 s
    xy: +0.249 → +0.130; 18 725 s Δφ: −3.2° → −0.3°; boxcar → hann).
  - **Per-window detrend (E2) is NOT the lever**: negligible on the tail
    (it removes the linear trend, not the periodic Sq line), and slightly
    harmful at the head (pass 83 → 67 with boxcar). Verdict: keep off —
    emptyUI's PW chain does not need aurora's per-window default.
  - **A per-level taper schedule combines the wins**: the dominant
    contamination differs by scale — white-ish fluxgate noise (post-PW) at
    level 0, where boxcar maximizes dof, vs line-spectra leakage at the
    day-scale windows where hann wins. boxcar L0–2 / hann L3–5 gives head
    0.050 (83/83) AND tail 0.010/0.036 (97/100); hann at level 0 destroys
    the 7.3 s band (Δ +4.3 decades, σ/|Z| = 8.1) — the schedule is
    physically motivated, not tuned. (Code path needed: compute_fc_levels
    takes one window_type; the investigation merges per-level fc dicts.)
  - **Level resolution**: deepest-wins assigns 11 916/18 725 s to level 5;
    the level-4 estimates at its own bands are clean (≤0.075 boxcar,
    ≤0.033 hann). The residual 11 916 s xy (+0.13) rides on ~6 independent
    windows — few-windows statistics, and the official σ/|Z| = 0.082
    there; E5/E7 territory (DOF correction, per-frequency ensemble
    statistic).
  **Refined recipe (SDG31, full band)**: boxcar + PW at levels 0–2, hann +
  PW at levels 3–5, full remote b+c, weight sort 0.7 at level 0 →
  head 0.034/0.080, short 0.007/0.022, tail 0.010/0.036;
  **reliable full band: median 0.009 / p90 0.029 / 98% pass** — vs the
  §16.3 original record 0.019/0.045/98%. Worst remaining rows: 9.1 s yx
  (−0.110, marginal) and the 11 916 s few-windows band. Figures:
  `sdg31_parity_final_recipe_vs_official.png`,
  `sdg31_parity_final_recipe_appres_phase.png`.
- **E2 — Per-window detrend** (enabled, ablation pending): hypothesis:
  closes the long-period tail bias.
- **E3 — EMTF Slepian multitaper** (research): multi-taper FC ensemble;
  only if E1 leaves a measurable in-band gap.
- **E4 — Exact EMTF band ladder as a node option** (M4 #9): the ablation
  harness uses `bands_override` internally; expose a band-table node.
- **E5 — Error-bar statistics + error floors**: add σ-agreement metrics to
  the SDG31 harness; then evaluate EMTF-style floors (C2/C3).
- **E6 — aurora FC round-trip** (blocked by A5): persist emptyUI FCs and
  compare amplitudes against aurora's own persisted FC layer for the same
  run — requires settling a shared normalization convention first.
- **E7 — True EMTF per-frequency coherence sort** (= M4 #7) and C2 DOF
  correction: statistics work, after the deterministic factors are settled.

Decision rule: adopt a factor into the **defaults** only when it improves
the paired short-period AND reliable-band medians without regressing the
pass rates; keep every adopted factor opt-in-able (provenance attrs) so
legacy workflows and the verified §16.3 record stay reproducible.

### 24.4 Workflow enablement — missing nodes/params for the parity recipe (2026-09-13)

The refined recipe (§24.3) cannot yet be expressed end-to-end in the node
UI. Recipe step → current node state → gap (effort):

1. **Merge remote runs (b+c)** — ✗ **blocking**. `Merge Runs`
   (`mth5.runs.merge`) has NO ts.run input ports: both modes require a
   `selection` payload whose producer node is M4-deferred, so merging
   SDH28-b + SDH28-c is impossible in the UI (the investigation merged
   physically via `RunTS.merge` in code). Needed (medium):
   (a) physical merge node taking `ts.run` inputs directly (pairwise
   `RunTS.merge`, gap-aware — 28-min interpolation caveat, ~0.2% windows);
   (b) production-grade: FC-level concatenation (aurora `merge_stfts`,
    `xr.concat` along time — fabricates nothing; makes the gap a no-window
   region).
2. **Compute FC taper vocabulary** — ✗ trivial. `window_type` select offers
   only `hamming`/`hanning`; the recipe needs **`boxcar`** (it wins at the
   short-period levels). The API already passes any scipy window name
   through `_window`; only the manifest vocabulary must grow.
3. **Per-level taper schedule** — ✗ medium. Recipe: boxcar L0–2 / hann L3–5;
   `compute_fc_levels` takes one global `window_type` (the investigation
   merged per-level fc dicts in code). Natural shape: `window_type` per
   decimation level (list/schedule param), or a `deep_level_taper`
   override for levels ≥ N. Physically motivated (white-noise scale vs
   Sq-line-leakage scale), not tuned.
4. **Coherence Sort level/period schedule** — ✗ medium. The node sorts ALL
   levels; the recipe uses weight-0.7 at level 0 only (sorting perturbs the
   deepest tail: p90 0.039 → 0.069 with boxcar). Needed: an
   `apply_below_period` (or level-schedule) param so the sort can be
   scoped to the noise-floor head.
5. **Exact band table** — ✗ M4 #9 (already backlog). The parity protocol's
   exact-official-bands table (`bands_override`) is low-level only; the UI
   offers l2u/uniform (l2u is statistically equivalent — this is for
   apples-to-apples comparisons).
6. **Estimate TF** — ✓ nothing missing (Robust, remote reference + `ref_fc`
   port, `min_bins_per_band`/`t_short`, inherit-from-sort).
7. **Rotate TF** — ✓ manual angle entry covers the −5.7° documented frame
   rotation (declination auto-pickup exists; the SDG31 −5.7° = IGRF 4.26° +
   1.4° site azimuth is site-specific, not in any archive the node reads).
8. **Align Runs / Merge TF Levels / Calibrate / Export** — ✓ nothing missing.

Priority: (1) is the workflow blocker (the recipe's single most important
factor — remote coverage — has no UI path today); (2) is a one-line
manifest change; (3)/(4) are the per-level schedule generalization; (5)
stays M4 #9. `overlap` also lacks a max validation (crashes mid-node for
overlap ≥ 1 — review item B13, now user-reachable while experimenting).

### 24.5 Implementation plan for §24.4 (decided 2026-09-13)

Decisions (user-confirmed): merge = **both routes, TS-physical first** (FC
concat as the later production-grade addition); **rebuild
`mth5.runs.merge` in place** (drop the dead `selection`/DataFrame path —
params-based run list like Extract Run); per-level taper = **minimal
override** (`window_type` base + `window_type_deep` + `deep_from_level`);
manual bands = **folded into Estimate TF** (`band_style="manual"`,
`enabled_when`-gated `band_centers` + `bands_per_level` → geometric-
midpoint edges, the harness-proven reconstruction).

- **P1 Merge Runs rebuild**: inputs `mth5`; params `survey`/`station`/
  `runs` (comma list; empty = all runs)/`gap_method`/`new_run_id`; output
  `merged_run: ts.run`. Core merge operates on a list of RunTS (unit-
  testable without an archive). **P1 v2 — Merge FC node (delivered this
  round)**: fixed `fc1..fc4` ports (burst-mode bulk mth5 ingestion is a
  recorded follow-up — it needs special care because a merged-TS cascade
  cannot decimate *across* burst slots; per-run cascades within a slot are
  fine, deep levels starve naturally in short slots); **best-effort
  intersect** validation (frequency grids intersected with tolerance,
  mismatched channels dropped with warnings, sample-rate mismatch errors);
  output feeds `Estimate TF`'s strict identical-grid check.
- **P1 v2b — clock-zero window anchoring (user-corrected: required)**:
  ComputeFC gains `clock_zero` ("" = off/run-start anchoring, "epoch" =
  Unix epoch, or an explicit ISO datetime): window starts anchor to
  ``t0 + k·step`` per level and the head partial window is dropped, so
  per-run FCs of *different* runs/stations share one window grid — the
  prerequisite for remote-reference pairing across separately-computed
  runs (gapped continuous runs, burst mode). Note: anchoring each station
  to "the first station's start" (the initial proposal) does not generalize
  — run starts are not whole window-steps apart, so the shared reference
  must be run-independent (epoch) or passed explicitly (ISO). Levels with
  fewer samples than one window (burst slots at deep levels) are skipped
  with a warning. Estimate TF keeps the **strict** identical-grid policy;
  a time-value-matching loosening is recorded for GPS-stray clocks on
  semi-permanent stations (not uncommon) — later option.
- **P2 ComputeFC per-level taper**: `window_type_deep` (select, empty =
  base) + `deep_from_level` (integer); per-level window recorded in the
  fc attrs.
- **P3 CoherenceSort scope**: `apply_below_period` — band-center periods
  at/above the threshold pass through unsorted (default empty = all, the
  recipe scopes the head remedy to level 0 without touching the tail).
- **P4 EstimateTF manual bands**: `band_style` + `"manual"`; shared
  `manual_bands(freqs, counts)` helper in `_processing.py` (descending
  frequency list + per-level counts → per-level `(f_hi, f_lo)` pairs at
  geometric midpoints — identical to the parity harness's
  `exact_official_bands`).
- **P5 verification**: synthetic multi-run e2e through the executor
  (merge → align → FC → scoped sort → estimate → rotate); manual-band
  parity against `bands_override`; per-level attr checks. SDG31 recipe
  workflow JSON template: after P1–P4 land.
- **P5 head-to-head (delivered 2026-09-14,
  `node_vs_harness_sdg31.py`)**: the refined recipe as a stock node graph
  (Load ×2 → Extract → **Merge Runs b,c** → Calibrate ×2 (archived
  responses apply to the merged raw run — the filter attrs survive
  `RunTS.merge`) → Align → Compute FC ×2 (boxcar/deep-hanning/PW) →
  Coherence Sort (weight 0.7, apply-below-20 s) → Estimate TF
  (Robust RR, **band_style="manual"** with the official centers +
  "6,6,6,6,4,2") → Rotate (−5.7°)), executed by the `Executor`, vs the
  recorded harness numbers. Result: **reliable full band identical
  (0.009 median / 0.029 p90 / 98% pass)**; short/tail identical; the only
  deviation is the 7.3 s band (node +0.180/−0.269 vs harness −0.046/
  −0.050) — the node sort bands with the l2u scheme (it has no band-table
  input; the harness sorted on the exact official table), so the head
  weights differ slightly. Extending the manual band table to the sort
  node (or feeding it one) would close that; the executor gained an
  opt-in `include_store` summary flag so scripts can resolve output
  handle ids to payloads. Metrics:
  `sdg31_node_vs_harness_metrics.json`.

### 24.6 Per-bin coherence statistic (E7/M4 #7 — the banding-independent sort)

**Grounding (2026-09-14).** aurora 0.6.2 has no reusable per-frequency
coherence routine: `compute_squared_coherence` (TRME lineage,
`m_estimator.py:315`) is a per-band explained-variance R² = 1 − SSR/SSYc
of the band regression on the *cleaned* data, per output channel — only
logged, never used for selection; `adjust_band_for_coherence_sorting`/
`get_band_for_coherence_sorting` are WIP band-widening placeholders
("min3" rule, never delivered). The EMTF-style per-frequency
window-ensemble statistic is therefore implemented fresh in emptyUI.

**Statistic (as implemented, opt-in `statistic="per_bin"`)**: for every
frequency bin f and window w, the multiple coherence² of each output on
the magnetic inputs estimated over a **sliding window ensemble**
(±`ensemble_windows` windows, center included — leave-one-out would
penalize isolated clean windows) with optional pooling over
±`smooth_bins` adjacent bins (a FIXED smoothing scale — deliberately not
the estimate's band table; restores dof at deep levels with few windows).
Adjusted-R² correction removes the p/(W−1) high bias of sample coherence
(the §16.2 C2 bias, corrected exactly for this statistic since the
ensemble count is known). Weights are per (bin, window) — finer than the
per-window scalars, frequency-dependent like EMTF's, and independent of
the band-average method (the band table survives only as the diagnostics
table's aggregation view). Failure mode trade-off: isolated clean windows
in noisy neighborhoods get down-weighted (conservative; the robust
estimator still sees the retained rows).

**SDG31 ablation (2026-09-14, recipe baseline: full remote b+c, exact
bands, −5.7°, all sorts scoped < 20 s, weight 0.7):**

| sort variant | head med/p90 (pass ρ/φ) | full med/p90 (pass) |
|---|---|---|
| band (l2u-banded sort) | 0.020 / 0.156 (67/67) | 0.008 / 0.029 (98/100) |
| per_bin ens=8 sm=0 | 0.223 / 0.399 (33/83) | 0.008 / 0.038 (98/100) |
| **per_bin ens=16 sm=0** | **0.043 / 0.064 (100/100)** | 0.008 / 0.037 (98/100) |
| per_bin ens=8 sm=1 | 0.161 / 0.341 (33/67) | 0.009 / 0.036 (98/100) |
| **per_bin ens=16 sm=1** | **0.040 / 0.043 (100/100)** | 0.008 / 0.037 (98/100) |

Findings: (1) **the per-bin statistic with ens=16 is the best head
variant — 100/100 within gate, and BAND-INDEPENDENT** (no exact-table
needed; the node-sort banding deviation from §24.5 P5 vanishes by
construction); 7.3 s: +0.041/−0.044 with sm=1. (2) Small ensembles (8 →
17-window segments) fail at the head — the noise-floor bins need the
ensemble dof for the debiased coherence to discriminate. (3) The debias
correction is what makes the thresholds transferable (the raw sample
coherence would sit at ~n_in/W ≈ 0.12 under the null). (4) Full-band
p90 (0.037) is marginally above the band statistic's 0.029 — the debias
makes the sort slightly stricter at short periods. **Recommendation**:
use `statistic="per_bin", ensemble_windows=16` in the recipe; node
default stays "band" pending the Plot Coherency counterpart (§20.1's
"the plot shows exactly the statistic the sort applies" holds only for
the band mode).

**Default flip (2026-09-14, user-directed).** The recommendation was
adopted as the DEFAULT everywhere user-facing:

- Coherence Sort node + `coherence_sort_fc`/`coherence_sort_fcs`:
  `statistic="per_bin"`, `ensemble_windows=16`;
- Coherence Sort / Estimate TF / Plot Coherency nodes +
  `estimate_tf`/`estimate_tf_levels`: `periods_per_octave=3`,
  `min_bins_per_band=20` (SDG31-recipe values, replacing the
  ablation-era 4/40);
- `_per_bin_coherence` helper default aligned to `ensemble_windows=16`.

Internal geometry helpers (`_period_bands`, `_period_bands_indexed`,
`_resolve_period_bands`, `_l2u_band_table`) keep their historical
defaults — every call site passes explicit values, so they never leak.
Pinned by `tests/test_defaults.py`; the legacy band-statistic tests now
pass `statistic="band"` (and `periods_per_octave=4` where their
assertions assume the old grid) explicitly. Caveat retained: Plot
Coherency renders the **band** statistic, so §20.1's "the plot shows
exactly the statistic the sort applies" now holds only when the sort is
explicitly set to `statistic="band"` — the node description documents
this. A browser e2e run with the per_bin default is pending.

### 24.7 Burst-mode data (MTU-5C / MTU-5A) — prior art & ingestion design (2026-09-14)

**Motivation.** The deferred follow-up to the Merge FC node (§24.5 P1 v2):
bulk ingestion of burst-mode archives — recorders that interleave
**scheduled high-rate slots ("mini-runs") with continuous low-rate
streams** — into the per-run-cascade + FC-merge pipeline.

**Hardware data model (Phoenix MTU-5C / MTU-5A).** Both recorders write
*several simultaneous streams at different sample rates*; only the low
rates are continuous. Rate families are mode-dependent — MTU-5C: sparse
24 kHz + continuous 150 Hz ("24 kHz continuous or decimation with sparse
24 kHz and continuous 150 Hz", GEF equipment spec); MTU-5A adds a fourth
rate: 24K/2.4K/150/15, where AMT tasks use 24K+2.4K+150 and MT tasks use
2.4K+150+15. Duty cycles can be extreme (the official MTU-5A test file
records 1 s per 10 min at 2400 Hz).

**Official IAGA test data** (`github.com/IAGA-DVI-DataStandards/
mth5-test-data`, master): `phoenix/phoenix_test_data.zip` (44 MB,
MTU-5C session folders: per-channel subdirs, `td_150` continuous +
`td_24k` segment files, `recmeta.json`) and `phoenix_mtu/
phoenix_mtu_test_data.zip` (21 MB, MTU-5A: `1690C16C.{TBL,TS3,TS4,TS5}`).
Both downloaded and decoded locally (2026-09-14); the MTU-5A file's TBL
schedule decodes as: TS5 = 15 Hz continuous 20.3 h; TS4 = 150 Hz in
16-second slots every 10 min (122 slots); TS3 = 2400 Hz in 1-second
slots every 10 min, staggered +5 min against TS4 — and the TBL's own
STIM/ETIM are placeholders (2009–2011), not the data extent.

**Prior art — mth5's representation: burst becomes runs.**
`mt_io.phoenix.phoenix_collection.PhoenixCollection.assign_run_names`
(mt_io/phoenix/phoenix_collection.py:325): files < 1000 Hz group into ONE
run split on temporal gaps; files ≥ 1000 Hz become **one run per
segment** (`sr24k_0001…`). No special burst container — the burst model
is *many short runs at multiple sample rates* in one station.
`PhoenixClient.make_mth5_from_phoenix` (mth5/clients/phoenix.py) writes
each as a RunGroup.

**Prior art — aurora's processing answer: never merge in the time
domain.** `aurora.pipelines`: one dataset row per run; per decimation
level each run is decimated + STFT'd **independently** (per-run cascade,
`clock_zero_type="ignore"` default) and `merge_stfts` (time_series/
spectrogram_helpers.py) plain-`xr.concat`s the per-run STFT arrays along
time — exactly the per-run-cascade + MergeFC decision of §24.5 P1 v2.
Two upstream weaknesses confirmed in code: (1) **chunk-positional RR
pairing** — `shape_check_spectrograms` (issue #289 glb/lub trim) +
`time_axis_check_spectrograms` assume local and remote share the same
chunk structure; our epoch-anchored `clock_zero` is strictly stronger
(pairable grids regardless of run partitioning); (2) **deep-level
starvation is punted** — run-decimation rows yielding <
`min_num_stft_windows` are marked invalid and skipped
(transfer_function_kernel.py:316, with an explicit WCGW note that short
chunks cannot reach deep levels); our skip-with-warning matches.
(3) **Mixed sample rates are NOT handled**: `prototype_decimate` blindly
applies the config factor to every run, so 150 Hz and 24 kHz runs land
on incompatible frequency grids in one concat — upstream users process
one rate family per config.

**Upstream gaps found in the installed stack (mt_io 2025-12 lineage).**
(1) **Latent correctness bug**: `MTUTSN.read()`
(mt_io/phoenix/readers/mtu/mtu_ts.py:435) densely concatenates all
25-byte-headed 32-byte-header data blocks (`ts[:,
iblock*n_scan:(iblock+1)*n_scan]`, skipping the block header) and stamps
the whole array with the file start + uniform rate — but **each 32-byte
block header carries a true timestamp** (verified: TS3 block 0 =
07:55:00, block 1 = 08:05:00). For burst files every slot after the
first gets a wrong, compressed implicit clock (121 slots over 10 h come
out as a dense 121-second array). (2) `PhoenixCollection` crashes on a
flat MTU-5A directory (AttributeError in `_set_df_dtypes`; it only
walks MTU-5C-style station folders). Net: **no working MTU-5A burst →
MTH5 path exists upstream**.

**Design direction (decided 2026-09-14).** Two distinct **load nodes**
for ingestion, both feeding the per-run cascade builder (ComputeFC per
run, epoch-anchored `clock_zero`) + Merge FC:

- `mth5.io.load_phoenix_5c` (MTU-5C): walk session folders via the
  collection convention; continuous rates → gap-split runs, high rates →
  run-per-slot; persist as an mth5 archive.
- `mth5.io.load_phoenix_5a` (MTU-5A): parse the TBL schedule
  (HSMP/L3NS/L4NS/SRL3-5) **and the per-block 32-byte headers** for true
  slot times (fixing the upstream dense-concat bug); emit run-per-slot
  at each burst rate + continuous runs.

The staggered slot schedules across rate families make the epoch-anchored
window grid essential; families meet at shared frequency coverage only at
Merge FC (the 2.4K cascade's deep levels overlap the 150/15 continuous
coverage), so the cross-cascade merge is where the full-spectrum TF
assembles — v1 scope: single-family many-slots first, then cross-family.
Test corpora: the two IAGA archives (44 MB / 21 MB, gitignored data
dir); a synthesized burst collection remains the fallback for e2e.

### 24.8 MTU-5A burst ingestion — format reference, upstream scheme, design (started 2026-09-14)

Focus of the first burst-mode delivery (§24.7): the **legacy MTU-5A**
format; MTU-5C (already-working upstream collection) follows once the
procedure is proven.

**Format reference (verified byte-level on the official IAGA test
recording `1690C16C`, mth5-test-data/phoenix_mtu).**

- `TBL` — station metadata + acquisition schedule + scalar calibrations,
  a series of 25-byte blocks (12-byte tag + 13-byte value, mixed types;
  decoded by `mt_io.phoenix.readers.mtu.mtu_table.MTUTable`). Key tags:
  site/serial (`SITE`/`SNUM`), location (`LATG`/`LNGG` DDM formats,
  `ELEV`, `DECL`), channel map (`CHEX`..`CHHZ`, 1-indexed positions),
  dipole lengths (`EXLN`/`EYLN`), gains (`HGN`/`HGNC`, `EGN`/`EGNC`,
  `HATT`, `HNOM` normalization mA/nT, `FSCV` full-scale V), coil serials
  (`HXSN`..), powerline (`LFRQ`), GPS satellites (`NSAT`), firmware
  (`HW`), and the **slot schedule**: `SRL3/SRL4/SRL5` = sample rates per
  level (here 2400/150/15 Hz), `L3NS`/`L4NS` = seconds recorded per slot,
  `HSMP` = slot period. `STIM`/`ETIM` are schedule placeholders in the
  test file (2009-01-01..2011-01-01) — **never use them as the data
  time period**; actual block timestamps are the truth.
- `TSn` — one file per rate level (TS3 = SRL3 rate, TS4 = SRL4, TS5 =
  SRL5; TS2/TSL/TSH also exist): a sequence of **data blocks**, each a
  32-byte tag followed by `n_scan × n_ch × 3` bytes of 24-bit
  little-endian samples (channel-interleaved per scan, sign via two's
  complement on 24 bits). The **per-block tag carries its own UTC
  timestamp** (bytes 0-5: sec/min/hour/day/month/year-2000), serial
  (8-9), `n_scan` (10-11), `n_ch` (12), tag length (13, 32 = MTU-5 /
  16 = V5-2000), status (14), **saturation flags** (15), sample length
  (17), **sample rate** (18-19), rate units (20), **clock status** (21),
  **clock error in seconds** (22-25). Empirically confirmed on the test
  file across all three rates (rate field matches per level, clock
  status constant 4, clock error 0).

**Upstream processing scheme (mt_io, installed) — and its burst gap.**

- `MTUTable` decodes the TBL into mt_metadata Survey/Station/Run objects
  (incl. per-channel metadata) and three **scalar** calibrations
  (`ex_calibration`/`ey_calibration`/`magnetic_calibration` — here
  1.9073e-4 counts→mV/km-ish and 2.7287e-7 counts→nT).
- `MTUTSN.read()` allocates `(n_ch, n_scan·n_block)` and **densely
  concatenates every block**, skipping the 32-byte per-block tags
  (`ts_fid.seek(32, 1)`) — only the FILE header's start time is kept.
  `to_runts()` wraps the dense array as ONE RunTS `sr{rate}_001` stamped
  with the file start and the nominal rate, multiplying the TBL scalar
  calibrations into the data and stamping units (mV/km, nT).
- Consequence for burst recordings: **every slot after the first gets a
  wrong, compressed implicit clock** (the test TS3: 121 one-second slots
  spread over 10 h come out as a dense 121-second array). Continuous
  rates (TS5; TS4 when configured continuous) are coincidentally correct.
- There is **no mth5 client for MTU-5A**: `mth5.clients.phoenix` is
  MTU-5C-only (folder collection), and `PhoenixCollection` crashes on a
  flat MTU-5A directory. The only MTU-5A→mth5 path is the single-file
  `read_phoenix` (with the dense-concat defect above).

**emptyUI design (two nodes, feeding the existing graph).**

1. `mth5.io.phoenix_mtu` ("Load Phoenix MTU-5A", mirrors LoadLemi's
   contract): params `data_path` (directory), `survey_id`/`station_id`
   (defaults from TBL `SRVY`/`SITE`), `save_path`/`mth5_filename`,
   `sample_rates` (comma list; empty = every TS file present),
   `calibration_path` (optional FAP files via `attach_filters`, like
   LEMI; TBL scalar calibrations always applied + recorded in run
   metadata), `gap_method` unused — runs are never merged. Core
   `read_phoenix_mtu(data_path)`:
   - parse the TBL once (schedule, metadata, calibrations);
   - stream each TS file block-by-block, reading the per-block tags:
     true UTC time, rate, `n_scan`/`n_ch` consistency, clock status /
     clock error, saturation flags — **the fix for the upstream
     dense-concat defect**;
   - group contiguous blocks (`t[k+1] == t[k] + n_scan/rate`) into
     segments; **one mth5 run per maximal segment**, id
     `sr{rate}_{NNNN:04d}` (the mth5/aurora convention, §24.7);
     continuous rates collapse to a single run per rate naturally;
   - metadata: TBL survey/station/run metadata + per-run sample_rate and
     time period from ACTUAL block times; units stamped mV/km / nT
     (TBL scalars applied); per-block clock/saturation anomalies
     aggregated into node warnings (feeds the GPS-stray awareness of
     §24.5 P1 v2b);
   - cross-check parsed segment times against the TBL slot schedule
     (`HSMP`/`L3NS`/`L4NS`); report deviations, never reject.
   Output `{"mth5": archive}` open in "a" mode (LoadLemi contract).

2. `aurora.fc.build` ("Build FC Database"): input `mth5` (+ the
   ComputeFC STFT parameter set + `survey`/`station`/`runs`); iterates
   the listed runs, `compute_fc_levels` per run (epoch-anchored
   `clock_zero` default **"epoch"** here — staggered slot schedules make
   run-start anchoring useless), `merge_fc_containers` per station →
   output `fc` container. Optional `persist` writes each run's FC levels
   into the archive's FC layer keyed by run (aurora
   `save_fourier_coefficients` pattern) for reuse/caching. This node is
   the "per-run cascade builder" — it also serves the gapped-continuous
   route without any new machinery. Rationale for a node rather than N×
   ComputeFC + Merge FC: burst recordings yield 100+ slot runs and the
   Merge FC ports are fixed (fc1..fc4); the builder owns the iteration.

**Downstream data flow (all existing nodes).** Load MTU-5A (local) →
Build FC; Load MTU-5A (remote) → Build FC; Estimate TF (`fc`,
`ref_fc`) — the epoch-anchored grids pair slots across stations despite
staggered schedules (flagship e2e of test_fc_merge generalizes to real
slots); Coherence Sort (per_bin default) → Rotate. Coverage model:
2400 Hz slots populate the top octaves (level-0 FCs only — 1 s of
samples per slot), 150 Hz slots the mid-band, 15 Hz continuous the
tail; Merge FC's level-set union assembles the full spectrum
cross-family, EMpower-style. Starved slot levels skip with warnings
(§24.5 P1 v2b behavior).

**Test plan.** Corpus: the official IAGA archive (21 MB, gitignored
data dir). Unit: block-header parsing (times/rates/clock fields vs the
empirically decoded values); segment grouping (TS5 → 1 run of 73036 s;
TS4 → 122 runs of 16 s; TS3 → 121 runs of 1 s); schedule cross-check;
archive structure (run ids `sr2400_0001...`). e2e: Build FC with
`clock_zero="epoch"` over all runs; single-station Estimate TF spanning
the families; RR e2e with a synthesized second station (the corpus is
single-station) to pin slot-level remote pairing. Negative controls:
truncated block (short read), inconsistent `n_scan` across blocks,
clock-error ≠ 0 → warnings not rejections.

**Core-logic dry run (2026-09-14, validated).** A prototype of the
block-stream → segment grouping (per-block tag timestamps, contiguous
`next_start == prev_end` regrouping) reproduces the expected structure
on the real recording exactly: TS3 → 121 segments of 1 s, TS4 → 122
segments of 16 s, TS5 → 1 segment of 73036 s (07:46:52..04:04:08) —
consistent with the TBL schedule (`L3NS=1`, `L4NS=16`, 10-min slots)
and confirming the per-block-tag fix against the upstream dense-concat
defect. The staggered starts across rates (TS5 07:46:52, TS4 07:50:00,
TS3 07:55:00) confirm run-start anchoring cannot pair these streams —
the epoch-anchored grid (§24.5 P1 v2b) is the required mechanism.

**Prior art — the multi-run problem in aurora's tracker (surveyed
2026-09-14; corroborates the design).** aurora #80 (closed 2021,
"Handling of multiple acquisition runs as a single processing run"):
the frequency-domain solution — per-run decimation cascades + STFT
concat — is exactly Build FC's route; the closing comment defers
time-domain merging to #152 (still OPEN: NaN-fill corrupts anti-alias
filters — cf. #328 where a few NaNs at run edges propagate to all-NaN
series through `prototype_decimate`; informal community preference is
median-fill plus a max-gap-length knob, unimplemented — the
`fill_gaps_by_*` hooks in `run_summary.py` are TODO stubs). Our Merge
Runs' `gap_method` interpolation matches the median-fill preference;
Build FC avoids the whole problem for burst data. #103 (short runs
crash deep decimation levels) is the upstream face of the starvation
our skip-with-warning handles. MTU-5A specifics upstream: the mth5
legacy notebook (`make_mth5_from_phoenix_legacy_mtu.ipynb`) assembles
runs manually with a TODO to accommodate the format in
`PhoenixCollection` (its run-assignment never landed — the §24.7
crash), `MakeMTH5` lists "Phoenix legacy MTU (TODO)", and the TBL tag
semantics are partly guesses (the notebook asks the community for
contributions — our byte-verified layout, including per-block clock
status/error/saturation fields, is a candidate contribution). MTU-5C
segmented readers track per-segment GPS time, `missing_count` and
saturation counts in a sub-header — the analog of our per-block tags,
relevant for the later 5C loader.

**Performance note (2026-09-14, updated after the browser e2e).** The
real-browser e2e (workflow: load all rates → Build FC → sort → estimate →
plot) exposed two mth5/mt_metadata cost centers far beyond parsing:

- **`RunGroup.to_runts` ≈ 7 s per run** — a pydantic
  `validate_assignment` + deepcopy storm (2M deepcopy calls per run,
  station metadata re-copied per channel). Build FC bypasses it:
  channels are read via `ChannelDataset.to_xarray` (0.17 s for 5
  channels — `metadata.to_dict` without validation) and wrapped in a
  duck payload (`.dataset`/`.sample_rate`/`.start`); ~40x faster.
- **`StationGroup.metadata` ≈ 20 s per access** on a burst archive (full
  HDF5 metadata rebuild incl. pyproj datum validation per attribute) —
  Build FC originally touched it once per run (~80 min for 244 runs:
  the "never finishes" report). Fixed by hoisting the access out of the
  run loop; per-run rate/start come from the channel attrs/time coords.
- `RunGroup.from_runts` ≈ 0.55 s per run in the loader — replaced by
  `add_run(run_metadata=...)` + `from_channel_ts` (5x cheaper). Ingest
  of the full corpus is ~2.5 min (one-time; mt_metadata object churn
  dominates — a future optimization target if slot counts reach
  thousands).

Measured pipeline (full corpus, user params: window 256, 3 decimation
levels, hanning + deep taper from L3, first-difference PW, epoch anchor):
ingest 159 s; **Build FC 265 s** (244 runs); Coherence Sort (per_bin)
7.4 s; Estimate TF 0.8 s → 78 periods. The merged FC database carries
**8 rate-keyed levels** (2400 / 600 / 150 / 37.5 / 15 / 3.75 / 0.9375 /
0.234 Hz) — the cross-family union assembling exactly as designed.

**Mixed-rate merge correctness (fixed 2026-09-14).** The first e2e
attempt crashed in `merge_fc_containers`: decimation-level INDICES do
not correspond across rate families (a 150 Hz run's L0 is 150 Hz while a
2400 Hz run's L2 is also 150 Hz; alphabetical run order interleaves
`sr15`/`sr150`/`sr2400` families arbitrarily). Levels are now keyed by
their ACTUAL sample rate (the only commensurable identity across
cascades) and the output re-keyed to consecutive ints in DESCENDING
rate order — single-rate inputs reproduce the input indices exactly
(backward compatible; the old same-index rate-mismatch error is
superseded: mixing rates is the feature, not an error).

**Open decisions — resolved (2026-09-14).** (a) run ids follow the mth5
convention `sr{rate}_{NNNN:04d}`; (b) Build FC persists per-run FCs OFF
by default (the param exists; turn on when the procedure is settled);
(c) TBL scalar calibrations are applied at ingest (upstream behavior);
filter/FAP + Calibrate-node handling is later work.

**Delivered (2026-09-14).** `emptyui_mth5.phoenix_mtu`: core
`read_phoenix_mtu` (TBL parse, per-block tag streaming with true
timestamps, contiguous segment grouping, run-per-segment archive build,
TBL scalar calibration at ingest, quality aggregation — clock status /
clock error / saturation / truncated tails as warnings, loose TBL slot
schedule cross-check) + `LoadPhoenixMtu` node (`mth5.io.phoenix_mtu`,
LoadLemi contract). `emptyui_aurora.process.BuildFC` (`aurora.fc.build`):
mth5 + the ComputeFC STFT parameter set, `clock_zero` default "epoch",
per-run `compute_fc_levels` → `merge_fc_containers`, identity attrs per
run, optional persist. Tests: synthetic binary corpus writer (CI-safe;
TBL tags encoded via `MTUTable.TBL_TAG_TYPES`, TS blocks per the
verified layout) covering block-tag decode, slot/continuous grouping,
archive structure, sample-rate filter, quality warnings, truncated
tails, schedule cross-check; real-corpus structure test (121/122/1
runs, skipif absent); Build FC node tests (epoch-anchored merged grid,
identity attrs, Estimate TF flow). Full suite 385 passing. The tz
alignment fix in `_clock_zero_head_skip` (naive epoch vs aware/naive
payload starts) came out of the Build FC tests.

**Browser e2e round (2026-09-15): speed + duplicate-period fixes.** The
user's e2e (load all rates → Build FC → sort → estimate) reported
hundreds of seconds for the load and >10 min without finishing for
Build FC. Root causes — upstream mt_metadata/mth5 pathologies plus one
burst-specific correctness bug:

- **pyproj CRS rebuild storm (the big one)**: mt_metadata's datum
  validator (`Location.validate_datum`) constructs a fresh
  `pyproj.CRS` on essentially EVERY metadata attribute update — 25,625
  CRS constructions (75 s) for a single Build FC pass over 121 slot
  runs, triggered by mth5's uncached `get_channel`/`metadata`/recache
  paths. Fix: `emptyui_mth5._compat.apply_crs_cache()` LRU-caches
  `pyproj.CRS.from_user_input` (CRS objects are immutable, datum strings
  repeat exhaustively; applied at package import). Build FC 5 slots:
  126 s → 22 s.
- **`RunGroup.to_runts` ≈ 7 s/run** (deepcopy + validation storm,
  station metadata re-copied per channel): Build FC reads channels via
  `ChannelDataset.to_xarray` (0.17 s/run) wrapped in a duck payload.
- **`StationGroup.metadata` ≈ 20 s per access** (full HDF5 metadata
  rebuild incl. pyproj): hoisted out of Build FC's run loop; per-run
  rate/start come from the channel attrs/time coords.
- **`RunGroup.from_runts` ≈ 0.55 s/run** in the loader →
  `add_run(run_metadata=...)` + `from_channel_ts` (5x cheaper).
- Ingest 160 → 48 s; Build FC 265 → 72 s; total pipeline ~2 min.

**Duplicate/overlapping long-period points (fixed).** A rigorous
cascade should tile the spectrum without duplicates; two causes: (a)
`_container_geometry` bailed (→ the l2u table silently fell back to the
per-level bin walk) because the burst families' rate steps are not a
single uniform cascade (37.5/15 = 2.5) — the geometry now only requires
a constant window and per-level df (the template applies per level
anyway; `factor` is None for non-uniform steps); (b) the misaligned
families (150/15 = 10, not a power of 2) interleave near-duplicate band
centers that the exact log2 dedup could not collide. Fixes at the
source: `_l2u_band_table` gains `_resolve_cross_level_claims` —
deepest-rate-first period tiling (a period is estimated by the deepest
level that covers it; shallower bands are clipped to the unclaimed
remainder), the same deepest-wins rule as the merge, applied so sort
diagnostics and estimate share one table — and `merge_tf_levels` dedups
tolerance-based within half the level's center spacing (1/(2·ppo)
octaves; aligned cascades still collide exactly). Aligned single-rate
cascades are untouched (the template's level spans abut — claims never
bite; SDG31 parity uses manual-band overrides anyway).

**Per-family decimation trim (user scheme, implemented).** The
continuous (lowest-rate) family carries the full `decimation_levels`
cascade; higher-rate families decimate only until their rate would
reach the next family's native rate (`d_max = ceil(log2(r/r_next)/
log2(factor)) − 1`): 2400 Hz slots decimate once (→ 600), 150 Hz slots
once (→ 37.5), the 15 Hz stream takes the full parameter. Beyond the
compute saving this prevents same-rate levels computed from different
families from double-covering identical acquisition windows.

Measured e2e (full corpus, user params): ingest 48 s; Build FC 72 s (8
rate-keyed levels); sort + estimate 7 s → 46 periods, strictly
monotonic, min gap 0.263 octave (the template's own ladder spacing — no
near-duplicates).

### 24.9 Unit-chain regression (NIMS/SDG31 UI workflow) + sort simplification (2026-09-15)

**Unit-chain regression (user report).** A hand-built SDG31 node graph
(no Calibrate, core Detrend in the chain) produced apparent resistivities
anywhere between 10^-6 and 10^8 ohm-m depending on `magnetic_unit` —
never right. Two compounding defects, both fixed:

- **Units stripped at the core-node boundary.** `coerce_run` (RunTS →
  core TimeSeries) dropped the per-channel `units` attrs, and the core
  filter nodes rebuilt payloads without them — Compute FC then ran blind
  (electric counts hit the legacy ×1e-6 "mV/km assumed"; magnetic counts
  stayed untouched). Fix: `TimeSeries` carries a `units` dict, `coerce_run`
  extracts it from the RunTS dataset attrs, Detrend/Bandpass preserve it,
  Align re-stamps it on its RunTS outputs, Compute FC reads it.
- **Raw counts passed through with only a log warning.** FDSN-built mth5
  archives store raw counts with the response attached (Calibrate
  mandatory); Compute FC's unrecognized-units branch warned and passed
  counts through. Fix: `_reject_counts` hard-errors on count-family units
  ("digital counts"/"count"/"counts") with an actionable message. The
  unitless path (synthetic fixtures, core payloads) is unchanged — LEMI
  workflows are unaffected (LEMI data arrive in physical units).

Reproduced magnitudes (Executor, the user's exact graph): mag=auto/h →
ρ max 111 (accidental counts/counts partial cancellation, ~0.1× low);
mag=b → ρ max 1.75e8 (mode 'b' force-converts counts as if nT → H ÷1.26e9
→ ρ ×1.6e17 vs the counts passthrough). With Calibrate inserted:
ρ 82–1.06e3, matching the official SDG31 maximum (~1e3).

**Coherence sort simplification: band-free sort (user decision, §24.9).**
The per-bin statistic is the common convention (EMTF et al.) and has been
the default since the §24.6 flip — the band-aggregated statistic had no
remaining weighting consumer. Removed: the `statistic` param (per-bin is
the only behavior), `periods_per_octave`, `band_style`,
`min_bins_per_band`, `t_short` (all band-geometry — the estimator owns
banding), and `coherence_trim` (band-level trimming is meaningless
without sort-side banding; hard per-window exclusion is `mode="drop"`).
The sort node surface shrinks to coherence_min/coherence_max/mode/
apply_below_period/ensemble_windows/smooth_bins. Estimate TF drops its
chain-walking inherit of band params from the sort — banding is the
estimator's concern, one parameter set, no duplication.

**Silent-drop grammar made loud (user report).** The sort never removes
bins from the FC (grid verified preserved), but a band whose every sample
was fully down-weighted (per-bin ensemble collapses to ≤ n_in valid
windows on starved levels → statistic scores 0 → weight 0 → NaN) left
the band regression with no usable rows: estimate_tf skipped it and
merge_tf_levels dropped the all-zero band silently — the TF just lacked
period points. Now: estimate_tf warns naming the skipped band periods
and retained-row counts; merge_tf_levels warns naming dropped degenerate
bands; the Estimate TF progress line reports merged/offered band-periods
and the processing notes record both counts. Kill + warn semantics
confirmed as the desired policy (a "can't estimate → pass through"
alternative rejected: it would re-admit un-QC'd data).

**Plot Coherency reworked to per-bin rendering** (per (bin, window)
window-ensemble coherence, median across windows, guidance band at the
sort bounds; one series per decimation level) — it renders exactly the
statistic the sort applies, per §20.1, and loses its duplicated banding
params and inherit declaration.

**Executor: stale params warn (user decision).** Saved workflows carrying
params a node no longer declares are processed unchanged — the executor
resolves only manifest params — but each stale param now emits a
node-level progress note ("ignoring params not declared by node type…")
instead of failing silently.
