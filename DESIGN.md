# emptyUI — Design Document

A node-based, ComfyUI-inspired visual workflow interface for magnetotelluric (MT)
data processing and visualization.

Status: **M0–M3 complete (2026-09-09)**; M4 (community & advanced backlog) in
progress — ranked by gain/effort in §12.

**How to read this document.** Sections §1–§12 are the **living contract** —
the current architecture and behavior, organized by subsystem (framework, backend,
frontend, data support, processing). Delivered design decisions are recorded in
`docs/history/` (`m2-delivery.md`, `m3-delivery.md`, `m4-delivery.md`), which
preserve the **original §13–§24 numbering verbatim** — historical section
references in code docstrings and commit messages resolve there. Where this
document consolidates a former dated section, the consolidation note points at
the history file.

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

## 2. Architecture Overview & Repository Layout

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

Repository layout (monorepo):

```
emptyui/
├── README.md
├── DESIGN.md                        # the living contract (this document)
├── docs/
│   ├── deployment.md                # remote/embedded deployment guide
│   ├── plugin-guide.md              # plugin authoring contract
│   └── history/                     # verbatim delivery records (M2/M3/M4);
│                                    #   preserve original §13–§24 numbering
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
│   ├── emptyui-mth5/           # archive/IO plugin: MTH5/LEMI/Phoenix, TF I/O, plots
│   │   └── src/emptyui_mth5/
│   ├── emptyui-aurora/         # processing plugin: FC/TF estimation, QC, sorts
│   │   └── src/emptyui_aurora/
│   └── web/                     # React frontend
│       └── src/
│           ├── graph/           # canvas, node rendering, edges
│           ├── panels/          # palette, properties, results
│           ├── api/             # REST/WS clients, generated types
│           └── stores/
├── examples/                    # sample workflows + tiny sample datasets
│   ├── workflow/                # ready-to-load workflow documents, kept in sync
│   │                            #   with the current node manifests (regenerate by
│   │                            #   loading and re-saving when surfaces change)
│   ├── data/                    # sample datasets; raw instrument corpora are
│   │                            #   gitignored (regenerable, see each subdir)
│   └── plugin-template/         # working plugin template
├── test/verification/           # recorded parity/metrics JSONs + the verification
│                                #   scripts that reproduce them (history records)
└── scripts/                     # dev up/down, codegen
```

Dependency policy:
- `emptyui` core deps: `fastapi`, `uvicorn`, `pydantic`, `numpy` only.
- `emptyui-core-nodes`: adds `matplotlib` (+ optional `pandas`, `scipy`).
- Everything else (mth5, aurora, mt_metadata, …): plugin extras.
- Frontend pinned via lockfile; API types are **handwritten** in
  `packages/web/src/api/client.ts` — the node manifest is plugin-open
  (unknown fields are ignored downstream), so catalog types stay
  deliberately loose; the server-side shapes the frontend depends on are
  pinned by a contract test (`packages/emptyui/tests/test_api_contract.py`),
  so removals/renames of API keys fail loudly.

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
| **Data I/O** | Bring data in/out of the graph; convert formats | Load MTH5, Load CSV, Save MTH5, Load Calibration, Import TF, Export TF, Read FC, Write FC |
| **Processing** | Transform data; produce new typed outputs | Detrend, Bandpass Filter, Calibrate, Align Runs, Compute FC, Build FC Database, Coherence Sort, Estimate TF, Merge TF Levels |
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
| `ts.runs` | Run Collection | list of `mth5.timeseries.RunTS` | Extract Run's `time range` mode: ALL runs intersecting a UTC window (the inclusive burst-mode selection); consumed by Compute FC (Build FC's route, §9.3) |
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
  color. The core tags ship as `CORE_PORT_TYPES` in the registry
  (`packages/emptyui/src/emptyui/registry.py`); node-declared `port_types`
  override per tag, and the frontend keeps a mirroring built-in table only
  as a fallback for older/offline backends. Fallback families: blue = time
  series, dark green = frequency domain (`fc`), red = transfer functions
  (`tf`), purple = metadata/config, gray/yellow = generic (`any`, `table`,
  `plot`).
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
LoadMTH5 ─mth5─▶ ExtractRun ─ts.run─▶ ComputeFC ─fc─▶ EstimateTF ─tf─┬▶ ExportTF
(LoadLemi / LoadCsv)                        │                        ├▶ PlotAppResPhase
                                              └─fc─▶ CoherenceSort ─fc─▶ EstimateTF
                                                                           ├─tf.level─▶ MergeTFLevels
                                                                           └─tf─▶ PlotTipper

LoadMTH5 ─mth5─▶ ExtractRun(selection=time range) ─ts.runs─▶ ComputeFC ─fc─▶ ...
                (all runs intersecting the UTC window — the inclusive
                 burst-mode selection; Build FC's route, §9.3)

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
  in nT like fluxgate B, so also converted to A/m; §10.5) — the `magnetic_unit`
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
  on/off, band style (`l2u` default / `uniform` / `manual`, §10.2),
  `min_bins_per_band`, `t_short`, periods-per-octave. Band edges are derived
  internally. Output `tf` + per-level `tf.level`. The TF is built aurora-style:
  rows/cols carry the **actual** channel names and the `channel_nomenclature`
  map is attached — a 2×2 impedance, or a 3×2 impedance + tipper. The remote
  reference (**RME_RR**, aurora-backed) pairs local/remote FC samples by row;
  by default (`pairing="intersect"`, §9.3) windows are matched by absolute
  start time and unmatched rows are dropped loudly on both sides; `strict`
  preserves the legacy identical-grid contract (both grids validated with
  clear errors); the runs must be aligned first (see
  `Align Runs`). When the ref fc's provenance records interpolated merge
  gaps inside the local span, the pairing warns (§3, M4 #2).
- **`Align Runs` node (`mth5.runs.align`)** — two `ts.run` inputs → two `ts.run`
  outputs, both normalized to an mth5 `RunTS` cropped to their **common
  acquisition overlap**. Run **collections** (Extract Run 'time range' mode,
  the Compute FC / Calibrate convention) are aligned member-wise: every
  member of both sides is cropped to the common overlap of the two
  collections' union spans, members entirely outside are dropped with a
  named warning, and extent rows are updated to the cropped spans (a
  single run acts as a 1-member collection whose output kind is
  preserved). Required for remote reference: same sample rate (aurora
  hard-errors on mixed rates; collections are WARNED — the FC pairing
  refuses real mismatches). Overlap is
  computed from absolute start/end times (GPS) when available, with a warning
  fallback to relative offsets (single runs only; collections require
  absolute spans). The remote station may be **magnetic-only**
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
  (coil archives are nT too, so hx/hy → A/m as well; §10.5).
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
  copy for `zmm` output. All supported file formats store the impedance in the
  SEG/EMTF convention ([mV/km]/[nT]) while internal TFs are SI (E in V/m,
  H in A/m) — mt_metadata's readers/writers copy impedance values verbatim, so
  `Import TF`/`Export TF` convert the impedance block (and its error; the
  tipper is dimensionless) at the format boundary.

### Merging multiple runs into one TF estimate

A single TF estimate is often built from **several acquisition runs** (e.g.
Phoenix MTU-5 segments). aurora's canonical mechanism allows multiple
`KernelDataset` rows and **concatenates their STFTs at the FC level** — it
does *not* merge in the time domain. emptyUI delivers this route through the
aurora plugin's **`Merge FC` node / `Build FC Database`** (§9.3), which
concatenate the per-run Fourier coefficients along time, per decimation
level — no fabricated samples; the idiomatic route for burst archives and
for building one TF from many runs.

The **`Merge Runs` node** (`mth5.runs.merge`) is the *physical* counterpart:
it combines the raw series of one station via `RunTS.merge()` into a new mth5
run group. It is explicitly opt-in because it **interpolates across gaps**
(gap policy is a param) — the fabricated samples stay limited to the gaps and
are stamped as provenance (`merged_gap_intervals`, hazard below). The
kernel-dataset `selection`-payload route once described here was never
delivered (its producer node remains M4-deferred; `View Selection` ships
hidden with no producer).

Constraints (both routes): all runs must share the same native sample rate,
and channel components must be consistent. Remote-reference multi-row
intersection is an open aurora TODO — emptyUI resolves it structurally
(pairing by the identical absolute window grid, §3 Estimate TF; the
`intersect` pairing mode of §9.3 covers residual clock/coverage
differences), and the residual silent hazard is warned about explicitly
(below).

**Remote-reference pairing hazard (M4 #2)**. The harmful variant is a remote
merge that bridged acquisition gaps by interpolation (physical `Merge Runs`):
local windows inside the gap pair with **fabricated** remote samples while
the identical-grid check still passes. The provenance chain: `Merge Runs`
records the bridged intervals on the merged payload
(`merged_gap_intervals`, plus `merged_source_runs`); `Align Runs` warns when
the aligned overlap includes them and carries the provenance through the
crop, `Compute FC` stamps it on the fc attrs, and the RR pairing sites
(`Estimate TF`, two-FC `Coherence Sort`) warn naming the affected windows —
with the remedy (Merge FC concatenates Fourier coefficients without
fabrication; or exclude the gap interval via the time-selector window).
`Align Runs` also warns when the overlap crop discards >10% of an input
(the E1b partial-remote silent loss).

**Burst-mode caveat.** For burst archives (scheduled high-rate slots +
continuous streams) inter-slot gaps are normal acquisition geometry, not
defects — the gap warning does not apply to the `Build FC` / `Merge FC`
route (which never fabricates samples; the epoch-anchored window grid pairs
slots across stations by design). A burst-mode generalization — per-slot
coverage QC derived from the per-window `run_id` coordinate — is a recorded
follow-up; the delivered warning is anchored to Merge Runs' interpolation
provenance, which the burst route does not produce.

### Data passing between nodes

Nodes exchange typed *handles*, not raw payloads: array-like containers →
xarray objects in an in-process store keyed by handle ID; tables → DataFrames;
files → paths in a per-run workspace; plots → artifact files. This keeps
WebSocket messages small and lets the same executor serve batch/CLI use later.

### Storage: file-based, backed by MTH5

No internal database. **MTH5 is the server-side container**:
- A workspace directory **per run** (`ctx.workspace`); intermediates a node
  chooses to persist are written into `<workspace>/workspace.mth5` (see
  `persist.py`). The persist path uses **one cached open handle per node
  execution** (history §24.8): the per-level open/close is quadratic — every
  `MTH5.close_mth5()` re-summarizes ALL summary tables (the fc summary
  recursively walks every persisted FC level; 294 levels ≈ 96 s of pure
  HDF5 walking). `persist_fc_level(m=...)` accepts the shared handle;
  `get_workspace_mth5`/`close_workspace_mth5` manage it. A persistent
  `~/.emptyui/workspace/` is a future item.
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
`file`/`directory` (and an array-coercing control for `list`), and the
remaining string-like widgets fall back to a single-line text input.

#### Tooltip / description / palette protocol

- **`tooltip`** — the hover tip; exactly **one sentence** saying what the node
  *does* (imperative, ≤ ~100 chars, no parameter lists; the
  first-sentence fallback truncates at 140 chars).
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
- **`enabled_when`** (per param): conditional-UX rule — the widget is editable
  only when the named sibling param currently holds one of the listed values:
  ```python
  Param("remote_reference", "boolean", default=False,
        enabled_when=("estimator_engine", ("Robust", "RME", "RME_RR")),
        cascade={True: (("estimator_engine", "Robust"),)}),
  ```
  Complements `cascade` (which *sets* values; `enabled_when` *gates* editing).
  Rendering-only — the executor never enforces it (nodes validate their own
  params).
- **`populated_when`** (per **port**, output or input): the port-analog of
  `enabled_when` — the port is populated only when the named sibling param
  holds one of the listed values. Rendering-only: the UI dims ports that
  will not be populated; the executor stores unpopulated outputs as None.
  Currently **dormant** (no manifest declares it): Extract Run's former
  dual outputs (`data` in `run id` mode, `runs` in `time range` mode) were
  collapsed into the single collection-carrying `data` output — collections
  ride the same `ts.run` port kind, so the dimming has no user. Kept for
  the ComfyUI-style use case: a node taking several mutually exclusive
  input types but only one at a time (e.g. object detection — a SEGM *or*
  SAM detector input, the sibling param selects which) dims the inputs the
  current mode does not take; input-port dimming is a small deliberate
  extension of the same frontend check when a manifest needs it.
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
  bundled plugin, `<org>.<domain>.<name>` for external plugins. The registry
  does not enforce the namespace today — it is a review-time gate; installing
  a plugin is an act of trust equal to `pip install` itself (L-C14).
- **Trust model (L-C14)**: a plugin's code runs with the server process's full
  privileges. The executor's guarantees are CONTRACTS, not sandboxing: nodes
  must be functional (never mutate inputs or previously returned outputs —
  §8.1), `Node` instances are shared across runs (instance attributes become
  cross-run state — keep per-run state in `ctx`), and `ctx.publish_artifact`
  may write into the current run's workspace. Anything beyond that
  (filesystem access, network, subprocesses) is the plugin's own business and
  the user's own risk; the settings dialog's plugin list is therefore the
  natural place to audit what is installed.
- **Open, namespaced port types**: plugins may introduce new types but must
  namespace them and declare UI metadata via the node's `port_types` ClassVar;
  reusing core tags is strongly preferred, and consumers duck-type documented
  payload attributes instead of importing plugin containers.
- **Self-tests required**: every plugin ships a pytest suite (see template).
- **Models declared, not bundled**: `models` in the manifest
  (id/filename/sha256/url/license); binaries live in a configured model
  directory (platform-provisioned in remote mode, §11) — never inside the wheel.
- Manifests include `version`; workflows record plugin versions for
  reproducibility. The full authoring contract (immutable-input execute
  contract, packaging layout, publishing checklist) is **`docs/plugin-guide.md`**
  with a working template in `examples/plugin-template/`.
- A future `emptyui plugin publish/index` flow can distribute community plugins
  as ordinary PyPI packages.

### 5.1 Plugin enable/disable

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

**Backend API** (see §7.3): `GET /api/plugins`,
`POST /api/plugins/{name}/enable|disable`.

**Config & persistence**
- `EMPTYUI_DISABLED_PLUGINS` sets the baseline at startup (the §11 platform knob).
- Runtime toggles are **in-memory only**; optional JSON-file persistence
  activates only when `EMPTYUI_SETTINGS_PATH` (or `create_app(settings_path=...)`)
  is configured; the default user-config-dir settings file is deferred. Unknown
  names are silently ignored.
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
  messages. Saved workflows carrying params a node no longer declares are
  processed unchanged (only manifest params are resolved); one node-level
  progress note lists all stale params per node — stale surfaces degrade
  loudly, never silently.

---

## 7. Backend — Execution, API & Runtime Settings

### 7.1 Execution model

- **Scheduler**: Kahn topological sort; branches run **sequentially**.
  Concurrent thread-pool execution is M4 #12 (`cpu_bound` hints).
- **Caching**: server-lifetime in-memory LRU (`NodeCache`, 256 entries). Cache
  key = **structural signature** computed in topological order:
  `sha256(type, manifest version, effective params, per-input upstream
  signatures)` — node ids excluded (copy/paste duplicates share entries);
  signatures recurse through the DAG so invalidation propagates. Data payloads
  are **never hashed** (xarray/RunTS impractical to digest); path params
  (`file`/`directory` widgets only) pointing at existing files hash
  `(path, size, mtime_ns)` (Nextflow-style; the path is taken
  **data-root-relative** when the file lives under the data root or project
  dir, so cache keys survive relocation — a plain copy refreshes `mtime_ns`
  and misses the cache (recompute, the safe direction); only mtime-preserving
  moves stay hits), so externally modified input files invalidate. On a cache
  hit the executor re-registers outputs and
  **copies artifact files into the new run's workspace**; vanished artifact
  files re-execute the node. Cached nodes
  emit `node_status` with `"cached": true`. **Immutability contract**
  (ComfyUI model): nodes must be functional — never mutate inputs or previously
  returned outputs. `DELETE /api/cache` drops all entries.
  Two node-declared opt-outs keep side-effecting/terminal nodes correct:
  `cacheable = False` (Save MTH5, Export TF — the side effect IS writing the
  file; a hit would report "done (cached)" with nothing written; the node
  re-executes every run, inputs still cached) and `invalidates_inputs = True`
  (Save MTH5's close-in-place closes the shared input archive — the executor
  evicts the direct-input producers' entries so later runs re-open the
  archive instead of serving the closed handle). Save As writes through a
  temp file + `os.replace` and the same-file check is normalization-proof.
- **Failed-run hygiene**: a failed/cancelled run closes the open MTH5
  payloads it left in its store — except payloads still served by the cache
  (never poison the cache); best effort, never masks the run's error. This
  keeps the §9.3 stale-flag recovery truthful: the in-process write-open
  registry (loader outputs, workspace persistence handles) dies with the
  process, so after a crash the flag is genuinely stale and auto-clearable;
  while a live writer holds an archive, `ensure_h5_openable` refuses
  `h5clear` and names the remedy.
- **Partial runs**: `POST /api/runs` accepts `targets: [node_id, …]` —
  "execute to here" runs the targets plus their ancestral closure; other nodes
  emit no events. Unknown targets rejected with 422. The web UI exposes
  "Run to here" in the properties panel.
- **Progress**: nodes call `ctx.progress(frac, message)`; forwarded to all
  connected WebSocket clients.
- **Failure**: the executor aborts the whole run (error event +
  `run_finished: failed`); per-branch continuation is future work.
- **Cancellation**: `POST /api/runs/{id}/cancel` sets a per-run
  `threading.Event` polled between nodes (including after the final node, so
  cancelling during the last node reports `cancelled`); toolbar **✕ Cancel**
  enabled while queued/running. A node cannot be interrupted mid-execution —
  cancellation takes effect at the next node boundary.
- **Payload summaries**: when the executor materializes a `ts.run`/`ts.channel`
  output handle (fresh or restored from cache) it emits
  `{"event": "payload", node, port, type, summary}` with a duck-typed extent
  summary — absolute ISO bounds (UTC) when the record carries a real start,
  relative seconds otherwise; placeholder epochs (1980/1970) degrade to
  relative. An open MTH5 archive summarizes as `kind: "mth5"` with per-run
  extents. These drive the frontend time selector (§8.3). Poll/WS replay
  carries them like artifact events.

### 7.2 Run lifecycle events (WebSocket)

```
{ "event": "run_started",   "run_id": "r1" }
{ "event": "node_status",   "node": "n2", "status": "running" }
{ "event": "node_progress", "node": "n2", "frac": 0.42, "msg": "filtering…" }
{ "event": "artifact",      "node": "n3", "id": "a7", "kind": "image",
  "handle": "h12", "url": "/api/artifacts/a7/preview" }
{ "event": "payload",       "node": "n2", "port": "out", "type": "ts.run",
  "summary": { … extent … } }
{ "event": "node_status",   "node": "n3", "status": "done" }
{ "event": "run_finished",  "run_id": "r1", "status": "completed" }
```

The `artifact` event carries the output `handle` id so viewer nodes resolve
exactly which figure arrives on which input edge, even when one node publishes
several figures.

### 7.3 REST API

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
| GET/PUT | `/api/settings` | Runtime settings view + live-knob edits (§7.4) |
| GET | `/api/files` | Server-side file picker (data-root clamped, §11.2) |
| GET | `/api/health` | Liveness probe (`{"status": "ok"}`) for platform gateways |
| WS   | `/api/ws` | Run events stream |

A run-level artifact gallery is not implemented — viewers on nodes are the
per-figure/table display, each with its own download link. MVP keeps state on
the file-based workspace (§3); a DB is added only if run history demands it.

### 7.4 Runtime settings

The ⚙ Settings dialog is the single settings surface; the backend is the
source of truth (`GET/PUT /api/settings`).

- **Precedence**: env/platform baseline wins → UI edits recorded on top →
  defaults. Knobs whose start value arrived from env/kwarg are
  **platform-baselined and not UI-editable** (PUT → 409 naming the knob) —
  confinement/serving semantics stay restart-level decisions.
- **Live-editable whitelist** (`LIVE_KNOBS` in `emptyui.settings`):
  `workspace_ttl_seconds` (null/0 = off), `max_concurrent_runs`
  (null/0 = unlimited), `cache_max_entries` (default 256). Applied
  immediately (TTL loop re-reads per pass, admission check reads the live
  holder, `NodeCache.resize()` trims). `RuntimeSettings` is the mutable holder
  the app closures read.
- **Deployment rows (read-only display)**: `data_root`, `serve_dist`,
  `cors_origins`, `root_path` — changing confinement/serving mid-flight is an
  architecture change; env + restart only.
- **Planned (reserved, inert)**: `project_dir`, `cache_max_gb`,
  `cache_ttl_days` — the §12.3 disk cache lands into this scaffolding.
- **Persistence**: one sectioned JSON file via `EMPTYUI_SETTINGS_PATH`
  (`disabled_plugins` + `settings` sections; atomic write; unknown keys
  ignored on load — files may travel). Opt-in: unset = memory-only, no file
  is created. Settings PUT validates against the whitelist (numbers ≥ 0 or
  null-for-off; null not allowed for knobs with a non-null default).
- **Frontend**: Settings dialog sections — Server & retention (live numbers,
  empty = off, explicit Save), Deployment (read-only code values), Planned
  (disabled rows, "planned" tag), Plugins (§5.1).
- **Policy (user decision)**: null/0 = off/unlimited; values ≥ 1 are real
  limits. Unbounded local mode is deliberate; the TTL knob is the only
  retention mechanism until the §12.3 disk cache lands.

---

## 8. Frontend

Stack: **Vite · React · TypeScript · @xyflow/react · Zustand · TanStack Query**

### 8.1 Components & state

- **GraphCanvas** — React Flow; maps workflow JSON ⇄ RF nodes/edges bidirectionally.
- **NodePalette** — grouped by `category`, populated from `GET /api/nodes`.
  New nodes are placed at the **center of the user's current viewport**
  (`screenToFlowPosition`, two-phase: the card is re-centered after it
  renders, since its height is unknown before the params mount) — a fixed
  origin disappears once the user pans/zooms away from it.
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

### 8.2 Canvas interaction protocol (ComfyUI parity)

- **Disconnect gestures**: drag a connected handle to the empty pane → unlink;
  drag either endpoint anchor of an edge → drop on a compatible handle re-routes,
  drop on the pane deletes; click an edge + `Delete`/`Backspace`. Reconnecting
  an input enforces one-edge-per-input: the old edge must survive the filter, so
  dropping the *only* edge is a delete, never a silent re-route.
- **Copy/paste**: `Ctrl/Cmd+C` snapshots the selected subgraph (nodes + internal
  edges); `Ctrl/Cmd+V` pastes with fresh ids, deep-copied params, remapped
  internal edges and a 24 px fan-out offset — external edges are *not* rewired.
  `Ctrl/Cmd+D` duplicates in place. Keystrokes in form controls are ignored.
- **Selection**: click = exclusive, `Ctrl/Cmd+click` = additive, `Shift+drag` =
  box selection (`selectionMode: Full`); `Delete`/`Backspace` deletes the full
  selection.
- **Initial view**: `fitView` capped at `maxZoom: 1` with generous padding
  (fraction-based, v12-compatible).
- **Node sizing**: standard node width 280 px (widened for the time bar; §8.3).

### 8.3 Time selector (time bar)

Selecting a time slot (e.g. to avoid noisy intervals) in records spanning hours
to days:

- **Extent source**: the backend payload summaries (§7.1) provide each
  connected `ts.run`/`mth5` port's extent; the frontend *intersects* the
  extents of the ports a param's `time_bar` binding lists (comma-separated;
  e.g. Align Runs shows the overlap of both inputs). Honest constraint: the
  extent is known only after the upstream node has run once.
- **`Param.time_bar`** (rendering-only): the input port name(s) whose summary
  drives the frontend time bar for this param pair.
- **`Param.suggest_from`** (rendering-only, §10.6): the input port name whose
  payload summary offers select-option suggestions (an fc database's
  `rates` tiers) — the frontend extends the param's options in an editable
  datalist after a run; pre-run the free-text placeholder applies.
- **`TimeBar` widget** (custom, zero deps): horizontal acquisition bar with two
  pointer-driven handles sandwiching the slot, shaded selection, extent tick
  labels, synced bidirectionally with the paired Start/End edit boxes (which
  remain the node's own string params — the bar only writes through their
  onChange). Guards: handles stay in the record, cannot cross, slot kept
  > 1 s (server rejects ≤ 1 s; widget clamps ≥ 2 s at second resolution);
  out-of-extent box values draw at the track edge (server clamps with a
  warning — degrade, never reject).
- **Hosts**: Plot Time Series (start/end of the streamed record), Extract Run
  (bar bound to the matched `mth5` run extent, absolute GMT; in `time range`
  selection the bar shows the union extent over all runs — the whole
  coverage the mode extracts), Align Runs (one
  pair over the aligned overlap — identical-interval outputs), Merge Channels
  extent; extent-unknown → passthrough with warning. A bounded window missing
  the resolved run entirely is rejected with an actionable message naming the
  matched run (the clamp would extract a useless 0-sample slot; history §24.12 L-M3
  keeps this one hard reject deliberately). Boundary convention (L-M10,
  deliberate): Extract Run's ISO crop is END-INCLUSIVE (`xarray.sel(slice)`)
  while the windowed processors (Align Runs / Build FC) use half-open
  `[t0, t1)` arithmetic — parity holds for fully-contained windows; the
  boundary sample differs by convention. Merge Runs is
  deliberately skipped: its per-run windows live in the selection table rows,
  not in a node-param pair.

---

## 9. Data & Instrument Ingestion

This section covers getting field data into the graph: channel nomenclature
(§3), the calibration framework, and the per-instrument loaders. Format-level
byte records and the ingestion research history are preserved in
`docs/history/` (§24.7/§24.8, §24.10–§24.12).

### 9.1 Calibration framework

`emptyui-mth5/calibration.py`:
- `parse_fap` (LEMI `.chn`/`.def`), `parse_json_calibration` (mt_io format),
  `build_fap_filters` (per-channel `FrequencyResponseTableFilter`, units
  nanoTesla / mV·km by channel prefix), `attach_filters` (mt_timeseries style —
  `run.filters` + per-channel `attrs["filters"]` in the AppliedFilter format
  that survives an MTH5 round trip; channels without a filter get a unity
  coefficient filter so `RunTS.calibrate()` passes temperature channels
  warning-free; channels that already carry a real archived response chain
  are never replaced by unity — the instrument response would otherwise be
  silently dropped from all downstream FC/TF math), `find_calibration_file`
  (auto-pickup, station-pair files
  preferred), `load_calibration` (dispatch by extension).
- **Below-table policy**: flat clamp (first-row modulus, zero phase, extended
  to 0.001·f_min) — log-extrapolation was worse.
- `Load LEMI-424`: `calibration_path` param + `extend_flat`; filters attached
  **before** `from_runts`, so the calibration is archived with the run. The
  `mth5.calibration.load` node feeds the explicit path of `mth5.calibrate`.

### 9.2 Calibration placement — frequency-domain, inside Compute FC

**Decision: calibrate in the frequency domain, inside Compute FC** — not a
separate downstream node, not the time-domain route. Rationale: the conventional
MT approach calibrates in the frequency domain (matching aurora's
`calibrate_stft_obj`); the MTU-5A makes the time-domain route impractical — the
induction-coil response falls off steeply at long periods, the burst archive has
~1000 short slots (per-slot FFT/iFFT round trips are wasteful), and per-frequency
division is trivial at every decimation level. STFT, prewhitening/recolor, and
response division are all linear frequency-domain operations — they commute.

**Mechanism.** Compute FC applies the response right after the STFT, before
returning/persisting the FCs — mirroring aurora exactly. No new port: the mth5
convention puts the response on the channel metadata
(`AppliedFilter(name, applied, stage)` → `/Survey/Filters`, typed filter
objects: ZPK / FAP / coefficient / time delay / FIR, evaluated as
`complex_response(f)`), which the ts.run payloads already carry. Applied-flag
semantics prevent double-calibration **at the FC level** (Compute FC): only
filters with `applied=True` are removed (divided out — the response IS part
of the recorded data); a prior time-domain `Calibrate` flips the flags to
`False`, so both routes coexist (the time-domain Calibrate node remains for
the classical route and explicit
control; NB mt_metadata's own time-domain `RunTS.calibrate` removes every
chain filter regardless of the applied flag — do not chain time-domain
Calibrates). The FC records the applied calibration in its attrs
(`calibration_applied`, e.g. `"hx=box+coil; ex=box"`).

The calibration chains ride the payload through the graph: core `TimeSeries`
carries `calibration` (`{channel: [(filter, applied), …]}`) coerced from RunTS
dataset attrs, preserved by `TimeSeries.copy`, the core Detrend/Bandpass nodes,
and rebuilt on Align Runs / Split → Merge Channels outputs.

### 9.3 Instrument loaders

| Loader | Notes |
|---|---|
| Load MTH5 | Native archives; runs/FCs/TFs |
| Load CSV | Generic series |
| Load LEMI-424 | `channel_nomenclature` select (lemi12/lemi34), FAP `.chn`/`.def` calibration auto-pickup |
| Extract / Split / Merge Channels, Align Runs, Merge Runs | Archive/series manipulation (§3) |
| **Load Phoenix MTU-5A** | Burst-mode ingestion — see below |
| TF Import/Export | EDI/XML/ZMM/AVG/J |

**HDF5 locking policy (uniform).** Every emptyUI open of an HDF5 archive —
loaders creating outputs, Load MTH5, the workspace persistence layer — passes
`locking=False` (mth5 forwards it to `h5py.File`). Handles live for the
server lifetime (cached payloads), so a default-locking handle would pin the
file against every other open (errno 35) until restart; worse, HDF5 rejects
**mixed** locking configurations across concurrent handles in one process
(*"file locking flag values don't match"*) — exactly the loader-creates +
Load-MTH5-reads flow inside one graph. With the uniform policy, concurrent
opens coexist and external tooling is never blocked by our handles; HDF5's
superblock consistency flag still rejects new opens of write-active files.

**Stale write-flag recovery.** A server killed while a loader/Save/persist
handle had an archive open for write leaves that flag set — every later open
fails until the flag is cleared. Load MTH5, Save MTH5 and the workspace
persistence layer pre-check the archive (`_compat.ensure_h5_openable`) and,
when `h5clear` is on PATH, clear the flag with a loud warning (the data is
intact as long as the last write transaction completed); without `h5clear`
the error names the remedy.

**NIMS (USArray) note.** Native NIMS ingestion is **not delivered** — USArray
NIMS recordings are automatically uploaded to the IRIS Data Center and
archived as miniSEED (a seismic-data format, repurposed here for MT), so
there is no local NIMS corpus to read. The `nims` *nomenclature keyword*
(mt_metadata `CHANNEL_MAPS`) remains fully supported — it names the channel
convention of IRIS-acquired archives, not a loader. The upstream mth5 FDSN
client (`mth5.clients.FDSN.make_mth5_from_fdsn_client`) already fetches
IRIS-held data into MTH5; wrapping it as a **`Load from FDSN`** node is a
recorded follow-up (§12.2). Until then, IRIS-sourced data is consumed from a
pre-fetched MTH5 via `Load MTH5` (see the `nims-remote-reference` example
workflow, which loads an archive fetched that way).

#### Phoenix MTU-5A burst ingestion

Burst-mode recorders interleave **scheduled high-rate slots** with **continuous
low-rate streams** (MTU-5A families: 2400/150/15 Hz; MTU-5C: 24 kHz/150 Hz).
Upstream has no working MTU-5A burst→MTH5 path (the stock reader densely
concatenates blocks and stamps a compressed implicit clock — each 32-byte block
header carries the true UTC timestamp/rate/clock status/saturation flags).

`read_phoenix_mtu(data_path, …)` (core) + `Load Phoenix MTU-5A` node
(`mth5.io.phoenix_mtu`, LoadLemi contract):
- Parse the TBL once (schedule, survey/station/run metadata, scalar
  calibrations — always applied at ingest; `STIM`/`ETIM` are placeholders,
  never the data extent — actual block timestamps are the truth).
- Stream each TS file block-by-block reading the per-block tags: true UTC
  time, rate, `n_scan`/`n_ch`, clock status / clock error, saturation flags.
- Group contiguous blocks (`t[k+1] == t[k] + n_scan/rate`) into segments;
  **one mth5 run per maximal segment**, id `sr{rate}_{NNNN:04d}`; continuous
  rates collapse naturally.
- Params: `survey_id`/`station_id` (defaults from TBL), `sample_rates`
  (comma list; empty = every TS file), `calibration_folder` (§9.2 ASC
  matching), `start`/`end` (UTC ISO or epoch; empty = full recording —
  filtered at ingest, per-second granularity, partial slots become short
  runs), `stream_session_hours` (0 = off; runs longer than the target are cut
  into consecutive fixed-length sessions on block boundaries so the
  continuous stream gets run-level QC granularity, §10.4),
  `session_by_slot` (cut the continuous stream at the TBL slot grid,
  every `HSMP` seconds, instead of a fixed length — **slot-granular QC**:
  a bad slot interval is masked without sacrificing the surrounding
  hours, so the surviving stream stays contiguous over much longer spans
  than hour-splitting allows. The cascade is unaffected by the session
  labels — `regroup_contiguous` cascades the kept contiguous span and
  spans never depend on QC outcomes — so finer session labels only
  refine rejection granularity; the Select Runs overlap guard bounds the
  collateral at ±1 FC window per dropped session. Falls back to
  `stream_session_hours` with a warning when the TBL carries no HSMP).
- Quality aggregation: per-block clock/saturation anomalies and truncated
  tails become warnings; the TBL slot schedule is cross-checked loosely
  (report, never reject). Runs with any saturated block carry a per-sample
  `saturated` auxiliary channel (§10.1).
- **Slot schedule output (second output `slot_schedule`, a `table`)**: the
  integer slot grid per burst family — the TBL schedule defines each
  family's slots (`HSMP` slot period in seconds, `L3NS`/`L4NS` seconds
  recorded at the BEGINNING of every slot); slot identity is derived from
  the actual block timestamps (tag truth) with the schedule as the
  cross-check: the slot period is the observed minimum spacing of
  consecutive run starts, a disagreement with `HSMP` warns (never
  rejects — the official IAGA corpus is known to stagger TS3/TS4 against
  the TS5 slot grid; per the standard the bursts should align with the
  slot start, an upstream documentation issue to be filed). One row per
  run (`run_id, sample_rate, slot_index, slot_start, slot_end, start,
  end, n_samples`); continuous families leave the slot columns empty and
  report one row per run (sessions included). This is the pairing
  foundation: two stations' runs pair by slot, and a slot *range*
  selection feeds Extract Run / Build FC directly.
- **Macro flow (current stage)**: `Load Phoenix MTU-5A → Extract Run →
  Align Runs → Compute FC` is the plausible path today — Extract Run
  exposes `ts.run` for any run (each burst slot, or the continuous
  stream as one run), so the classic single-rate alignment/cropping flow
  works on burst archives. For the burst-native route
  (`Load → Build FC`), the **slot-based selection layer** is delivered:
  the loader's `slot_schedule` table + Build FC's UTC window (whole-slot
  selection) — the same window on two stations is the coarse pairing.
  **Remote reference with burst slots is FC-level**: although a 1-s slot
  is too short for meaningful time-domain RR, the pairing happens in the
  frequency domain — as long as the local and remote FCs from the time
  slots are matched (identical epoch-anchored window grids, §10.1),
  RME_RR pools the paired rows across all windows and the slots are
  fully usable. Per-slot `ts.run` extraction is also the input surface
  for time-domain extensions (e.g. NN denoising).
- **Window-grid intersection pairing (fine layer — delivered)**: the
  pairing sites (`Estimate TF`, two-FC `Coherence Sort`) accept a
  `pairing` mode — **`intersect` (default)** matches local/remote windows
  by absolute start time (one-to-one, closest pairs first) and drops the
  unmatched rows on BOTH sides with a loud per-level report (counts plus
  the affected slots via the per-window `run_id` coordinate);
  **`strict`** preserves the legacy identical-grid contract (Align Runs /
  parity harnesses). `pair_tolerance_s` (default 0 = exact match) covers
  GPS-stray clocks: window starts shift by the receiver's clock error and
  the closest pairs still match. The mapping to the official Phoenix
  processing (SSMT2000) is deliberate: the **frequency grid stays strict
  in both modes** — a window-settings/duty-cycle mismatch refuses to
  work, exactly as upstream — while the time dimension follows SSMT2000's
  "drop the slot whose clocks cannot match" semantics (never
  interpolate). The default flip is safe: identical grids take the fast
  no-op path, so the Align Runs and parity workflows behave exactly as
  before. Not a mandatory feature — the slot-level Build FC window (the
  coarse layer above) remains the de-facto workflow; the fine layer only
  trims the residual coverage differences that whole-slot selection
  cannot remove.

#### Build FC Database (`aurora.fc.build`)

Input `mth5` (+ the ComputeFC STFT parameter set + `survey`/`station`/`runs`);
iterates the listed runs, `compute_fc_levels` per run (epoch-anchored
`clock_zero` default **"epoch"** — staggered slot schedules make run-start
anchoring useless), `merge_fc_containers` per station → output `fc` container.
Optional `persist` writes each run's FC levels into the archive's FC layer
keyed by run (off by default).

- **Slot-aware run selection (`start`/`end` params)**: a UTC window keeps
  every run whose time period intersects it — **whole runs, never cropped**
  (a burst slot is either in or out; the loader's `slot_schedule` table maps
  UTC windows to slot indices). This is the slot-based selection layer
  driving Build FC, and the same window on two stations' Build FC nodes is
  the **coarse remote-reference pairing** (§9.3): both stations compute FCs
  over the same acquisition window, in slot granularity, without any
  time-domain alignment machinery.

- **Cascade spans (`regroup_contiguous`)**: maximal contiguous spans of
  same-rate runs are cascaded as ONE payload (adjacency within two samples,
  from the run summary columns) — the physically right unit for unbroken
  acquisitions; gapped slot runs stay separate. Hourly sessions alone cannot
  form deep decimation levels (an hour caps at ~0.23 Hz with window 256);
  cascading the kept contiguous span restores the long-period tail. Build FC
  builds spans from ALL sessions (before any QC — spans never depend on QC
  outcomes), reads each span exactly once, and stamps per-window `run_id` at
  session granularity.
- **Per-family decimation trim**: the continuous (lowest-rate) family carries
  the full `decimation_levels` cascade; higher-rate families decimate only
  until their rate would reach the next family's native rate — prevents
  same-rate levels from different families double-covering identical windows.
- **Performance hardening** (mth5/mt_metadata pathologies): `pyproj.CRS`
  LRU cache (`_compat.apply_crs_cache`), `Unit`-object LRU cache
  (`_compat.apply_unit_object_cache` — unit validation rebuilds Unit
  objects from a DataFrame lookup per attribute update), memoized
  dynamic metadata-class enhancement
  (`_compat.apply_metadata_class_cache` — mth5's group/channel init
  builds a NEW pydantic model class per init; the deterministic
  enhancement is memoized per input class, ~80 s on a ~1000-run burst
  ingest), `ChannelDataset.to_xarray` channel
  reads (~40× faster than `RunGroup.to_runts`), station metadata hoisted out
  of the run loop, `add_run`+`from_channel_ts` archive writes, and loaders
  that write station/survey summaries directly from the known segment
  geometry (mth5's child-walking `update_metadata` costs ~40 s per property
  access on a ~1000-run archive). The **group
  `.metadata` properties are the same storm** — `station.metadata`/
  `survey.metadata` rebuild the FULL hierarchy through pydantic on every
  access (~30-40 s on a 145-run burst archive): identity (ids) is read free
  from the HDF5 group paths (`_group_id`) and scalars (location) from the
  group's HDF5 attrs — Apply sites: Extract Run run-id mode (fast read +
  `_CachedRunTS` — channels are built lazily once per downstream access,
  ~0.2 s vs ~7-19 s), Merge Runs (per-run fast read + registry re-resolve
  on the merged output), Build FC, Save FC (`_fc_identity_attrs`). Run-level
  `run.metadata` accesses (~10 ms) and error-path messages stay on the
  metadata properties.

#### Extract Run time-range selection (ts route)

Extract Run carries two selection modes: **`run id`** (legacy surface — one
run, cropped by the time bar; empty id falls back to the first run-summary
row, warned loudly on multi-run stations) and **`time range`** (the
default) — the inclusive selection for burst-mode
instruments: **ALL runs of the station whose period intersects the UTC
window** are extracted (empty start/end = every run of the station), each
cropped to the window overlap (the
time-selector semantic — clamped, never fabricated). A `sample_rates`
param scopes the families (empty = every family: the staggered burst slots
plus the continuous stream). The single `data` output port carries the
**`ts.runs` run collection** when range mode selects runs (a single run in
run-id mode rides the same port as a 1-member collection; §4's
runtime-shape hints tell the frontend which form is present).

Compute FC consumes the collection on **Build FC's route**: per-run
cascades (`compute_fc_levels`) merged along time per decimation level
(`merge_fc_containers`), with Build FC's **per-family decimation trim**
(higher-rate families decimate only until their rate would reach the next
family's native rate — history §24.8) and **per-window `run_id`** stamping (§9.3).
A multi-run collection without `clock_zero` anchoring warns naming the
remedy (`epoch`): run-start grids differ per run, which is fine without
remote reference and refuses loudly at the RR pairing sites. Parity
anchor: for a window that fully contains the runs, Extract(range) +
Compute FC produces **identical FC levels** to Build FC (regression
`test_compute_fc_collection.py::test_collection_route_parity_with_build_fc`).

The classic single-run flow keeps the legacy semantics — including the
first-run fallback — but the fallback now **warns loudly** whenever the
station holds more than one run (naming the matched run, the station's run
count, and its longest runs), and a bounded window that misses the resolved
run's extent entirely rejects with an actionable message naming the matched
run, its extent, and the station's longest runs (a burst archive's first
summary row is a short slot; the continuous stream is what users want).

Run-id mode also rides the **fast channel read** (history §24.8): the run's
channels are read via `to_xarray` (crop-before-wrap when the time bar
bounds the window) and wrapped into a real `RunTS` via a cached-channels
subclass — `RunTS(dataset)` construction is ~0.2 s vs `to_runts`'s ~7-19 s
mt_metadata deepcopy/validation storm, and the cached `__getattr__` avoids
rebuilding a fresh `ChannelTS` on every attribute access downstream. The
wrap carries the archive's survey/station/run ids (read free from the HDF5
group paths — the `station.metadata`/`survey.metadata` properties are the
same pydantic storm) and the resolved filter registry, so Calibrate/Split/
Merge Channels and Compute FC behave identically downstream (parity-
checked: identical channel data, FC levels, and calibrated output).
Conversely, the singular-semantics ts tooling (Split/Merge Channels, Align
Runs, Calibrate, Detrend/Bandpass, ts plots) **rejects collections loudly**
with node-appropriate remedies — collection mapping and per-family viewing
are M4 backlog #21.

### 9.4 Universality — session labels for continuous instruments

The QC architecture validated on MTU-5A (session labels → Select Runs /
Residual Sort → drop-with-overlap-guard → `regroup_contiguous` cascade
spans) is instrument-agnostic; the "slot" is really a **session label on a
regular grid**. Two mechanisms make it universal:

- **Build FC `session_hours`**: tiles every run into fixed-length,
  contiguous time intervals labelled `{run id}_sNNNN` (0 = off). Sessions
  never cross run boundaries; the cascade spans are built from ALL
  sessions and never depend on QC outcomes; source runs are read lazily
  (at most once per span) and sliced by UTC time. With
  `regroup_contiguous` a run's own sessions merge back into one cascade
  span (they are contiguous) while gapped runs stay separate — so the
  deep levels are unaffected by the label granularity. Without regroup,
  each session is its own cascade and deep levels starve (the documented
  burst behavior). This is what keeps a noisy hour from condemning an
  entire acquisition day: LEMI/NIMS runs are whole days, and the QC
  granularity is now a parameter.
- **Compute FC stamps the per-window `run_id` coordinate** in the classic
  single-run flow (`Extract → Compute FC`) too — whole runs get their own
  id (all-or-nothing QC; stamped after persistence so the verified mth5
  round trip is untouched). Session-level QC routes through Build FC's
  `session_hours`.

**Session length guidance.** The statistical unit is FC windows per
session, not wall time: ~1 h at 1 Hz with window 256 gives ~28 windows
(50% overlap) — the validated MTU-5A stream recipe; 10 min gives ~4.7
(too noisy for scoring at 1 Hz, fine at 512 Hz where a 10-min session
holds thousands of windows). Rule of thumb: each session should hold
≥ ~20–30 FC windows at the QC reference band. The deep level bounds how
fine is useful — one deepest-level window spans ~68 min of raw 1 Hz data
(5 levels × factor 4), and each dropped session costs ±1 window per
level — so finer than ~10 min mostly buys scoring noise.

With this, the burst and continuous instruments share one scheme: the
same QC nodes, the same pairing (`pairing="intersect"` handles the
residual clock/coverage differences for both), and the slot schedule
remains a Phoenix-specific refinement (the loader's `slot_schedule`
table + `session_by_slot`, which simply sets the session grid to the
physical slot rhythm).

---

## 10. Processing Semantics

The physics contract of the TS → FC → TF chain. Ablation histories and
verification records live in `docs/history/` (§14, §16, §24).

### 10.1 Compute FC — the STFT chain

emptyUI implements **taper → FFT only** (boxcar default, 50% overlap,
`boundary=None`); the remaining aurora steps are deliberately absent:

- **Prewhitening (default on)**: `prewhitening="first difference"` (the
  default) applies the time-domain first difference and the exact
  first-difference inverse `÷ (1 − e^{−j2πfΔt})` recoloring (clean-signal
  FCs preserved; the DC bin is NaN — recoloring is undefined there.
  History: aurora's idealized `j2πf` recoloring does not exactly invert its
  central-difference operator and leaves a low-frequency amplitude bias —
  the exact inverse filter is used instead). Ablation verdict (SDG31 vs
  the official EMTF product): the
  controlling factor at short periods is **DC/edge leakage suppression** —
  the current default's entire deficit is concentrated at 7.3–15.1 s; a
  zero-endpoint taper (hann) or prewhitening closes it; boxcar without
  prewhitening is catastrophic (~1 decade, all periods). Defaults are now
  **boxcar + first-difference PW** (the dof-maximizing recipe, coherent
  since PW is on; empty = off keeps the legacy emptyUI behavior) — the
  recipe for noise-floor-dominated stations is boxcar + PW at
  short-period levels (history §24.3 E1).
- **Per-level taper schedule**: `window_type` (base) + `window_type_deep` +
  `deep_from_level`. Physically motivated, not tuned: white-ish fluxgate
  noise dominates level 0 (boxcar maximizes dof), the diurnal Sq line lies
  inside day-scale windows (hann suppresses its sidelobe leakage). Per-level
  window recorded in the fc attrs.
- **Per-window detrend (opt-in, off by default)** — `per_window_detrend`
  ("linear") detrends each STFT window before tapering, keeping window-level
  trends out of the low-frequency bins. The default stays off: emptyUI
  detrends globally via the upstream `Detrend` node, and per-window
  detrending was ablated as NOT a lever for the long-period tail
  (history §24.3 E2).
- **No clock-zero truncation by default**: `clock_zero` ("" = run-start
  anchoring, "epoch" = Unix epoch, or explicit ISO) anchors window starts to
  `t0 + k·step` per level and drops the head partial window, so per-run FCs
  of different runs/stations share one window grid — the prerequisite for
  remote-reference pairing across separately-computed runs (burst slots).
  Levels with fewer samples than one window are skipped with a warning.
- **FC normalization**: scipy 'spectrum' scaling (÷ win.sum()); aurora
  additionally applies a linear spectral-density calibration factor. All
  ratio-based products (Z, tipper, coherence, OLS σ) are scale-equivariant
  and unaffected; persisted FC amplitudes and spectrogram dB are in
  emptyUI's own convention (history §14.1, item A5).
- **Saturation rejection (`reject_saturated`, default on)**: every FC window
  whose flagged-sample FRACTION exceeds `saturated_max` (default 0.01) is
  NaN'd in all in/out FCs; the mask max-pools through decimation;
  the count is recorded in attrs. A no-op for payloads without the mask.
  Verified: surgical — medians barely move, daytime garbage windows vanish;
  official-EDI parity preserved (history §24.12).
- **Calibration application**: §9.2 (frequency-domain, applied-flag-aware,
  DC column NaN when the response vanishes there).

### 10.2 Frequency grid (band styles)

`Estimate TF` banding parameter set: `band_style` (`l2u` default /
`uniform` / `manual`), `periods_per_octave` (3), `min_bins_per_band` (20),
`t_short` (0 = template head).

- **`l2u` (default)** — fixed reference template + log-uniform
  window-starvation coarsening + optional head clip, reproducing the official
  hand-crafted USArray grid strategy (SDG31, ppo≈3) for any (window, factor):
  the SAME decoded template is applied to EVERY decimation level (periods
  scale by the cascade factor); a level whose band sample counts fall below
  `min_bins_per_band` is rebuilt as a coarser uniform table (endpoints
  included, overlaps dropped until disjoint or ≤2 bands). `t_short` drops
  bands shorter than the given period (applied at a single choke point —
  `_resolve_period_bands` — for BOTH schemes and the metadata-less fallback,
  so it is honored everywhere the param is advertised). The l2u grid is
  **window-invariant** (the template scales ×window/128 — the window only
  changes FC bin resolution). Verified: SDG31 rotated metrics 0.020 median /
  0.071 p90 / 93% pass vs the exact official table's 0.019/0.045/98%
  (history §14.9 for the full experiment record).
- **`uniform`** — the uniform log2-grid partition: shortest period Fs/4 per
  level, longest window/4, band centers on a global log2 grid anchored at
  4/Fs so consecutive levels land on the same absolute periods; levels
  overlap and the merge keeps the deepest level covering each period
  (more cycles = better long-period resolution). Legacy alias `"log"`.
- **`manual`** — explicit band table: comma-separated band-center frequencies
  + per-level counts → per-level `(f_hi, f_lo)` edges at geometric midpoints
  (the harness-proven reconstruction of the official grid, for
  apples-to-apples comparisons). `enabled_when`-gated params.
- Legacy `"emtf"` is an alias of `"l2u"`; the retired `"index"` mode raises a
  clear error.
- **Cross-level tiling**: deepest-rate-first period tiling
  (`_resolve_cross_level_claims`) — a period is estimated by the deepest level
  that covers it; shallower bands are clipped to the unclaimed remainder.
  `merge_tf_levels` dedups tolerance-based within half the level's center
  spacing (aligned cascades still collide exactly; burst mixed-rate families
  dedup near-duplicates).

### 10.3 Coherence Sort (per-bin statistic)

The node (`aurora.process.coherence_sort`, between Compute FC and Estimate TF)
scores FC windows with the **EMTF-style per-frequency window-ensemble
multiple coherence** — the band-free sort:

- **Statistic**: per (frequency bin, window), the multiple coherence² of each
  output on the magnetic inputs estimated over a sliding window ensemble
  (±`ensemble_windows` windows, default 16, center included) with optional
  pooling over ±`smooth_bins` adjacent bins (a fixed smoothing scale,
  deliberately not the estimate's band table). Adjusted-R² debias removes the
  sample-coherence high bias (a small residual under the null remains,
  ~1/(W−p), empirically immaterial at ens=16). Weights are per (bin, window) —
  finer than per-window scalars and independent of any band table.
- **Params** (the whole surface): `coherence_min` (0.25), `coherence_max`
  (1 — the default upper bound; a tighter bound flags perfect coherence
  as a noise signature: few DOF or coherent
  noise), `mode` (`weight` default / `drop` / `off`), `apply_below_period`
  (empty = all levels; band-center periods at/above the threshold pass
  unsorted — scope the remedy to the noise-floor head without perturbing the
  healthy deep tail), `ensemble_windows` (16), `smooth_bins`.
- **Two-FC design**: accepts `fc` + `ref_fc` and outputs both (mirroring
  Estimate TF) because remote reference *replaces* the local magnetic
  channels: single-FC mode uses local outputs' multiple coherence on the
  local magnetic inputs; two-FC mode uses the local outputs' coherence on the
  remote magnetic inputs combined (minimum) with the local↔remote magnetic
  transfer coherence. Both criteria are reported in the diagnostics table.
  The two FCs must share identical time/frequency grids (validated, same
  errors as Estimate TF). The shared per-row weight is baked as √w into both
  output FCs (weighted-LS semantics: one weight per observation row).
- **Loud failure semantics**: a band whose every sample was fully
  down-weighted leaves the band regression with no usable rows — `estimate_tf`
  warns naming the skipped band periods and retained-row counts;
  `merge_tf_levels` warns naming dropped degenerate bands; the Estimate TF
  progress line reports merged/offered band-periods and the processing notes
  record both counts. Kill + warn (a "can't estimate → pass through"
  alternative was rejected: it would re-admit un-QC'd data).
- **Known caveat (uncorrected)**: overlapping windows and shared noise
  realizations mean the effective DOF is smaller than the nominal row count;
  no Egbert-style DOF correction is applied to the estimate's σ² (M4 #7
  statistics work). The per-bin debias handles the statistic itself.
- `Estimate TF` skips non-finite FC rows when gathering band rows (required
  by `drop` mode, and gives correct RR semantics: a row is used only if both
  local and remote FCs are finite).

### 10.4 Run-level QC (Select Runs, Residual Sort)

Noise rejection at **run/session granularity**, before estimation. Motivation
(Weckmann et al. 2005; Szarka 1988): power and coherence alone are misleading —
cultural noise is often correlated across E and H, defeating robust processing
and coherence sorting (measured: stronger sorting *amplified* this site's
scatter — negative result, history §24.13). The site's solar plant makes
daytime noise track output, not the clock, so blind day/night removal is
wrong — rejection is **data-driven with a soft diurnal prior**.

- **Select Runs** (between Compute FC and Coherence Sort): per-level scoring;
  per-run per-bin median of the per-window |Z|² (bivariate E/H from the run's
  own calibrated FCs) within the reference band (default 1–10 s); station-wide
  clean level per bin = a configurable percentile (default 50) over the runs'
  medians — self-calibrating (storms raise every run equally and cancel;
  cultural noise raises single runs). **One-sided up check** (noise only ever
  inflates E): drop if run/level > `deviation_max` (default 2) scaled by the
  soft diurnal prior (`prior_factor(h) = 1 + (prior_max−1)·0.5·(1 + cos(2π(h−12)/24))`,
  UTC offset default 8 h) — daytime runs are judged up to `prior_max`×
  stricter, night runs by the full threshold; a measured-clean day still
  passes, a noisy night is still dropped. Separate **instrument-fault check**,
  prior-independent: drop if level/run > `dead_max` (default 2). Runs without
  in-band bins pass through unscored. Per-run diagnostics published as a table
  artifact; kept/dropped lists ride the fc attrs; relative-time payloads
  disable the prior with a warning. **Overlap guard**: windows of neighbor
  sessions that overlap a rejected one picked up its samples through the
  decimation cascade and are dropped too (one window duration wide).
- **Residual Sort** (`aurora.process.residual_sort`, the orthogonal
  complement): a
  first-pass TF predicts each FC window's outputs from the pooled magnetic
  vector; the per-window unexplained-power fraction is aggregated per run
  (median over windows, then median over the run's bands after per-band
  normalization against the station-wide median). **Independent, cascadeable
  node** — inputs are the fc payload and the first pass's `tf levels`; the
  chain BuildFC → EstimateTF → Residual Sort → EstimateTF is a loop-free DAG.
  `estimate_tf_levels` returns each level's `band_intervals` (the pooling
  key). **One-sided up-check only** (a LOW unexplained fraction means the
  data fit the model WELL, not dead channels — the inherited down-check
  wrongly flagged 340 good-fit slots and cost the long-period tail).
  Semantics: a tight-but-wrong run DOES drop (it disagrees with the model,
  with a bounded structured fraction rather than the explosive score of
  incoherent noise); empty period scope warns and passes through; unscored
  runs pass through; the shared soft diurnal prior applies.
- **Stage-1 separation study (negative result, kept as the record)**:
  magnetic polarization concentration (Weckmann's strongest discriminator)
  is LOW everywhere on the validation corpus and the major-axis angle rotates
  with time (natural source geometry) — the polarization feature is dropped
  from the workflow plan; the cluster-spread feature is a diagnostic column
  only. The QC effort belongs to the residual lever (history §24.13).

### 10.5 Units & scaling

- **Magnetic**: B/H mode derived from nomenclature/channel names (bx/by →
  fluxgate B→H; hx/hy → coil H, archived in nT like fluxgate B). Conversion
  is **units-first, mode-second**: explicit A/m attrs → leave untouched
  (whichever mode — the payload already carries H); an explicit induction
  unit (T/mT/µT/nT/pT) → scale by its own prefix ÷ μ₀; only payloads with
  **no** usable unit attrs fall back to the mode alone ('b' converts —
  fluxgate is archived in nT; 'h' leaves — the legacy unitless synthetic
  convention). Regression: `test_magnetic_units.py` (incl. the `b`+A/m
  precedence case, issues.md H2).
- **Electric (`Compute FC`, attr-aware)**:

  | condition | action |
  |---|---|
  | `e_dipole_length_m` set | raw-potential path: `x · 1e-3 / dipole` (warning if attrs canonicalize to mV/km — contradiction) |
  | attrs = milliVolt per kilometer | ×1e-6 (LEMI archives; legacy path) |
  | attrs = Volt per meter | passthrough (already physical) |
  | attrs = milliVolt per meter | ×1e-3 |
  | attrs = milliVolt (raw potential) | **hard error**: set `e_dipole_length_m` |
  | unrecognized | warning + passthrough (never silently scale what cannot be interpreted) |
  | no attrs (unitless) | ×1e-6 — legacy behavior (synthetic fixtures, core payloads); **warns** on every use: missing attrs are indistinguishable from a stripped payload |

  The fc records `electric_scaling` in attrs. Unitless payloads behave exactly
  as the legacy path (all synthetic fixtures and the USArray verification
  unaffected; regression `test_electric_units.py`).
- **Counts rejection**: FDSN-built mth5 archives store raw counts with the
  response attached (Calibrate mandatory); Compute FC **hard-errors** on
  count-family units ("digital counts"/"count"/"counts") with an actionable
  message instead of a silent passthrough (the unitless path is unchanged —
  LEMI workflows are unaffected).
- **Core-node boundary**: `coerce_run` preserves the per-channel `units` dict
  through the core Detrend/Bandpass nodes; Align re-stamps it on its RunTS
  outputs (a dropped-units regression once produced 10⁻⁶–10⁸ Ω·m output).
- **Calibration relabeling**: `RunTS.calibrate` rewrites each channel's units
  attr to `channel_response.units_in`; gain-correction FAPs (units_in ==
  units_out) keep the archival label, and declared unit transformations are
  honored by the attr-aware path.

### 10.6 Display & plot conventions

Publication-quality standard (USArray convention; reference figure
`examples/data/nims-usarray-test/SDG31b_H28coh.png`). `Plot AppRes & Phase`
and `Plot Tipper` stay separate
nodes — no combined plot.

- **TF error propagation**: `_solve_band`/`build_tf` propagate
  `transfer_function_error` + `tipper_error` (std of each complex element —
  what mt_metadata's EDI writer squares into `.VAR`). RME/RME_RR populate
  `cov_ss_inv`/`cov_nn`; OLS uses the classical `(XᴴX)⁻¹·σ²` covariance.
- **Error-bar math** (aurora/mtpy formulas): `σ_|Z| = σ_Z/√2`;
  `σ_ρ/ρ = 2·σ_|Z|/|Z|`; `σ_φ(rad) = σ_|Z|/|Z|`; tipper per Re/Im panel:
  `σ_component = σ_T/√2`.
- **Phase display**: off-diagonal components folded into (0°, 90°) by ±180°
  (full `mod 180`, both directions), ylim 0–90°, ticks every 15°; **diagonal
  components keep signed phase in [-180°, 180°]**. A value no ±180° fold can
  place inside (0°, 90°) is **reported, not silently dropped**: one loud
  warning names the component, the clipped point count and their value range
  (issues.md H5). The axis stays at 0–90° — parity with USArray/aurora
  `plot_config`, never an auto-expanding ylim; raw values stay in the View
  TF table.
- **Axes**: x = Period (secs) log10 (decade ticks, shared across panels);
  ρ_a = log10 (Ω·m); φ = linear (Degrees); tipper = Re/Im, ylim [-1, 1];
  **equal per-decade scale** (one decade of period = one decade of ρ_a — MT
  convention).
- **Markers & colors**: XY = ○ open circle (blue), YX = ✕ (red) + distinct
  linestyles (colorblind-safe); XX = square, YY = diamond. Tipper: Re = ○ /
  Im = ✕ per component panel (panels = Re/Im, series = components — the exact
  mirror of AppRes & Phase). Symbols only, shared log-period x-limits,
  boxed legend.
- **Style** (shared module `emptyui-mth5/plotting.py`; minimal mirrored
  `emptyui-core-nodes/_plotstyle.py` — core stays mth5-free): font sizes
  ~9–10 pt, dpi ≥ 150, `bbox_inches="tight"`, PNG + optional PDF.
- **TF station metadata**: ComputeFC attaches survey/station/run + lat/lon/elev
  to fc attrs; EstimateTF/MergeTFLevels pass them to `build_tf`, which sets
  `tf.survey_metadata`/`station_metadata`/`run_metadata` — the single point of
  truth for plot titles AND EDI/XML/ZMM export headers.
- **Title & provenance**: EMTF/USArray-style
  `<station id><run id> [ LON = …; LAT = … ]` plus a processing-provenance
  subtitle; `print_params` boolean (default off) toggles the provenance line,
  which is anchored to the measured x-label bottom and word-wrapped to the
  axes width.
- **Plot Time Series** (`core.plot.timeseries`): stacked channel plot;
  accepts a **run collection** (backlog #21 Tier 1) — a `sample_rate` param
  (default `auto` = the lowest rate present, the continuous stream) picks
  the family to view, and all runs of that family plot as segments on the
  shared absolute time axis (gaps stay visible; the time bar crops each
  run — clamped — runs entirely outside are skipped with one aggregated
  warning). The suggestion datalist offers the collection's distinct rates
  after a run (`suggest_from`).
- **Plot Coherency** (`aurora.plot.coherency`): renders the **per-bin
  statistic** the sort applies (per (bin, window) window-ensemble coherence,
  median across windows, one series per decimation level, guidance band at
  the sort bounds); fc-direct — works without a sort node in the graph. On a
  **merged burst FC database** a `sample_rate` param selects the rate family
  (default `"auto"` = the lowest NATIVE family — the continuous stream,
  e.g. 15 Hz Phoenix / 1 Hz LEMI; a single-rate fc keeps every cascade
  level) — the families' cascades overlap on one period axis otherwise.
  The remote's tier is matched by rate, not index (different stations may
  have different family ladders). The available tiers are streamed in the
  fc payload summary (`kind: "fc"`, `rates`) and offered as select
  suggestions (`Param.suggest_from`) after a run.
- **Plot Phase Tensor** (`mth5.plot.phase_tensor`): Caldwell-style PT
  ellipses vs period colored by skew (Φ = X⁻¹·Y from Z = X + iY; ellipses
  constructed in display space and mapped back — data-space ellipses are
  log-warped). No PT error bars (MVP; noted in the plot label).
- **Plot Spectrogram (fc)** (`aurora.plot.spectrogram`): input `fc` — power
  = 10·log10|FC|², coefficients reused, no second FFT; `channel` param with
  inherited-comma-split candidates; `decimation_level` indexes the cascade
  STEP within the selected rate family (0 = the family's native tier —
  original data; 1..k one decimation deeper each; −1 = the deepest
  available step; descending tier order reproduces legacy cascade indices
  exactly); the
  same burst-aware `sample_rate` tier selection (explicit rate wins; the
  title annotates the rate); the legacy ts-based node is hidden but still
  executable.
- **Plot Tipper**: `tipper_ylim` param ("auto" | "min,max"), mirroring
  `rho_ylim`.

---

## 11. Deployment

emptyUI is **not** a standalone multi-user app. The target remote scenario is
emptyUI's FastAPI backend (+ built web frontend) embedded **as a module** into
a platform that already provides authentication — one backend instance **per
user or team**, mounted under a sub-path of the platform's URL space. The
design goal is a small **compatibility contract**, not an auth subsystem. The
operational guide is **`docs/deployment.md`**.

### 11.1 Trust model

| Concern | Owner |
|---|---|
| Authentication (certs, SSO, session cookies), TLS termination | platform gateway |
| User/team management, authorization, routing, rate limiting | platform |
| Data provisioning (original time series, archives — system of record) | platform |
| emptyUI process lifecycle (one per user/team) | platform |
| Identity-blind execution, confined file access, proxy-compatible API/WS, retention & limits | emptyUI |

**Identity-blindness contract** — emptyUI contains no auth code and never
enforces identity. The gateway authenticates every request (including
WebSocket upgrades — browsers cannot set `Authorization` headers on WS, so
gateway-side cookie/cert auth is the natural fit). Per-user isolation is
achieved by construction (one process per user/team).

### 11.2 Storage realms

- **Data root (read-only)**: the platform provisions original data into a
  configured directory. Input path params and the `/api/files` picker resolve
  **only inside** the data root; nodes never write here.
- **Output workspace (writable)**: per-run workspaces hold artifacts, plots
  and TF/export files — served through the artifact endpoints; nothing outside
  the output workspace is writable.
- **Project directory (stable, planned §12.3)**: a user-designated writable
  directory holding long-lived, relocatable state — the `.emptyui-cache/` disk
  cache. `EMPTYUI_PROJECT_DIR` designates it (unset → disk cache disabled).

Local development keeps unrestricted behavior (no data root — the
whole filesystem is browsable and addressable; the files browser merely
STARTS at the process cwd); confinement activates only in remote mode.
Manifests mark path-kind params as `in`/`out` so a single resolver confines
inputs to the data root and forces outputs into the workspace (outputs
normalize + validate identically — `../` escapes are rejected).

### 11.3 Serving, mounting, proxies

- **Single origin**: `create_app` optionally serves the built `dist/` bundle —
  no CORS in remote mode; session cookies flow naturally to REST and WS.
- **Sub-path mount**: mountable under e.g. `/tools/emptyui/` — FastAPI
  `root_path` plus a runtime-derived frontend API base (works at `/`, under a
  sub-path and in dev with no rebuild); the WS URL derives from that base.
- **Proxy compatibility**: standard `X-Forwarded-*` handling; WS upgrade
  proxying with generous idle timeouts; the polling fallback is a deliberate
  asset — the UI keeps working when proxies drop WS connections.

### 11.4 Operational policy (config knobs)

- **Workspace retention**: optional TTL + periodic cleanup coupled with
  artifact GC; expired workspaces invalidate their artifact entries; cleanup
  only removes worker-complete terminal runs. Disabled TTL preserves
  local-dev behavior.
- **Max concurrent runs** per instance (queued or HTTP 429), bounding
  memory/CPU in a shared platform; default unlimited for local dev.
- **Config surface**: env-first knobs (`EMPTYUI_DATA_ROOT`,
  `EMPTYUI_SERVE_DIST`, `EMPTYUI_CORS_ORIGINS`, `EMPTYUI_BASE_PATH`,
  `EMPTYUI_WORKSPACE_TTL`, `EMPTYUI_MAX_CONCURRENT_RUNS`,
  `EMPTYUI_DISABLED_PLUGINS`, `EMPTYUI_SETTINGS_PATH`) mirrored as
  `create_app(...)` keyword arguments for module embedding (kwargs win).

### 11.5 Explicit non-goals
- Authentication/user management inside emptyUI (the gateway's job).
- Multi-user shared process — scale out is "more instances", not "a bigger
  instance".
- File-upload API — data is provisioned by the platform; uploads are a
  possible follow-up.
- Database persistence for runs/artifacts — the file-based workspace (§3)
  stands until run history demands more.

---

## 12. Roadmap, Open Questions & Planned Designs

### 12.1 Status

| Milestone | State | Scope |
|---|---|---|
| M0 — Skeleton | ✅ | monorepo scaffold, CI, core Node API + registry + schema validation + echo node, web shell |
| M1 — First real pipeline | ✅ | `load_csv → bandpass → timeseries/spectrogram`, topo executor with WS progress/artifacts, manifest-driven properties panel, workflow save/load |
| M2 — MT ecosystem integration | ✅ (2026-09-06) | mth5 plugin (Load/Save, ExtractRun, channels, TF I/O, Merge Runs, FC read/write), aurora plugin (ComputeFC/EstimateTF/MergeTFLevels; OLS/RME/RME_RR), plot standardization (§10.6) |
| M3 — Power features | ✅ (2026-09-09) | nomenclature-aware auto channel configuration; remote reference end-to-end; run cancellation + artifact download; headless Playwright suite; robustness batches; caching + partial execution; calibration framework + LEMI; coherence sort; plugin guide + template; plugin enable/disable; remote-server contract |
| Post-M3 | ✅ | l2u automatic frequency grid (default), TF rotation node, per-bin coherence sort (default), MTU-5A burst ingestion + Build FC, Phoenix ASC calibration, saturation QC, run-level QC (Select Runs / Residual Sort), settings, time selector, plot nodes |
| M4 — backlog | **in progress** | ranked table below |

Processing-correctness milestones (all verified): unit normalization
(magnetic nT→A/m both archive types, electric mV/km→V/m — §10.5); impedance
phase convention matches aurora (Zxy≈+45°, Zyx≈−135°); USArray SDG31
ground-truth verification (median |Δlog₁₀ρ| = 0.019 after the documented
−5.7° rotation, 98% within gate on the exact official grid; history §16.3);
MTU-5A parity with the official Phoenix TF (Δlog₁₀ρ ~0.005, history §24.12);
data-driven full-band recipe reaching 416.6 s parity without any clock rule
(history §24.13).

### 12.2 M4 — community & advanced backlog (ranked by gain/effort)

| # | Task | Gain | Effort | Notes |
|---|------|------|--------|-------|
| 1 | ~~Plot Coherency + Plot Phase Tensor nodes~~ | — | — | **delivered** (§10.6) |
| 2 | ~~Merge Runs: remote multi-row warning~~ | — | — | **delivered**: gap-provenance warning chain (Merge Runs → Align Runs → fc attrs → RR pairing sites), §3 |
| 3 | **Batch/queue runner** | High | Medium | multi-station processing is the real repetition pattern; headless `POST /api/runs` + a small queue UI/CLI |
| 4 | **In-band smoothing taper (Parzen/Hann)** | Medium | Low | `√weight` FC rows by band-center proximity before regression; composes with all estimators — genuine enhancement (neither EMTF nor aurora does this) |
| 5 | **Disk-persisted result cache** | Medium | Medium | designed (§12.3); implementation pending |
| 6 | **Calibration absolute-gain verification** | Medium | Low | process one site independently (mtpy/other) and compare — validation, not a feature |
| 7 | **DOF correction + error statistics** | Medium | Medium | estimate-side DOF inflation (§10.3 caveat), error-bar agreement metrics, EMTF-style error floors |
| 8 | **mtpy golden-image plot comparison** | Low-Med | Low | test-only dev extra; guards plot regressions |
| 9 | **Hand-authored band-config files** | Low-Med | Medium | load exact EMTF band-setup files as a preset layer (manual bands already cover parity comparisons) |
| 10 | **Paradigm B: cross-decimation FC pooling** | High | High | pool raw FCs across levels per band for statistically stronger long-period estimates; novel, unproven, highest scientific upside |
| 11 | **FC→TS inverse node** | Low | Low-Med | inverse STFT/overlap-add for frequency-domain denoising demos |
| 12 | **Executor thread-pool concurrency** | Low | Medium | `cpu_bound` hints; MT pipelines are 5–10 mostly-linear nodes |
| 13 | **Sub-workflows (nested DAGs)** | Medium | High | **deferred**: typical MT pipelines are 5–10 mostly-linear nodes; copy/paste + example templates + the batch runner cover the repetition patterns; revisit only if workflows outgrow them |
| 14 | **Docs site** | Low-Med | Med-High | plugin guide + deployment guide exist; a hosted site is onboarding, not functionality |
| 15 | **Plugin index/publish flow** | Low | High | PyPI + the plugin-guide checklist already cover distribution |
| 16 | **ML plugin GPU querying** (`ctx.resources.gpu_count`) | Low | Trivial | only matters once ML noise-suppression plugins exist |
| 17 | ~~Improved auto frequency grid~~ | — | — | **delivered** (§10.2 l2u) |
| 18 | **MMT (multi-station MT) estimator** | Low-Med | Med-High | **long-term** (Kelbert et al. 2026): Egbert/Smirnov–Egbert extension of RR using multiple sites; aurora 0.6.2 has only single-remote estimators; do after the basic chain + frequency grid are pinned |
| 19 | **Universal saturation/QC mask channel convention** | Low-Med | Low | generalize the MTU-5A `saturated` auxiliary-channel convention — any loader may emit a per-sample binary quality mask; Compute FC / Build FC already consume it generically |
| 20 | **Load from FDSN (IRIS) node** | Medium | Low-Med | wrap `mth5.clients.FDSN.make_mth5_from_fdsn_client` — USArray NIMS recordings are auto-uploaded to IRIS as miniSEED (§9.3 note); a web-loading node replaces the never-delivered native NIMS loader |
| 21 | **Run-collection (`ts.runs`) semantics across the graph** | Medium | Medium | burst-mode corpora make collections the norm (Extract Run `time range`, §9.3). Tier 3 (delivered with the feature): singular-semantics nodes (Split/Merge Channels, Align Runs, Calibrate, Detrend/Bandpass, remaining ts plots) reject collections loudly. Tier 1 (delivered): Plot Time Series accepts collections — a **sample-rate selection** (default `auto` = the lowest rate present, e.g. 15 Hz Phoenix / 1 Hz LEMI/NIMS) picks the family to view, and all runs of that family plot as segments on the shared absolute time axis (gaps stay visible; the time bar crops each run — clamped — and runs entirely outside are skipped). The fc plot nodes carry the same tier selection (§10.6). Tier 2 (delivered): **element-wise maps** — Detrend/Bandpass/Calibrate
accept collections, the recipe applied to every member and the collection
shape preserved (extents rows unchanged — identity is not the recipe's
business; `mapped_by`/`filter` provenance rides each member's dataset
attrs). Bandpass **quick-rejects members whose sampling scope cannot
support the cutoffs** (e.g. a 50–100 Hz bandpass on 1 Hz data) — skipped
loudly, the whole node rejecting only when NO member is in scope (a
silently empty map would lie downstream); Calibrate maps through each
member's own archived responses (explicit-filter collection mapping is
M4 backlog). The maps' one-port outputs ride the §4 runtime shape hints.

### 12.3 Planned — portable local disk cache (designed, not yet implemented)

User decisions: **workspace-embedded** (the cache travels with the data
folder), **type-native serialization** (no pickle), **opt-in per node**,
**GB + TTL combined** eviction. Two-tier design — the existing in-memory
`NodeCache` stays the fast path; the disk tier adds restart persistence and
relocation. The cache lives in a **user-designated project directory**, not
the ephemeral per-run workspaces — a third storage realm (§11.2).

- **Project directory**: `EMPTYUI_PROJECT_DIR` (env) / `create_app(project_dir=…)`
  — a stable, writable directory the user designates, typically where the
  survey data lives. Cache root: `<project_dir>/.emptyui-cache/`. Unset →
  disk cache disabled (no silent `~` writes).
- **Key design (relocation-friendly)**: the executor's structural sha256 (§7.1)
  plus payload-format version, node plugin version, emptyui version — version
  bumps invalidate cleanly. File params become relocatable: the signature uses
  the path relative to the data root/project dir when the file lives under
  them (already implemented in §7.1 cache keys); absolute stat signature
  otherwise.
- **Layout & codecs**: `entries/<key[:2]>/<key>/{meta.json, payload/…,
  artifacts/…}`; payload kinds → netCDF (fc per level), XML (tf), JSON/npz
  (tables/arrays). Codec registry; kinds not in the registry skip disk caching
  with a log note (plugin-extensible later). Atomic writes; deserialize
  failures → invalidate + re-execute (never half-serve).
- **Opt-in per node**: `disk_cache: ClassVar[bool] = False` on `Node`; `True`
  for ComputeFC and EstimateTF (the real recomputation cost). Plots/tables
  stay memory-only. The manifest records the flag.
- **Executor integration**: `mem.get` → `disk.get` (materialize a CacheEntry,
  touch atime) → the same restore path as memory hits; `execute → mem.put`,
  and `disk.put` when manifest.disk_cache. Disk I/O failures log and continue —
  **the cache must never break a run**. Stale disk entries age out (keys embed
  the manifest version).
- **Eviction**: `EMPTYUI_CACHE_MAX_GB` (default 10) — LRU by atime until under
  budget; `EMPTYUI_CACHE_TTL_DAYS` (default 30). Cleanup on server start +
  piggybacked on the workspace-TTL loop; `DELETE /api/cache?tier=disk|memory|all`.
- **Tests (planned)**: round-trip per codec; restart persistence; relocation
  (move project dir + data root → still hits); eviction; corruption →
  re-execute; opt-in flag respected; injected I/O error → the cache never
  breaks a run.

### 12.4 Open questions

- **FC→TS inverse node** (M4 #11) — plugin-tier coercion when needed.
- **ML plugin GPU availability** (M4 #16) — `ctx.resources.gpu_count`?
- **Sub-workflows** (M4 #13) — deferred, see the backlog note.
- **Authentication model** — resolved by design (not implementation): emptyUI
  stays identity-blind behind the platform gateway; see §11.

---

## Appendix — history map

| Former DESIGN.md sections | Now |
|---|---|
| §1–§6 | unchanged in this document (§1–§6) |
| §7 execution, §8 API, §22 settings | §7 |
| §9 frontend, §23.4 time selector | §8 |
| §10 repo layout | §2 |
| §11 roadmap, §12 open questions, §21 disk cache | §12 |
| §13–§14 (M2 plan/decisions/banding history) | `docs/history/m2-delivery.md` |
| §15 plot standardization, §23.1–23.3 | §10.6 |
| §16 M3 (calibration, sort, USArray verification) | §9.1–9.2, §10.3; verification record in `docs/history/m3-delivery.md` |
| §17 remote deployment | §11 |
| §18 code review record | `docs/history/m4-delivery.md` |
| §19 electric scaling | §10.5 |
| §20 plot nodes | §10.6 |
| §21 disk cache | §12.3 |
| §22 settings | §7.4 |
| §23.4 time selector | §8.3 |
| §23.5 t_short | §10.2 |
| §24 physics parity | §9.2–9.3, §10.1–10.4; full research record in `docs/history/m4-delivery.md` |

References to §13–§24 are written `history §N` in code docstrings and commit
messages and resolve in the history files, which preserve the original numbering
verbatim.
