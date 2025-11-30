# emptyUI — Design Document

A node-based, ComfyUI-inspired visual workflow interface for magnetotelluric (MT)
data processing and visualization.

Status: v0.2 — design frozen for M0/M1 (2026-08-25).
Changes after v0.1: aurora-informed port type system, node categories,
TS→FC→TF stage split, MTH5-backed file storage, backend-decided port colors
and ComfyUI-style port rendering conventions (§3).
Implementation status: **M0 complete; M1 complete; M2 complete** — mth5/
aurora plugins delivered; ComputeFC (TS→FC windowed STFT) and EstimateTF
(FC→TF cross-power LSQ) implemented and verified end-to-end through the
executor on real 1 Hz LEMI-424 and BGS long-period data (Load LEMI → Detrend
→ ComputeFC → EstimateTF → Export EDI / Plot AppRes&Phase / Plot Tipper).
**M3 partial (2026-09-05/06)** — nomenclature-aware automatic channel
configuration, remote reference end-to-end (Align Runs), run cancellation,
artifact download, and the headless Playwright regression suite are
delivered; caching/partial execution, sub-workflows, remote-server docs and
the plugin packaging guide remain (see §11). **M2 is now complete** — the
plot-standardization task (§15) is delivered.
Processing correctness milestones:
- **Unit normalization**: magnetic channels converted nT→T→H (B→H) and
  electric mV/km→V/m; missing nT→T factor was the root cause of non-physical
  apparent resistivity (now ~10–10³ Ω·m).
- **Impedance phase convention**: verified against aurora on the official
  synthetic halfspace (`test12rr`); all three estimators (OLS/RME/RME_RR) are
  aurora-backed and share aurora's convention (Zxy≈+45°, Zyx≈−135°).
- Detrending moved to the dedicated `core.filter.detrend` node (mean/linear/
  none); `e_dipole_length_m` optional (off by default — LEMI output is mV/km).
- **Channel nomenclature** (2026-09-05): instrument naming is handled through
  mt_metadata's canonical `CHANNEL_MAPS` (§3 "Channel nomenclature") instead
  of hardcoded name tables; the magnetic B/H mode is derived automatically and
  TFs are built aurora-style (actual channel names + nomenclature map).
Estimator: **robust RME/RME_RR implemented** (aurora-backed) with remote
reference support; multi-level decimation/FC persistence, Calibrate, table
viewer and tippers delivered. See §13/§14 for the (historical) plan records.

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
| **Display** | Render data to `plot` outputs (no data deps downstream) | PlotTimeSeries, PlotSpectrogram, PlotAppResPhase, PlotTipper, PlotCoherency, PlotPhaseTensor |

Display nodes emit a **`plot` output port** (rendered-figure reference), not
typed data — so they can't feed processing nodes, but their figures can branch
to `Preview Plot` / `Save Plot` nodes. This mirrors ComfyUI's IMAGE port +
Preview/Save Image pattern: multiple results are each handled by their own
viewer node (drag-resizable, positioned anywhere on the canvas) instead of a
single fixed results pane. There is **no run-level gallery** — the
node-embedded viewers are the primary (and only) display.

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
3. `tf.level[]` → `tf` (merge; resolves duplicate periods across sampling
   rates / decimation levels — the multi-level merge case).
4. Format coercions for popular types ship in core (`SaveEDI`, `LoadEDI`,
   …); exotic import/export converters are plugins.

### Pipeline stage split: TS → FC → TF

The aurora processing run is **not** one monolithic node. It is split at the
Fourier-coefficient boundary, mirroring both the conceptual MT pipeline
(decimation → windowing → time/frequency transform → power spectra → TF
estimation) and aurora v0.3.x's own ability to persist and reuse FCs:

- **`ComputeFC` node (TS → FC)** — input `ts.run`; params: channel
  nomenclature (auto/default/lemi12/lemi34/phoenix123/musgraves — presets the
  channel names via `cascade`), input/output channel names, decimation levels
  + factor, window samples/overlap/type. Output `fc`. Internally expands
  uniform decimation levels sequentially (level sample rates derived from
  factor × initial rate; constant window in samples, aurora-style D2).
  Persists computed FCs into the workspace MTH5's FC layer. Outputs may be the
  electric pair and/or a vertical magnetic channel (e.g. `e1,e2,bz`) to also
  produce the vertical-field geomagnetic transfer function (tipper). The
  magnetic B/H mode is **derived automatically** from the nomenclature /
  channel names (bx/by → fluxgate B→H; hx/hy → coil H) — the `magnetic_unit`
  select defaults to `auto` and exists only for legacy saved workflows.
  **Channel auto-configuration**: empty input/output channel fields resolve
  from the data and its nomenclature — inputs = the horizontal magnetic pair,
  outputs = the electric pair **plus** the vertical magnetic channel when the
  run contains one (tipper by default); a run with no electric channels yields
  a tipper-only FC, and an explicitly requested output missing from the data
  is dropped with a warning (best-effort vertical extension; missing inputs
  stay a hard error). Matching is case-insensitive (CSV `Ex,Ey,Hx,Hy`). When
  the nomenclature keyword is known from the upstream chain (Load LEMI-424 →
  Extract Run → Compute FC), the frontend auto-fills the fields from the
  inherited keyword (defaults may be overridden by inheritance; explicit user
  choices never are).
- **`EstimateTF` node (FC → TF)** — inputs `fc` + optional `ref_fc` (remote
  reference); params: estimator (`Robust`/`Least Squares`), remote-reference
  on/off, periods-per-octave. Band edges are derived internally
  (`_period_bands`). Output `tf` + per-level `tf.level`. The TF is built
  aurora-style (§3 "Channel nomenclature"): rows/cols carry the **actual**
  channel names (e1/e2/hz × hx/hy or the LEMI equivalents) and the
   `channel_nomenclature` map is attached to the mt_metadata TF — a 2x2
   impedance, or a 3x2 impedance + tipper. The remote reference (**RME_RR**,
   aurora-backed) pairs local/remote FC samples by row, so the ref fc must
   share the local fc's **frequency AND time grids** — both are validated
   (clear errors); the runs must be aligned first (see `Align Runs` below).
- **`Align Runs` node (`mth5.runs.align`)** — two `ts.run` inputs → two
  `ts.run` outputs, both normalized to an mth5 `RunTS` cropped to their
  **common acquisition overlap**. Required for remote-reference processing:
  the two stations must share the same sample rate (aurora hard-errors on
  mixed rates), and after alignment both FCs have identical time grids so
  RME_RR pairs the correct windows. The overlap is computed from absolute
  start/end times (GPS) when available, with a warning fallback to relative
  offsets. The remote station may be **magnetic-only** (bx,by) — a Compute FC
  on such a run resolves to a reference FC (magnetic channels as outputs) and
  only the magnetic channels are used by EstimateTF's `ref_fc`. The node
  also computes the inter-station coherence of the horizontal magnetic field
  (64–1024 s band) and **warns when it is weak** — a clock offset or an
  un-filled data gap breaks station alignment and silently degrades remote
  reference (mt_io's LEMI reader indexes files by *row*, not by timestamp, so
  recording gaps must be filled at conversion — see
  `scripts/convert_bgs_lemi.py`, which interpolates over gaps so one row ==
  one true second).

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

### Channel nomenclature (mt_metadata CHANNEL_MAPS)

Instrument channel naming is **not** re-invented: `emptyui-mth5/nomenclature.py`
builds on mt_metadata's canonical `CHANNEL_MAPS`
(`default`/`lemi12`/`lemi34`/`phoenix123`/`musgraves`/`nims`) as the single
source of truth for standard ⇄ actual channel names. Following the
aurora/mt_metadata design, channels are **never renamed in the data** — a RunTS
keeps `e1`/`bx`/... names; the nomenclature keyword is carried as metadata
(fc attrs, TF map) and used only to interpret names:

- **Unit normalization** (B vs H) is derived from the nomenclature/channel
  names, not a manual switch: `bx/by/bz` → fluxgate B→H conversion; `hx/hy/hz`
  → coil H (unchanged).
- **`Load LEMI-424`** exposes a `channel_nomenclature` select (lemi12/lemi34):
  LEMI instruments can carry four E channels (e1..e4); the keyword selects
  which electric pair is read into the MTH5 (mt_io `e_channels`).
- **`Split Channels` / `Merge Channels`** resolve the standard ex/ey/hx/hy/hz
  ports onto the actual names via the nomenclature (exact-name lookup, never
  `getattr` on an absent component).
- **`ComputeFC` channel auto-configuration**: empty input/output channel
  fields resolve from the data + its nomenclature — the horizontal magnetic
  pair as inputs, the electric pair plus the vertical magnetic channel
  (tipper) as outputs when present; a magnetic-only run yields a tipper-only
  FC; requested-but-missing outputs are dropped with a warning (missing
  inputs stay errors). The frontend auto-fills the fields from an
  upstream-inherited keyword (Load LEMI-424 → Extract Run → Compute FC).
- **Display nodes accept standard names / auto-select** (`core.plot.*`):
  `Plot Time Series`'s channel spec and `Plot Spectrogram`'s channel field
  resolve through the inferred nomenclature — standard component names
  (ex/ey/hx/hy/hz) map to the instrument names (e.g. `ex` → `e1` on LEMI
  data), matching is case-insensitive, and `Plot Spectrogram` defaults to
  the first electric channel when the field is empty (no more hardcoded
  `"Ex"` that fails on LEMI/phoenix runs). Core nodes keep the plugin
  dependency policy: the resolver lazily imports mt_metadata's canonical
  `CHANNEL_MAPS` with a mirrored fallback
  (`emptyui-core-nodes/_channels.py`).
- **`ComputeFC`** records the resolved keyword on each fc's attrs
  (`channel_nomenclature`, `magnetic_unit`), so `EstimateTF`/`MergeTFLevels`
  can map back without re-entry. FCs read from an mth5 archive are enriched
  with the same attrs by `mth5.fc.read`.
- **`EstimateTF`** builds the mt_metadata TF the way aurora does
  (`TF(channel_nomenclature=...)`): coordinates are the **actual** instrument
  names and `tf.channel_nomenclature` carries the standard→actual map, so
  aurora's `TTFZ`/`TransferFunctionCollection` can consume emptyUI TFs and the
  instrument naming is preserved as provenance. All TF accessors
  (`tf.ex`/`tf.hx`, `has_impedance`, `impedance`, `tipper`) are
  nomenclature-aware; display nodes select components via `tf.ex/ey/hx/hy`.
- **Exporters**: EDI/XML/AVG standardize on write (mt_metadata's writers
  always emit standard channel names); the ZMM writer in mt_metadata 1.0.10
  assumes standard names, so `ExportTF` relabels to a standard-coordinates
  copy for `zmm` output.

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

- The server maintains a workspace directory **per run** (a
  `tempfile.mkdtemp()` workspace, `ctx.workspace`). Every intermediate result
  a node chooses to persist is written as an HDF5 group inside a workspace
  MTH5-compatible tree (`<workspace>/workspace.mth5`, see `persist.py`);
  handles reference these paths. A persistent `~/.emptyui/workspace/`
  (surviving across runs) is a future item.
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

#### Tooltip / description / palette protocol (for new nodes & plugins)

Manifests carry two audience-separated text fields plus a visibility flag;
the frontend is purely generic about them:

- **`tooltip`** — the hover tip on palette buttons and node headers. Exactly
  **one sentence** saying what the node *does* (imperative, ≤ ~100 chars,
  no parameter lists, no semantics discussion). Example:
  `"Down-weight or drop low-coherence FC windows before TF estimation."`
- **`description`** — the full details, shown in the right-hand properties
  panel when the user clicks the node. Anything beyond the one sentence
  belongs here (parameters, conventions, caveats, provenance).
- **Fallback**: when `tooltip` is empty, the backend manifest resolves it to
  the **first sentence of `description`** (`emptyui.api.first_sentence`) —
  so plugins that don't set a tooltip stay consistent automatically. To make
  that fallback readable, write every `description` so its first sentence
  works as a standalone tip.
- **`hidden`** — utility/test nodes (e.g. the `core.util.echo` smoke test)
  set `hidden = True`: they are excluded from the palette by default but
  remain registered — saved workflows containing them still load, validate
  and execute.
- **Palette ordering protocol**: categories display in fixed order
  (`Data I/O`, `Processing`, `Display`, then any custom categories
  alphabetically); nodes are sorted alphabetically by title within each
  category. Category assignment follows the §3 table — figures are not
  data, so plot-export nodes (`Save Plot`) are **Display**, not Data I/O.

Plugins that follow these four rules need zero frontend changes — the
palette, tooltips and properties panel are generated from the manifests.

### Param auto-change (`cascade`) & upstream auto-fill (`inherit`)

Two manifest-driven UI conveniences keep the frontend schema-agnostic:

- **`cascade`** (per param): when the user changes this param to a given value
  (key), the UI sets the listed `(other_param, value)` pairs. Keys serialize as
  strings for JSON (`bool` → `"true"`/`"false"`). Used, e.g., for `Compute FC`'s
  channel-nomenclature select to preset the input/output channel names per
  mt_metadata nomenclature, and for `Estimate TF`'s remote-reference to force
  the Robust estimator:

  ```python
  Param("channel_nomenclature", "select", default="auto",
        options=("auto", "lemi12", "lemi34", "phoenix123", "musgraves", "default"),
        cascade={
            "lemi12": (("input_channels", "bx,by"), ("output_channels", "e1,e2")),
            "lemi34": (("input_channels", "bx,by"), ("output_channels", "e3,e4")),
            ...
        }),
  Param("remote_reference", "boolean", default=False,
        cascade={True: (("estimator_engine", "Robust"),)}),
  ```

  The magnetic B/H mode is **not** a manual choice: it is derived from the
  channel nomenclature / actual channel names (`bx/by` → fluxgate B→H;
  `hx/hy` → coil H) — the `magnetic_unit` select defaults to `auto` and exists
  only for legacy saved workflows.

- **`inherit`** (per node): `{input_port: {target_param: upstream_param}}`. When
  an upstream node is connected to that port, those params render as dropdowns
  populated from the upstream node's params and are auto-filled if empty.
  Same-name inheritance is just an identity mapping. Used for `Compute FC` to
  pull `survey`/`station`/`run` from an `Extract Run`, and for mth5-sourced
  nodes to map e.g. `survey ← survey_id` from `Load LEMI-424`:

  ```python
  inherit: ClassVar[dict[str, dict[str, str]]] = {
      "data": {"survey": "survey", "station": "station", "run": "run",
               "channel_nomenclature": "channel_nomenclature"},
  }
  # Extract Run pulling from Load LEMI-424 (name mapping):
  inherit: ClassVar[dict[str, dict[str, str]]] = {
      "mth5": {"survey": "survey_id", "station": "station_id",
               "channel_nomenclature": "channel_nomenclature"},
  }
  ```

  `channel_nomenclature` is inheritable end-to-end: `Load LEMI-424` → `Extract
  Run` → `Compute FC` chain-walks the keyword (e.g. `lemi12`) so downstream
  nodes are pre-populated; the run-time fallback infers it from the actual
  channel names when no upstream value exists.

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
2. **Only declared params auto-fill.** A param is auto-filled **only if** it
   appears as a target in the node's own `inherit` mapping. Everything else —
   root-node defaults, radio/select config (e.g. `Detrend` method, `Compute FC`
   magnetic unit, `Estimate TF` estimator) — stays fully manual. This keeps the
   "set once at the source, inherit downstream" intent: a user types the
   identity at the first node, and every declared downstream param is then
   filled from it.
3. **Editable until a source provides a value.** `resolve` does **not** return a
   node's *own* value as an inherited option (that value is just the current
   field contents). So a declared param connected to a source that provides
   nothing (e.g. `Extract Run` → `Load MTH5`, which has no `survey_id`) stays a
   plain editable text field — even after the user types into it. It only
   becomes a dropdown once an upstream node actually yields a value. The
   inherited dropdown is an **editable combobox** (datalist), so a suggested
   value can always be overridden by typing.
4. **Recursive chain-walk.** For the entry node the resolver walks its declared
   mapping (or pass-through) upstream; at each intermediate hop it *does*
   consider that hop's own value as a candidate. The walk is transitive across
   arbitrary processing nodes even when an intermediate node declares nothing
   for the param (pass-through fallback) — e.g. `Compute FC → Detrend → Extract
   Run → Load LEMI-424` still resolves `survey`, so inserting a future
   time-domain node (de-noising, filtering) between the source and `Compute FC`
   never loses the identity.
5. **Name mapping.** Each hop uses the *current* node's mapping to decide which
   upstream param to look for, so heterogeneous sources (e.g. `survey_id`) are
   bridged without changing the downstream node.
6. **Cycles & fan-in.** Use a visited set (the graph is a DAG but be defensive)
   and union candidates across parallel sources feeding a port.
7. **Same-name default.** Prefer the identity mapping; only declare a mapping
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
- **Failure**: *design* — a failed node halts its dependents only; other
  branches continue. *Current executor (M1/M2)*: a node failure aborts the
  whole run (error event + `run_finished: failed`); per-branch continuation is
  future work.
- **Partial runs**: user can run a single node ("execute to here"), enabled by
  the caching scheme.
- **Cancellation**: *design* — client sends a cancel message; executor checks a
  token between node steps. *Current*: implemented — `POST /api/runs/{id}/cancel`
  sets a per-run `threading.Event` that the executor polls between nodes
  (including after the final node, so cancelling during the last node reports
  `cancelled` rather than `completed`); the toolbar **✕ Cancel** button is
  enabled while a run is queued/running. The executor still cannot interrupt a
  node mid-execution — cancellation takes effect at the next node boundary.

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
| GET  | `/api/artifacts/{id}` | Full artifact download (`Content-Disposition: attachment`) |
| POST | `/api/runs/{id}/cancel` | Request cancellation; executor checks the token between nodes |
| WS   | `/api/ws` | Run events stream |

> Cancellation and artifact download are implemented (see rows above). A
> run-level artifact gallery is still not implemented — viewers on nodes are
> the per-figure/table display, each with its own download link.

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
  viewers can attach to the same figure/table. Each viewer exposes a **⤓
  download** link for the full artifact (`GET /api/artifacts/{id}`) next to
  the inline display.
- **Toolbar** — run, cancel (✕, enabled while a run is queued/running);
  new/open/save workflow. Run-level artifact gallery is not implemented —
  viewers on nodes are the per-figure/table display, each with its own
  download link.

State split:
- Zustand store: workflow document (single source), selection, connection drafts.
- TanStack Query: server resources (node catalog, artifacts).
- WebSocket client: ephemeral run state overlaid onto node statuses.

End-to-end testing:
- **Playwright Test runner** (`packages/web/e2e/`, headless, full-stack): the
  config starts both the Vite dev server and the FastAPI backend via
  `webServer` (backend resolves the repo-root `.venv` or falls back to
  `python3`), so the suite exercises the real UI + executor + artifact
  pipeline. Run with `npm test` (see README); CI runs it headless on
  ubuntu-latest (`.github/workflows/e2e.yml`).
- **Playwright MCP** (`@playwright/mcp` via opencode `mcp` config) is the
  interactive/exploratory counterpart for agent-driven browser debugging.

Connection rules (port types, one edge per input port, cycle prevention)
enforced client-side for UX, re-validated server-side before every run
(duplicate edges into one input port are a validation error — the executor
must not silently overwrite).

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
- **M2 final task — publication-quality plot standardization** ✅ delivered
  2026-09-06 (see §15): shared plot style, error bars, TF station metadata,
  MT/USArray display conventions (phase 0–90° fold for off-diagonal,
  tipper Re/Im, equal per-decade scale, ○ XY / ✕ YX markers). **M2 is now
  complete.**

**M3 — Power features**
- ✅ **Robustness batch ("Batch B")** — delivered 2026-09-07; validation &
  data-quality guards (no design changes):
  - Fan-in duplicates: multiple edges into one input port are rejected in
    `validate_workflow` (§9 server-side re-validation) — the executor's
    last-wins overwrite is no longer reachable from the server.
  - Legacy mth5 API: `SpectraToFCGroup`/`MergeRuns` now resolve
    survey/station/run through `runs._resolve_survey`/
    `_resolve_station`/`_resolve_run` (dual modern/legacy layout support;
    regression: `StationGroup.runs_group` does not exist in mth5 0.6.x).
  - `Load CSV`: time-column-only files error cleanly; ISO datetime time
    columns set `start_datetime` + derive the rate (lazy pandas import);
    predominantly non-monotonic time columns error (median-based detection
    stays robust to single strays); NaN/Inf guard with a clear error at
    `Compute FC` (via `to_channels`), `Bandpass Filter` and `Detrend`
    entry (`ensure_finite` in core-nodes types) — NaN propagates silently
    through scipy `decimate`/`stft` otherwise.
  - Tipper plot ylim [-1, 1] auto-expands when data legitimately exceeds it.
  - Doc note (§16.2): the coherence sort is a per-window broadband
    (bin-ensemble) coherence heuristic, not EMTF's window-ensemble per
    frequency; a true EMTF-style variant is an M4 item.
  - Completed earlier in the same spirit (2026-09-07, "Batch A"):
    remote-reference partial-ref-container now errors instead of silently
    self-referencing; JSON calibrations keep per-channel frequency grids and
    unmatched channels warn instead of passing through silently; artifact
    filenames are node-instance-unique (`RunContext.artifact_path`); the
    `Calibrate` explicit-filter path delegates to `RunTS.calibrate` (proper
    detrend/zero-pad/windowing instead of bare FFT division).
  - **Test suite reorganization (2026-09-07)**: error-surface tests grouped
    into one `pytest.mark.errors` file per package (`test_errors.py`,
    `test_core_errors.py`, `test_mth5_errors.py`, `test_aurora_errors.py`;
    marker registered in the root `pytest.ini`) so `-m "not errors"` gives a
    fast feature/physics pass and `-m errors` runs the edge suite (~5 s).
    Default invocation and CI are unchanged (full suite). Rule: a test moves
    iff its primary assertion is `pytest.raises`/error-status/degenerate
    input; feature, physics-convention and value-recovery tests stay in the
    feature files (backward-compat estimator names stay with the estimator
    tests).
    remote-reference partial-ref-container now errors instead of silently
    self-referencing; JSON calibrations keep per-channel frequency grids and
    unmatched channels warn instead of passing through silently; artifact
    filenames are node-instance-unique (`RunContext.artifact_path`); the
    `Calibrate` explicit-filter path delegates to `RunTS.calibrate` (proper
    detrend/zero-pad/windowing instead of bare FFT division).
- ⏳ **Instrument calibration framework + coherence sorting** (planned
  2026-09-07, see §16) — the next M3 tasks:
  - General calibration framework with LEMI-417/424 support (manufacturer
    `.chn`/`.def` FAP files → per-channel `FrequencyResponseTableFilter`
    → `RunTS.calibrate()`); verified on the BGS pair that the short-period
    collapse is the instrument noise floor, and calibration is a modest
    (~×6 ρ, ±30° phase) correction near Nyquist.
  - EMTF-style coherence sorting: filter/downweight low-coherence FC
    windows before the robust regression (aurora 0.6.2 only *logs* squared
    coherence; its "coherence sorting" helpers only widen bands).
- ⏳ Caching + partial execution; sub-workflows; remote-server mode docs
  (same API, deployed elsewhere); plugin packaging guide + example template.
  *Caching is designed in (content-addressed handle keys, §7) but not
  implemented — the executor re-runs the whole DAG every time. Sub-workflows
  and the plugin guide/template have no implementation; `docs/` is empty.*
- ✅ **Dynamic "select from" options**: delivered (2026-09-05) via the
  **chain-walking inheritance** mechanism of §4 instead of the originally
  sketched `options_from`/`POST /api/options` design: `Compute FC`'s
  survey/station/run (and channel nomenclature) auto-fill as editable
  comboboxes from the connected upstream chain, frontend-only, with no
  backend options endpoint. Selection-style nodes (`Extract Run`, `Split
  Channels`) resolve the upstream source server-side at run time.
- ✅ **Remote reference end-to-end** (2026-09-05/06): `Align Runs` node
  (`mth5.runs.align`), `estimate_tf` frequency+time grid validation,
  magnetic-only remote stations, coherence warning guard; verified on the
  BGS NO46+NX37 pair (see §3).
- ✅ **Run cancellation + artifact download** (2026-09-05): per-run cancel
  token polled between nodes, `POST /api/runs/{id}/cancel`, toolbar Cancel
  button, `GET /api/artifacts/{id}` attachment download (see §7/§8).
- ✅ **Headless E2E regression suite** (2026-09-05): Playwright Test runner
  in `packages/web/e2e/` with a CI workflow (see §9).

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
- Sub-workflow semantics (nested DAGs) — still open; no implementation
  (M3 roadmap item).
- Authentication model for remote deployments (defer; design API so middleware
  can be inserted).

---

## 13. Upcoming Plan — Remaining M2 (2026-08-29)

Status: **M2 delivered** (2026-09-05); every item in the inventory below is
done. This section is retained as the historical plan record — the agreed
priority order and divide-and-conquer tracks for finishing M2. It supersedes
the one-line "next milestone" note above.

### Remaining M2 deliverables

Inventory against §11. ~~Struck~~ items are delivered; the gaps:

| # | Deliverable | Where | Status |
|---|-------------|-------|--------|
| 1 | Robust **RME/RME_RR** estimator | `_processing.estimate_tf` + `process.py` | ✅ **done** — OLS/RME/RME_RR all aurora-backed; `remote_reference` supported (ref `fc` or self-reference/dummy) |
| 2 | **Multi-level decimation + FC persistence** | `compute_fc_levels` + `persist.py` | ✅ **done** — cascading decimation, per-level FC persisted to workspace MTH5 (§14) |
| 3 | **MergeTFLevels** (`tf.level` → `tf`) | `merge_tf_levels` + `process.py` | ✅ **done** — per-level estimate + deepest-wins merge of overlapping octave periods (§14) |
| 4 | **Table viewer** | `mth5.table.selection`/`mth5.table.tf` + `core.display.view_table` + frontend | ✅ **done** — JSON table artifacts rendered inline |
| 5 | **Calibrate** node | `mth5.calibrate` + `calibrate.py` | ✅ **done** — `ts.run` (+ optional `calibration`) → calibrated `ts.run`; uses mth5 `RunTS.calibrate()` channel responses, or frequency-domain response removal for an explicit filter |
| 6 | **Publication-quality plot standardization** | `mth5.plot.*`, `core.plot.*` + shared style module | ✅ **done** — §15; M2 complete |

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
- **P1 — Multi-level decimation + FC persistence + MergeTFLevels.** ✅ delivered
  (2026-08-29). Second value:
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
Track B (backend, arch)  Multi-level ComputeFC + FC persistence + MergeTFLevels   (largest)              ✅
Track C (frontend)       Table viewer + "View Selection" Display node            ✅
Track D (backend, I/O)   Calibrate node + calibration port wiring                                       ✅
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

Status: **delivered** (all four phases landed 2026-08-29; see the phase notes
below). This section is the historical plan for broadband MT — processing more
than one decimation level so the apparent resistivity/phase curves span
decades. It is grounded in reading the pinned aurora 0.6.2 / mth5 0.6.8 source
(§14.1).

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

**Phase 4 — Integration & verification** ✅ delivered
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
**every** level when an absolute start is available and skips with a warning
otherwise. Round-trip fc→mth5→fc verified (datetime required). Note
(2026-09-05): the mth5 read-back path is now complete — `mth5.fc.read`
enriches the read-back fc with the metadata `EstimateTF` needs (sample rate,
channel roles via the nomenclature) and `estimate_tf` normalizes mth5's
`(time, frequency)` layout.

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
near-duplicate periods by keeping the **deepest** level → one broadband
`tf` (per the final spec below). `EstimateTF` now loops levels and returns the
merged `tf` (D5) plus a `tf_levels` (`tf.level`) output; `MergeTFLevels` is
functional (merges a `tf.level` container). Verified: multi-level merged TF
keeps aurora's phase convention on the halfspace (Zxy≈+45°, Zyx≈−135°);
end-to-end on LEMI (4 levels → 64 merged periods → EDI).

**Period banding rework (2026-08-29, feedback-driven).**
- **Periods per octave** (`periods_per_octave`, default 4) replaces `periods_per_decade`;
  bands are placed on a log2 (power-of-2) grid, matching FFT/decimation windows (128/256).
- **Octave-aligned, shared boundaries (`_period_bands`)**: each decimation level spans from
  its Nyquist period (2/fs) to the next level's Nyquist period (2·factor/fs); band centers sit
  on a global log2 grid including both end points, so level *i*'s last period **exactly equals**
  level *i+1*'s first period. Levels therefore tile continuously (no gaps), and the merge simply
  de-duplicates the single shared boundary period. This
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

---

## 15. M2 final task — Publication-quality plot standardization (delivered 2026-09-06)

**Motivation.** The Display nodes (`mth5.plot.appres_phase`, `mth5.plot.tipper`,
`core.plot.timeseries`, `core.plot.spectrogram`) were proof-of-concept level:
no error bars (TF errors were never stored), no MT display conventions (phase
plotted raw, no y-limits policy), components distinguished by color only
(unreadable for color-blind readers), titles carried no site/processing
metadata, DPI was low (110) and style was inconsistent across nodes.

**Status: delivered.** P1–P3 landed; see §15.2. Conventions implemented as
decided in §15.4 (USArray markers/axes, tipper Re/Im with ylim [-1, 1],
diagonal phase signed [-180, 180°], equal per-decade scale, error bars, TF
station metadata carried for plots *and* exports).

**Reference figures.** `SDG31b_H28coh.png` (USArray convention — the primary
style reference: log10 Period × log10 ρ_a, linear φ, ○ XY / ✕ YX, error
bars, site + LON/LAT title) and `202511.PNG` (MTpy-style combined response —
used for the tipper Re/Im panel layout). We do **not** build a combined
plot — `Plot AppRes & Phase` and `Plot Tipper` stay separate nodes — but we
adopt the conventions that make the figures publication/archive-worthy.

**Scope.** All existing Display nodes. Out of scope: new plot types (Plot
Coherency, Plot Phase Tensor — M4), multi-station overlays, interactive plots.

### 15.1 Gap analysis (what exists vs. what is needed)

1. **TF error propagation (enabler)** — `build_tf` (`_processing.py`) writes
   only `tf.transfer_function`; `tf.impedance_error`/`tf.tipper_error` stay
   `None`, so no error bar can be drawn. aurora's estimators expose
   `inverse_signal_covariance` (`cov_ss_inv`) and `noise_covariance`
   (`cov_nn`); `TTFZ.standard_error()` = sqrt(|cov_ss_inv · cov_nn|) and
   `TTFZ.apparent_resistivity` derives `rho_se`/`phi_se` from it. Plan:
   carry the per-observation std through `_solve_band` → `build_tf` into
   `transfer_function_error` (2×2) and `tipper_error` (1×2) so both plot nodes
   and EDI export can consume errors. Spike first: confirm which estimators
   populate the covariances (RME path uses `IterControl(return_covariance=…)`;
   OLS via `estimate_ols` may not — fall back to the classical
   `(XᴴX)⁻¹·σ²` covariance from residuals).
2. **Error-bar math (adopt aurora/mtpy formulas)** — with `σ_Z` = std of the
   complex impedance element and `σ_|Z| = σ_Z/√2` (real/imag std, aurora
   convention):
   - ρ = |Z|²/(μ₀ω) [SI]; σ_ρ/ρ = 2·σ_|Z|/|Z| (equivalently aurora's
     `rho_se = 2·√(T·ρ/5)·Z_se` in MT units).
   - σ_φ(rad) = σ_|Z|/|Z| (aurora `phi_se = (180/π)·Z_se/|Z|`).
   - Tipper is plotted **Re/Im** (see below), so the `tipper_error` (std of
     the real/imag parts) is used **directly** — no relative-error
     propagation needed.
3. **Phase display convention (decided 2026-09-06)** — off-diagonal
   components (Zxy, Zyx) are folded into (0°, 90°) by ±180° (aurora
   `RhoPlot` and the USArray reference both show off-diagonal phases in the
   first quadrant; the USArray figure uses 0–80°). y-limits 0–90°, major
   ticks every 15°. **Diagonal components (Zxx, Zyy) keep signed phase in
   [-180°, 180°]** — the first-quadrant fold is only defined for the
   off-diagonal pair.
4. **Style** — one shared style module (in `emptyui-mth5/plotting.py` for TF
   plots; a mirrored minimal `emptyui-core-nodes/_plotstyle.py` for core
   plots, since core must not depend on mth5) providing:
   - `apply_style(fig, ax)` — font sizes (publication-safe ~9–10 pt),
     linewidth, marker size, grid (both axes, light), figure size, dpi ≥ 150,
     `bbox_inches="tight"`, PNG + optional PDF.
   - **Axes (USArray convention)**:
     - x = **Period (secs), log10** — decade ticks + minor ticks, shared
       across panels.
     - ρ_a = **log10 (Ω·m)** — decade ticks.
     - φ = **linear (Degrees)** — 0–90 ylim (off-diagonal) or [-180, 180]
       (diagonal).
     - Tipper = **Re/Im, linear, ylim [-1, 1]** — normal tipper magnitude
       should not exceed 1.
     - **Equal per-decade scale (MT convention)**: one decade of period must
       occupy the **same physical length** as one decade of ρ_a (e.g.
       `ax.set_aspect(1)` on log axes, or a figure sized from the decade
       spans; verify the exact mechanism against the pinned matplotlib
       version). This is what makes the ρ_a vs period curve comparable to
       published MT responses.
   - **Markers & colors (USArray convention)** — mimic the USArray reference
     (`SDG31b_H28coh.png`): **XY = ○ (circle, blue), YX = ✕ ("x", red)**,
     plus distinct linestyles (solid/dashed) so components remain
     distinguishable without color (colorblind-safe). Proposed for the
     components USArray does not plot: XX = square, YY = diamond (same
     blue/red families). Tipper: within each component panel distinguish
     **Re = ○ / Im = ✕** (as in the 202511.PNG tipper panels) and
     components by panel.
   - Labels with units: "Period (secs)", "ρ_a (Ω m)", "φ (Degrees)",
     "Tipper Re/Im (dimensionless)".
   - **Legend**: framed legend inside the axes, marker+label ("○ XY /
     ✕ YX"), never color-only.
5. **TF station metadata (prerequisite for plots *and* TF export)** —
   `build_tf` creates a bare `TF()` and never sets
   `tf.station_metadata`/`tf.survey_metadata`, so plot titles cannot show
   the site/LON/LAT and exported TF files (EDI/XML/ZMM) carry **no station
   header** — a serious gap for archive/export, since EDI station lines
   (id, lat/lon, elevation, declination) are what make a file usable
   standalone. `ExportTF`'s `_to_standard` already copies
   `tf.station_metadata`/`tf.survey_metadata` onto the standardized export
   copy (`tf.py`), so populating the TF at build time is the **single point
   of truth** — plots and exports both read it from the TF. Wiring:
   `ComputeFC` already resolves survey/station/run (`runts_context` for FC
   persistence) and the `ts.run` carries the mth5 station metadata
   (location.latitude/longitude/elevation) — attach the identity
   (survey/station/run + lat/lon/elevation) to the **fc attrs** at ComputeFC
   time, then `EstimateTF` → `build_tf` sets `tf.survey_metadata` /
   `tf.station_metadata` (mt_metadata Station: id, latitude, longitude,
   elevation, declination). `EstimateTF` currently has no survey/station
   params — it must read the identity from the fc (inherited from the
   upstream chain) rather than require re-entry. Note (M3): the future
   content-addressed **cache** (DESIGN §7) will also be able to restore
   any intermediate data (including the fc/TF metadata) directly, which
   mitigates loss of provenance — the cache structure design is deferred,
   but artifacts/FCs should carry provenance now so the cache can key on
   it later.
6. **Title & metadata** — USArray-style: `Site: <station id> [ LON = …;
   LAT = … ]` plus a subtitle/annotation with run id and processing
   provenance (estimator, remote reference on/off, decimation levels,
   periods per octave, nomenclature). Built from the TF station metadata of
   item 5 (lat/lon are not available today). Figure metadata (`Software:
   emptyUI`, `Title`, `CreationDate`) via `fig.savefig(metadata=…)`.
7. **Legend** — component labels (XY/YX, Tzx/Tzy), not raw channel names;
   actual instrument names stay as provenance in the TF metadata (a
   footnote/subtitle line can state the channels if they differ from
   standard).
8. **Consistency** — all TF plot panels share the same period vector,
   x-limits and figure geometry (shared axes); identical style constants
   across nodes; consistent artifact filenames.

### 15.2 Phasing (small, independently verifiable)

- **P1 — TF enrichment (enabler). ✅ delivered.** aurora's RME/RME_RR
  populate `cov_ss_inv`/`cov_nn` during estimation
  (`IterControl(return_covariance=True)`, the default); OLS does not, so the
  classical covariance `(XᴴX)⁻¹·σ²` from residuals is computed instead.
  `_solve_band`/`build_tf` now propagate `transfer_function_error` +
  `tipper_error` (std of each complex element — exactly what mt_metadata's
  EDI writer expects: it writes the square as `.VAR`). **Station identity is
  wired end-to-end**: ComputeFC attaches survey/station/run + lat/lon/elev
  (primitives) to the fc attrs, EstimateTF/MergeTFLevels pass them to
  `build_tf`, which sets `tf.survey_metadata`/`tf.station_metadata`/
  `tf.run_metadata` (incl. processing notes) — the single point of truth for
  plot titles **and** EDI/XML/ZMM export headers (verified: EDI now carries
  `DATAID`/`LAT`/`LON`/`SURVEY` + `*.VAR` lines). `AlignRuns._crop_to_runts`
  now carries station/survey/run metadata onto the aligned runs (they were
  being reset to defaults). `RunContext.workspace` is normalized to `Path`
  so plot/table nodes work through the executor (which passes a str).
- **P2 — TF plot nodes. ✅ delivered.** Shared style module
  (`emptyui-mth5/plotting.py`) + refactored `PlotAppResPhase`/`PlotTipper`:
  error bars (errorbar, capped), off-diagonal phase fold to (0, 90) with
  ylim 0–90/15° ticks (diagonal signed [-180, 180]), USArray marker/color
  encoding (○ XY / ✕ YX + distinct linestyles), equal per-decade scale
  (`equal_decade_layout`), labels/units, USArray title with site + LON/LAT
  and processing-provenance footer, dpi 150. `PlotTipper` now renders
  **Re/Im panels with ylim [-1, 1]** (replacing the |T|/phase panels).
- **P3 — core plot nodes. ✅ delivered.** `PlotTimeSeries`: absolute
  datetime x-axis (ConciseDateFormatter; relative seconds fallback), channel
  unit hints (mV/km electric, nT magnetic via `_plotstyle.channel_unit_hint`),
  station/run in title. `PlotSpectrogram`: unit-aware power colorbar label,
  station/run in title. Shared minimal `_plotstyle.py` (core stays mth5-free).
- **P4 — verification & docs. ✅ delivered.** Tests assert the conventions
  (off-diagonal ylim (0, 90), diagonal [-180, 180], tipper [-1, 1],
  errorbar containers present, ○/✕ markers, equal per-decade scale < 5%
  tolerance, station in title, log scales, dpi ≥ 150); visual review of the
  BGS NO46+NX37 RR workflow and the LEMI time series against the USArray
  reference; this section and README updated.

### 15.3 References

- **`SDG31b_H28coh.png` (USArray convention)** — primary style reference:
  log10 Period (secs) × log10 ρ_a (Ω m), linear φ (Degrees, first quadrant
  — our fixed ylim is 0–90), error bars, ○ XY (blue) / ✕ YX (red), framed
  legend, `Site [ LON = …; LAT = … ]` title.
- **`202511.PNG` (MTpy-style combined response)** — tipper panel layout
  reference: Re/Im panels per tipper component, error bars, y in [-1, 1].
- aurora 0.6.2 `transfer_function.TTFZ` (`apparent_resistivity`,
  `standard_error`, `RhoPlot.phase_sub_plot`/`rho_sub_plot`) — the pinned
  reference implementation of error-bar math and the phase-fold convention;
  aurora itself notes (Issue #209) these QC plotters should eventually be
  replaced by MTpy.
- mtpy `PlotMTResponse` — source of the tipper panel layout; **not a
  dependency** (optional dev extra for comparison tests only).
- mt_metadata 1.0.10 `TF.impedance_error` / `TF.tipper_error` /
  `TF.transfer_function_error` — the storage target for errors.

### 15.4 Decisions (2026-09-06)

- x-axis: **Period (secs), log10** — decided; consistent with the TF table
  node and the USArray convention.
- Tipper: **Re/Im panels, ylim [-1, 1]** — decided; replaces the current
  |T|/phase panels.
- Diagonal phase: **signed, [-180°, 180°]** — decided; the first-quadrant
  fold applies only to off-diagonal components.
- Markers/colors: **USArray convention** — ○ XY, ✕ YX, blue/red, plus
  distinct linestyles; XX = square / YY = diamond (proposal, not in the
  USArray reference).
- Open: add `mtpy` as a test-only dev extra for golden-image comparison?

---

## 16. M3 pending tasks — instrument calibration framework & coherence sorting (planned 2026-09-07)

Two independent follow-ups identified while investigating the BGS
NO46/NX37 short-period results. They are the **next M3 tasks** (before the
caching/sub-workflow M3 backlog); each lands separately.

### 16.1 General calibration framework + LEMI-417/424 calibration support

**Grounding (researched 2026-09-07).**
- The LEMI logger internally corrects the recorded signal and outputs
  physical units (nT, µV/m) **assuming unity gain and zero phase**; the
  deviation from unity grows near Nyquist (short periods) — per EPIC's
  `lemi2seed` docs, the calibration file provides the frequency-dependent
  correction factor (converted FAP → poles/zeros per logger/sensor pair,
  which is a **matched, individually calibrated pair**).
- **File formats**: the manufacturer's `.chn` (LEMI-417-era, e.g. `lemi.chn`
  dated 2009 — the NO46 unit is a LEMI-417) is an FAP file: header
  `'#FP' 12 5 / NTF,NCH DR-TRANSFORMED (MOD,ARG)` = 12 frequencies
  (0.25–1000 Hz) × 5 channels (`#HX #HY #HZ #EX #EY`), complex response as
  modulus + argument (degrees). The first column is **frequency (Hz)**.
  The LEMI-424-era format is the KMS `.def` file (used by the proprietary
  KMS-200 software); same FAP-style content, different wrapper.
- **No public parser exists** for either format: mt_io/mth5 only accept
  per-channel JSON (`{"Calibration": {"gain", "Freq": [...], "Re": [...],
  "Im": [...]}}`), and `LEMICollection.get_calibrations` globs only
  `*.json` — manufacturer files are silently ignored today. EPIC's
  `lemi2seed` ships a pre-baked PZ pickle instead.
- **Empirical check on the BGS pair (2026-09-07)**: the short-period ρ_a
  collapse (T=4–16 s: ρ~0.1 Ω·m, phase→0, E–B coh²=0.01–0.1) is primarily
  the **instrument noise floor** — B PSD ≈ 0.011 nT/√Hz at T=4 s equals
  the LEMI-424 fluxgate spec (≤10 pT/√Hz); the LEMI-424 is a long-period
  system (designed T=20–100,000 s). Applying the `.chn` (frequency
  interpretation, **flat/clamped below 0.25 Hz** — log-extrapolation makes
  results worse) scales ρ by ~×6.25 (E/H response ratio 0.012/0.03 at
  0.25 Hz) and shifts phase ~±30° — a real but modest correction; it cannot
  recover sub-noise-floor periods.

**Plan.**
- General calibration framework in `emptyui-mth5`: parse manufacturer FAP
  files (`.chn`/`.def`, MOD/ARG) **and** the mt_io JSON format into
  per-channel `mt_metadata FrequencyResponseTableFilter`s, and attach them
  to the run the way mt_timeseries expects — register in `run.filters` +
  `dataset[ch].attrs["filter.name"]` (naive `run.by.channel_response = …`
  is silently lost: `run.by` returns a copy; `RunTS.calibrate()` then
  applies them). Filters need a non-empty `name` and mt_metadata-compatible
  units (`nanoTesla`, `mV/km`).
- Expose as a `calibration` input/param: `Load LEMI-424` auto-pickup of
  calibration files (data dir / `calibration_path` param), plus an
  explicit calibration-file node feeding the existing `mth5.calibrate`
  node. `.chn` values are approximate when they do not match the specific
  unit — per-pair files are preferred (log/annotation when generic).
- Decide: interpretation of the FAP `gain` field (static factor) and
  below-table extrapolation policy (default flat = unity/zero-phase
  assumption at long periods).
- Verify against a reference dataset (e.g. an independently processed /
  EDI-published site) to settle the absolute-gain question.

**Implemented (2026-09-07).**
- `emptyui-mth5/calibration.py`: `parse_fap` (`.chn` LEMI-417 FAP and the
  KMS-200 `.def` wrapper — same FAP block, leading wrapper lines skipped;
  `#FP <NTF> <NCH>` header, MOD/ARG rows, first column = Hz), `parse_json_calibration`
  (mt_io `{"Calibration": {"Freq", "Re", "Im"}}`), `build_fap_filters`
  (per-channel `FrequencyResponseTableFilter`, units nanoTesla / mV·km by
  channel prefix, `gain=1.0`, `sequence_number=1`), `attach_filters`
  (mt_timeseries style: `run.filters` + per-channel `attrs["filters"]`
  AppliedFilter format — the modern format that survives an MTH5
  round trip; channels without a filter get a unity coefficient filter so
  `RunTS.calibrate()` passes temperature channels through warning-free),
  `find_calibration_file` (auto-pickup, station-pair files preferred),
  `load_calibration` (dispatch by extension).
- `Load LEMI-424`: `calibration_path` param (file or directory; empty =
  auto-pickup in the data directory) + `extend_flat` param; filters are
  attached **before** `from_runts`, so the calibration is archived with the
  run and survives `Extract Run`.
- New `mth5.calibration.load` node ("Load Calibration"): parses a
  calibration file into a `calibration` payload for the `mth5.calibrate`
  node's explicit path.
- Below-table policy implemented as **flat clamp** (first-row modulus, zero
  phase, extended to 0.001·f_min) — log-extrapolation was found to be
  worse (default `extend_flat=True`).
- Verified on the BGS NO46 LEMI-417 pair: `lemi.chn` attached through the
  archive, `RunTS.calibrate()` applies it, TF ρ_a increases by ≈×6.25
  (the E/H response ratio squared, as measured) and phase shifts ≈5–15° —
  consistent with the earlier empirical check; the noise-floor short
  periods remain below the spec (calibration is a modest, real correction,
  not a recovery).
- Open: absolute-gain verification against an independently processed
  reference site; `.def` wrapper parsing is heuristic (no sample in the
  repo — raise a clear error if the FAP block is not found).

### 16.2 EMTF-style coherence sorting (filter low-coherence FCs)

**Grounding (researched 2026-09-07).**
- aurora 0.6.2 has **no sample-level coherence sorting**: the robust
  M-estimator computes squared coherence (R²) per output channel
  (`m_estimator.compute_squared_coherence`) but only **logs** it;
  `get_band_for_coherence_sorting`/`adjust_band_for_coherence_sorting`
  (`frequency_band_helpers.py`) are **band-widening** helpers ("min3" rule,
  marked WIP). emptyUI's only coherence logic is the `Align Runs`
  inter-station guard (magnetic–magnetic, 64–1024 s).
- **Measured on the BGS pair**: E–B coh² ≈ 0.01–0.1 at T=4–16 s (signal
  below the fluxgate noise floor — no recovery), 0.03–0.3 at T=11–30 s
  (transition zone — coherence sorting *can* help), 0.6–0.9 at T>30 s.
  Remote reference is also incoherent at short periods (190 km separation).
- Grounding for the method: Egbert (1997, EMTF) and Chave & Thomson (2004,
  BIRRP) use coherence-based identification/downweighting of contaminated
  samples in robust MT processing; neither is implemented in the pinned
  aurora, so this is added in emptyUI (pre-regression, external to aurora's
  residual-based M-estimator).

**Plan (2026-09-07, feature design).**

*Design decision (2026-09-07): a dedicated node, not `Estimate TF` params.*
Coherence sorting is a distinct FC→FC transformation (Unix philosophy —
one thing, one node): **`aurora.process.coherence_sort`** ("Coherence
Sort"), category Processing, placed between `Compute FC` and `Estimate
TF`. It lives in `emptyui-aurora` because it reuses the `_period_bands`
banding (the same band geometry `Estimate TF` uses —
`periods_per_octave` is exposed and documented to match, default 4).

**Two-FC design (user decision 2026-09-07).** The node accepts **both**
FCs — `fc` + `ref_fc` inputs, `fc` + `ref_fc` outputs (mirroring
`Estimate TF`'s port structure) — because remote reference *replaces*
the local magnetic channels with the remote ones: local E/H coherence is
not the right quality measure when RME_RR is used. Sorting must use the
coherence that actually governs the downstream estimate:
- **single-FC mode** (no `ref_fc`): per-window multiple coherence of the
  local outputs (E, Bz) on the **local** magnetic inputs (original plan).
- **two-FC mode**: per-window multiple coherence of the local outputs on
  the **remote** magnetic inputs (the RR-relevant signal coherence),
  combined conservatively with the **local↔remote magnetic transfer
  coherence** (H_local vs H_remote): the reference can only clean the
  local H to the extent the two magnetic fields are coherent, so both are
  required for a good RR window. The node uses the **minimum** of the two
  — **fixed** (no `coherence_source` param; decision 2026-09-07): when
  remote reference is in play, the data are *already* facing bad
  coherence (low or very high), so the local E/H coherence is generally
  the wrong criterion; the conservative `both` is the only sensible
  choice, and the single-FC mode already provides the local-only view
  for single-site workflows. Both criteria are reported in the
  diagnostic table so the binding one is visible. Either FC alone is not
  sufficient for RR quality — this is the "local E/H fine but RR bad"
  (and vice versa) case.
- The two FCs must share identical **time and frequency grids** — the
  node validates them itself with the same errors as `Estimate TF`
  ("runs are not aligned — use Align Runs").
- **Shared per-row weight**: the single combined weight is baked as √w
  into **both** output FCs, making remote-reference row weighting exactly
  correct (one weight per observation row, as weighted LS requires).
  Running two separate one-FC nodes on the two stations would *not*
  achieve this (the earlier approximation). Note the remote station's
  quality now participates in the local sorting — correct for RR, but
  the diagnostic must make it visible.

*Node params (defaults per user decision 2026-09-07):*
- `coherence_min` (float, default `0.25`): lower bound — windows below are
  down-weighted (noise-dominated FCs; the BGS short-period bands).
- `coherence_max` (float, default `0.98`): upper bound — windows at/above
  are down-weighted: **perfect multiple coherence can be a noise
  signature** (few degrees of freedom / overfit, or coherent noise driving
  both E and H), so coh ≈ 1 must not automatically mean "good data".
- `mode` (`weight` default / `drop` / `off`): `weight` multiplies both
  FCs by √w per (band, window) with a **single shared weight per row**
  (weighted-LS semantics; aurora's robust weights act on top; no
  `Estimate TF` change needed); `drop` sets excluded samples to NaN in
  both FCs.
- `coherence_trim` (boolean, default `False`): **band-edge trimming** —
  set frequency bins whose band median coherence is below
  `coherence_min` to NaN in both FCs, with a warning listing the
  affected bands.
- `periods_per_octave` (default `4`, banding consistency with
  `Estimate TF`).

*Coherence statistic*: per-window **multiple coherence** (n_bins ≥ n_in +
1; the band design guarantees several harmonics at level 0), on the
already-normalized samples, per decimation level. Reference: local
magnetic inputs (single-FC mode) or remote magnetic inputs combined with
the local↔remote magnetic transfer coherence (two-FC mode, min). Sparse
bands fall back to the minimum pairwise coherence; all-zero/degenerate
windows score 0 (conservative). Per output channel, the conservative
minimum across outputs is used. Degenerate bands (n_bins < n_in + 1) fit
the regression exactly, so their coherence is trivially 1.0 — the
`coherence_max` suspicion rule is **not** applied there (implemented
2026-09-07), otherwise the longest-period bands would be dropped
unconditionally.

> **Semantics note (Batch B, 2026-09-07).** The statistic is a per-window
> multiple coherence over the **frequency-bin ensemble inside one STFT
> window** — a conservative broadband QC heuristic measuring how
> predictable a window's outputs are across the band's harmonics. It is
> **not** EMTF/BIRRP's window-ensemble coherence computed *at each
> frequency* (ensemble across windows). The two agree qualitatively
> (noise-dominated windows score low) but not numerically; a true
> EMTF-style per-frequency window-ensemble variant remains an M4 item.
> The node description states this so users don't equate the diagnostics
> table with EMTF's coherence plots.

*`Estimate TF` change (small, generic — not coherence-specific)*: skip
non-finite FC samples when gathering band rows. Required by
`drop`/`coherence_trim`, and gives correct remote-reference semantics for
free: a row is used only if **both** the local and remote FCs are finite
(the conservative union — RR pairing stays row-aligned because the time
grid is never dropped).

*Diagnostics*: the node's `table` output — per-period median coherence
for **each** criterion (E–H_local, E–H_remote, H_local–H_remote when
applicable), the retained/total window counts and the applied weight
range, so the binding criterion is visible. A coherence-vs-period plot
(USArray style per §15) is a later add-on.

*Implementation locations*:
- `emptyui-aurora/_processing.py`: `_simple_coherence`, `_multiple_coherence`,
  `_coherence_weight`, `coherence_sort_fc` / `coherence_sort_fcs`
  (per-window multiple coherence per band/level + weight/drop/trim, both
  reusable by the node); non-finite-row skip in the `estimate_tf` band
  gather loop.
- `emptyui-aurora/process.py`: `CoherenceSort` node (params, ports,
  register) + `_fc_io_channels` helper (channel recovery from fc attrs /
  nomenclature fallback).
- Tests + `mth5.coherence`-style table rendering (reuse `View Table`).

*Verification*:
- Synthetic FCs with injected low-coherence windows: weights exclude /
  down-weight them and recover the true impedance; `weight` vs `drop`
  compared; upper bound tested with an overfit/degenerate band (few
  windows → coh ≈ 1).
- Two-FC mode: synthetic stations with a noisy local H (low E–H_local
  but high E–H_remote coherence) must be **retained** for RR (the
  single-FC mode would wrongly drop them) and the RR estimate must
  recover the true impedance; conversely, a window with low E–H_remote
  or low H_local–H_remote must be dropped; shared √w baked into both
  FCs verified.
- Real BGS RR workflow: T=11–30 s should improve (coherence ↑, error
  bars ↓) while long periods are unchanged; `coherence_trim` removes the
  noise-floor bands with a warning; remote-reference row pairing intact.
- Defaults are active (min 0.25 / max 0.98, weight mode) but loose — the
  BGS long-period results must remain stable, and existing unit tests
  must still pass (update expectations only where the sorting is
  demonstrably better).

