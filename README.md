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
# One-shot setup: creates .venv, installs all packages editable (with their
# [dev] extras) + the example plugin, and runs npm install in packages/web.
# Add --with-browsers to also download the Playwright Chromium build.
./scripts/install.sh

# …or do it manually:
uv venv --python 3.12 .venv
uv pip install -e "packages/emptyui[dev]" -e "packages/emptyui-core-nodes[dev]" \
               -e "packages/emptyui-mth5[dev]" -e "packages/emptyui-aurora[dev]" \
               -e "examples/plugin-template[dev]"
cd packages/web && npm install

# Run tests
.venv/bin/python -m pytest packages -q

# Fast development passes (tests are grouped by role, see below)
.venv/bin/python -m pytest packages -q -m "not errors"   # happy path
.venv/bin/python -m pytest packages -q -m errors         # edge cases (~5 s)

# Start API server on http://127.0.0.1:8000
./scripts/dev-backend.sh

# Frontend — dev server on http://localhost:5173 (proxies /api to :8000)
cd packages/web && npm run dev

# …or launch both at once (Ctrl+C stops both cleanly)
./scripts/dev.sh
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
  off-diagonal), Plot Tipper (Tzx/Tzy real & imaginary panels). Verified
  end-to-end through the executor on real 1 Hz
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
  M2 is now complete — all §11 M2 deliverables delivered (see also the
  plot standardization item below).
- **M2 — Phase 4 polish** (delivered 2026-09-05):
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
  - **Plot standardization** (delivered 2026-09-06): publication-quality TF
    plots per DESIGN.md §15 — USArray conventions (log10 Period × log10 ρ_a,
    phase 0–90° fold for off-diagonal, diagonal signed [-180, 180°], tipper
    Re/Im with ylim [-1, 1], ○ XY / ✕ YX markers, equal per-decade scale,
    error bars with caps, site + LON/LAT title and processing-provenance
    footer, dpi 150). TF error propagation (estimator covariances →
    `transfer_function_error`/`tipper_error`) and TF station metadata
    (survey/station/run + lat/lon/elevation from the run) are wired through
    ComputeFC → EstimateTF → build_tf, so both plots **and** EDI/XML/ZMM
    exports carry errors and a complete station header. Core time-series/
    spectrogram plots got the same treatment (datetime axis, unit hints,
    identity titles, dpi 150).
- **M3 — delivered so far**:
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
  - **Remote reference end-to-end**: new **Align Runs** node
    (`mth5.runs.align`) crops two time series to their common acquisition
    overlap (same sample rate required; absolute GPS times preferred,
    relative-offset fallback with a warning) so their FCs can be paired —
    the prerequisite for `Estimate TF`'s Robust + Remote Reference (RME_RR).
    `estimate_tf` now validates that the remote fc shares the local fc's
    **time grid** as well as the frequency grid (clear "align the runs"
    error). A magnetic-only remote station is supported: empty output
    channels resolve to a reference FC (magnetic channels as outputs). Align
    Runs also measures inter-station magnetic coherence and warns when it is
    weak (clock offsets / un-filled data gaps break alignment — mt_io's LEMI
    reader indexes by row, so `scripts/convert_bgs_lemi.py` interpolates over
    recording gaps: one row == one true second). Tested end-to-end on the BGS
    LEMI sites NO46 + NX37 (48 h common window, decimated FCs, RME_RR, EDI
    export) plus an offline synthetic two-station test and a coherence
    regression guard. The bundled BGS pair is **truly continuous**: the
    common window starts at 2024-02-29 11:14:00, after NO46's only recording
    gap (1254 s); converter gap-filling remains as a safety net for other
    data.
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
  - **Correctness batch ("Batch A", delivered 2026-09-07)**: remote-reference
    fc containers missing a decimation level now error instead of silently
    degrading to a self-reference; JSON calibrations keep per-channel
    frequency grids and unmatched filter keys warn (case-insensitive channel
    match) instead of silently passing through; artifact/diagnostic filenames
    are node-instance-unique (`ctx.artifact_path`) so same-type display nodes
    no longer overwrite each other's figures; `Calibrate` with explicit
    filters delegates to `RunTS.calibrate` (proper detrend/zero-mean/
    zero-pad/windowing, no bare-FFT circular-convolution artifacts).

### Remaining work (see DESIGN.md §11/§12/§16)

- **M3 — next tasks (planned, independent)**:
  - **Robustness batch ("Batch B")** — **delivered (2026-09-07)**:
    validation & data-quality guards, no design changes:
    - Fan-in duplicates rejected: multiple edges into one input port are now
      a `validate_workflow` error (previously silently overwritten, last wins).
    - Legacy mth5 API removed: `SpectraToFCGroup`/`MergeRuns` resolve ids via
      the dual-layout `_resolve_survey`/`_resolve_station`/`_resolve_run`
      helpers (regression: `StationGroup.runs_group` does not exist in mth5
      0.6.x — the old physical-merge path crashed with AttributeError);
      `Merge Runs`' dormant `selection` mode is documented as such.
    - `Load CSV` robustness: time-column-only files error cleanly; ISO
      datetime time columns set `start_datetime` and derive the rate (via
      pandas, lazy import); predominantly non-monotonic time columns error;
      NaN/Inf in any channel now raise a clear error (naming channel + first
      bad sample) at `Compute FC`, `Bandpass Filter` and `Detrend` entry
      instead of propagating silently through scipy `decimate`/`filtfilt`/
      `stft` (`Detrend method=none` stays a passthrough).
    - Tipper plot ylim auto-expands above [-1, 1] when |T| legitimately
      exceeds 1 (strong 2D/3D conductivity contrasts); default unchanged.
    - Coherence-sort semantics documented: per-window broadband
      (bin-ensemble) coherence heuristic, not EMTF's window-ensemble per
      frequency (DESIGN §16.2 note; true EMTF-style variant is M4).
  - **Test suite reorganization (2026-09-07)**: edge-case/error-surface tests
    (primary assertion is `pytest.raises` / error status / degenerate input)
    are grouped into one marked file per package (`test_errors.py` for the
    framework, `test_core_errors.py`, `test_mth5_errors.py`,
    `test_aurora_errors.py`; marker registered in the root `pytest.ini`).
    `pytest -m errors` runs the error surfaces in ~5 s. CI still runs the
    full suite by default; shared fixtures (`archive`, `fake_runts`) moved
    to the mth5 `conftest.py`.
  - **Graph interaction hardening (2026-09-07, F1)**: self-loops and cycles
    are rejected live during the drag (`isValidConnection`, ComfyUI-style
    feedback), re-checked in `onConnect`, and reported explicitly server-side
    ("cannot connect to itself"); the executor fails loudly on cycles even
    when called without validation. E2E: `packages/web/e2e/interactions.spec.ts`.
  - **Correctness batch ("Batch A", delivered 2026-09-07)**: remote-reference
    fc containers missing a decimation level now error instead of silently
    degrading to a self-reference; JSON calibrations keep per-channel
    frequency grids and unmatched filter keys warn (case-insensitive channel
    match) instead of silently passing through; artifact/diagnostic filenames
    are node-instance-unique (`ctx.artifact_path`) so same-type display nodes
    no longer overwrite each other's figures; `Calibrate` with explicit
    filters delegates to `RunTS.calibrate` (proper detrend/zero-mean/
    zero-pad/windowing, no bare-FFT circular-convolution artifacts).
  - **Content-addressed caching + partial execution** (2026-09-08): node
    results are cached across runs in a server-lifetime in-memory LRU keyed
    by structural signatures (`type, version, params, upstream signatures` —
    ComfyUI-style recursive invalidation; file params hashed by
    path/size/mtime, Nextflow-style; data payloads never hashed). Unchanged
    nodes skip execution and report **cached** (green chip on the node);
    published artifacts are restored into the new run's workspace so viewers
    keep working. `POST /api/runs` accepts `targets` ("execute to here": the
    selected node plus its ancestry only) exposed as **▶ Run to here** in the
    properties panel; `DELETE /api/cache` drops all entries (DESIGN §7).
  - **Plugin packaging guide + example template** (2026-09-08): the
    ecosystem contract in `docs/plugin-guide.md` — entry-point packaging,
    namespaced node/port types (open type system: plugins may introduce
    namespaced types with declared UI metadata, collected into the catalog
    and used to color edges), the §4 manifest/tooltip protocol, the
    functional (immutable-input) execute contract, **required self-tests**,
    **models declared not bundled** (id/filename/sha256/license in the
    manifest; binaries in a configured model directory), versioning and a
    publishing checklist. A working, installable template lives in
    `examples/plugin-template/` (`pip install -e` + its nodes appear under
    the Example category).
  - **Disconnect + copy/paste (2026-09-07, F2/F3)**: ComfyUI-parity canvas
    gestures — drag a connected handle to the pane to unlink, drag edge
    endpoints to re-route (validated) or drop on the pane to delete,
    `Ctrl/Cmd+C/V/D` for subgraph copy/paste/duplicate (fresh ids,
    deep-copied params, remapped internal edges, pasted copy independent of
    the original), full-selection delete (multi-node + selected edges), and
    edge hover/selected styling. Initial `fitView` capped at 100% zoom.
    Interaction protocol documented in DESIGN.md §9; 11 e2e tests cover all
    gestures.
  - **Instrument calibration framework + LEMI support** — **done** (DESIGN.md
    §16.1): `emptyui-mth5/calibration.py` parses manufacturer FAP files
    (`.chn`/`.def` MOD/ARG, and the mt_io JSON format) into mt_metadata
    filters; `Load LEMI-424` gains `calibration_path`/auto-pickup (filters
    archived with the run, surviving `Extract Run`), and the new
    `mth5.calibration.load` node feeds the existing `Calibrate` node.
    Verified on the BGS pair: ρ ×≈6.25, phase ≈5–15° — a modest real
    correction; the short-period collapse is the LEMI noise floor. Open:
    absolute-gain check against an independently processed reference site.
  - **EMTF-style coherence sorting** — **done** (DESIGN.md §16.2): the
    `aurora.process.coherence_sort` node down-weights/drops low-coherence FC
    windows (and supports remote-reference two-FC mode) before the robust
    regression, with a per-band coherence diagnostic table.
- **M3 — not yet started**:
  - Plugin enable/disable switches — **design done** (DESIGN §5.1): tick-box
    settings dialog (third-party plugins only; bundled `core`/`mth5`/`aurora`
    always on), palette reveal/hide via catalog refetch, in-process hot
    reload of the registry ("backend restart" without a process respawn),
    settings-file persistence + `EMPTYUI_DISABLED_PLUGINS` env baseline,
    silent ignore when nothing is installed; implementation pending.
  - Remote-server mode — **contract delivered** (DESIGN §17 + §17.6 items
    1–3 & 6: data-root confinement with path_kind params, same-origin dist
    serving + configurable CORS, runtime sub-path API base, config module +
    `docs/deployment.md`); workspace TTL cleanup and max-concurrent-run
    limits are implemented and configurable for embedded deployments.
  - Sub-workflows — **deferred to M4**: comparable systems (ComfyUI
    subgraphs, n8n sub-workflows, Node-RED subflows, KNIME metanodes) earn
    their complexity at decluttering/sharing scale that typical 5–10 node MT
    pipelines don't have; copy/paste + example-workflow templates + an M4
    batch runner cover the real repetition patterns (DESIGN §12).
- **M4 — community/advanced**:
  - Plugin index/publish flow, docs site, batch/queue runner, more
    visualization nodes (Plot Coherency, Plot Phase Tensor are designed but
    not implemented).
  - Sub-workflows (nested DAGs) — deferred from M3; rationale in DESIGN §12.
  - Disk-persisted result cache (2026-09-08: deferred from M3 — not
    essential to running the process; the cross-run cache stays in-memory
    and clears on restart).
  - Aurora-style band config files (exact centre frequencies + bandwidths)
    instead of the always-auto-derived period table.
  - Paradigm B: cross-decimation-level FC pooling (see DESIGN §14.5) — a
    novel extension, not required for the current pipeline.

### UI conventions (stable)

- **ComfyUI paradigm**: node parameters render inline on the node body; the
  right-hand panel shows metadata (node id, version, ports, description, deps)
  and a Delete action. Path widgets include a server-side file browser ("…"
  button). Hover protocol: one-sentence tooltips on palette buttons and node
  headers (full details on click in the panel); the palette shows categories
  in fixed order (Data I/O, Processing, Display) with nodes alphabetized,
  and hides utility/test nodes by default (see DESIGN.md §4 protocol).
- **Plots as data**: Display nodes emit a typed `plot` port; `Preview Plot`
  nodes render figures inline (drag-resizable) and `Save Plot` exports them —
  ComfyUI's Preview/Save Image pattern. There is no run-level gallery; the
  node-embedded viewers are the primary (and only) display (DESIGN.md §9).
