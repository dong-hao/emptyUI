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

### Commit-log convention

```
Type: one-line summary — the what and where (DESIGN §ref)

Short paragraph: the problem/motivation — why the change exists.

- bullet: the substantive change (mechanism, not narration)
- bullet: …one per logical step

Tests: what covers the change (test names/cases, or "none needed" for
message-only/docs chores + why).
```

`Type` is one of `Feat:`, `Fix:`, `Perf:`, `Chore:`, `Docs:`. Keep the
subject under ~72 characters where possible; the DESIGN.md section
reference anchors the change to the design record.

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

Current state: **M0–M3 complete (2026-09-09)**; post-M3 deliveries — the
**l2u automatic frequency grid** is now the `Estimate TF` default
(2026-09-11, DESIGN.md §14.9) and the **TF rotation node** is delivered
(2026-09-11, DESIGN.md §11/§16.3). All remaining work is the M4 backlog below;
full architecture, contracts and verification records are in
[DESIGN.md](DESIGN.md).

### Implemented

**Framework**
- Node API + manifest-driven UI: new nodes need zero frontend changes; plugin
  system via entry points, namespaced port types, plugin enable/disable
  (DESIGN.md §5.1); node manifests carry tooltip/description/palette protocol,
  `cascade` auto-change and chain-walking `inherit` (§4).
- DAG executor: topological scheduling, WebSocket run events/progress,
  content-addressed structural caching with artifact restore, partial runs
  ("Run to here"), run cancellation, artifact store/download (§7–§9).
- Runtime settings (§22): ⚙ dialog sections (plugins / live knobs —
  retention TTL, run limit, cache size — / read-only deployment rows /
  planned §21 cache knobs); `GET/PUT /api/settings`, whitelist-validated,
  env-baseline-wins precedence, opt-in JSON persistence via
  `EMPTYUI_SETTINGS_PATH`.
- Remote-server contract (DESIGN.md §17.6 items 1–6): data-root confinement,
  same-origin serving + CORS control, sub-path mounting, workspace TTL +
  artifact GC, max-concurrent-runs guard, config module + deployment guide.

**MT pipeline** (all verified end-to-end on real data)
- Data I/O: Load MTH5 / CSV / LEMI-424 (nomenclature select, calibration
  auto-pickup), Extract / Split / Merge Channels, Align Runs, Merge Runs,
  Calibrate (incl. LEMI FAP `.chn`/`.def`), TF Import/Export (EDI/XML/ZMM/AVG/J).
- Processing: ComputeFC (multi-level decimation, tipper, nomenclature-aware
  automatic channel configuration), EstimateTF (OLS / robust RME / RME_RR;
  `l2u` default band grid with window-starvation coarsening + `t_short` head
  clip, `uniform` legacy scheme), MergeTFLevels, Coherence Sort
  (two-FC remote-reference aware), Rotate TF (true-north rotation, EDI
  declination parsing with angle fallback).
- Display: Plot AppRes & Phase, Plot Tipper, Plot Time Series, Plot
  Spectrogram, TF/Selection tables — publication-quality conventions
  (DESIGN.md §15: USArray markers, phase fold, error bars, station metadata,
  equal per-decade scale, dpi 150).

**Correctness (verified)**
- Unit normalization (nT→A/m for fluxgate *and* coil archives, mV/km→V/m);
  impedance phase convention matches aurora (Zxy≈+45°, Zyx≈−135°).
- USArray ground-truth verification (4P.SDG31 vs the official EMTF TF,
  DESIGN.md §16.3): after the documented −5.7° true-north rotation, median
  |Δlog₁₀ρ| = 0.019 with 98% within the gate on the exact official grid; the
  delivered l2u default grid is statistically equivalent (0.020/0.071/93%).
  Verification also found and fixed two real bugs (coil-unit conversion,
  Align Runs dropping channel units).
- **EMTF physics-parity ablations (DESIGN.md §24, 2026-09-13)**: opt-in
  first-difference prewhitening (exact inverse recoloring) + per-window
  detrend on ComputeFC. First ablation: the current default's whole deficit
  vs the official product is DC/edge leakage at T = 7.3–15 s; prewhitening
  OR a zero-endpoint taper closes it (94%/88% → 100%/100% within-gate,
  short-period p90 0.065 → 0.016 decades); boxcar without prewhitening is
  catastrophic (~1 decade) — quantifying the classical prewhitening
  motivation. Default flip deferred until the detrend ablation and the
  persisted-FC convention are settled.
- **Short-period gap closed (DESIGN.md §24.3 E1b, 2026-09-13)**: the
  residual deficit at 7.3–11.6 s was remote-data coverage, not estimator
  physics — the official EMTF-FCU product (`SDG31b_H28coh` tag) uses all
  SDH28 remote data; aligning against SDH28-b only had cropped away ~29%
  of the local run. Full remote coverage fixes 9.1–11.6 s outright; 7.3 s
  closes with mild coherence selection (the FCU's own "coh" lever).
  Self-reference underestimates ρ by up to 5.8 decades below 15 s —
  quantifying the fluxgate noise floor.
- **Long-period tail explained (DESIGN.md §24.3, 2026-09-13)**: the diurnal
  Sq variation inside the 9.1 h/36.4 h deep-level windows leaks through
  boxcar's sidelobes — a per-level taper schedule (boxcar+PW at short-period
  levels, hann+PW at deep levels) combines the head and tail wins. Refined
  full-band recipe vs the official EMTF: median |Δlog₁₀ρ| 0.009, p90 0.029,
  98% within gate (original record: 0.019/0.045/98%). Per-window detrend
  ruled out as a tail lever (it is not in emptyUI's chain and not needed).
- **Burst-mode prior art surveyed (DESIGN.md §24.7, 2026-09-14)**: Phoenix
  MTU-5C/MTU-5A burst data (scheduled high-rate slots + continuous low-rate
  streams) is represented upstream as run-per-slot (mth5 convention) and
  processed as independent per-run FC cascades + time concat (aurora
  legacy) — matching our Merge FC design. Decoded the official IAGA test
  archives; found a latent timestamp bug in the MTU-5A reader (block
  headers carry true slot times, upstream discards them) and no working
  MTU-5A ingestion path. Design: two load nodes (5C collection walk,
  5A TBL+block-header parser) feeding per-run cascades + Merge FC.
- **MTU-5A burst ingestion delivered (DESIGN.md §24.8, 2026-09-14)**:
  `Load Phoenix MTU-5A` node (TBL + per-block-tag parsing — fixes the
  upstream dense-concat timestamp defect; run-per-contiguous-segment
  `sr{rate}_NNNN`, TBL scalar calibration at ingest, clock/saturation
  warnings) + `Build FC Database` node (`aurora.fc.build`: per-run FC
  cascades merged along time, clock zero defaults to "epoch" — staggered
  slot schedules make run-start anchoring useless; per-run FC persistence
  off by default). Validated against the official IAGA 1690C16C recording
  (121 × 1 s @ 2400 Hz slots, 122 × 16 s @ 150 Hz slots, 20.3 h
  continuous @ 15 Hz) with a CI-safe synthetic-corpus test suite. MTU-5C
  loader follows once the procedure is proven.
- **MTU-5A TBL calibration constants verified; per-coil response gap mapped
  (DESIGN.md §24.10, 2026-09-15)**: the TBL's magnetic chain (FSCV=6.4 V,
  HGN=12, HATT=0.233, HNOM=1000 mA/nT → 2.729e-7 nT/count, ±2.29 nT full
  scale) and electric chain (EGN=40, 100 m dipoles → 1.907e-4 mV/km/count)
  verified end to end with physically sensible sample magnitudes. Key gap:
  one scalar chain serves three different coils (serials coil1693/94/95);
  the per-coil frequency responses are being converted from proprietary
  formats to ASCII to land in the Calibrate route (FAP-style, keyed by coil
  serial) — the scalar stays at ingest, per decision (c).

- **MTU-5A ASC calibration delivered end to end (DESIGN.md §24.12,
  2026-09-17)**: the manufacturer ASC tables (box `*_CLB.asc`, one coil file
  per `*_CLC.asc`) are normalized shape corrections — their headers carry
  exactly the TBL constants (E-gain/H-gain, 1000 mV/nT), so the TBL scalar
  stays the absolute scale and the ASC filters correct the frequency
  response on top. `Load Phoenix MTU-5A` auto-matches the tables by the TBL
  serials (SNUM, HXSN/HYSN/HZSN), registers them in `/Survey/Filters/fap`
  and references them per channel (box stage 1, coil stage 2,
  `applied=True` = the response is in the recorded data); Compute FC now
  divides the Fourier coefficients by the chained complex response
  (aurora-style, applied-flag-aware, DC column NaN when the response
  vanishes there) — the chains ride the payload through the core nodes and
  Align Runs, and compose safely with the time-domain Calibrate node.
  Verified end to end on the 1690618B corpus (all three coils + box
  matched, zero warnings).

- **MTU-5A parity with the official Phoenix TF; coupling decided (DESIGN.md
  §24.12, 2026-09-17)**: against the official MT-Editor EDI for the same
  one-night window (`B0940330_z.edi`, self-reference, rotation 0), the
  burst pipeline reproduces the official TF to **median Δlog₁₀ρ +0.0005/+0.0045
  (IQR 0.015/0.027 decades), Δφ +0.2° (IQR 0.7°) over 0.003–500 s** after
  the SI↔EDI conversion (×795.77) and the 180° sign convention. The official
  applies the box CLB table **as-is** (including its low-frequency tail) —
  confirmed by A/B against the alternative (coupling-neutralized) variant,
  which deviates −0.11…−0.35 decades at 182–505 s. Full-record QC findings
  also recorded: June 21 daytime is saturated (block tags), the MTU cancels
  the SP offset on its slot paths but not on the continuous stream, and
  stronger coherence sorting amplifies the long-period scatter (negative
  result) — the trustworthy recipe is session-based.
- **Saturation QC: zero-weight saturated FC windows (DESIGN.md §24.12,
  2026-09-17)**: MTU-5A ingest now emits a per-sample `saturated`
  auxiliary channel from the block tags; the mask max-pools through
  decimation and Compute FC (default) NaNs every FC window overlapping a
  flagged sample (`reject_saturated` on/off, attrs count). The clean-record
  TF is now smooth 0.003–109 s (previously garbage >10 s) and the official-EDI
  parity is preserved (Δlog₁₀ρ +0.002/+0.002, Δφ +0.1°) — rejection is
  surgical: medians barely move, the daytime garbage windows vanish.

### Current issues (known problems, from the 2026-09-11 code review)

Fixed by the review batches (each with a regression test): Rotate TF
registration, the output-path confinement escape, Coherence Sort's
banding mismatch with Estimate TF (`min_bins_per_band`/`t_short` are now
exposed and verified identical), the plugin-toggle/discovery races, the
cancelled→completed status flap, the WS client leak, `merge_tf_levels`
crashing on payloads without `Z_err`, NaN guards on unused auxiliary
channels, the Align Runs coherence guard ignoring LEMI `bx/by` names, and
the missing absolute-ohm amplitude test (analytic ~1% anchor); frontend:
radio/datalist DOM id collisions, unvalidated workflow load (format/version
+ stale-edge pruning), the lost-event window at run start, the inherit
  auto-fill re-render loop, Run-to-here double-start, and the permanent
  `untitled` save name; post-review: unit-aware electric scaling
  (DESIGN.md §19 — V/m payloads pass through, raw-potential data error
  instead of silently mis-scaling, `electric_scaling` recorded in fc
  attrs). Still open:

- **Unbounded growth by default** — with `workspace_ttl_seconds=None` (local
  default) run state, event lists and temp workspaces are never cleaned; the
  `NodeCache` pins payloads by reference and keeps stale entries. (Decision
  2026-09-12: TTL only for now; a portable local disk cache is a later
  design.)
- **Coherence Sort statistic default is per-bin (2026-09-14) — now the only
  behavior (2026-09-15)** — the EMTF-style per-frequency window-ensemble
  multiple coherence (sliding ±16 windows, adjusted-R² debiased), fully
  band-independent: the sort node is band-free (no
  `statistic`/`periods_per_octave`/`band_style`/`min_bins_per_band`/
  `t_short`/`coherence_trim` params — banding belongs to Estimate TF,
  DESIGN.md §24.9), diagnostics are per frequency bin, and Plot Coherency
  renders the same per-bin statistic (§20.1). The executor warns on stale
  params in saved workflows instead of silently ignoring them.
  Band-geometry defaults stay on Estimate TF (3 periods/octave, 20 min bins
  per band).
- Viewer bodies stay **deliberately literal** (ComfyUI parity, decided
  2026-09-12): producers emit `plot` ports and stay compact — previewing is
  a user choice via the Preview/Save/View Table utility nodes; generalizing
  inline viewers is parked until table workflows mature.

### M4 — community/advanced backlog (ranked by gain/effort; full detail in DESIGN.md §11)

1. ~~Plot Coherency + Plot Phase Tensor nodes~~ — **delivered 2026-09-12**
   (DESIGN.md §20): fc-direct coherence plot (numerically identical to the
   Coherence Sort statistic) and Caldwell-style phase-tensor ellipses
   colored by skew; no new runtime dependencies.
2. **Merge Runs: remote multi-row warning** — medium gain, trivial effort: warn
   when a local run spans multiple remote runs.
3. **Batch/queue runner** — high gain, medium effort: multi-station processing is
   the real repetition pattern; reuses the executor as-is.
4. **In-band smoothing taper (Parzen/Hann)** — medium gain, low effort (§14.8):
   `√weight` FC rows by band-center proximity; composes with OLS/RME/RME_RR.
5. **Disk-persisted result cache** — medium gain, medium effort (§7): cache
   keys/restore exist; avoids FC/STFT recomputation across restarts.
6. **Calibration absolute-gain verification** — medium gain, low effort (§16.1):
   process one site independently (mtpy/other) and compare.
7. **True EMTF per-frequency coherence sort** — medium gain, medium effort
   (§16.2): window-ensemble per-frequency variant for EMTF parity.
8. **mtpy golden-image plot comparison** — low-medium gain, low effort (§15):
   test-only regression guard for plot conventions.
9. **Hand-authored band-config files** — low-medium gain, medium effort: exact
   band-table node option for apples-to-apples comparisons; the l2u scheme
   (§14.9) already reproduces the official grid strategy automatically.
10. **Paradigm B: cross-decimation FC pooling** — high gain, high effort (§14.5):
    novel, research-y, unproven — the highest scientific upside.
11. **FC→TS inverse node** — low gain, low-medium effort (§12): inverse
    STFT/overlap-add for denoising demos.
12. **Executor thread-pool concurrency** — low gain, medium effort (§7):
    pipelines are 5–10 mostly-linear nodes.
13. **Sub-workflows (nested DAGs)** — medium gain, high effort (§12): revisit
    only if workflows outgrow copy/paste + templates + the batch runner.
14. **Docs site** — low-medium gain, medium-high effort.
15. **Plugin index/publish flow** — low gain, high effort: PyPI + the
    plugin-guide checklist already cover distribution.
16. **ML plugin GPU querying** (`ctx.resources.gpu_count`) — low gain, trivial
    effort (§12): only matters once ML plugins exist.
17. **Improved auto frequency grid** — **delivered 2026-09-11** (§14.9): the
    `l2u` scheme is the Estimate TF default (fixed reference template +
    log-uniform window-starvation coarsening + optional head clip).
18. **MMT (multi-station MT) estimator** — low-medium gain, medium-high effort,
    **long-term** (Kelbert et al. 2026): Egbert/Smirnov–Egbert extension of RR
    using multiple sites; aurora 0.6.2 has only single-remote estimators.

### UI conventions (stable)

- **ComfyUI paradigm**: parameters render inline on the node body; the
  right-hand panel shows metadata (id, version, ports, description, deps) and
  Delete. Hover protocol: one-sentence tooltips; palette in fixed category
  order (Data I/O, Processing, Display), nodes alphabetized, utility nodes
  hidden by default (DESIGN.md §4).
- **Plots as data**: Display nodes emit a typed `plot` port; `Preview Plot`
  renders inline (drag-resizable) and `Save Plot` exports. No run-level gallery
  (DESIGN.md §9).
