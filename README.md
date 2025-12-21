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

Current state: **M0–M3 complete (2026-09-09)**; post-M3 deliveries — the
**l2u automatic frequency grid** is now the `Estimate TF` default
(2026-09-11, DESIGN.md §14.9) and the **TF rotation node** is delivered
(2026-09-12, DESIGN.md §11/§16.3). All remaining work is the M4 backlog below;
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

### M4 — community/advanced backlog (ranked by gain/effort; full detail in DESIGN.md §11)

1. **Plot Coherency + Plot Phase Tensor nodes** — high gain, low effort: standard
   MT QC/interpretation products; data and plot patterns already exist.
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
