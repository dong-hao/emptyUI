# emptyUI

A node-based, ComfyUI-inspired visual workflow interface for magnetotelluric
(MT) data processing and visualization. See [docs/DESIGN.md](docs/DESIGN.md) for the
architecture and design decisions.

## Layout

```
packages/
├── emptyui/              # Python core: node API, registry, DAG executor, FastAPI server
├── emptyui-core-nodes/   # bundled minimal node plugin
├── emptyui-mth5/         # archive/IO plugin: MTH5/LEMI/Phoenix, TF I/O, plots
├── emptyui-aurora/       # processing plugin: FC/TF estimation, QC, sorts
└── web/                  # React + React Flow frontend (Vite)

examples/
├── workflow/             # ready-to-load example workflow documents (kept in sync
│                         #   with the current node manifests; regenerate by loading
│                         #   and re-saving when node surfaces change)
├── data/                 # sample datasets (raw instrument corpora are gitignored;
│                         #   see the notes in each subdirectory)
└── plugin-template/      # working plugin template

**Session QC & gating demo** — `examples/workflow/phoenix-mtu5a-session-qc.emptyui.json`
runs the full data-driven QC story on the tracked `smoke` Phoenix
archive: Mask TS (time-domain quality mask: Hampel despiking, constant-run
and gap detection) → Compute FC with `session_hours=0.25` +
`regroup_contiguous` (session-granular FCs; §9.4) → Select Runs (per-session
scoring with the soft diurnal prior) → Estimate TF. The run diagnostics and
mask report are viewable on their View Table nodes. When sessioning pays:
the statistical unit is FC windows per session — aim for ≥ ~20–30 windows
per session at the QC reference band (§9.4); a noisy session then costs ±1
window per level instead of the whole recording. The demo is pinned by a
headless execution test (`test_demo_session_qc.py`).

test/verification/        # recorded parity/metrics JSONs + the verification
                          #   scripts that reproduce them (DESIGN history sections)
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

# CI runs the same two groups (.github/workflows/tests.yml) on every
# push/PR, plus the Playwright e2e suite (.github/workflows/e2e.yml)

# Start API server on http://127.0.0.1:8000
./scripts/dev-backend.sh

# Frontend — dev server on http://localhost:7860 (proxies /api to :8000),
# also on every interface (0.0.0.0:7860) for remote browsers. Narrow it:
#   cd packages/web && npm run dev -- --host 127.0.0.1
cd packages/web && npm run dev

# …or launch both at once (Ctrl+C stops both cleanly)
./scripts/dev.sh
```

### Commit-log convention

```
Type: one-line summary — the what and where (docs/DESIGN.md §ref)

Short paragraph: the problem/motivation — why the change exists.

- bullet: the substantive change (mechanism, not narration)
- bullet: …one per logical step

Tests: what covers the change (test names/cases, or "none needed" for
message-only/docs chores + why).
```

`Type` is one of `Feat:`, `Fix:`, `Perf:`, `Chore:`, `Docs:`. Keep the
subject under ~72 characters where possible; the docs/DESIGN.md section
reference anchors the change to the design record. The message must
describe the diff it accompanies — a message narrating changes that are
not in the commit falsifies the decision record. Historical references (history §13
and above) are written `history §N` and resolve in `docs/history/`, which
preserves the original numbering verbatim; current-contract references are
written `docs/DESIGN.md §<n>` against docs/DESIGN.md's living numbering (§1–§12).
Both records live under `docs/` since 2026-10-17; commit messages written
before that date cite the repo-root `DESIGN.md`/`issues.md` paths — do not
rewrite them (the same precedent as `docs/history/`'s verbatim records).

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

Current state: **M0–M3 complete (2026-09-09)**; M4 (community & advanced
backlog) in progress. The full feature status, the ranked M4 backlog and the
processing-correctness record (unit chain, USArray SDG31 ground-truth
verification, MTU-5A official-product parity) live in [docs/DESIGN.md](docs/DESIGN.md)
§12; dated delivery records are preserved in `docs/history/`.

Feature summary:

- **Framework** — manifest-driven nodes (zero frontend changes for new node
  types), plugin system with enable/disable, DAG executor with structural
  caching, partial runs, cancellation, WebSocket events, runtime settings,
  remote-server contract (data-root confinement, sub-path mounting, TTL/GC,
  run-limit guard).
- **MT pipeline** (verified end-to-end on real data) — Mask TS (time-domain
  quality gating: a pooled `quality_mask` aux channel from Hampel despiking,
  constant-run/clip detection, interval exclusion and gap reporting;
  Compute FC consumes it at the window level), Load MTH5/CSV/LEMI-424/
  Phoenix MTU-5A, channel tooling, Align/Merge Runs, frequency-domain
  calibration inside Compute FC, multi-level Compute FC (taper schedule,
  prewhitening, clock-zero anchoring, saturation rejection), Estimate TF
  (OLS/robust/RME_RR, l2u/uniform/manual band grids), Coherence Sort (per-bin
  EMTF-style statistic), Select Runs + Residual Sort (data-driven run-level
  QC with a soft diurnal prior), Rotate TF, TF import/export.
- **Display** — publication-quality plots (AppRes & Phase, Tipper, Coherency,
  Phase Tensor, Spectrogram, Time Series) with error bars, USArray
  conventions and station metadata.
- **Verification anchors** — USArray SDG31 vs the official EMTF TF
  (median |Δlog₁₀ρ| = 0.019 after the documented −5.7° rotation, 98% within
  gate); MTU-5A burst pipeline vs the official Phoenix EDI (Δlog₁₀ρ ~0.005
  over the full band). Records + reproduction scripts in `test/verification/`.

The M4 backlog (batch runner, disk cache, cross-level FC pooling, MMT, …) is
ranked in docs/DESIGN.md §12.2.

### UI conventions (stable)

- **ComfyUI paradigm**: parameters render inline on the node body; the
  right-hand panel shows metadata (id, version, ports, description, deps) and
  Delete. Hover protocol: one-sentence tooltips; palette in fixed category
  order (Data I/O, Processing, Display), nodes alphabetized, utility nodes
  hidden by default (docs/DESIGN.md §4).
- **Plots as data**: Display nodes emit a typed `plot` port; `Preview Plot`
  renders inline (drag-resizable) and `Save Plot` exports. No run-level gallery
  (docs/DESIGN.md §8).

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — the living design contract, organized by subsystem
  (§7 backend, §8 frontend, §9 data/instruments, §10 processing semantics,
  §11 deployment, §12 roadmap).
- [docs/plugin-guide.md](docs/plugin-guide.md) — plugin authoring contract.
- [docs/deployment.md](docs/deployment.md) — remote/embedded deployment.
- `docs/history/` — verbatim delivery records (M2/M3/M4) preserving the
  original history §13–history §24 numbering; write references to them as `history §N`
  in code docstrings and commit messages.
