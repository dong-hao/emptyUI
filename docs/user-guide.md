# emptyUI user guide

Recipes for the common processing chains. Load any workflow with the
toolbar **Load** button; ready-to-run examples live in
`examples/workflow/`. Node parameters edit inline on the node body —
hover a parameter for its one-sentence tooltip. The toolbar also has
**🕘 History** — a session-only list of finished runs (Open/Export/Delete/
Clear all); it is cleared on restart and there is no undo/redo.

> **Data availability**: only the `phoenix-mtu5a-smoke` and
> `phoenix-mtu5a-session-qc` examples ship with data (both use the tracked
> 3-hour smoke archive). The remote-reference
> recipe below can be exercised on a fresh clone by first regenerating
> the NIMS archives with
> `python examples/data/nims-usarray-test/build_mth5.py` (an EarthScope
> FDSN download). The `phoenix-mtu5a-day-12h` and `lemi424` examples
> need local corpora that do not ship — see the availability table in
> the README.

## Recipe: single-station processing

The bundled demo (`phoenix-mtu5a-smoke.emptyui.json`) is exactly this:

```
Load MTH5 → Extract Run → Compute FC → Coherence Sort → Estimate TF → plots
```

Expected: three Preview Plot artifacts — time series, AppRes/phase, and
spectrogram.

- **Load MTH5** — point at an `.mth5` archive.
- **Extract Run** — pick survey/station/run (`auto` = the only one); a
  time range subsets the recording.
- **Compute FC** — Fourier coefficients: `Window (samples)`/`Decimation
  factor`/`Decimation levels` define the frequency tiers (defaults suit
  150 Hz BBMT); `Persist to workspace` writes them back into the
  archive (MTH5 FC groups).
- **Coherence Sort** — per-bin EMTF-style weighting; `Sort below period
  (s)` sorts only bins shorter than that period (empty = sort
  everything).
- **Estimate TF** — band-averaging: `Periods per octave`,
  `Bands per level`, `Min bins per band` control the density/stability
  trade-off.

## Recipe: remote reference (RR)

Two stations, one chain:

1. **Load MTH5** twice (local + remote station), **Extract Run** on each.
   (The shipped `nims-remote-reference` example uses Merge Runs →
   Calibrate → Detrend before Align Runs instead of Extract Run on each.)
2. **Align Runs** — crops the remote run collection to the local time
   window; gap provenance is carried through and checked at pairing.
3. **Compute FC** on the local extract, and a second **Compute FC** on
   the aligned remote collection.
4. **Coherence Sort** (optional) weights the paired bins.
5. **Estimate TF** with `Remote Reference` ✓ — its `ref fc (remote)`
   port takes the remote FC. The robust estimator becomes RME_RR.
   `Remote pairing: intersect` matches FC windows that exist in both
   stations (raise `Pair tolerance (s)` to bridge small gaps).

## Recipe: session QC & gating

`phoenix-mtu5a-session-qc.emptyui.json` — data-driven run selection:

```
Load MTH5 → Extract Run → Mask TS → Compute FC → Select Runs → Estimate TF
```

- **Mask TS** builds a pooled `quality_mask` aux channel: Hampel
  despiking (`Hampel threshold (MADs)`, 8 by default), constant-run and
  gap detection, plus manual `exclude_intervals` — `start/end` ISO
  datetime pairs separated by `;`. Compute FC consumes the mask at the
  window level; deliberate exclusions do not warn.
- **Compute FC** with `Session length (h)` = 0.25 and
  `Regroup contiguous runs` on yields session-granular FCs (§9.4).
- **Select Runs** scores sessions per band with a soft diurnal prior;
  the run diagnostics and mask report are readable on **View Table**
  nodes. Aim for ≥ ~20–30 FC windows per session at the QC reference
  band — a noisy session then costs ±1 window per level instead of the
  whole recording.

## Plots and tables

Plot nodes take a `plot` port: **Plot AppRes & Phase** (impedance +
optional tipper), **Plot Tipper**, **Plot Coherency**, **Plot Phase
Tensor**, **Plot Time Series**, **Plot Spectrogram** (per FC level, FC
time series), and **Plot Coherence Map** (time–frequency QC view; tick
`Print parameters` only for algorithm-comparison figures). **Preview
Plot** renders any of them inline; **Save Plot** exports the incoming
plot in its source format (bundled plot nodes emit PNG; svg/pdf require a
matching source).
**View Table** shows tabular outputs (run diagnostics, TF tables).

## Exporting results

- **Export TF** — `edi` (default), `xml`, `zmm`, `avg`; impedance is
  written in SEG/EMTF units. (The `j` writer is not implemented yet.)
- **Write FC / Save MTH5** — persist Fourier coefficients or whole
  archives; **Read FC** reads persisted FCs back for later processing.
- **Load Calibration** — attach calibration filters to archives loaded
  without them.

## Troubleshooting

- **Server unreachable in the browser** — is it running? Remote browser
  → start with `--host 0.0.0.0`; port busy → `--port 8080`.
- **"path is outside the data root" / "absolute output paths are not
  allowed in remote mode"** — a `--data-root` server only resolves input
  paths inside that directory; use paths under it (saved workflows may
  carry project-relative paths).
- **A run fails on one node** — the node shows the error copyable in
  the properties panel; fix the parameter and press Run again (only
  the affected subtree recomputes).
- **Plots look stale** — plots re-render only when their inputs
  change; nudge a parameter upstream or re-run.
- **Data files for the demos** — `examples/data/` (the smoke archive is
  tracked; larger corpora are gitignored, see the notes there).
