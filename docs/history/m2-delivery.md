# emptyUI delivery record — Milestone 2

Verbatim record of the M2-era design and delivery decisions, preserved
from DESIGN.md with the original section numbering (§13–§15). Cross-
references to §1–§12 inside these records use the pre-restructure
numbering (e.g. §11 = MVP roadmap, §9 = frontend design); the living
contract lives in DESIGN.md.

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
   samples (n_windows × band width) fall below `min_bins_per_band`
   (default 40 in the internal helper; the Estimate TF node exposes **20**
   since the §24.6 default flip — every call site passes its value
   explicitly, `test_defaults.py` pins the split),
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
Reference figure: `SDG31b_H28coh.png` (USArray convention — primary style
reference; repo root). We do **not** build a combined plot — `Plot AppRes &
Phase` and
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
