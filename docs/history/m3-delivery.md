# emptyUI delivery record — Milestone 3

Verbatim record of the M3-era design and delivery decisions, preserved
from DESIGN.md with the original section numbering (§16–§17). Cross-
references to §1–§12 inside these records use the pre-restructure
numbering (e.g. §11 = MVP roadmap, §9 = frontend design); the living
contract lives in DESIGN.md.

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
  `EMPTYUI_CORS_ORIGINS`, `EMPTYUI_BASE_PATH`, `EMPTYUI_WORKSPACE_TTL`,
  `EMPTYUI_MAX_CONCURRENT_RUNS`, `EMPTYUI_DISABLED_PLUGINS`,
  `EMPTYUI_SETTINGS_PATH`)
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
