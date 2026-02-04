# emptyUI — review findings (2026-09-18)

Review of the implementation against DESIGN.md / README / docs/. Verification
status at the time of review:

- Python suite: **445 passed, 1 skipped** (95 s) — green.
- Browser e2e (`npx playwright test`): **17 passed, 1 failed** — the failure is
  the stale banding-inherit test (H4 below), empirically reproduced.

**Fix rounds:** H1–H4 are resolved — `SelectRuns` / `ResidualSort`
registered (`plugin.py`) with a new wiring guard
(`tests/test_plugin_registration.py`), `ComputeFC` forwards `saturated_max`
with a node-level regression test (`test_fc_calibration.py`), the stale
e2e test was replaced by a §24.9 negative-regression test, and the HEAD
commit message was amended to describe its actual diff. M1–M4 are resolved
as well — DESIGN §24.13 / README recipe text refreshed (hour-level QC lives
in Select Runs since `623e541`), the `BuildFC.decimation_levels` and
`MergeFC` manifest descriptions corrected, the schema's unknown-param
hard-reject removed in favor of the documented §24.9 tolerate+warn policy
(end-to-end test added), and malformed workflow JSON now returns 422 with
pydantic's field errors instead of a 500. M5–M8 (the physics batch) are
resolved with corrected records — see their entries below (M5's trigger
was overstated and re-verified; M6/M7/M8 shipped with empirical
convention/declination/geometry fixes). M9–M14 are resolved too — see the
entries below (M12's guards were empirically upgraded from warnings to
hard errors; the reconnect-occupancy fix exposed reconnectEdge's
error-007 contract). Suite after the fixes: **Python 473 passed / e2e 18
passed**. The universal saturation-mask follow-up is recorded as M4 #19 in
DESIGN.md §11 and the README. A documentation-inconsistency sweep of the
low list is also resolved — see the "Documentation-inconsistency subset —
FIXED" block in the low section (§8/§17.4/§14.9/§24.6/§4 completion and
corrections, deployment.md env table, README commit-convention guard,
garbled/misleading docstrings, gitignore hygiene).

Everything else in the documented contract (API routes, caching/partial
runs/cancellation, plugin toggles, remote-mode knobs, settings, unit physics,
l2u banding, per-bin coherence sort, Phoenix MTU-5A chain) is implemented as
specified. The findings below are the exceptions.

---

## High severity (verified genuine)

### H1 — `Select Runs` and `Residual Sort` are implemented but never registered — **FIXED**

- **Evidence**: `packages/emptyui-aurora/src/emptyui_aurora/process.py:772`
  (`SelectRuns`, `aurora.process.select_runs`) and `:990` (`ResidualSort`,
  `aurora.process.residual_sort`) are fully implemented and unit-tested
  (`tests/test_select_runs.py`, `tests/test_residual_sort.py`,
  `test_fc_build.py:304`), but `plugin.py:15-23` registers only
  `ComputeFC, BuildFC, CoherenceSort, EstimateTF, MergeFC, MergeTFLevels` +
  the two plot nodes. Live check: `build_default_registry().status_of(...)`
  → `unknown` for both.
- **Impact**: DESIGN §24.13 and the README present both nodes as *delivered*
  (including the flagship self-test chain "ingest → Build FC → Select Runs →
  Coherence Sort → Estimate TF"). That chain is not expressible in the UI.
  `BuildFC.regroup_contiguous`'s description even routes users to Select Runs
  for hour-level QC (`process.py:377-384`).
- **Root cause**: the exact §18.1 failure mode (`mth5.tf.rotate` was
  implemented/tested but never registered) recurring — and the mth5 plugin
  has a registration guard test (`emptyui-mth5/tests/test_registration.py`)
  while the aurora plugin has none.
- **Fix plan**:
  1. `plugin.py`: import `SelectRuns, ResidualSort` and `registry.add` both.
  2. Add `packages/emptyui-aurora/tests/test_registration.py` mirroring the
     mth5 guard: assert the full registered type set (all 10 aurora types)
     against the plugin module — this closes the bug class, not the instance.
  3. Verify: registry lists `aurora.process.select_runs` /
     `aurora.process.residual_sort` as `active`; smoke the §24.13 chain in a
     test (`Executor` run of Build FC → Select Runs → Coherence Sort →
     Estimate TF on the synthetic corpus already exists piecemeal; one e2e
     test through the executor would pin the UI-expressible path).

### H2 — `Compute FC` silently ignores its own `saturated_max` param — **FIXED**

- **Evidence**: the param is declared at
  `packages/emptyui-aurora/src/emptyui_aurora/process.py:185` (UI-visible,
  `enabled_when=("reject_saturated", (True,))`) and accepted in
  `execute` (:217), but the `compute_fc_levels(...)` call at :223-236 omits
  it — the helper default `0.01` (`_processing.py:415`) always applies.
  `Build FC` forwards it correctly (`process.py:592`), so the two nodes
  disagree on the same exposed knob. No test covers `saturated_max` on
  either node (grep: zero references in tests).
- **Impact**: changing "Saturated max" on Compute FC does nothing, with no
  warning — a silent dead knob, violating the project's fail-loudly policy.
  Note: the param itself was added in HEAD commit `b31b509`, whose message
  (H3) does not describe this change.
- **Fix plan**:
  1. Add `saturated_max=saturated_max` to the `compute_fc_levels` call in
     `ComputeFC.execute` (one line, `process.py:223-236`).
  2. Regression test: call `ComputeFC().execute(...)` with
     `saturated_max=0.5` and assert `levels[0].attrs["saturated_max"] == 0.5`
     (the attr now exists since `b31b509`) and that the rejection counts
     respond to the threshold; the same assertion for Build FC (already
     forwarded) pins both nodes.
  3. Optional: a generic guard (executor warning when an `enabled_when`-gated
     param is dropped by a node) is a larger change; the direct fix plus test
     suffices for now.

### H3 — HEAD commit `b31b509` message does not match its diff — **FIXED**

- **Evidence**: the message says *"Add conservative window-overlap guard for
  run drops (fixes #412)"* and describes neighbor-run window drops, a
  frequency-domain overlap guard, and per-drop reason metadata — none of
  which appears in the diff. The actual diff is the `saturated_max`
  fraction-threshold feature (`_saturated_window_mask` ratio rule +
  `ComputeFC`/`BuildFC` params + fc attr + widened test fixture). The
  message also claims "added overlap-drop cases" tests that do not exist,
  references issue **#412** which does not exist in this repository (no
  remote, no other branch, no other mention of 412 anywhere), and violates
  the README's commit-log convention (no `Type:`, no DESIGN §ref).
- **Impact**: the commit log — the project's primary decision record per the
  README convention — is falsified at HEAD. Anyone reconstructing §24.12 QC
  behavior from `git log` will look for a feature that is not there and miss
  the one that is.
- **Fix plan**:
  1. `git commit --amend` HEAD with a corrected message, e.g.:
     `Feat: fraction-threshold saturated-window rejection (saturated_max) — §24.12 QC refinement`
     describing: the any-overlap rule deleted the long-period tail whenever
     saturation is frequent (73-min window vs one clipped sample); windows
     are now zero-weighted only when the flagged fraction exceeds
     `saturated_max` (default 0.01); param exposed on Compute FC / Build FC;
     `saturated_max` recorded in fc attrs.
  2. The run-drop-overlap work the message describes may exist somewhere
     else (another checkout/repo) — confirm before discarding the wording;
     if it was ever applied here, it is not in this history.
  3. Going forward: the repo has no remote/CI gate for message-vs-diff
     sanity; the README convention checklist could gain a "message describes
     this diff" line item.

### H4 — stale e2e test: banding inherit removed by §24.9 still asserted — **FIXED**

- **Evidence**: `packages/web/e2e/app.spec.ts:94-129` selects "Band style"
  on Coherence Sort and expects it to flow into Estimate TF. §24.9 removed
  exactly this (band-free sort; the Estimate TF inherit declaration was
  deleted at `emptyui_aurora/process.py:1366-1369`). Empirically reproduced:
  full e2e run → `1 failed / 17 passed`, the failure timing out waiting for
  the removed param (`app.spec.ts:125`). The spec was last touched at
  commit `f24d17e` — the commit that *added* the inherit — and never
  updated.
- **Impact**: the headless regression suite (and `.github/workflows/e2e.yml`
  CI) is red; §24.6's own caveat — "a browser e2e run with the per_bin
  default is pending" — was never closed out.
- **Fix plan** (choose one; (b) recommended):
  1. (a) Delete the test — the behavior was deliberately removed; the
     inherit machinery remains covered indirectly.
  2. (b) Replace it with a **negative regression test** pinning the §24.9
     surface: connect Coherence Sort → Estimate TF, assert the sort node
     renders **no** "Band style"/banding widgets, and Estimate TF's
     `Band style` stays at its `l2u` default (i.e. inheritance is truly
     gone). This keeps the §24.6/§24.9 decision under regression guard so a
     future revert of the surface is caught.
  3. While there: the §24.9 caveat means Plot Coherency renders the band
     statistic only when the sort is explicitly `statistic="band"` — the
     sort no longer exposes `statistic` at all, so double-check the Plot
     Coherency description/manifest wording still matches the code
     (`plots.py` renders per-bin since §24.9 — OK, but any residual
     description text referencing the band statistic should go).

---

## Medium severity

- **M1** ~~Stale §24.13/README docs~~ — **FIXED**: DESIGN §24.13 and README
  now record that hour-level QC folded into Select Runs (commit `623e541`);
  the historical scorer text is retained as the superseded record and the
  committed recipe updated (`stream_session_hours=1`, `regroup_contiguous=True`,
  Select Runs defaults, sort `apply_below_period=20`).
- **M2** ~~Stale manifest descriptions~~ — **FIXED**: `BuildFC.decimation_levels`
  now reads "Deepest decimation level: 0 = native rate only" (+ the §24.8
  per-family trim note) and `MergeFC` documents the rate-keyed mixed-rate
  merge instead of the retired rate-mismatch error; the dead `dec_of_run`
  block it stemmed from is removed.
- **M3** ~~Stale-param semantics broken by schema~~ — **FIXED** (policy =
  the documented §24.9 tolerate+warn): `schema.py` no longer hard-rejects
  unknown params; the executor's stale-param note is now reachable through
  `POST /api/runs`, pinned end-to-end by `test_stale_params_accepted_with_warning`.
- **M4** ~~Malformed workflow JSON → HTTP 500~~ — **FIXED**: `app.py` also
  catches pydantic's `ValidationError` and returns 422 with the field
  errors; `test_malformed_workflow_returns_422_not_500` covers the
  malformed-envelope cases.
- **M5** ~~`RotateTF` hard-requires a tipper~~ — **FIXED (record corrected)**:
  verification showed the claimed trigger (EDI imports) is factually wrong —
  EDI/XML readers store 5×5/3×2 with a zero hz row, so tipper-less EDIs
  (verified on the real USArray SDG31 EDI) and tipper-less `build_tf` TFs both
  rotate fine. The `hz_i is None` path is defensively unreachable in the
  bundled stack; hardened anyway (conditional zero-filled hz row, consistent
  with `_rotate_arrays`' tolerance) and regression-tested.
- **M6** ~~EDI import ignores `SIGNCONVENTION`~~ — **FIXED**: Import TF gains a
  `sign_convention` param (`as-is` default / `negate (180 deg)` /
  `conjugate`). Empirics (stored §24.12 parity data): the Phoenix MT-Editor
  EDI needs **negation** (Δφ +0.24°), not conjugation (−67°); the USArray
  EMTF EDI declares exp(+iωt) yet its data already matches emptyUI as-is —
  so headers cannot be auto-trusted; `as-is` warns when the field declares
  the + family, conversions append a processing note.
- **M7** ~~declination explicit-0.0 vs absent~~ — **FIXED**: layered logic —
  strict-matched Decl field is authoritative (explicit 0.0 survives, model
  not clobbered); a declination KEY line the strict regex cannot parse
  defers to mt_metadata's parser (no Hx override); the Hx AZM fallback fires
  only when no declination field exists anywhere.
- **M8** ~~FC provenance round-trip loses attrs~~ — **FIXED (scoped)**: a
  JSON-attr shortcut was implemented and **rejected after verification** —
  mt_metadata's `read_metadata` parses every group attribute and unknown
  keys crash ANY foreign consumer of the archive. Delivered instead: the
  window/decimation geometry rides mth5's own FC metadata model
  (`time_series_decimation` + `short_time_fourier_transform`, written by
  Write FC and `persist_fc_level`, restored by Read FC/`load_fc_level` with
  enum-value normalization), station identity resolves from the archive
  hierarchy, and `magnetic_unit`/nomenclature re-derive from channel names.
  `electric_scaling`/`calibration_applied` deliberately stay in-memory
  provenance (the FC model cannot express them; FC amplitudes are already
  calibrated). Bonus fix: Read FC with an empty run id now falls back to the
  first run (pre-existing gap, same contract as the write node).
- **M9** Frontend: (a) ~~`fitView` no-op + `padding: 40` fraction semantics~~ —
  **FIXED**: the declarative `fitView` prop (consumed on the empty initial
  node set in React Flow v12) was replaced by an explicit fit when the graph
  first becomes non-empty, with `padding: 0.1` under v12's fraction
  semantics (the inverted comment removed); (b) ~~polling dedupe drops
  multi-port `payload` events~~ — **FIXED**: the dedupe key now includes
  `port` + summary; (c) ~~edge reconnect violates one-edge-per-input~~ —
  **FIXED**: the reconnect path now drops edges occupying the new target
  input (same rule as onConnect); the dragged edge must survive the filter
  (reconnectEdge looks it up by id — missing → error 007 → the edge is
  lost); (d) ~~`EMPTYUI_BASE_PATH` unimplemented~~ — **FIXED**: read in
  `Settings.from_env` → `root_path`; (e) ~~deployment.md "toggling is
  planned"~~ — **FIXED** (wording updated); (f) ~~palette nodes 250 px~~ —
  **FIXED** (280 px everywhere).
- **M10** ~~`band_intervals` only populated for the l2u primary path~~ —
  **FIXED**: the non-l2u branch resolves the same table `estimate_tf` uses
  (`_resolve_period_bands` on the normalized, DC-filtered fc) and passes it
  as `bands_override` — guaranteed 1:1 alignment with the Z rows. Pairs are
  widened by one ULP so single-bin bands survive the
  `_bands_from_freq_pairs` mapping (searchsorted excludes edge-exact
  bins). Residual Sort no longer silently skips uniform/fallback levels
  (regression test runs the whole chain).
- **M11** ~~`overlap ≥ 1` user-reachable crash~~ — **FIXED**:
  `compute_fc_levels` rejects `overlap` outside [0, 1) with an actionable
  message before any scipy call (verified crash: `noverlap must be less
  than nperseg`); the manifest bound (0.99) still passes.
- **M12** ~~Merge Runs lacks §3 guards~~ — **FIXED (empirically
  upgraded)**: verification showed `RunTS.merge` cannot combine UNEQUAL
  channel sets (raw xarray `MergeError`) nor OVERLAPPING acquisitions
  (non-monotonic index error) — both are now hard errors with actionable
  messages naming the offending run; duplicate run ids in the params list
  are deduped with a warning (previously silently double-weighted).
- **M13** ~~View Selection dead + broken fallback~~ — **FIXED**: the node is
  `hidden = True` (palette-hidden, still executable for plugin-produced
  selection payloads) and `_as_dataframe` now accepts a real pandas
  DataFrame or a payload with `.df`; the dict fallback (crash-on-use) is a
  loud `TypeError`.
- **M14** ~~Settings null/0 conflation~~ — **FIXED**: null is rejected with
  422 for knobs with a non-null default (`cache_max_entries`), still valid
  for the None-default "off" knobs; `0` clamps to 1 identically on live PUT
  and restart-from-file (the falsy `or 256` conflation removed).

---

## Low severity (abridged; full detail available on request)

**Documentation-inconsistency subset — FIXED (docs round):**
- §24.6's "corrected exactly" adjusted-R² claim reworded: the implemented
  form leaves a ~1/(W−p) residual under the null vs the textbook
  null-zeroing form (L5) — empirically immaterial at ens=16; the ablation
  record was measured with the implemented form (formula change deferred:
  it would shift sort-threshold semantics vs the recorded ablations).
- §14.9's `min_bins_per_band` "default 40" clarified: 40 is the internal
  helper default; the Estimate TF node exposes 20 since §24.6
  (`test_defaults.py` pins the split).
- §8 API table completed: `GET/PUT /api/settings` (§22), `GET /api/files`
  (§17.2) and `GET /api/health` were implemented but undocumented.
- §17.4's env-knob list completed (adds `EMPTYUI_CORS_ORIGINS`,
  `EMPTYUI_DISABLED_PLUGINS`, `EMPTYUI_SETTINGS_PATH`; `EMPTYUI_BASE_PATH`
  is now implemented AND documented, plus a deployment.md table row).
- §4 tooltip guidance notes the 140-char fallback truncation (vs the
  ~100-char author guidance).
- Garbled `_resolve_band_style` docstring (L9) rewritten; `SaveMth5`'s
  "record its path" description corrected (the param is a mismatch guard,
  not a move); `Registry.rebuild`'s "never partial" docstring claim scoped
  to the `_nodes` swap (the plugin tables update in separate assignments).
- README commit-log convention gains the H3 lesson: the message must
  describe the diff it accompanies.
- Repo hygiene: `*.emptyui.json` gitignored (the two untracked user
  workflows in the repo root) and a root-level `test-results/` entry
  added.

**Remaining behavior items (low):**

- Adjusted-R² formula change — optional follow-up to the L5 doc fix above;
  requires re-running the §24.6 ablation thresholds first.
- Clock-zero: no input validation (raw pandas `ValueError` on typos) and a
  future clock zero silently anchors at run start (`_processing.py:792`,
  `:805-810`).
- Cache keys hash pre-coercion params (`executor.py:219-227`) — avoidable
  misses; file-param mtime invalidation silently misses for relative paths
  in remote mode (`executor.py:197-203`).
- `Registry.rebuild` three-assignment swap + unlocked `/api/nodes` read —
  minor race, message overclaim (`registry.py:135`, `app.py:272`).
- Broken third-party plugin crashes discovery/startup (`registry.py:170-175`)
  — no graceful degradation.
- `/api/health` implemented but undocumented (§8); `EMPTYUI_CORS_ORIGINS` /
  `EMPTYUI_SETTINGS_PATH` implemented but missing from §17.4's list.
- Headless cache restore writes artifacts to CWD when workspace is None
  (`executor.py:274`, `context.py:39`).
- Align Runs bare-number fallback mis-parses seconds as epoch nanoseconds
  (`emptyui_mth5/runs.py:135-136`); mixed channel lengths produce ragged
  RunTS silently (`runs.py:474-475`).
- `lemi.py:118-131` loop-variable leak (NameError on empty runs list);
  silently empty archive for a directory with no `.TXT` files.
- `phoenix_mtu.py`: nonexistent calibration folder → raw `FileNotFoundError`;
  schedule cross-check false positives at window edges; run vs channel
  metadata end-time off-by-one-sample; empty-serial warning wording.
- Dead code: `dec_of_run` (`process.py:429-438`), duplicate `_select_channels`
  (`emptyui_core_nodes/plots.py:29-40`), unreachable starvation branch
  (`_processing.py:1683-1685`), `DEFAULT_KEYWORD` export unused, garbled
  docstring (`_processing.py:1090-1092`), single-level `compute_fc` drifting
  from `compute_fc_levels` (tests-only).
- Core-nodes: `LoadCsv` imports pandas unguarded (optional dep); duplicate
  headers silently collide; auto time-column detection matches any header
  containing "time"; `SavePlot` name_prefix glob/`/` hazards; Detrend
  "none" shares channel arrays (no copy).
- Frontend: `TimeBar` clamp-order edge cases and bare-number semantics
  mismatch server-side; `ParamField` can emit transient NaN; 6-hex node ids
  without collision check; stale-edge pruning skipped when catalog hasn't
  loaded; Settings dirty check can't clear a 0-valued knob; properties panel
  omits manifest `models`.
- Repo hygiene: untracked user workflows in the repo root
  (`mtu5test.emptyui.json`, `nims-rr.emptyui.json`) — gitignore or move to
  `examples/`.

---

## Suggested order of work

1. ~~**H1 + H2 + H3 + H4**~~ — **done** (registration + guard, param
   forwarding, message amend, e2e replacement).
2. ~~**M1–M4**~~ — **done** (doc refresh, manifest descriptions,
   tolerate+warn policy, 422-on-malformed).
3. ~~**M5–M8**~~ — **done** (physics batch, with corrected records).
4. ~~**M9–M14**~~ — **done** (frontend/config batch, empirically verified).
5. Low list: the documentation-inconsistency subset is **done**; the
   remaining behavior items (clock-zero validation, cache-key coercion,
   plugin-crash degradation, LoadCsv/SavePlot/TimeBar polish, …) as
   capacity allows.
