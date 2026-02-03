# emptyUI — review findings (2026-09-18)

Review of the implementation against DESIGN.md / README / docs/. Verification
status at the time of review:

- Python suite: **445 passed, 1 skipped** (95 s) — green.
- Browser e2e (`npx playwright test`): **17 passed, 1 failed** — the failure is
  the stale banding-inherit test (H4 below), empirically reproduced.

**Fix round (same day):** H1, H2, H4 and H3 are resolved — `SelectRuns` /
`ResidualSort` registered (`plugin.py`) with a new wiring guard
(`tests/test_plugin_registration.py`), `ComputeFC` forwards `saturated_max`
with a node-level regression test (`test_fc_calibration.py`), the stale
e2e test was replaced by a §24.9 negative-regression test, and the HEAD
commit message was amended to describe its actual diff. Suite after the
fixes: **Python 448 passed / e2e 18 passed**. The universal saturation-mask
follow-up is recorded as M4 #19 in DESIGN.md §11 and the README.

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

- **M1** Stale §24.13/README docs: commit `623e541` removed
  `reject_noisy_runs` (+ QC band/threshold knobs) from Build FC — hour-level
  QC folded into Select Runs — but DESIGN.md §24.13 (lines ~3513/3538) and
  README ("Build FC gains hour-level QC before any cascade (`reject_noisy_runs`)")
  still document the removed param as "the committed recipe".
- **M2** Stale manifest descriptions: `BuildFC.decimation_levels` claims
  "0 = auto (decimate until fewer samples than one window remain)"
  (`process.py:326-329`) — code computes `min(decimation_levels, d_max)`, so
  0 = no decimation, same as ComputeFC. `MergeFC` still advertises "sample
  rates must agree (error)" (`process.py:1382-1385`) although §24.8 made
  mixed rates the feature (`_processing.py:1874-1883`).
- **M3** Stale-param semantics broken by schema: `schema.py:87-92` hard-422s
  unknown params before a run, so the executor's stale-param warning
  (`executor.py:502-513`, §24.9: saved workflows "processed unchanged" with
  a warning) is unreachable via `POST /api/runs`. Pick one policy.
- **M4** Malformed workflow JSON → HTTP 500 instead of 422:
  `server/app.py:402-406` catches only the internal `ValidationError_`;
  pydantic's `ValidationError` from `Workflow.model_validate` escapes as 500
  (verified: missing `type`, non-list `edges`, non-list `nodes`).
- **M5** `RotateTF` hard-requires a tipper (`emptyui_mth5/tf.py:248-251`)
  although `_rotate_arrays` handles `hz is None` — tipper-less EDIs (very
  common) fail with a misleading "ex/ey/hz required" error; a conditional
  2×2 block fixes it.
- **M6** EDI import ignores `SIGNCONVENTION` (`tf.py:133-149`): an
  `exp(+iωt)` product (e.g. the Phoenix `B0940330_z.edi` whose 180° flip the
  §24.12 record corrected by hand) imports with mirrored phases and no
  warning; also interacts with the 0–90° phase fold (−45° → clipped).
- **M7** §24.2 declination fix is partial (`tf.py:145-147`): `if not d.value:`
  cannot distinguish "field absent" from "explicit 0.0" (mt_metadata
  defaults the header declination to 0.0); an explicit zero in an unparsed
  format can be silently overwritten by the Hx-AZM fallback.
- **M8** FC provenance round-trip loses attrs (`emptyui_mth5/fc.py`): only
  `units` survives write→read; `magnetic_unit`, `electric_scaling`,
  `calibration_applied`, window geometry, station identity all drop —
  breaking merge warnings and the §23.3 provenance line for read-back FCs
  (§3's "enriched by `mth5.fc.read`" only partially holds).
- **M9** Frontend: `fitView` is consumed on the empty initial node set (no-op)
  and `padding: 40` is a *fraction* in React Flow v12, not px
  (`packages/web/src/App.tsx:360-364`); polling-fallback dedupe omits `port`
  from the `payload` event key (`App.tsx:218-233`) — multi-output nodes lose
  all but one extent summary in the polling path; edge **reconnect** does
  not enforce one-edge-per-input (`src/stores/workflow.ts:181-187`);
  `EMPTYUI_BASE_PATH` documented in DESIGN §17.4 but not implemented in
  `config.py`; `docs/deployment.md:110` still says plugin toggling "is
  planned" (delivered §5.1); palette-added nodes are 250 px vs loaded 280 px
  (§23.4).
- **M10** `band_intervals` only populated for the l2u primary path
  (`emptyui_aurora/_processing.py:2484`) — Residual Sort silently skips
  `uniform`/metadata-less-fallback levels (`process.py:1070-1073`).
- **M11** `overlap ≥ 1` remains server-reachable: manifest `max=0.99` is
  frontend-only (`process.py:134`, `:333`); neither `compute_fc_levels` nor
  the executor validates `Param.min/max` — review item B13 half-open.
- **M12** Merge Runs lacks §3's documented guards: no overlap warning, no
  channel-consistency check, duplicate run ids accepted silently
  (`emptyui_mth5/merge.py:39-98`).
- **M13** View Selection is registered but dead: no bundled node produces a
  `selection` payload (producer M4-deferred; `tables.py:26-46`). Hide or
  remove; its dict fallback path would also crash (`_as_dataframe`).
- **M14** Settings edge cases: `cache_max_entries` accepts `null` (violating
  §22's "null not allowed for non-null defaults") and `0` conflates with
  unset across restart (`settings.py:176-178`, `app.py:198-199`).

---

## Low severity (abridged; full detail available on request)

- Adjusted-R² debias off-by-one vs the null-zeroing textbook form
  (`_processing.py:1192-1193`) — "exact" claim in §24.6 slightly overstated.
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
2. **M1–M4** — doc/manifest refresh + the two API-consistency fixes.
3. M5–M14 and the low list as capacity allows; the physics items (M5–M8)
   matter most for correctness-sensitive users.
