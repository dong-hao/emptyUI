# emptyUI — Issue Register

**Open findings only.** Two sources merged:

1. the **2026-09-29 full-codebase review** (docs ↔ implementation ↔ tests) —
   findings written as `H*` / `M*` / `L*`;
2. the still-open items of the **2026-09-26 register** — re-verified today and
   re-issued under their new severity rank; their legacy IDs are kept in the
   entry titles for cross-reference.

Solved entries are **removed** from this register (the fixes they recorded
land in commits `aaca9ad` … `4e46132`, 2026-09-26…29). For continuity, the
closed set is: A1–A5, B1–B3, C1–C5, D1–D2, E1, E2, E4, F1, G1–G2, G6, G11,
G12 (the relative-path matrix half; its remaining UI half is M19), and the
legacy H-series leftovers H1, H2, H6 (those are **2026-09-26** IDs — none of
them is the live H2 fixed this session). **ID collision warning:** the old
register also lettered its sections A…H — those IDs appear here only as
`legacy …` inside a live entry (or in the closed list above); the live
`H*`/`M*`/`L*` numbers below are new and rank by severity.

Status legend: `[ ]` is **open**, `[x] FIXED (2026-09-30)` is closed this
session and kept in place with its remedy note for continuity (it will be
deleted with the rest of the register on cleanup). The session's fixes
landed as `dd0dfed` (H2, H5) · `09b8a59` (H1) · `e6735ad` (H3, H4) ·
`d4a2f02` (H6). Open entries are ranked
**High → Medium → Low by impact/severity**, not by subsystem. Section refs
in parentheses use the living docs/DESIGN.md numbering (§1–§12); `history §N`
resolves in `docs/history/`.

> **Note (2026-10-25):** the 2026-09-28 "stays gitignored" decision is
> **historical** — L17 is done and the file is tracked (`git ls-files
> docs/issues.md` confirms). The register is versioned from here on; the
> stale note is kept only as provenance. The header's "solved entries are
> removed" sentence is likewise stale (the `[x]` entries are retained) —
> see L29.

**Verification baseline (2026-09-29, this review):** pytest **630 passed** /
0 failed (87 s) · `ruff check .` **0** (pinned `ruff==0.16.4`) ·
`npm run build` (`tsc -b && vite build`) clean · Playwright **28 tests /
9 spec files** collectable · CI runs both pytest marker groups, the lint job
and the e2e suite. 42 registered node types.

**Re-verified after the fixes (2026-09-30):** pytest **643 passed** (+13
regression tests over the 630 baseline: 3 in `test_magnetic_units.py`, 3 in
`test_plot_style.py`, 2 in `test_remote_reference.py`, 5 in
`test_errors.py`) · `ruff check .` **0** · `npm run build` clean ·
Playwright **31 passed / 10 spec files** (+1 delete-selection test for
H4, the H3 toolbar assertion, +2 failure-surface tests for H6). H2/H5 alone
were 636; H1 adds the 2 provenance tests; H3 adds the 5 failure-reason
tests (H6 and H4 are frontend fixes — 0 pytest, +2 e2e). Marker groups now
split `-m "not errors"` **570** / `-m errors` **73**.

**Re-verified after the Build FC retirement (2026-10-15, M25):** pytest
**748 passed** (0 failed, ~90 s) · `ruff check .` **0** · `npm run check`
clean (vitest 38) · Playwright **34 passed**. Node types: 42 → 40
(`aurora.fc.build`, `aurora.fc.merge` removed).

**Re-verified after Mask TS (2026-10-15, M26):** pytest **774 passed**
(0 failed, ~90 s) · `ruff check .` **0** · `npm run check` clean
(vitest 38) · Playwright **34 passed**. Node types: 40 → 41
(`mth5.qc.mask_ts` added).

**Re-verified after the L20 demo (2026-10-15):** pytest **775 passed**
· `ruff check .` **0** (the demo test executes the shipped document
headless; the Hampel default retune is recorded in M26's surface).

**Correction (2026-10-15, found during L21):** the "recorded JSONs
unchanged" claim above was wrong for the CLB coupling records — they
were last written pre-M25 (`b1845de`, Build FC era) and the uniform
route shifted the marginal deep-band grid (46 → 44 periods; shared
values to 3.4e-15). Re-recorded at L21.

**Re-verified at the 2026-10-25 full-codebase review** (docs ↔
implementation ↔ tests): pytest **727 passed** (`-m "not errors"`) +
**74 passed** (`-m errors`) = **801 collected**, 0 failed (~72 s) ·
`ruff check .` **0** · `npm run typecheck` / `npm run lint` clean ·
Vitest **38 passed** (6 files) · Playwright **36 tests / 11 spec files**
collectable · **42 registered node types** (core 9, mth5 22, aurora 9,
example 2) · all five shipped example workflows validate against the
current manifests. The gates are green: the findings below are **not**
gate failures — they are correctness, contract and drift gaps the gates
do not cover. **Open after this review: 7** (M28, L27–L32; H8 fixed
2026-10-25, H9/H10/M27/M30 fixed 2026-10-26, M29 fixed 2026-10-27,
M31 planned 2026-10-27); the `[x]` entries are retained for continuity.

---

## High — silent wrong results, invisible failures, data loss

- [x] **H8 — the coherence map's time axis double-shifted window centres
  to window ends (silent misregistration, worst at deep levels). FIXED
  (2026-10-25).** (Found 2026-10-25, full-codebase review; the L18 node.)
  `plots.py:511-526` (`_coherence_map_matrix`). The FC `time` coordinate
  is **already** the window centre — the settled L11 convention, and the
  function's own docstring said so (`plots.py:495-497`);
  `process.py:288-297` proves it by subtracting `half_win` to recover a
  start. The function then added a second half-window (`centres = t +
  half`, `t + dt / 2`), so the map was labelled with window **ends**.
  Measured on the fixture: level 0 labels `[16, 32, 48]` s (16 s = half
  of the 32 s window) became `[32, 48, 64]` s; level 2 shifted by 256 s.
  `pcolormesh(..., shading="auto")` treats the labels as centres, so the
  whole map slid right relative to the data. Silent: no warning, no
  error, a plausible-looking image.
  **Remedy (done 2026-10-25):** `_coherence_map_matrix` now returns the
  fc's own `time` coordinate unchanged (`centres = np.asarray(local
  ["time"].values)`); the half-window/`dt` computation is deleted. The
  docstring states the labels are the fc's centres, not derived here.
  **Tests:** `test_coherence_map_window_centres` rewritten — it used to
  **pin the bug** (`expected = starts + half`); it now asserts
  `centres == paired["time"]` on (a) the relative float-seconds fixture,
  (b) an absolute-time (datetime64) fixture via the new
  `_two_fc_levels(start_datetime=...)`, and (c) the deepest level
  (half-window 256 s, so a regression cannot hide in a tolerance). Red
  first: 63/63 mismatches, 16 s shift.
  **Verification:** pytest **801 passed** (~71 s), `ruff check .` 0.

- [x] **H9 — `Calibrate` mutated its input payload in place (cache
  poisoning; DESIGN §7.1). FIXED (2026-10-26).** (Found 2026-10-25.)
  `calibrate.py:256-267` + `calibration.py:293-353`. The explicit-filter
  path took the input object itself (`runts = data if hasattr(data,
  "dataset") ...`) and called `attach_filters(runts, ...)`, which writes
  `run.dataset[ch].attrs["filters"]` (`calibration.py:328,342`), assigns
  `run.filters = run_filters` (`:344`) and clears the channel cache
  (`:345-352`). A cached or shared `RunTS` was therefore modified by a
  downstream node: the same input hash could yield different results
  depending on execution order, and a second `Calibrate` on the same
  payload saw an already-stamped run (the M8 guard only catches
  `_refuse_if_calibrated`, not this path's attrs). The dict calibration
  path also stamped missing names/units onto the caller's filter objects
  (`_channel_filters`).
  **Remedy (done 2026-10-26):**
  1. New `_private_copy(runts)` (`calibrate.py:110-130`) — a fresh
     `RunTS` with `dataset.copy(deep=True)` and a shallow copy of the
     filter registry. `RunTS.copy()` alone is not enough: `set_dataset`
     stores the Dataset by reference, so the copy would share the
     DataArray attrs. `execute` attaches to that copy
     (`calibrate.py:296`); the caller's payload and channel cache stay
     untouched.
  2. `_channel_filters` copies each dict-path filter
     (`model_copy(deep=True)`, `copy.deepcopy` fallback) before stamping
     names/units (`calibrate.py:167-171`), so the calibration payload is
     immutable too.
  **Tests:** `test_calibrate_does_not_mutate_the_input_runts` (snapshot
  of every channel's attrs + the registry; a sentinel archived response
  on the unmatched channel proves the copy carries the registry and the
  output still applies it);
  `test_calibrate_twice_on_the_same_payload_is_order_independent` (two
  gains on one payload, each output matches its own gain, the input
  keeps `filters == []` and an empty registry);
  `test_calibrate_does_not_mutate_the_calibration_payload` (a bare
  filter keeps empty name/units). Red first: the input gained `cal_ex`;
  the bare filter gained `cal_ex` / `milliVolt per kilometer`.
  **Verification:** pytest **804 passed** (~71 s), `ruff check .` 0.

- [x] **H10 — `ImportTF`'s `sign_convention` was a silent no-op for every
  format except EDI. FIXED (2026-10-26).** (Found 2026-10-25.)
  `tf.py:219-233` declares the param for all imports; `tf.py:260-302`
  applied it only inside `if p.suffix.lower() == ".edi":`. For `.xml`,
  `.zmm`, `.avg`, `.j` the user selected "negate" or "conjugate", the
  node reported success, and the TF was unchanged — a silently wrong
  polarity/phase convention downstream (the exact class H2 fixed for
  units). The only `sign_convention` tests were EDI
  (`test_tf_rotate.py:441,452`).
  **Remedy (done 2026-10-26):** the negate/conjugate block moved out of
  the EDI branch and now runs for every format before the EDI-only
  header/declination parsing (`tf.py:260-272`). The EDI SIGNCONVENTION
  warning stays EDI-only (it reads the header text).
  **Tests:** `test_import_sign_convention_applies_to_every_format`
  (`test_tf_rotate.py`), parameterized over the real sample files shipped
  with the pinned `mt_metadata==1.0.10`
  (`data/transfer_functions/tf_xml.xml`, `tf_zmm.zmm`, `tf_jfile.j`,
  `tf_avg.avg` — in the wheel RECORD, so a stable fixture; the test skips
  if a future pin drops them). The assertions are relative — as-is vs
  negate vs conjugate — so they do not depend on each file's own
  convention. Red first: all four formats returned identical arrays for
  all three options (112/700 elements mismatched on the negate check).
  The export round-trip was rejected as a fixture source: the minimal
  synthetic TF does not survive the XML/ZMM/AVG writers (missing
  metadata, azimuth rotation, pydantic enum), so the library samples are
  the honest real-file route.
  **Verification:** pytest **808 passed** (~71 s), `ruff check .` 0.

- [x] **H7 — Phoenix calibration absence loads silently (and the
  collected warnings are UI-invisible). FIXED (2026-10-13).** (User
  report, 2026-10-13.) The MTU-5A loader collected calibration problems
  in `res["warnings"]` and logged them — server log only, invisible to a
  browser session (M15: `node_progress.msg` is stored but rendered
  nowhere) — and the total-absence case ("no ASC tables provided at all
  — stay silent", `_prepare_asc_calibration`) did not even log: the load
  completed with the TBL scalar chain only and a missing
  frequency-dependent shape correction, i.e. a silently wrong TF.
  **Remedy (done 2026-10-13):**
  1. The no-ASC-at-all path now warns (naming the folder and the
     consequence) instead of returning `{}` in silence.
  2. New **`ctx.warn(message)` → `node_warning` run event**: the generic
     degradation channel (RunContext, like `ctx.progress`). The Phoenix
     loader's `execute` routes every warning through it (also logged —
     the log keeps the full record; headless embedding degrades to the
     log alone). Other loaders can adopt it the same way.
  3. Frontend: `node_warning` events collect per node (deduped,
     order-preserved, reset per run); the node card shows an amber ⚠
     marker (full text on hover) and the properties panel renders the
     readable list — the user's "warning on the node, or the right
     panel", both.
  **Tests:** `test_phoenix_mtu.py` — the no-ASC corpus warns (RED
  pre-fix: the archive test even asserted `warnings == []` on the ASC-less
  synthetic corpus, pinning the silence) and the serial-mismatch case
  warns per channel (passed pre-fix — the collection existed, only its
  surface was missing). `test_node_warnings.py` (3) — the event contract
  (emitted raw, node-scoped), the log co-presence, the headless
  fallback. `run.test.ts` (2) — per-node collection + dedupe + reset.
  DESIGN §7.2 documents the event.
  **Verification:** pytest **714 passed**, `ruff check .` 0, `npm run
  check` clean, Playwright **31 passed**.
  (M15's dead `node_progress.msg` surface is a separate, still-open
  item — this fix adds the warning channel; progress text rendering
  remains M15's.)

- [x] **H1 — the M4 #2 gap-provenance warning chain is dead on the default
  route. FIXED (2026-09-30).** Extract Run's default `selection="time range"`
  returns a *collection*, and the warning sites were wired only into the
  single-run body (`runs.py:1058-1068` vs `_align_collections`' carry-only at
  `:1248`); Compute FC's collection route copied no provenance at all
  (`process.py:402` vs `:289-296`), so `_warn_remote_gaps` could never fire for
  a collection-produced ref fc; `merge_fc_containers` kept only the first
  contributor's attrs, losing member-2+ gaps; and the single route's copy
  swallowed `AttributeError` in a bare `except: pass`.
  **Remedy:**
  1. `_align_collections` now runs the same chain as the single body —
     `_warn_merge_gaps_in_overlap` (window stamp normalised through `_utc`)
     and `_warn_if_uncorrelated`, which is collection-aware via the new
     `_aligned_channel_names` / `_aligned_channel_values` (members
     concatenated in time order; the coherence guard requires a single,
     shared rate). `_merge_gap_attrs` unions a collection's members' gap
     provenance. Partial coverage needed no new site: the collection
     route already warns with the same 90% threshold and its own remedy
     (drop warning, `runs.py` per-side loop).
  2. `process.py` gained `_merge_provenance_attrs(data, logger)` — a
     `dir()`-probe copy of the three `merged_*` keys that **logs** a read
     failure instead of hiding it (E4); both the single route and the
     collection route call it (the collection falls back to the
     collection-level attrs when a member carries none).
  3. `_processing.py` exports `_MERGE_PROVENANCE_ATTRS`, and
     `merge_fc_containers` unions those keys over every contributor
     (`_union_merge_provenance_attrs`) instead of keeping `base.attrs`.
  4. DESIGN §3 now states the chain must hold **on both routes**.
  **Tests (the missing middle):** `test_single_route_stamps_gap_provenance_on_the_fc`
  (pins what already worked — passes on pre-fix code) and
  `test_collection_route_stamps_gap_provenance_on_the_fc`
  (fails on pre-fix code at the Align warning, and with only the Align fix
  at `ref_fc.attrs["merged_gap_intervals"]` → `KeyError`).
  **Verification:** pytest **638 passed**, `ruff check .` 0.

- [x] **H2 — `_magnetic_to_h` ignores declared units in `b` mode. FIXED
  (2026-09-30).** Was: `if b_to_h: return x * 1e-9 / MU0` ran *before* the unit
  string was consulted, so a `b`-mode channel whose attrs say **A/m** was still
  scaled by ≈1.257e-3 → impedance wrong by ≈10³, contradicting DESIGN §10.5.
  **Remedy:** `_magnetic_to_h` rewritten as units-first, mode-second — explicit
  H unit (`Ampere per meter`, `A/m`, mA/m, µA/m) → untouched; explicit
  induction unit (T/mT/µT/nT/pT or bare `nt`) → scaled by its own prefix ÷ μ₀;
  only *no usable unit attrs* falls back to the old mode default (‘b’ converts,
  ‘h’ leaves), preserving `test_fluxgate_nt_always_converts` and the
  unitless-legacy contract. Mode key now canonicalized before matching
  (`_canonical_unit` preserves registry casing). DESIGN §10.5 rewritten to
  state the precedence rule. Regression: `test_fluxgate_explicit_am_is_not_converted`,
  `test_fluxgate_tesla_unit_converts_at_its_own_scale`,
  `test_magnetic_to_h_precedence` (`test_magnetic_units.py`, 8 passed).

- [x] **H3 — a failed run reaches the UI with no reason at all. FIXED
  (2026-09-30).** Was: pre-key execution failures emitted a bare
  `run_finished` with no `error` field (`executor.py:603`, `:625`, `:655`);
  the worker then saw a terminal summary and suppressed its own detailed
  event (`app.py:551-562`); and the frontend kept only `event.status`,
  dropping `event.error` even when present (`run.ts:152-154`). Net: red
  **failed** with an empty `.error-msg`, and the reason only survived in
  `GET /api/runs/{id}`'s top-level `error`, which no client reads.
  **Remedy (all three layers):**
  1. `executor.py` puts `error` — and `node` when one node is responsible —
     on **every** failed terminal event, all six sites: cycle
     (`:606-609`), unknown type (`:631-634`), missing-input/coercion
     (`:665-668`), output materialization (`:759`), node execution
     (`:839-842`), cache restore (`:878-881`).
  2. The worker logs the exception before building its summary
     (`app.py:553`, `logger.warning("run %s failed: %s", …)`), so the
     reason exists even for a run whose event stream a client never saw.
  3. `run.ts` keeps `runError` — set from `event.error` on
     `run_finished: failed`, cleared by `setRunId`/`reset`
     (`run.ts:158-166`) — and the toolbar renders it as
     `message ?? runError` inside `header .error-msg` (`App.tsx:513-515`),
     so the badge can no longer turn red silently.
  DESIGN §7.2 now states the contract ("failure reasons ride the terminal
  event"; a `failed` status with no reason is a contract violation).
  **Tests (the missing middle):** 4 parametrized
  `test_run_finished_carries_the_failure_reason[cycle|unknown_type|
  param_coercion|node_error]` (the pre-key paths that never reached a node,
  previously untested) plus
  `test_failed_run_reason_reaches_the_api_and_the_worker_log` (the reason
  survives `GET /api/runs/{id}` *and* the worker log). All 5 fail on the
  pre-fix tree; e2e asserts the toolbar copy in
  `pipeline.spec.ts` "a node error is shown as copyable text…".
  **Verification:** pytest **643 passed**, `ruff check .` 0, Playwright
  **29 passed**.

- [x] **H4 — "Delete selected" can delete a node the user is not looking at.
  FIXED (2026-09-30).** Was: `deleteSelected` unioned React Flow's selection
  with `selectedNodeId` (`workflow.ts:310-312`), but `selectedNodeId` is only
  cleared by pane-click, deletion, new/load (`workflow.ts:318,419,427`) —
  never when the RF selection changes. Click node A (panel shows A) →
  Shift-box-select B → toolbar **Delete selected** (`App.tsx:521`, enabled by
  either flag) removed A *and* B, with **no undo anywhere in the app**.
  **Remedy (the label is the contract):** `deleteSelected` now removes the
  visible RF selection only — `n.selected` flags, no `selectedNodeId` union
  (`workflow.ts:306-322`) — and the button's `disabled` is computed from the
  same flags (`App.tsx:519-527`), so the button is enabled by exactly what it
  will delete. The properties panel stays a *display* focus (it never
  silently arms a delete). DESIGN §8.2's Selection bullet now says so
  explicitly.
  **Test (the missing middle):** `interactions.spec.ts` "Delete selected
  removes exactly the highlighted nodes" — click A, Shift-box-select B,
  delete → A survives, B is gone, and the panel still shows A. Fails on
  pre-fix code at the "Load CSV survives" assertion (both nodes deleted).
  **Verification:** pytest **643 passed**, `ruff check .` 0,
  `npm run build` clean, Playwright **29 passed**.

- [x] **H5 — the off-diagonal phase panel silently drops real data. FIXED
  (2026-09-30).** Was: `_fold_phase_deg` folded only the negative side
  (`p<0 → p+180`) while `set_ylim(0, 90)` clipped everything >90° with no
  warning — a convention-flipped import could render a completely blank phase
  panel, and the repo's own fixture already hid its XY series (`Zxy = 3 − 1j`
  → 161.6°, invisible in every generated figure). **Remedy:** the fold is now a
  complete `np.mod(p, 180)` in both directions; values no fold can place in
  (0°,90°) trigger **one loud warning per figure** naming the component, the
  clipped point count and their value range (`plotting.py:367-376`, logger
  `emptyui`); the dead `phase_all` accumulator was deleted and the module
  docstring updated. The axis deliberately stays 0–90° (USArray/aurora
  parity) — the visibility guarantee is carried by the warning, not an
  auto-expanding ylim; raw values remain in the View TF table. DESIGN §10.6
  now states this. Regression: `test_fold_phase_deg_is_a_mod180`,
  `test_in_quadrant_phase_draws_every_point`,
  `test_out_of_quadrant_phase_is_reported_not_silent`
  (`test_plot_style.py`, 20 passed).

- [x] **H6 — a catalog fetch failure produces a silent white app. FIXED
  (2026-09-30).** Was: `useQuery` for `GET /api/nodes` had no error branch
  (`App.tsx:78-85`), the palette rendered `catalog?.nodes ?? []`, and
  `main.tsx:8-14` had **no React error boundary**. Backend down or a 500 at
  boot = white canvas, empty palette, disabled Run, zero message; any
  render throw white-screened the whole app.
  **Remedy (nothing fails silently):**
  1. `App.tsx:85-100` reads the catalog query's status instead of
     discarding it — `catalogPending` / `catalogError` feed the palette and
     the toolbar's `.error-msg` slot (`message ?? runError ?? catalogError`,
     `:525`), so the boot failure is named where the user is looking.
  2. `NodePalette` replaces the ambiguous empty list with the two states it
     used to hide: `"Loading node catalog…"` while the request is in flight
     (`:68`) and an error card — "Node catalog unavailable" + the reason +
     **Retry catalog** → `refetch()` (`:71-77`). An empty palette now always
     means "the catalog says so".
  3. `main.tsx:26-29` wraps the whole tree in `AppErrorBoundary`
     (`AppErrorBoundary.tsx:22`) — React only recovers from a render throw
     if a boundary sits *above* it. The fallback is a `role="alert"` crash
     card (message, collapsed stack, **Copy details**,
     **Reload emptyUI**; reload is a clean workflow because nothing is
     kept in the browser). The e2e fault switch `?fault=render` lives in
     `FaultInjector` behind `import.meta.env.DEV`, so it is compiled out of
     production bundles (verified: absent from `dist/` after `npm run
     build`, while the card's own copy is present).
  DESIGN §8.1 records both contracts (palette states + the boundary and
  its fault switch).
  **Tests (the missing middle):** `error_states.spec.ts` —
  "a failed catalog fetch surfaces in palette and toolbar; Retry
  recovers" (route → 500, both surfaces carry `GET /nodes: 500`,
  un-route, Retry repopulates the palette and clears both messages) and
  "a render failure lands on the crash card, not a white screen"
  (`?fault=render`, then a fault-free reload boots). Both fail on the
  pre-fix tree.
  **Verification:** pytest **643 passed**, `ruff check .` 0, `npm run build`
  clean, Playwright **31 passed**.

---

## Medium — correctness risk, robustness, gate gaps

- [x] **M27 — `MaskTS`'s `hampel_threshold` default disagreed between the
  manifest (8.0) and `execute` (3.0) — L7 recurrence. FIXED
  (2026-10-26).** (Found 2026-10-25.) `qc.py:185-186` declares
  `default=8.0` with a long rationale ("h=3 masks ~5% of samples and
  collapses the FC band coverage"); `qc.py:491` defaulted the method
  argument to `3.0`. The executor resolves the manifest default, so graph
  runs used 8.0; every direct `MaskTS().execute(...)` in
  `test_mask_ts.py` (10 call sites) used 3.0. Tests therefore exercised a
  different gate than production, and a headless/embedding caller that
  omitted the kwarg silently got the over-gating value the manifest
  explicitly rejects.
  **Remedy (done 2026-10-26):** the method default is now `8.0`
  (`qc.py:491`), matching the manifest — the same manual-alignment
  pattern L7 used for `print_params`. No algorithm change.
  **Tests:** `test_mask_ts_default_threshold_is_the_manifest_value` —
  omitting the kwarg gates identically to `hampel_threshold=8.0`, and
  the 3.0 run flags strictly more on clean Gaussian noise. Red first:
  default == 3.0 (11/512 flagged) vs 8.0 (0/512).
  **Verification:** pytest **809 passed** (~71 s), `ruff check .` 0.
  **Follow-up (open, not this fix):** the user reports MaskTS "wasn't
  performing very well in real-world scenarios" (2026-10-26). This entry
  only aligns the default; the field gating quality (false positives on
  clean natural-source data, missed spikes, FC band coverage) needs a
  corpus-driven review before any criterion/threshold change. Do not read
  this fix as a quality endorsement.

- [ ] **M28 — `Load Phoenix MTU-5` holds the whole recording in memory
  (`blocks = list(...)`) and the batch decoder is unbounded in bytes.**
  (Found 2026-10-25; re-measured 2026-10-27 on the real 1690618B corpus.)
  `phoenix_mtu.py:625` wraps the streaming `_iter_blocks(path, warnings)`
  in `list(...)`, so the entire file is resident before the MTH5 write
  starts; `_DECODE_BATCH_BLOCKS = 16384` (`:81`) bounds the decode batch
  by block COUNT, not bytes.
  **Measured geometry (real corpus, 84.4 h):** TS3 2400 Hz = 1,012 blocks
  (506 slots × 2 s, 17 min of data); TS4 150 Hz = 8,096 blocks (506 × 16
  s); TS5 15 Hz = 303,742 blocks (one 84.4 h continuous run). The 2400 Hz
  data is slot-mode, so the original "12 h at 2400 Hz ≈ 3.3 GB" premise
  was wrong; the 303k-block file is the 15 Hz stream, not the 2400 Hz
  file (the `_iter_blocks` docstring at `:94` says otherwise — fix it
  with the code).
  **Measured peak RSS (pre-fix, delta above a ~205 MB interpreter
  baseline):**
  `list(_iter_blocks(TS5))` alone **419 MB** (182 MB data + 221 MB array
  objects + 15 MB tags + 17 MB tuples); the 2400 Hz file alone **399 MB**
  (batch buffers, not the list); 15 Hz-only ingest **823 MB**; a 12 h
  window **903 MB** (the full 84 h files are materialized before the
  window filter, so cropping does not bound the peak); full corpus
  (1,013 runs) **1,548 MB**. A synthetic 24 kHz file with 256 blocks
  peaks at **973 MB** and extrapolates linearly to **~3.8 GB** at 1,012
  blocks — the GB-scale case is the 24 kHz AMT batch decoder, not the
  `list`.
  **Impact:** Medium — the 15 Hz continuous stream scales linearly with
  duration (419 MB per 84 h for the list, plus the `runs` accumulation);
  the 24 kHz batch can reach ~3.8 GB for a 1,012-block file. Both are
  crashes, not silent, but they arrive after a long read.
  **Plan (2026-10-27, user):**
  1. **A — byte-bounded batch decode. DONE 2026-10-27.** A 64 MiB
     transient budget (`_DECODE_BATCH_BYTES`) caps the batch by bytes as
     well as the 16,384-block count (`_decode_batch_blocks`): 29 blocks
     at 24 kHz, 294 at 2400 Hz, 4,698 at 150 Hz, 16,384 at 15 Hz (block
     cap). Decode time is unchanged within noise (TS3 -0.6%, 24 kHz
     -4.2%, TS5 -4.1% with the batch unchanged); peak RSS TS3 399→243 MB,
     24 kHz 973→359 MB at 256 blocks (extrapolates to ~1.0-1.1 GB at
     1,012 blocks), TS5 unchanged (~454 MB, list-dominated). Test:
     `test_decode_batch_is_byte_bounded`.
  2. **B — single-pass streaming. DONE 2026-10-27.** The window filter
     and quality aggregation now run as the blocks are decoded, and
     `_group_segments` consumes the same stream; the segments hold only
     the in-window chunks. Measured: 12 h window 903→377 MB (-58%),
     15 Hz-only 823→719 MB (-13%), full corpus 1,548→1,512 MB (-2.3%);
     ingest time 6.3→5.9 s / 5.1→4.8 s / 30.1→29.5 s. The existing
     corpus round-trip and window-crop tests stay green.
  3. **C — stream the MTH5 write per run. PARKED.** The full corpus is
     still ~1.5 GB, dominated by the `runs` accumulation (all 1,013
     runs held until the write). Do it only if 84 h corpora matter for
     the deployment; the 12 h case is now 377 MB.
  **Effort:** A was XS (done); B was M (done); C is M (parked).
  **Test:** B kept the existing corpus round-trip and window-crop tests
  green; C would need a streaming-write spy.

- [x] **M29 — an empty `save_path` wrote the MTH5 next to the input data
  (`phoenix_mtu.py:775`, `lemi.py:61,71`). FIXED (2026-10-27).** (Found
  2026-10-25.) Both loaders did `Path(save_path) if save_path else <data
  dir>`; LEMI passed `save_path=None` to the client and then re-derived
  `client.save_path`, which the client resolves into the data directory.
  In remote mode `PathPolicy._resolve_output` maps an empty
  `path_kind="out"` param to the run workspace, so the contract held
  there — but a local/headless run without a workspace wrote outputs into
  the read-only-by-intent input tree, and a re-run could overwrite a
  source archive.
  **Decision (2026-10-27, user):** option B — a dedicated subfolder, not
  the workspace (TTL-reclaimable; the MTH5 is a persistent product) and
  not a hard error (the casual empty-path use stays).
  **Remedy (done 2026-10-27):**
  1. New `_paths.resolve_output_dir(save_path, data_path)`
     (`_paths.py:18-30`): explicit `save_path` wins; empty resolves to
     `<data_path>/emptyui_output`, created with parents. A path that
     exists as a file raises `FileExistsError`.
  2. Phoenix uses it (`phoenix_mtu.py:780`) and logs the target
     (`:783`); LEMI passes the resolved directory to `LEMI424Client`
     (`lemi.py:66`) — never `None`, which the client would turn back
     into `data_path` — and logs the target (`:77`).
  3. The shipped LEMI example sets `save_path: "examples/output"` on both
     nodes (gitignored, `.gitignore:33`), so the example stops writing
     into the data tree.
  **Tests:** `test_output_paths.py` (empty → subfolder, explicit wins,
  file-path raises); `test_read_phoenix_mtu_default_output_folder`
  (empty → `<data>/emptyui_output`, no MTH5 beside the raw files);
  `test_load_lemi_default_output_folder` (the client receives the
  subfolder, not `None`; no MTH5 beside the raw data). Red first: the
  LEMI client saw `None` and wrote `lemi_data/from_lemi424.mth5`; Phoenix
  wrote into the data root.
  **Verification:** pytest **814 passed** (~70 s), `ruff check .` 0.

- [x] **M30 — `merge_tf_levels` built a 1-D `Z_err` when a level omitted
  it, crashing TF assembly with an opaque `IndexError`. FIXED
  (2026-10-26).** (Found 2026-10-25.) `_processing.py:3074-3077`:
  `err = np.zeros(len(Z))` was 1-D, so each `e` was a scalar and
  `Z_err = np.stack(...)` (`:3106`) had shape `(n_periods,)` instead of
  `(n_periods, n_out, n_in)`. `tf_from_estimates` then indexes
  `Z_err[:, ri, ci]` (`:2899`) → `IndexError: too many indices`. The
  comment says "treat as zeros", so the intent is a zero-filled array of
  `Z`'s shape.
  **Remedy (done 2026-10-26):** `np.zeros_like(Z, dtype=float)` for the
  missing-error branch and `np.full(Z.shape, float(err))` for the scalar
  branch (`_processing.py:3079,3083`), so both broadcast to `Z`'s shape.
  **Tests:** `test_merge_tf_levels_without_z_err_merges_as_zeros`
  (`test_aurora_errors.py`) now asserts `Z_err.shape == Z.shape` for the
  missing and scalar cases (it previously checked values only, so the 1-D
  output slipped through). Red first: `(2,) != (2, 2, 2)`.
  **Verification:** pytest **808 passed** (~71 s), `ruff check .` 0.

- [x] **M31 — the plugin guide documented a model-integrity control that
  does not exist (`EMPTYUI_MODELS_PATH`, mandatory `sha256`). CLOSED
  (2026-10-27): marked planned.** (Found 2026-10-25.)
  `docs/plugin-guide.md:185-215` said weights live in a configured
  `EMPTYUI_MODELS_PATH` and that `sha256` is **mandatory** — "results
  must be reproducible and tamper-evident" — and the checklist (`:262`)
  repeated it. Nothing reads `EMPTYUI_MODELS_PATH`; `models` is a
  declared `ClassVar` with a docstring (`api.py:232-236`) and no
  resolution or hash verification anywhere in `packages/`. A plugin
  author following the guide shipped a manifest that was silently
  unenforced.
  **Decision (2026-10-27, user):** mark it **planned**, not implement —
  there is no ML plugin consumer yet, and a resolver with no user would
  be speculative (the M23 precedent for `X-Forwarded-*`).
  **Remedy (done 2026-10-27):**
  1. `docs/plugin-guide.md` §8 now states the resolver is **planned, not
     enforced** (no `EMPTYUI_MODELS_PATH` read, no `models` resolution,
     no `sha256` verification); the fields are the forward-compatible
     contract, and `execute` owns the missing-model error until the core
     resolver lands. The checklist item carries the same note.
  2. `docs/DESIGN.md` §11.4 records the planned contract (env dir + hash
     verification) as **not implemented**.
  3. `api.py`'s `models` docstring says resolution/verification is
     planned.
  **Tests:** none needed — docs/contract wording only; the code path is
  unchanged (there is no resolver to test).
  **Verification:** `ruff check .` 0 (docs + docstring only); full pytest
  **814 passed** unchanged.

- [x] **M24 — Align Runs cannot be wired to Extract Run's collection
  output (burst-mode RR route blocked in the UI). FIXED (2026-10-13).**
  (User report, 2026-10-13.) Extract Run's 'time range' mode (the burst
  corpus route) emits a `ts.runs` payload; the frontend's wiring check
  (`workflow.ts:147-150`) refuses a ts.runs payload into an input without
  `collection_ok` (DESIGN §4 one-port convention) — and Align Runs'
  `data1`/`data2` lacked the flag, although the BACKEND has aligned
  collections member-wise for a while (`_align_collections`,
  `test_align_runs_collections_*`). Compute FC (flag present) worked —
  hence the impression that "only Align Runs refuses". Merge Runs takes
  `mth5` (unaffected); Split/Merge Channels correctly lack the flag
  (singular semantics, loud reject).
  **Remedy (done 2026-10-13):** `collection_ok=True` on both Align Runs
  inputs (manifest-only — the wiring check reads the catalog, zero
  frontend changes).
  **Tests:** `test_runs.py::test_align_runs_declares_collection_ok_inputs`
  (red pre-fix); the backend collection behavior was already pinned by
  the 7 `test_align_runs_collections_*` tests.
  **Verification:** pytest **715 passed**, `ruff check .` 0.

- [x] **M1 — no server-side validation of param *values* (DESIGN §6 drift).
  FIXED (2026-10-12).** `validate_workflow` checked structure only
  (`schema.py:56-181`) and the executor coerced types but never
  ranges/options/`optional` (`executor.py:491-537`). Verified today: a
  workflow whose `core.filter.bandpass` node carries `{"order": -5}`
  (declared `min=1, max=10`) passed `validate_workflow` with **no errors**
  and would POST a 200. Widgets guard the UI, but §11 makes the API the
  trust boundary in remote mode.
  **Remedy (done 2026-10-12):**
  1. `emptyui/params.py` — the executor's `_coerce_param` moved verbatim to
     a shared `coerce_param(param, value)` (one copy, per the M5
     discipline); `Executor._coerce_param` delegates.
  2. `schema.py` gains `_param_value_problems`, called per node in the
     loop: every manifest param present in the document is coerced with
     the **same** function (validation and execution can never disagree on
     what a value means), then checked against the declared
     `min`/`max` (number/integer), `options` (select/radio) and
     `optional` (a non-optional default-less param absent — or explicit
     null, or an empty file/directory widget — is "not provided").
     Fractional values on integer widgets are rejected. Messages follow
     the existing `node '<id>' (<type>): param '<name>' …` shape.
  3. Documented policies preserved: manifest defaults are not re-checked
     (plugin-declared, not client-supplied); stale params a node no
     longer declares stay tolerated (§24.9, executor warns); option
     checks skip `suggest_from` params (runtime-extended option lists)
     and inherit targets (upstream-fed comboboxes). `enabled_when` stays
     rendering-only.
  4. `POST /api/runs` now 422s on the issue's exact probe
     (`core.filter.bandpass` with `order: -5`), naming the param.
  **Tests:** `packages/emptyui/tests/test_param_validation.py` (18) — 10
  rejection cases RED on the pre-fix tree (including the issue's
  `order=-5` probe), 8 coercion-parity/policy guards GREEN before and
  after (string-carried numbers, numeric select options, boundary values,
  stale-param tolerance, aggregation). Plus
  `test_errors.py::test_out_of_range_param_value_returns_422` (the API
  surface). All four shipped example workflows validate clean against the
  real catalog (no false rejections).
  **Verification:** pytest **686 passed**, `ruff check .` 0, Playwright
  e2e suite green (no frontend change; run as the server behavior
  changed).

- [x] **M2 — no run timeout/watchdog: one hung node wedges the instance.
  FIXED (2026-10-13).** Cancellation was polled only between nodes
  (`executor.py:667`, `:899`) and there was no timeout anywhere in the
  package; the admission slot was released only in the worker's `finally`
  (`app.py:563-572`) and TTL cleanup refused non-`worker_done` runs
  (`app.py:243-255`). With the UI-editable `max_concurrent_runs=1` a single
  stuck node returned 429 forever and its workspace never expired.
  **Decision (user, 2026-10-13): a default deadline, editable in Settings,
  enforced per node** (600 s — generous, and heavy nodes override upward).
  **Remedy (done 2026-10-13):**
  1. **Knob** `node_timeout_seconds` (default 600, `0` = off) in
     `LIVE_KNOBS`/`RuntimeSettings`; env `EMPTYUI_NODE_TIMEOUT`;
     `create_app(node_timeout_seconds=...)`; env/kwarg values baseline the
     knob (PUT → 409). Applied live — the next worker call reads the
     holder. Settings dialog renders it generically from the view.
  2. **Per-node budget**: every node gets the full budget, reset at node
     boundaries — a long pipeline of quick nodes is never killed; only a
     single stuck node is. `Executor.run(node_timeout_seconds=...)` (None
     = off, the headless default — the exact pre-watchdog behavior).
  3. **Per-node override**: `Node.timeout` ClassVar recorded in the
     manifest (contract-pinned) — `None` inherits the knob, `0` opts out,
     `> 0` is the node's own budget. The heavy raw-data/FC nodes (the
     mth5 loaders/extract/align/merge/calibrate/fc-write, ComputeFC,
     BuildFC, EstimateTF, the sorts, MergeFC, LoadCsv, PlotTimeSeries)
     declare 3600 s so genuine long computations survive the default.
  4. **Mechanism**: the node runs on a **daemon thread**
     (`_execute_with_deadline`); on expiry it is *abandoned* — Python
     cannot kill threads, so the thread dies with the process and never
     blocks interpreter exit (a `ThreadPoolExecutor` was rejected for
     exactly that reason) — and the run ends with a **`timeout`
     terminal status** whose `error` names the node, the budget and the
     remedy (H3 contract). The admission slot is released and TTL cleanup
     accepts the run; payloads are swept like any failed run. A
     `ThreadPoolExecutor`-per-node was also rejected (it would re-raise
     correctly but join at interpreter exit). `ctx.abandoned` voids the
     abandoned thread's late `progress`/`publish_artifact` calls.
  5. **Frontend**: `run.ts` treats `timeout` as a reason-carrying terminal
     status (toolbar shows it, H3-style); the poll loop stops on it; the
     badge gets an amber style.
  **Tests:** `packages/emptyui/tests/test_watchdog.py` (10) — hung node →
  `timeout` summary/event naming the node, downstream never starts, the
  run ends promptly; node-declared opt-out and override; the
  unset-knob inline behavior pinned; abandoned-ctx unit; the M2 wedge
  end-to-end (max_concurrent_runs=1: the run after a timed-out one is
  admitted, not 429); knob default/edit/0/null/validation and the env
  409. `test_api_contract.py` pins the `timeout` manifest key (red
  pre-fix). All 10 watchdog tests red before the fix (missing kwarg /
  missing knob / missing status).
  **Verification:** pytest **695 passed**, `ruff check .` 0, `npm run
  check` clean (tsc + eslint + vitest 17), Playwright **31 passed**.

- [x] **M3 — the collection route leaks a write-open workspace HDF5. FIXED (2026-10-10).**
  `_execute_collection` opens the cached archive at `process.py:405` and
  closes at `process.py:426` with **no try/finally**; any exception in
  `persist_fc_level` / `merge_fc_containers` / the level loop leaks the
  handle — exactly the stale write-flag condition
  `_compat.ensure_h5_openable` exists to repair. `ComputeFC.execute`
  (`process.py:300-315`) and `BuildFC` (`process.py:857,1046-1048`) are
  correct; this one route is not. → try/finally + a leak regression test.
  Fix: the close moves to a `finally` around the collection call in
  `ComputeFC.execute` (the single-run path's pattern), and `_execute_collection`
  no longer closes on success, so the handle has one owner. Test:
  `test_compute_fc_collection_leak.py` makes `persist_fc_level` raise and checks
  the handle is gone and not write-open; red before the fix, green after.

- [x] **M4 — a cache hit hands downstream a path into the *producer* run's
  workspace. FIXED (2026-10-13).** The executor copied the artifact file into
  the new workspace (`executor.py:710`) but re-registered the **original**
  output dict (`executor.py:693` `store.put(port.type, entry.outputs.get(...))`),
  so consumers that read `value["path"]` (e.g. Save Plot) resolved run A's
  file while the event advertised run B's (`executor.py:724`). If TTL
  reclaimed A mid-run (`app.py:256-268`) the read failed although a fresh
  copy existed.
  **Remedy (done 2026-10-13):**
  1. The restore block copies the artifact files FIRST, collecting
     `{original_src → new_dest}`.
  2. `_relocated_output(value, dest_by_src)` — any output value that is a
     dict whose `path` matches a copied artifact's src is re-registered as
     a **shallow copy** with the path rewritten to this run's file. The
     match is by normalized path, not the recorded `port` field
     (`_record_artifacts` records only the FIRST port an artifact dict was
     returned under — aliased outputs would slip through). The cache
     entry's dict is never mutated (the §7.1 immutability contract).
  3. The `artifact` event path and the stored value path now derive from
     the same map — consistent by construction. Headless restores (no
     workspace, no copy) keep the cache path; same-workspace restores are
     a no-op (`samefile`).
  4. Drive-by: `_artifact_dest` coerces `Path(workspace)` — it assumed a
     `Path` while `run()` also accepts `str` (the server always passes
     `Path`, so it never bit; the new tests pass a str).
  5. Test-fixture note: deleting the producer's workspace *between* runs
     is NOT a discriminating scenario — the vanished-src guard
     (`executor.py:742`) re-executes the producer. The real hazard window
     is after the restore, before the consumer runs.
  **Tests:** `packages/emptyui/tests/test_cache_artifact_path.py` (4) —
  the store value points at this run's copy; the event path equals the
  stored value path; the cache entry's dict is not mutated (guard, green
  before and after); the honest end-to-end — run 2's consumer re-executes
  (changed param → producer hit, consumer miss) and reclaims run A's
  workspace BEFORE reading (the deterministic mid-run TTL simulation):
  must complete reading run B's copy, `FileNotFoundError` on the pre-fix
  tree. All 4 red before the fix.
  **Verification:** pytest **699 passed**, `ruff check .` 0.

- [x] **M5 — triplicated/mirrored knowledge, now with concrete instances.
  CONSOLIDATED (2026-10-14, all eight groups; one issue-commit — each
  group's diff was kept separable, deviating from the one-group-per-commit
  plan note for review economy).**
  1. EstimateTF's inline fc classification -> `_fc_io_channels` (the
     estimator can no longer disagree with QC about inputs).
  2. The decimation-trim formula -> `_family_decimation` (both call sites
     now share the Build FC rule).
  3. The three identity-attr builders -> canonical `FC_IDENTITY_KEYS` in
     emptyui_mth5.fc; `_identity_attrs` and `_station_identity_attrs` key
     off it (the mth5 hdf5 builder references it).
  4. plot_impedance_response's twin rho/phase/ok loops -> one
     `_panel_curve` computed once (limits and plotted data cannot
     disagree).
  5. The placeholder-epoch literal -> executor's named
     `UNKNOWN_EPOCH_STARTS` mirror, pinned to core-nodes' canonical by
     `test_executor_epoch_mirror_matches_core_nodes`.
  6. The nomenclature mirror is BY DESIGN (core-nodes is the single
     implementation; mth5 re-exports) — the residual hazard, the
     plugin-free `_FALLBACK_MAPS`, is pinned to mt_metadata's canonical
     by `test_fallback_maps_match_mt_metadata`.
  7. The two `apply_style` copies -> core's `_plotstyle` (mth5's had
     already drifted — the refined EMTF-reference grid now lives in the
     ONE copy and every figure family gets it).
  8. The frontend PORT_COLORS mirror -> pinned to registry.py's
     CORE_PORT_TYPES by a vitest that reads the Python source.
  Gates: pytest 739, vitest 33, Playwright 34, ruff clean.
  *(legacy E3)*. Candidates for shared constants/dedup, each a drift hazard:
  the fc in/out-channel classification is implemented inline in
  `EstimateTF.execute` (`process.py:1789-1799`) **and** as the helper
  `_fc_io_channels` (`process.py:1067-1082`) used by the three QC nodes —
  a divergence makes the estimator disagree with QC about which channels
  are inputs; the decimation-trim formula exists twice
  (`process.py:377-382` vs `:961-966`); three identity-attr builders
  (`process.py:54`, `_processing.py:2899`, `mth5/fc.py:106`);
  `plot_impedance_response` computes rho/phase/ok in two near-identical
  loops (`plotting.py:279-300` axis fit vs `:316-343` the draw) that must
  stay in sync or the limits and the plotted data disagree;
  placeholder-epoch lists live in
  `types.py`, `_timewindow.py`, `executor.py:178`; nomenclature mirror
  `core/_channels.py` vs `mth5/nomenclature.py`; two `apply_style` copies;
  frontend `PORT_COLORS` mirrors `registry.py:21-35`.

- [x] **M6 — FC defaults and a dead duplicate diverge from the shipped
  recipe.** Two halves:
  (a) `compute_fc` (`_processing.py:242-408`) has **zero production
  callers** (every call site is a test) yet reimplements the STFT→FC path
  with *different* semantics — window length in **seconds** vs samples,
  no saturation QC, no decimation cascade — so ~17 test modules exercise a
  code path the product never runs;
  (b) the shipped entry `compute_fc_levels` itself defaults to
  `window_type="hamming"`, `prewhitening=""` (`_processing.py:418,423`)
  while the node surface defaults to boxcar + first-difference PW
  (`process.py:229,232`); runs are correct only because the node always
  passes both — any other direct caller silently gets the legacy recipe
  (§10.1 calls non-PW boxcar "catastrophic"). → align the library defaults
  with §10.1, and delete or delegate `compute_fc` so the fixtures test the
  shipped path.
  Status (2026-10-10): part (b) FIXED — `compute_fc_levels` defaults are now
  boxcar + first-difference prewhitening (DESIGN §10.1); `test_fc_defaults.py`
  is red with the old defaults and green after. Four calibration/saturation
  tests and the M8 guard test pin or tolerate the designed DC NaN. Part (a) is
  FIXED (2026-10-11): `compute_fc` is deleted. Its 80 call sites moved to the
  test helper `compute_fc_seconds` (aurora `tests/_synthetic.py`, mth5
  `tests/_fc.py`), which delegates to `compute_fc_levels` with the same legacy
  recipe (no decimation, prewhitening off). The retired function's guards are
  kept: prewhitening is validated up front in `compute_fc_levels`, and a record
  shorter than one window raises in the helper (`compute_fc_levels` returns an
  empty dict there).
  Note (found 2026-10-11, not fixed): `compute_fc_levels` returns `{}` silently
  when a record is shorter than one window (the level loop logs a warning and
  `break`s, `_processing.py` near the `n_usable < nperseg` check). The retired
  `compute_fc` raised. Production callers (`ComputeFC`, `BuildFC`) need a check,
  since an empty result may propagate with no error. Open: raise or check at the
  caller, with a test.
  Follow-up (2026-10-11, `cb77d6d`): the fixtures run the shipped prewhitening
  default; the no-PW baselines are pinned explicitly. Window moved to boxcar (`d98ffbe`); the two deliberate hamming lines stay.
  (the taper-override and legacy-behaviour tests keep hamming on purpose).

- [x] **M7 — the inter-station coherence diagnostic is a silent no-op above
  64 Hz. FIXED (2026-10-10).** `nperseg = min(4096, …)` (`runs.py:1415`) bounds the resolution to
  `df ≥ fs/4096`, while the band filter needs a bin at `f ≤ 1/64 Hz`
  (`runs.py:1419`) → for `fs > 64 Hz` (the repo's own Phoenix **150 Hz and
  2400 Hz** families) `band.any()` is always false and the function returns
  silently (`runs.py:1420-1421`). Early-outs `n_common < 256` and missing
  hx/hy (`runs.py:1407-1412`) are equally silent, and all constants (`256`,
  `4096`, `1/1024`, `1/64`, `0.5`, coverage `0.9`) are hardcoded. Tests only
  use sr = 1 and 8 Hz. → express the band as seconds, not Hz-vs-nperseg; log
  every skip; test sr=150.
  Fix: the band is in seconds (64-1024 s), the series is decimated to a 1 Hz
  analysis rate, segments are 2048 s, and the minimum overlap is 1024 s. Every
  skip is logged at info. Tests in `test_coherence_diagnostic.py` (sr 1, 8, 150;
  weak coherence; short-overlap and missing-channel skips): four red before the
  fix, green after. Note: segments are now 2048 s at every rate (before: 4096
  samples, i.e. 4096 s at 1 Hz and 512 s at 8 Hz), so the reported median can
  differ from the old run at those rates.

- [x] **M8 — chained time-domain `Calibrate` double-corrects, unguarded and
  untested. FIXED (2026-10-10).** DESIGN §9.2 explicitly warns "do not chain
  time-domain Calibrates" because upstream `RunTS.calibrate` removes *every*
  chain filter regardless of the `applied` flag. Probe: one pass halves the
  amplitude correctly (flags go `False`); a second pass halves it again
  (ratio 0.5) and flips the flags back to `True`, so a later Compute FC would
  divide a third time. Fix: `_refuse_if_calibrated` in `calibrate.py` refuses
  any payload with an `applied=False` filter (single path and per collection
  member), naming the channels and the remedy. Tests in
  `test_calibrate_chain.py`: two refusal tests (red before the fix) and a
  `Calibrate → Compute FC` guard that matches the FC route (green before and
  after).

- [x] **M9 — the clock-zero grid labels windows the data does not start on (silent phase error in remote reference). FIXED (2026-10-10).**
  `_clock_zero_head_skip` (`_processing.py:928-932`) rounds the sample skip
  (`n_skip = round((k0*step - dt)*sr)`) but returns the unrounded
  `k0*step - dt` as the time offset. Labels therefore sit exactly on the
  `t0 + k·step` grid, while the data starts up to half a sample away from it
  whenever the record start is off the sample grid (`dt·sr` non-integer).
  Two stations with different sub-sample offsets then differ by up to one
  sample. At a band's upper edge (`f = Fs^(l)/4`) the RME_RR phase error is
  `2π·f·δt`, up to about π/2 in the worst case. The default route is reached:
  Build FC's `clock_zero` defaults to `epoch` (`process.py:567`). Silent:
  pairing matches the labels, so no check fires. Conditional on an off-grid
  record start; the real-archive magnitude is not measured.
  **Remedy (done 2026-10-10):** `_clock_zero_head_skip` returns the rounded
  skip as the offset, so labels are the actual sample times (within half a
  sample of the grid). `_pair_remote_fc` floors the pairing tolerance at one
  sample of the level (`_one_sample_s`), so sub-sample-offset records still
  pair 1:1; strict mode and identical grids are unchanged. Manifest
  descriptions and DESIGN §10.1 updated. Behaviour change: records that
  previously failed to pair because of a sub-sample offset now pair within one
  sample. Tests: `test_remote_reference_numerics.py` (5 M9 tests; the label and
  floor tests fail on the pre-fix tree) and `test_pair_order.py` (M11, below).
  M11 tests fail on the pre-fix tree). Full suite 649 passed; ruff clean.
  Test: (1) synthetic 1 Hz record starting at a grid point + 0.4 s: window 0's
  label must equal the actual time of its first sample (fails pre-fix, Δ ≈
  0.4–0.6 s). (2) two noise-free synthetic stations with sub-sample offsets 0
  and 0.5 sample: the paired windows must be the same physical windows, and
  the recovered Z phase must match the offset-free run within 1° (fails
  pre-fix). (3) integer-offset inputs: FC output bit-identical to pre-fix.

- [x] **M10 — the remote-gap safety check compares window centres with gap intervals (silent miss at deep levels). FIXED (2026-10-10).**
  `_warn_remote_gaps` (`_processing.py:1286`) tests the raw fc `time`
  coordinate, which is the scipy segment centre (`process.py:1049` documents
  this). BuildFC shifts back by half a window before attributing windows
  (`process.py:1049-1060`); the M4 #2 gap check (DESIGN milestone, not this
  register) does not. The offset is `N/(2·Fs^(l))`: about 2 min at level 0
  (1 Hz, N = 256) and about 34 min two decimations deep. A window whose start
  lies in an interpolated gap but whose centre lies after it is not flagged,
  which is the silent degradation M4 #2 exists to catch. Conditional on Merge
  Runs gap provenance (opt-in route).
  **Remedy (done 2026-10-10):** `_warn_remote_gaps` shifts the fc centre labels
  back by `N/(2·fs)` (`_centre_to_start_s`) before the membership test, so a
  window is flagged by its start. If the fc does not record
  `num_samples_window` and `sample_rate`, no shift is applied. Tests:
  `test_remote_gap_centres.py` (four M10 tests; the two start-only tests
  fail on the pre-fix tree, the guard and centre-regression tests pass on both).
  The existing remote-gap tests in `test_remote_reference.py` pass unchanged.
  Test: (1) a ref fc whose `merged_gap_intervals` ends between a window's start
  and its centre: the warning must fire (fails pre-fix). (2) a level-2
  synthetic fc: the flagged set must equal the set computed from window starts.
  (3) the existing H1 provenance tests still pass.

- [x] **M11 — tolerance pairing returns windows in match-distance order; the coherence ensemble then runs over unrelated windows. FIXED (2026-10-10, prerequisite of M9).**
  The tolerance branch of `_pair_remote_times` (`_processing.py:1146`) sorts
  candidates by |Δt| (`cand.sort()`, `:1178`) and returns the pairs in that
  order (`:1187`). `_pair_remote_fc` applies `isel` in that order, and
  `coherence_sort_fc` (`_processing.py:1656`) builds its sliding ensemble by
  array index (`lo = idx - w0`, `hi = idx + w0 + 1`), so "±16 adjacent
  windows" become arbitrary windows. Estimate TF is order-independent and is
  not affected. Reachable only with `pair_tolerance_s > 0` (default 0). Silent.
  **Remedy (done 2026-10-10):** the tolerance branch returns pairs sorted by
  local index (stable argsort), so the Coherence Sort ensemble is
  chronological. Needed before M9, because the default path now always reaches
  the tolerance branch. Test: `test_tolerance_pairs_come_back_in_local_time_order`.
  Test: (1) `_pair_remote_times` with tol > 0 on inputs whose nearest-match
  order is not chronological: the returned local indices must be strictly
  increasing (fails pre-fix). (2) Coherence Sort on a remote grid shifted by
  0.4 s with tol = 1 s: per-bin coherence must equal the exact-grid result
  (fails pre-fix). (3) Estimate TF output unchanged for the same inputs.

- [x] **M12 — every run blanks all viewers and time bars. FIXED (2026-10-14).**
  Remedy: `setRunId` no longer clears `artifacts`/`payloads`/`nodeStates`
  (only run identity and the failure reason start fresh), and `startRun`'s
  own `reset()` call was REMOVED — it alone would have re-blanked the
  viewers on every run. Wiping now happens on document load / New only
  (`handleLoad`/`handleNew`, which already called `reset()`). A node's own
  stale warnings/progress/error clear when THAT node re-executes
  (`node_status: "running"` — the executor emits no per-node "queued"); a
  cache hit (`done` + `cached`) skips "running" and keeps prior state.
  Untargeted nodes keep their chips after a targeted run — MORE truthful
  than the old wipe (their results genuinely were not recomputed); the
  cached-chip e2e expectation moved 2 -> 3 with the reasoning in a comment.
  During verification the pipeline e2e exposed a **latent L5 regression**
  (the cache size probe raised NameError on RunTS payloads — mt_timeseries'
  `__getattr__` quirk); fixed separately (commit `d3fb2ef`).
  Tests: `run.test.ts::a re-run keeps artifacts and payload extents visible`
  + `...warnings clear when THAT node re-executes` (both red-first);
  full gates: pytest 731, vitest 23, Playwright 31.

- [x] **M13 — the frontend still hardcodes node/category/port literals.
  FIXED (2026-10-14; decision: manifest-declared run-visibility flag).**
  Remedy, all five: (1) a `terminal` ClassVar/manifest field — declared
  `True` on every Display node (12 across 6 plugins), pinned by
  `test_display_nodes_declare_terminal` against the real registry; the
  run-visibility notice tests `outputs.length === 0 || m.terminal`;
  (2) viewer port rows render from `manifest.inputs/outputs` via a single
  generic `ViewerPorts` (the four Handle components are gone) — a viewer
  plugin declaring other port names now renders handles matching its own
  edges; (3) the legacy-edge migration's `mth5.io.extract_run` literal is
  the exported `EXTRACT_RUN_NODE_TYPE`; (4) card sizes are one source
  (`DEFAULT_NODE_SIZE`/`VIEWER_NODE_SIZE` in workflow.ts; App re-centering
  subtracts half the declared default); (5) select suggestions lead with
  the param's manifest default, not a hardcoded "auto". Plugin guide's
  manifest table documents `terminal`.
  Tests: `test_api_contract.py` MANIFEST_KEYS + the terminal pin (red
  first — the key was absent); full gates: pytest 732, vitest 24,
  Playwright 31, ruff/tsc clean.

- [x] **M14 — frontend quality gates are open. CLOSED (2026-10-18): part 1
  fixed `f666b3a` (2026-10-10); part 2 — the CI `web` gate and the deferred
  e2e holes — delivered with the GitHub migration (`bd7a1aa` + the e2e
  commit).** The CI gap is gone: `tests.yml` gained a `web` job parallel to
  pytest/lint — `npm ci`, `npm run check` (tsc -b + eslint + vitest, the
  local contract unchanged) and `npm run build` (vite bundling, which the
  e2e dev-server boots never exercise) — so a type error can no longer
  ship green. The deferred e2e holes landed across the interim deliveries:
  port-type/`accepts`/cycle rejection and one-edge-per-input are pinned by
  unit tests (`workflow.test.ts`) AND e2e (`interactions.spec.ts`); the
  error/notice surfaces (`error_states.spec.ts`), the WS `offline` state
  (`ws_outage.spec.ts`), accessibility (L1's `a11y.spec.ts`) and the
  delete/copy-paste gestures all have dedicated specs. The last two named
  holes closed with their own tests: palette search narrows by title AND
  type tag and restores on clear; the `FileBrowser` dialog browses the
  server filesystem (repo-root listing → select a tracked archive into the
  path param) and Esc-closes without committing (L-F11).
  **Migration note (2026-10-17, pre-push): RESOLVED — no LFS needed.**
  The earlier "~100+ MB" figure is stale: the tracked
  `smoke_phoenix_mtu.mth5` measures **14 MB** since the `8fa54de`
  smoke_2h → smoke rename (the current 3-hour archive), far under
  GitHub's 100 MB hard blob limit. A plain push carries it; the next
  largest tracked blobs are `test/verification/polarization_features.json`
  (5 MB) and `docs/screenshot.png` (0.6 MB). The raw corpora (LEMI 401 MB,
  Phoenix TS*/day_12h 182 MB, NIMS h5 41 MB, the regenerable
  `mtu_clb_*.mth5` 110 MB) are all gitignored as designed — the
  tracked-data list is already minimal. Verified 2026-10-17
  (`git ls-files` sizes + `git count-objects`); example-data
  availability is documented in the README's table.

- [x] **M15 — liveness/recovery gaps in the run channel. FIXED (2026-10-14).**
  Remedy, all four parts: (a) the socket NEVER gives up — after
  `MAX_RETRIES` the badge goes `offline` (the outage stays loud) but a slow
  capped retry (15 s) continues underneath, so the connection self-heals
  when the backend returns; `startRun` also re-arms the socket (the
  documented "rerunning calls ensureSocket()" is now true).
  (b) The poll counts consecutive failures: 5 in a row stops the poll and
  surfaces "Lost contact with the backend — run status may be stale" (a few
  transient blips stay silent). (c) `handleNew` now stops the in-flight
  poll like `handleLoad` (§8 mirror). (d) `node_progress.msg` renders on
  the card under the bar (`.mt-progress-msg`) and in the properties panel —
  §7.2's progress text is live state.
  Tests: `ws.test.ts::the socket keeps retrying after MAX_RETRIES...`
  (red-first: the old code stopped at 6 attempts); full gates: vitest 24,
  Playwright 31, typecheck+lint clean.

- [x] **M16 — backend test blind spots that hide the above. CLOSED
  (2026-10-16, commit `c9d9373` — the last residue, the export-format
  tests, is done; see the Status paragraph for the full timeline).**
  Phase 0 part FIXED (2026-10-10, commit `447c7bb`).
  (a) **WS delivery is effectively untested** — FIXED at `447c7bb`
  (see the Status paragraph; `test_ws_delivers_run_events_under_the_
  lifespan` asserts the full event stream through a real socket).
  (b) ~~Nothing pins the failure-reason contract (H3)~~ — pinned by H3's 5
  tests in `test_errors.py` (2026-09-30); a `run_finished` that fails without
  `error` now fails CI.
  (c) `MANIFEST_KEYS`/`PARAM_KEYS` in `test_api_contract.py:21-33` lag the
  manifest: `cacheable`, `invalidates_inputs` and `suggest_from` are emitted
  (`api.py:154,266-267`) and `suggest_from` is *consumed* (`MTNode.tsx:332`)
  but a rename would not fail the contract.
  (d) `tests/conftest.py:12-16` builds apps from the real environment — a
  developer with `EMPTYUI_DATA_ROOT`/`EMPTYUI_SETTINGS_PATH` exported breaks
  isolation tests; no `monkeypatch.delenv` anywhere.
  (e) `test_server.py:87-89` asserts a status on a **single immediate GET**
  (races the worker; every other test in the file polls `_wait_status`).
  (f) Physics gaps: no end-to-end provenance test (H1), no double-Calibrate
  test (M8), no `b`-mode A/m test (H2), no phase-panel *value* assertion
  (H5), `ExportTF` covers only edi/zmm — `xml`/`avg`/`j` untested, no
  concurrency test (two runs sharing cache entries and one shared `Node`).
  Status (2026-10-10, `447c7bb`): fixed here — (a) WebSocket delivery under the
  lifespan; (c) the contract keys `cacheable`, `invalidates_inputs` and
  `suggest_from`; (d) an autouse fixture that clears `EMPTYUI_*` in
  `packages/emptyui/tests`; (e) `test_run_echo_workflow` waits with
  `_wait_status` (a 0.3 s node reads `running` straight after the POST). Each
  fix has a mutant that the pre-fix test passed and the new test fails. Still
  open, with the fixes that need them: the `xml`, `avg` and `j` exports (L8),
  the concurrency test (M2, M4). The double-Calibrate test (M8) is done.

  **Concurrency + multi-client residue CLOSED (2026-10-16, commit
  `24f2575`)** — `test_concurrency.py` (5 tests, design approved in
  session): (H-A) two concurrent runs through ONE shared node instance
  (the registry's singleton, the server's own model) emit only their own
  values; (H-B) an identical rerun restores from the cache and two
  concurrent identical runs stay correct (correctness pinned — the cache
  does not dedupe in-flight runs); (H-C) three connected clients each
  receive the identical ordered stream; (H-D) a client closing its tab
  mid-run — the run completes, the survivor receives it, a subsequent
  run still broadcasts, /api/health stays ok. **Measured finding, now a
  node-author contract**: a node that writes an instance attribute and
  reads it back after a yield point DOES cross under concurrent runs
  (observed run A emitting run B's value) — the executor is
  thread-per-run by design, so nodes must be stateless. The barrier
  stress variant was rejected as CI-flaky. Remaining in M16: the
  `xml`/`avg`/`j` export tests (L8's surface).

- [x] **M17 — workspace TTL cleanup runs synchronously on the event loop.
  FIXED (2026-10-13).** `cleanup_loop` awaited the plain sync
  `cleanup_expired_runs` — path resolution, cache scans and `shutil.rmtree`
  inline (`app.py:271-280` → `:256-268`); reclaiming a workspace of MTH5
  archives stalled the same loop that drives every WS broadcast
  (`app.py:65`), and `ignore_errors=True` (`app.py:268`) leaked directories
  silently.
  **Remedy (done 2026-10-13):**
  1. The loop awaits `asyncio.to_thread(cleanup_expired_runs)` — a
     multi-GB rmtree no longer touches the loop thread. A pass guard
     (`try/except` + `logger.exception("workspace reclaim pass failed")`)
     keeps the reaper alive whatever one pass raises (previously an
     exception would have silently killed the task for the process
     lifetime — the ignore_errors removal makes hardening it necessary).
  2. `shutil.rmtree(workspace)` without `ignore_errors`, wrapped in
     `except OSError: logger.exception("workspace reclaim failed for
     %s", ...)` — failed deletions name the workspace instead of leaking
     it; later workspaces are still reclaimed.
  3. `app.state.cleanup_expired_runs` stays sync for embedding callers.
  **Tests:** `packages/emptyui/tests/test_ttl_cleanup.py` (3) —
  (a) the loop stays responsive DURING a reclaim (a fake rmtree sleeping
  3 s stands in for a multi-GB deletion; a concurrent `GET /api/health`
  must answer in < 1 s — measured 0.99 s blocked pre-fix, microseconds
  post-fix; margins sized for a loaded CI machine);
  (b) a failing deletion is logged naming the workspace (silently
  swallowed pre-fix);
  (c) the real-rmtree guard (the thread offload still deletes).
  All red pre-fix except (c). Test-authoring note: the first draft of (a)
  hung itself — its tidy-up iterated `doomed` while the still-patched fake
  appended to it; the snapshot + pre-captured real rmtree fix it.
  **Verification:** pytest **702 passed** (one pre-hardening flake of the
  latency probe under full-suite load; margins widened and 9 consecutive
  greens since), `ruff check .` 0.

- [x] **M18 — every node re-renders and re-walks inheritance on every drag
  frame. FIXED (2026-10-14), measured.**
  **Measured (20-node graph, one 20-step drag of a single node,
  render/walk counters probed on MTNode, temp e2e harness — stripped after):**
  renders during the drag 970 -> **54** (~18x; the residue is the dragged
  node itself), inheritance-context rebuilds 920 -> **0**. Graph-build
  churn (20 add+place steps) also fell 9400 -> 1410 renders.
  **Remedy, two drivers addressed:**
  1. The zustand `nodes` subscription: the inheritance context now keys on
     a `graphRev` structure revision in the workflow store — bumped only by
     topology/param changes (add/remove/edge/param/paste/load), never by
     position/dimension/select updates (the per-frame drag path); the memo
     reads the arrays via `getState()`. The auto-fill effect keys on it too.
  2. React Flow itself: its NodeWrapper's internal selector returns the
     node OBJECT, replaced every frame by position changes
     (`{...n, position}`), and it re-invoked the plain-function node
     component. `MTNode` is now `memo`-ized — through a drag every prop
     stays referentially stable, so only the dragged node re-renders.
  Tests: the measurement harness verified behavior before/after (above);
  full gates: vitest 24, Playwright 31, typecheck+lint clean.

- [x] **M19 — the UI never produces relative paths (legacy H8, G12's other
  half). FIXED (2026-10-15, with M23 — decision 2026-10-15:
  relativize on save).** Remedy: `serializeWorkflow` relativizes every
  `path_kind: "in"` file/directory param that lives under the project
  directory or the data root (longest anchor wins) when saving AND when
  starting a run — the document stores the anchor-relative form, and the
  executor resolves relative inputs against `project_dir` at run time
  (server CWD when unset, the status quo). `GET /api/files` keeps
  returning absolute entries (the picker inserts paths; the save pass
  does the relativizing — one conversion point, not two). Output paths
  (`out`) are never relativized. Cache signatures use the
  anchor-relative form (§7.1), so a relocated project stays a cache hit.
  Tests: `workflow.test.ts` (relativization + anchor precedence +
  as-is-without-anchors), `test_project_dir.py` (signature relocation
  stability + end-to-end resolution).

- [x] **M20 — a duplicate node type takes the whole instance down.
  FIXED (2026-10-13).** `discover_plugins` degraded on
  import/`load()`/generic register failures (`registry.py:221-231,243-251,262-272`)
  but deliberately re-raised `ValueError` (duplicate type)
  (`registry.py:238-242`) → `create_app()` threw at startup
  (`app.py:224-226`) and every toggle 409s. One mis-packaged third-party
  plugin then bricked the server, at odds with the "never crash startup"
  doctrine three lines above. Tested as intentional
  (`test_plugins.py:303-304`).
  **Decision (user, 2026-10-13): degrade with an error row in
  `GET /api/plugins`.**
  **Remedy (done 2026-10-13):**
  1. The enabled path's duplicate `ValueError` degrades like any other
     registration failure: the plugin's PARTIAL adds are rolled back
     (`_rollback_partial` — also fixed the pre-existing generic-failure
     leak where a plugin registering N nodes then crashing left N−1
     registered), a warning is logged, and the plugin is skipped whole.
     First registration wins the type (consistent with §3's port-type
     rule). The disabled path already degraded — the asymmetry was the
     bug.
  2. Failed plugins are now recorded in `registry._plugins` — previously
     the error info existed only in discovery's local list and the API
     never saw it (the "shows up disabled with a note in /api/plugins"
     comment was aspirational). All three failure paths (import,
     duplicate, generic) now surface their error rows.
  3. `_toggle_plugin` (M-C3 invariant preserved): a re-enable of a
     degraded plugin returns 200 with the error row (never a 500/409)
     and SKIPS the persistence — the settings file keeps the plugin
     disabled, matching the live registry. The 409 handler remains for
     manual-node contract violations.
  4. Frontend: `PluginInfo.error` typed; the Settings dialog's note cell
     shows the error (title tooltip for long text) instead of
     "disabled".
  **Tests:** `test_plugins.py` — the intentional-raise test rewritten
  (`test_enabled_duplicate_plugin_degrades_with_error_row` red pre-fix),
  all-or-nothing rollback (`fake.ok` NOT half-registered), failed-plugin
  API visibility (import + duplicate rows), and the HTTP surface
  (catalog intact, re-enable → 200 + error row).
  `test_robustness.py::test_toggle_failure_keeps_settings_file_consistent`
  re-contracted: M-C3's file invariant now holds via the persistence
  guard (200 + error row + file still disabled). Caught by the staged
  index verification — the old contract failed there first.
  **Verification:** pytest **706 passed**, `ruff check .` 0, `npm run
  check` clean, Playwright **31 passed** (one unrelated load flake
  re-run green).

- [x] **M21 — the coherence sort materializes GB-scale intermediates.
  FIXED (2026-10-13).** `_min_coh` (`_processing.py:1471-1486`) copied the
  whole input into `Xz`/`Yz` and built `cs_hh` (F·W·I·I) and `cs_he`
  (F·W·I·O) complex128 cumsum arrays plus their pad seeds — all with no
  chunking over bins or windows — a 12-hour 150 Hz level-0 FC (F≈129,
  W≈50k) was order 1–2 GB transient, then `pinv` over F·W 2×2 blocks. No
  guard, no warning, no test at realistic sizes.
  **Measured (tracemalloc peak, the register's rule: numbers first):**
  12 h/150 Hz level 0 (F=129, W=50625, 2 in + 2 out): **5563 MiB**
  single-FC / **6011 MiB** two-FC — ~3× the register's 1–2 GB estimate
  (the S_*/pinv/t einsum temporaries stack on the cumsums); wall 23 s /
  46 s (memory was the killer, not compute).
  **Remedy (done 2026-10-13):** windows are processed in **haloed chunks**
  sized from a transient budget (`_COH_BUDGET_BYTES = 256 MiB`, derived
  per call from F and the channel counts) — a chunk's haloed slice
  `[c0−w0, c1+w0)` contains every window's full ±ensemble_windows
  ensemble, so the statistic is unchanged; a chunk covering all windows
  IS the pre-fix path (bit-for-bit — small fixtures never chunk, all 13
  pre-existing sort tests pass unchanged). The chunking logs one info
  line naming F, W, the channels and the avoided transient.
  **Measured after:** 12 h single-FC **1084 MiB** (5.1×), two-FC
  **1533 MiB** (3.9×); a 48 h recording (4× the corpus) completes at
  **2467 MiB** — the remaining floor is the input stacks, which scale
  with the data; the transient is bounded. Chunked `pinv` is slightly
  faster (cache locality): 21.1 s vs 23.1 s.
  **Tests:** `test_coherence_sort.py` —
  `test_chunked_statistic_matches_the_single_chunk_path` (NaN bins +
  scattered NaNs, smooth_bins pooling, two-FC; allclose 1e-9 — per-chunk
  cumsum baselines differ in low-order bits),
  `test_chunking_activates_and_is_logged`,
  `test_realistic_size_stays_bounded` (F=129, W=10000 — the register's
  "no test at realistic sizes"; red pre-fix at 1099 MiB > 900 bound,
  green at 713 MiB; the bound is deterministic — tracemalloc allocations,
  not RSS).
  **Verification:** pytest **709 passed**, `ruff check .` 0.

- [x] **M22 — exact remote-reference pairing is quadratic in the number of windows. FIXED (2026-10-10).**
  `_pair_remote_times` (`_processing.py:1200`) runs `if j in ri_out` on a
  Python list for every matched row, a linear scan per row. Burst archives
  reach roughly 10^4–10^5 windows per level, and the check runs at every
  pairing site (Estimate TF, two-FC Coherence Sort, every level). Not timed.
  → track used remote rows in a boolean array, keeping the first match (the
  L-A12 semantics).
  Remedy (done 2026-10-10): a boolean `used` mask over the remote rows replaces
  the list membership test. Measured before the fix: 9 s at 5·10^4 windows,
  37 s at 10^5. After: 0.012 s, 0.024 s and 0.26 s at 5·10^4, 10^5 and 10^6.
  Tests: `test_exact_pairing.py` (keep-first duplicate test, which
  passes on both trees; the 5·10^4-window timing guard, which fails on the
  pre-fix tree).
  Test: (1) timing guard: 10^5 synthetic windows pair within a few seconds;
  measure before and after the fix. (2) duplicate local times: local
  `[t0, t0, t1]`, remote `[t0, t1]` must pair to `[(0,0), (2,1)]` exactly.

- [x] **M23 — DESIGN contract items with no implementation anywhere.
  RESOLVED (2026-10-15, per-item dispositions — user decision 2026-10-15).**
  1. **Plugin versions: recorded.** Every catalog manifest carries
     `plugin_version` (the registry stamps it from the contributing
     plugin's distribution metadata; manual/test nodes carry none); the
     frontend records it per node on save; the executor compares record
     vs installed at run time and a mismatch emits `node_warning` (amber
     card marker + properties panel) AND a server-log line, naming both
     versions and the remedy — the run always uses the installed
     codebase, loudly. Per-node-definition versions deliberately NOT
     recorded (a plugin version is what a user can keep track of).
     Contract: `test_api_contract.py::test_plugin_manifest_carries_plugin_version`;
     behavior: `test_project_dir.py` (drift warns on execute AND on
     cache hits; matching/absent/unattributed record warns nothing).
     The four shipped example workflows are stamped.
  2. **models: marked planned** (DESIGN §5) — nothing ships model
     weights yet; the surface returns when one does.
  3. **X-Forwarded-\*: marked planned** (DESIGN §11.3) — no deployment
     web server exists yet (§11.1 gateway model); it lands when one does.
  4. **project_dir: implemented.** `EMPTYUI_PROJECT_DIR` env /
     `create_app(project_dir=…)`; a deployment row (graduated from
     `PLANNED_KNOBS`, shown as the EFFECTIVE anchor — server CWD when
     unset); `PathPolicy` anchors relative `in` paths (data root takes
     precedence); cache signatures are anchor-relative (relocation
     stable, §7.1/§12.3). Tests: `test_project_dir.py` (14).

- [x] **M25 — the Build FC path retired; Compute FC is the ONE TS→FC
  route. DONE (2026-10-15, user decision after the D1–D10 review).**
  Rationale: the TS/TS-collections → FC scheme is the standard (every
  example workflow uses it), clearer for practitioners, and the
  collection route already WAS Build FC's internals (parity was pinned
  while both existed). Decisions: session tiling lives on Compute FC
  (D1); the span criterion is reused verbatim (D2, plus one guard —
  overlapping units never join a span, which aligned collections would
  otherwise concatenate); Extract Run keeps cropping (D3, head/tail
  windows only under epoch anchoring); `clock_zero` default stays `""`
  (D4); the E dipole is stamped on the payload by Extract Run (D5, L-A4's
  new home); `aurora.fc.build` and `aurora.fc.merge` are deleted, their
  engine `merge_fc_containers` remains the shared library call (D6/D7).
  **Remedy:**
  1. The Build FC orchestration was extracted verbatim into node-free
     helpers (`_tile_sessions`, `_contiguous_spans`, `_concat_span_channels`,
     `_stamp_span_run_ids` in `process.py`) and wired into
     `ComputeFC._execute_collection` — new manifest params `session_hours`
     and `regroup_contiguous`; a lone `ts.run` input is wrapped as a
     one-run collection so the classic single-run flow gets session QC too.
  2. Extract Run stamps `e_dipole_length_m` on every payload
     (`_dipole_length_m` from the archive's electric-channel metadata);
     Compute FC falls back to the attr when its param is unset (both
     routes).
  3. `test_fc_build.py` deleted; the uncovered behavior ported to
     `test_compute_fc_sessions.py` (17 tests: session labels, regroup
     spans, start attribution, the no-fabrication guard, the single-run
     wrap, the dipole attr, helper unit tests incl. the overlap guard).
     The old parity anchor became
     `test_collection_route_is_memberwise_cascades` (route == manual
     member-wise engine calls).
  4. The verification record re-ran on the uniform route:
     `clb_coupling_comparison.py` now drives Extract Run → Compute FC —
     **bit-for-bit identical** TF records (46/46 periods, |Δlog₁₀ρ| =
     0.0, both CLB variants); `investigate_polarization.py` migrated the
     same way. The `coherence_sort_*.json` diagnostics flap at ~1e-13
     (numpy reduction FP noise) and were left at their recorded values.
  5. Docs: DESIGN §3/§9.3/§9.4/§10.4/§12 rewritten around the uniform
     route (the retirement noted at §12.1); `merge.py` docstring updated.
  **Verification:** pytest **748 passed**, ruff clean (full suite).
  Follow-up recorded: L20 (the session-QC demo) and L21 (Extract Run
  burst-mode performance).

---

- [x] **M26 — time-domain quality gating: the Mask TS node (user
  decision 2026-10-15, after the gating-landscape review).** The review
  (existing layers: the in-Compute-FC saturation fraction, Coherence
  Sort, Select Runs, Residual Sort) found the missing piece: no
  time-domain MARK producer outside the Phoenix loader, and no
  user-interval exclusion surface. Design (decisions: `quality_mask`
  naming, pooled criteria, configurable-with-defaults, repair deferred
  to a separate node, "Mask TS" name anticipating a future "Mask FC"):
  1. `mth5.qc.mask_ts` writes a pooled per-sample `quality_mask` aux
     channel (data channels untouched; criteria OR-composed): UTC
     interval exclusion, Hampel despiking (per-member auto window
     min(60 s, span/8) floored at 16 samples — burst slots get
     slot-local windows), constant-run detection (scale-free exact
     equality: clips, zero-filled gaps, stuck sensors), NaN flagging,
     axis-gap report+warn (no samples exist to flag), existing-mask
     union, loud >50% warning. Report table rides the `report` port +
     an artifact. Collection mapping preserves the shape.
  2. `MASK_CHANNELS = ("saturated", "quality_mask")` (canonical in
     core-nodes; M5 discipline): Compute FC unions every present mask
     into the fraction rule (`mask_channels` attr records the
     contributors); the decimation max-pool, span concat and Merge Runs
     padding treat the pair uniformly. `reject_saturated` param names
     kept (saved-workflow compat).
  3. **Latent bug fixed**: the core nodes previously filtered EVERY
     channel — `Load Phoenix → Detrend → Compute FC` corrupted the
     `saturated` mask silently. `TimeSeries.aux_channels` + aux
     passthrough in Detrend/Bandpass (regression-pinned).
  Tests: `test_mask_ts.py` (14), `test_mask_consumption.py` (5),
  `test_aux_passthrough.py` (7 — incl. the latent-bug regression).
  Full gates: pytest 774, ruff 0, vitest 38, Playwright 34.

## Low — polish, hygiene, dormant surfaces

- [ ] **L27 — the inherited/suggested combobox is unlabelled (L1
  regression).** (Found 2026-10-25.) `ParamField.tsx:203-232`: the
  `selectOptions` branch renders `CommitInput` without `id={fieldId}` or
  `ariaLabel`, although `CommitInput` accepts both (`:29-60`, `:114`). The
  visible `<label htmlFor={fieldId}>` (`:387`) therefore points at no
  element, and the control has no accessible name — on the most common
  param type once inheritance/suggestions are active.
  **Impact:** Low — accessibility regression; no data effect.
  **Effort:** XS (minutes) — pass `id={fieldId}` and
  `ariaLabel={param.label ?? param.name}`.
  **Test:** a Vitest/Playwright assertion that `getByLabel(param.label)`
  resolves the combobox input.

- [ ] **L28 — the settings live-knob label does not wrap its input.**
  (Found 2026-10-25.) `SettingsDialog.tsx:24-40`: `LiveKnobRow` renders
  `<label className="plugin-row"><span>{label}</span></label>` and a
  separate `<input type="number">` with no `id`; the label names nothing.
  **Impact:** Low — accessibility.
  **Effort:** XS (minutes) — `id` on the input plus `htmlFor` on the
  label (or wrap the input inside the label).
  **Test:** `getByLabel` for a live knob in the settings dialog.

- [ ] **L29 — documentation drift after the 2026-10-15/18 deliveries.**
  (Found 2026-10-25.) Mechanical, but it misleads contributors:
  - `docs/issues.md` header still says the file "stays gitignored"
    (`:30-35`) although L17 is done and the file is tracked; the
    "solved entries are removed" sentence contradicts the kept `[x]`
    entries; the baseline blocks stop at 775/34 while the tree is at
    801/36.
  - `docs/DESIGN.md` §12.1 still says "M4 — backlog **in progress**"
    though the M4 delivery rows are historical; §12.2's open rows
    (batch runner #3, disk cache #5, DOF stats #7, Parzen taper #4,
    FC→TS #11, thread pool #12, sub-workflows #13, plugin manager #14,
    FDSN #20, MMT #18) are the real backlog and should say so.
  - `docs/plugin-guide.md:125` still lists `selection` as a core port
    type (removed by L2).
  - `docs/deployment.md` omits `EMPTYUI_NODE_TIMEOUT` and
    `EMPTYUI_PROJECT_DIR` (`config.py:86`, DESIGN §11) and presents
    `X-Forwarded-*` as required (`:79-81,144`) while DESIGN §11.3 marks
    the handling **planned** (M23 decision).
  **Impact:** Low — no runtime effect; high onboarding cost.
  **Effort:** S (hours) — one docs commit, no code.
  **Test:** none (docs); optionally a grep-based CI check for the
  removed `selection` token and the env-var list.

- [ ] **L30 — repository hygiene: an untracked 385 MB corpus file is not
  ignored, and the template ships build clutter.** (Found 2026-10-25.)
  - `test/test_mtu.mth5` (385 MB, untracked) is **not** covered by
    `.gitignore` — only `test/verification/*.mth5` is (`:62`); one
    `git add -A` commits it.
  - `examples/plugin-template/{src,tests}/__pycache__/` are present.
  - The `example` plugin is installed by `scripts/install.sh:62,70` and
    CI and appears in the palette under "Example"; its nodes
    (`example.ts.gain`, `example.demo.rms`) are not `hidden`, although
    the plugin-guide checklist (`:259-260`) asks for `hidden` on test
    nodes.
  **Impact:** Low — repo size and palette clutter; the gitignore gap is
  one command away from a 385 MB history entry.
  **Effort:** XS (minutes) — add `test/*.mth5` (or the exact file) to
  `.gitignore`, delete `__pycache__`, set `hidden = True` on the demo
  nodes (or gate the template install behind a dev extra).
  **Test:** `git check-ignore -v test/test_mtu.mth5`; a registry test
  asserting the example nodes are hidden.

- [ ] **L31 — the plugin template declares MIT while the repository is
  Apache-2.0.** (Found 2026-10-25.)
  `examples/plugin-template/pyproject.toml:7` has
  `license = { text = "MIT" }`; the repo ships Apache-2.0. The template
  is the copy-me surface, so every derived plugin inherits the wrong
  license.
  **Impact:** Low (legal/metadata) — but it propagates by design.
  **Effort:** XS (one line) — `license = { text = "Apache-2.0" }` (and
  check the template's `LICENSE`/classifier if present).
  **Test:** none; a packaging check if CI has one.

- [ ] **L32 — no undo/redo and no run history in the graph editor
  (decision to revisit, not a defect).** (Found 2026-10-25.) DESIGN §8.2
  documents the omission; the practical effect is that a mis-drag or a
  delete is unrecoverable, and there is no gallery of previous runs to
  compare against. Batch/queue, disk cache and auth are already tracked
  in DESIGN §12.2 (#3, #5) and are not duplicated here.
  **Impact:** Low-Medium (usability) — the highest-friction gap for new
  users after the correctness items.
  **Effort:** L for undo (graph history + store integration), M for a
  run-history list (the run records already exist server-side).
  **Decision needed:** whether undo is in scope before the GitHub
  migration, or the editor stays append-only with save/load as the
  recovery path.
  **Test:** if implemented, undo/redo round-trips a node add, move,
  delete and edge change; history lists completed runs with their
  artifacts.

- [x] **L1 — accessibility gaps (systemic). FIXED (2026-10-14).**
  Remedy, all four clauses: (1) every param control is labeled — the
  visible label is a real `<label htmlFor>` (radio keeps its wrapped
  labels and gains `role="radiogroup"` + `aria-label`; the file widget's
  browse button and both dialogs' close buttons are named); previously
  only the default widget branch carried an aria-label, so most node
  params were nameless to assistive tech. (2) Run status/notice are
  `role="status"`, errors `role="alert"`. (3) A shared `useDialogFocus`
  hook: dialogs focus their first control on open, Tab/Shift+Tab cycle
  INSIDE (aria-modal alone trapped nothing), Escape closes, focus
  returns to the invoker. (4) Contrast: `.muted-small` 0.5->0.75 opacity,
  the dark-ground variants #aaa->#a0a0a0 at opacity 1 — the old blend
  measured ~3:1 against AA's 4.5:1.
  Verified by a new `a11y.spec.ts` (labeled params via getByLabel, live
  regions, the focus-trap cycle + Escape + focus restore). One honest
  test fix surfaced: `click an edge and press Delete` had passed for the
  WRONG reason — its midpoint click landed on a param LABEL and selected
  the NODE (Delete then removed node+edge); with labels correctly
  forwarding clicks to their inputs, the test now probes its bbox for a
  point that hit-tests to the edge path itself (dense grid) and asserts
  true edge selection. Full gates: pytest 736, vitest 32, Playwright 34. `ParamField` passes `aria-label`
  only in the default branch (`ParamField.tsx:343`); `boolean` (`:224`),
  `select` (`:238`), number (`:265`), file/directory (`:280`), code (`:312`),
  `list` (`:322`) and the datalist combobox (`:205-214`) are unlabeled, and
  the visible label is a bare `<span>` (`:354`) — most node params are
  nameless to assistive tech (only `radio` wraps its input in a `<label>`).
  Run status/notice/error have no
  `role="status"`/`aria-live` (`App.tsx:508-510`). `FileBrowser`/`SettingsDialog`
  set `aria-modal` but never trap focus or mark the background inert
  (`FileBrowser.tsx:47-55`, `SettingsDialog.tsx:115-123`) and use bare "✕"
  close buttons. `.muted-small { opacity: 0.5 }` + `#aaa` on dark ≈3.3:1
  (`styles.css:480-484,673-676`) fails WCAG AA. → labels, live regions,
  focus management, contrast pass.

- [x] **L2 — `selection` port registered with no producer (legacy D3).
  FIXED (2026-10-14; decision: hide until a producer exists).** Remedy:
  `selection` removed from `CORE_PORT_TYPES` and the frontend
  `PORT_COLORS` mirror, with a comment naming the return condition (the
  M4-deferred Kernel Dataset producer); `ViewSelection` stays registered
  but hidden (unchanged). Tests: `test_port_types_has_no_producer_less_
  selection` + `catalog.test.ts` mirror pin — both red-first.

- [x] **L3 — `artifact` event `handle`/`path` inconsistencies. CLOSED (2026-10-14; backend half `7284aeb`, frontend + handle contract this commit).** Fresh runs
  emit `handle: null` when the artifact dict isn't an output
  (`executor.py:862`) while cache restores emit `""` (`executor.py:725`);
  both emit the absolute server-side `path` to every WS client
  (`executor.py:861`), and §7.2 promises `handle` as the authoritative
  router — the frontend meanwhile resolves artifacts by input edge and
  falls back to "the source node's last artifact" (`MTNode.tsx:87-98`),
  which can show the wrong figure for multi-output sources. → normalize the
  handle to `""`/absent, stop leaking absolute paths, match on
  `targetHandle`.

- [x] **L4 — the event cap can drop `run_started`.** `app.py:519-520`
  caps the replayed event list at 5000, so a late poller can see a truncated
  stream with no start marker (self-describing replay lost).

- [x] **L5 — the cache is bounded by entry *count* only.** `NodeCache` clamps
  the bound to ≥1 and evicts LRU by count (`cache.py:46-47,61-63,65-73`);
  256 large xarray/MTH5 payloads can OOM the process while `cache_max_gb`
  stays a reserved-but-inert knob (`settings.py:55-58`).

- [x] **L6 — settings durability/HTTP polish.** `save()` does not `fsync`
  before `os.replace` (`settings.py:122`) — a crash leaves a corrupt file
  silently defaulted away (`settings.py:89-92`); `OSError` from
  `set_disabled`/`set_settings` escapes as a bare 500 (`app.py:349`,
  `:435`); `cache_max_entries: 0` documented as "off" clamps to 1 and the
  view then reports `1` (`app.py:436-442`).

- [x] **L7 — `print_params` defaults disagree between manifest and method.**
  `PlotTipper`: `Param(..., default=False)` (`plots.py:97`) vs
  `execute(..., print_params=True)` (`plots.py:106`);
  `PlotPhaseTensor`: `plots.py:148` vs `:156`. The executor injects the
  manifest default so runs are correct, but direct `node.execute(...)` calls
  (as tests do) silently flip a documented-off switch.

- [x] **L8 — `ImportTF` always applies the EDI→SI conversion, no opt-out.**
  `EDI_TO_SI` applied unconditionally (`tf.py:25,239`, warned but not
  suppressible): a file already stored in SI units would be double-scaled
  with no way to say so. Round-trip on real EDIs is exact today.

- [x] **L9 — documentation drift. FIXED (2026-10-14).**
  §10.6 now names the shipped tick grid (20°, `np.arange(0, 91, 20)`;
  diagonal 45°); the M0-era comments in `store.py`/`executor.py` describe
  the current behavior (persistence shipped; the milestone tags are
  history); the bare-`§13+` references — 191 across 53 files by today's
  count (the register's 270 predates the later fixes) — are prefixed
  `history §` by a scripted pass (docs/DESIGN.md, docs/issues.md, docs/history and
  the appendix table excluded; appendix-row `| §N |` definitions kept).
  A post-pass census: zero bare references, one README prose collision
  fixed. Gates: pytest 739, vitest 33, Playwright 34, ruff clean.

- [x] **L10 — DESIGN §9.4 session arithmetic does not match its own formula.**
  docs/DESIGN.md:1284 says one deepest-level window spans about 68 min of raw
  1 Hz data "(5 levels × factor 4)". With a window of 256 and a factor of 4
  per level, the deepest window spans 256·4^L seconds at 1 Hz: about 3 days
  at L = 5. The 68 min figure corresponds to L = 2. The session-length
  guidance that depends on it is off by orders of magnitude.
  → confirm the default decimation count and window in the node manifests,
  then correct the figure or the level count.
  Test: a small test reads the default decimation count and window from the
  manifests and asserts that the DESIGN figure matches within 10%.

- [x] **L11 — window labels are documented as starts but produced as centres. DECIDED 2026-10-10: centres kept; reader audit DONE (2026-10-14).**
  DESIGN §10.1 (docs/DESIGN.md:1335) and the `compute_fc_levels` docstring say
  clock-zero "anchors window starts" to `t0 + k·step`. The code labels each
  window with its scipy segment centre (`process.py:1049`). The code is
  internally consistent (BuildFC shifts back), but the docs, M10, and any new
  consumer assume starts.
  **Decision (2026-10-10): keep the centre labels.** Rule for new code: an fc
  time label is the window CENTRE (first sample + N/(2·fs)). To test a window's
  data, use its start, `label − N/(2·fs)` (`_centre_to_start_s`; BuildFC's
  `half_win`). Persisted FC archives store centres, so changing the convention
  would need a migration. Docs updated: the `compute_fc` / `compute_fc_levels`
  docstrings, the pair-tolerance descriptions, and DESIGN §10.1. The
  `_clock_zero_head_skip` docstring correctly speaks of starts. Left as
  delivered: `docs/history/m4-delivery.md` says "window starts" (historical
  record). Open follow-up: the reader audit below (not blocking).
  Usage survey (2026-10-10): every FC carries centre labels (`compute_fc`,
  `compute_fc_levels`), persisted as centres in the mth5 FC layer
  (`persist.py`). Corrected for: BuildFC session attribution only
  (`process.py:1056-1067`). Read as centres, uncorrected: SelectRuns and
  ResidualSort absolute run start for the diurnal prior (`process.py:1361`,
  `:1617`); SelectRuns dropped-session bounds (`process.py:1447-1452`);
  `_warn_remote_gaps` (M10); the spectrogram time axis (`plots.py:439`).
  Pairing is unaffected (both sides carry the same offset). The saturation
  mask indexes window starts directly. The `start_datetime` attr is the series
  start, not a window start.
  Test: a property test that the first window's `time` minus the first
  sample's time equals `N/(2·Fs^(l))` on a synthetic record, so any change
  of convention is visible.

- [x] **L12 — assorted frontend defects. FIXED (2026-10-14, one sweep).**
  Remedy, all ten: (1) `deserializeWorkflow` names missing
  `nodes`/`edges` arrays (red-first unit tests); (2) `serializeWorkflow`
  throws on a handle-less edge instead of inventing `'data'` wiring
  (red-first); (3) a `docRev` store counter re-fits the canvas on EVERY
  document load (the empty->non-empty transition only covered the first);
  (4) Save revokes the object URL after 10 s, not synchronously;
  (5) `.error-msg` sites carry `title` and modal errors wrap;
  (6) the pre-run notice stays silent while the catalog is pending;
  (7) Settings rejects a non-numeric knob by name instead of
  NaN->null-clearing it; (8) stale `.properties label*`/`.mt-node-body`
  CSS deleted; (9) NodePalette's untruncated tooltip fallback removed
  (the server already derives tooltip from the description's first
  sentence); (10) pipeline/app/interactions specs import the shared
  `_flow` helpers — the ~160-line inline copy and the triplicated
  `addNode` are gone. Full gates: vitest 32, Playwright 31, tsc/lint
  clean. Loading a second document with
  nodes already on screen skips `fitView` (`App.tsx:59-66`), so the graph
  can land off-screen; Save revokes the object URL synchronously after
  `a.click()` (`App.tsx:410`); `deserializeWorkflow` doesn't check
  `nodes`/`edges` exist (`workflow.ts:552,581-588`) → generic "Load failed";
  `serializeWorkflow` defaults a missing handle to `"data"`
  (`workflow.ts:521-523`) instead of failing; `.error-msg` is
  `nowrap+ellipsis` with no `title` (`styles.css:619-626`) so long 422
  details are unreadable; the pre-run notice claims "results won't be
  visible" whenever the catalog hasn't loaded (`App.tsx:314-319`);
  `SettingsDialog` turns an empty-to-`Number()` NaN into `null` via
  `JSON.stringify` (`SettingsDialog.tsx:91`); stale CSS for removed
  surfaces (`.properties label*` `styles.css:88-103`, `.mt-node-body`
  `:149-154`); `NodePalette` re-implements the tooltip first-sentence
  fallback the server already performs, untruncated (`NodePalette.tsx:65`);
  e2e helper duplication — `_flow.ts:1-3` admits ~160 lines copied from
  `pipeline.spec.ts`, and `addNode` is redefined in 4 files.

- [x] **L13 — `port_types` "first declaration wins" depends on entry-point
  iteration order (registry.py:121-123): two plugins declaring the same tag
  could flip the port color across installs. FIXED (2026-10-14) — landed
  inside the M5 consolidation commit `2df11f5` (registry.py IS one of the
  mirrored-knowledge sites).** Remedy: `manifests()` iterates node types
  SORTED, so "first declaration" is the alphabetically-first node type —
  deterministic across installs. Test:
  `test_port_type_declaration_is_install_order_independent` registers a
  conflicting pair in both orders and pins the same winner (red-first:
  the reversed order used to flip it).

- [x] **L14 — `examples/output/` held ~503 MB of unreferenced build products
  (legacy H3). DELETED (2026-10-14, user decision — GitHub-bound size).**
  All four archives removed (the lemi424 one is the WRITE target of the
  shipped example workflow — regenerable by running it); the directory is
  empty now. `nims-usarray-test/` (L16) no longer exists — already clean.

- [x] **L19 — select dropdowns offer blank (unnamed) options (user report
  2026-10-14: Compute FC's prewhitening showed "first difference" plus a
  blank). FIXED (2026-10-14).** Six blank options existed, all on Compute
  FC + Build FC (window_type_deep / prewhitening / per_window_detrend),
  all meaning "off" — but window_type_deep's "" meant "same as the
  shallow taper", not "none", so the labels are per-param honest:
  ("same as shallow", "hamming", "hanning", "boxcar"),
  ("none", "first difference"), ("none", "linear"). The sentinels alias
  to the canonical "" internals at the top of compute_fc_levels —
  pre-relabel documents (saved with "") behave identically — and
  ParamField's select now includes a stored out-of-list value (L-F9)
  while never rendering a blank option. A catalog-wide pin
  (test_no_select_param_offers_a_blank_option) guards every future
  manifest. Tests: the pin + test_blank_option_sentinels_alias_to_off
  (bit-identical FCs for "" vs the sentinels), both red-first; full
  gates: pytest 736, vitest 30, Playwright 31.

- [x] **L15 — no shipped example exercises the burst-native `Build FC`
  route (legacy H4). OBSOLETE (2026-10-15, with M25): the route itself was
  retired — the burst story is now the uniform `Extract Run (time range)
  → Compute FC` collection path, which the phoenix examples already
  exercise. The follow-up demo (session QC on a burst collection) is
  L20.**
- [x] **L20 — a shipped example demonstrating session QC + gating.
  DONE (2026-10-15).** `examples/workflow/phoenix-mtu5a-session-qc.emptyui.json`
  on the tracked `smoke` corpus (stream family only — the 16-s/2-s
  slots are statistically meaningless QC units, a README-noted choice):
  Load MTH5 → Extract Run (time range, sample_rates=15) → Mask TS
  (defaults) → Compute FC (session_hours=0.25, regroup_contiguous,
  clock_zero=epoch) → Select Runs (defaults) → Estimate TF → plot, with
  View Table nodes on the mask report and the run diagnostics. Measured
  result (the "improved by QC" question): the corpus is NIGHT data but
  its 21:15–23:00 stretch is genuinely noisier — at the PRODUCTION
  defaults (deviation_max=2.0, diurnal prior on) four sessions drop
  (deviations 1.0–1.4 kept vs 2.4–4.9 dropped), with the documented
  §9.4 trade-off visible (the dropped tail costs the deepest bands).
  The exploration also RETUNED the Hampel default (3 → 8 MADs): h=3
  over-gates heavy-tailed natural series (5.3% masked → 21 → 6 periods;
  h≥5 is band-neutral, Δlog ρ ≤ 0.014 vs plain). Pinned by
  `test_demo_session_qc.py` — the shipped document executed headless
  through the real executor (corpus-guarded, ~3 s).
- [x] **L21 — Extract Run is slow on burst-mode archives. DONE
  (2026-10-15).** Profiled first (the register's rule): a full-collection
  extraction from `day_12h` (145 members) took **15.6 s**, metadata-bound —
  347k h5py attribute reads via per-channel `get_channel` storms (898
  channels), 146 `run_summary` DataFrame walks (one per resolved run, ~30
  ms each), and per-run `run_obj.metadata`/dipole validations. The data
  reads themselves are ~2.5 s. Three fixes, all bit-faithful:
  `read_run_channels` reads channel HDF5 datasets directly (raw attrs +
  `make_dt_coordinates`; `filters` JSON-parsed) — pinned equal to
  `get_channel().to_xarray()` on every channel of both test archives
  (`test_read_run_channels_matches_slow_path`); `_dipole_length_m` reads
  raw attrs; `_resolve_run` resolves a named run directly instead of
  rebuilding the summary per run (`test_full_extraction_avoids_metadata_
  storms` pins 0 `get_channel` calls and ≤1 summary walk). Result:
  **15.6 s → 1.94 s (8×)** for day_12h full extraction; the CLB
  verification suite reproduces clean-HEAD output EXACTLY.

  Side finding while verifying: the recorded CLB/coupling baselines were
  STALE since M25 — the committed 46-period records don't reproduce even
  at `ee375da` (the M25-era route change shifted the marginal deep-band
  grid: 6 bands lost, 4 re-centered; shared-period values agree to
  3.4e-15, coherence records differ on 21/1280 rows at ≤1.3e-12) — the
  M25 "recorded JSONs unchanged" claim was never validated for the
  coupling script. All four records re-recorded at the canonical uniform
  route (deterministic: two runs at `ee375da` agree; my tree equals
  clean HEAD exactly).

- [x] **L16 — `nims-usarray-test/` leftovers (legacy H5). OBSOLETE
  (2026-10-14): the directory no longer exists — nothing to purge.

- [x] **L22 — Align Runs is slow on one-day burst pairs. DONE
  (2026-10-16, commit `afcba20`).** User report: on a one-day
  remote-reference pair (south_china, 598+598 members) align took
  40-50 s while post-L21 Extract Run takes ~8 s. Profiled first: align
  measured **97.9 s, 99.5% in `_crop_to_runts`** — 5993 ChannelTS +
  1196 RunTS rebuilds (63.7 s + 20.3 s) of mt_metadata validation
  storms for what is an index slice. Collection members now stay duck
  payloads (dataset `isel` crop, attrs/filters/provenance intact);
  singles and RunTS members keep the rebuild (output-kind contract).
  Result: **0.32 s (305×)**, verified bit-identical (align output AND
  downstream Compute FC) against the old implementation on the user's
  corpus; pinned by `test_align_collections_never_rebuild_runts`.

- [x] **L23 — node audit for the L21/L22 synopsis; chain storms fixed.
  DONE (2026-10-16, commit `d0ccccf`).** Stage-timed the full one-day RR
  chain on the user's corpus: extract 6.9+7.6 s, align 0.33 s (both
  post-L21/L22), Compute FC 3.65 s, coherence sort 13.7 s, Estimate TF
  0.5 s — and **Compute FC persist cost 23 s more than the math**.
  Fixed: (a) the coherence sort's batched SVD pinv over ~1.6M 2x2
  matrices (11.1 s) -> closed-form adjugate for the hermitian-PSD 2x2
  case with an SVD fallback for rank-deficient entries (13.7 -> 2.0 s;
  retained counts identical, coherences within 7.8e-16); (b) a latent
  Extract Run bug — the filter-registry dedupe only replaced a member's
  fresh registry when the name-set was new, so 4784 distinct filter
  objects survived instead of 8 and defeated any caching; plus
  `_cached_complex_response` memoizing per (filter, grid) — FC output
  verified bit-identical. **Left open by design**: persist's 23 s (mth5's
  FCDataset metadata validation per channel per level inside
  `FCDecimationGroup.from_xarray` — fixing means bypassing mth5's writer
  or changing the persisted structure; user decision needed) and
  Calibrate's per-member `_member_runts` rebuild (same synopsis; the
  node is absent from the RR chain). Chain total ~65 s -> ~33 s.

- [x] **L24 — Compute FC persist cost 23 s more than the FC math.
  DONE (2026-10-16, commit `0e49cb4`).** Path A (user decision — keep
  mth5's writer, drop the redundant work): `_write_fc_channels`
  replicates `FCDecimationGroup.from_xarray`'s per-channel metadata and
  `add_channel` calls with the duplicate check as one `groups_list`
  lookup instead of a channel_summary DataFrame walk per channel;
  `_update_decimation_metadata` performs update_metadata's aggregation
  from the shared channel metadata; `_ensure_survey/_station` cache on
  the open handle. Persist 26.8 -> 18.6 s on the one-day RR corpus
  (the remaining floor is mth5's FCChannelDataset validation ~1 ms x 10
  channels x 1200 calls — path B, raw h5py writes, noted if ever
  needed). Verified identical twice: a subtree diff against
  from_xarray's output (test_persist_channel_writer_matches_mth5_
  from_xarray) and 1197 reloaded runs/levels bit-identical via the
  repo's load_fc_level on the real corpus. Chain total: ~65 s ->
  ~41 s (extract 14 + align 0.2 + FC 6 + sort 2 + estimate 0.5 +
  persist 15.6 + estimate overhead).

- [x] **L25 — Load Phoenix MTU-5 slow on long acquisitions. DONE
  (2026-10-16, commit `336ba1b`).** Profiled on the 12-h corpus (145
  runs, 736 channels): 14.8 s, of which 5.0 s was ChannelTS
  constructions built only to be unwrapped by from_channel_ts — the seg
  loop now carries (name, type, data, metadata) tuples written via
  RunGroup.add_channel, with the channel end normalized to the last
  sample (what the ChannelTS constructor used to produce). Ingest
  14.8 -> 12.1 s, archives content-identical (every dataset + attr;
  only mth5's derived summary tables differ — row order + per-file
  references); CLB records unchanged from the d0ccccf baseline. The
  remaining time is NOT metadata: ~2/3 is the raw TS block decode
  (linear in samples) and ~1/3 mth5's per-channel/per-run writer floor.
  Follow-up the same day (`3856d9b` was L24): the block decode
  VECTORIZED — bounded 16384-block batches, 22.8x on the decode
  (measured bit-identical), header guard batched, yield contract
  unchanged — ingest 12.1 -> 6.8 s; archives content-identical, all
  four verification records byte-identical. Remaining: ~1 s tags + ~4 s
  mth5's writer floor (path B noted).

- [x] **L17 — `docs/issues.md` is gitignored (legacy H7). DONE (2026-10-16):
  the rule is dropped and the register ships.**
  **Sanity check (the caveat below), run before the un-ignore:** 58
  register entry IDs; 118 distinct IDs cited across docs/DESIGN.md + 317
  commits — every current-register citation resolves. The only
  non-resolving tokens belong to the 2026-09-18 review round
  (H1–H4/M1–M14 + the B/L-A/L-C/L-F/L-M sub-item families), whose
  register was DELIBERATELY removed at `9c03d20` ("the review record —
  resolved") — code comments citing those IDs point at that commit, not
  at this register (documented here so the distinction survives).
  False positives in the scan: `E731` (a ruff code), `P2` (a §24.5
  phase label), `A1-A4`/`G4'` (old-round IDs). The check is against
  entries accidentally deleted rather than moved to a scratch file —
  none were.

- [x] **L18 — coherence-map display node. DONE (2026-10-16, commit
  `4e0ace6`) — implemented per the settled design below.**
  `aurora.plot.coherence_map` (Display, terminal): two-FC (fc + ref_fc —
  the conservative min of the E–remote-H signal and the local↔remote
  transfer coherences over the intersect-paired windows) and single-FC
  (E–H signal) modes; the sort's EXACT criterion via the shared
  `_per_bin_coherence` (pinned value-for-value); window-centre time
  axis (L11); the spectrogram's cascade-step level convention with the
  ref tier matched at the same rate (actionable error when missing —
  hit on the real corpus during verification). Verified on the
  south_china 630/634 chain: the 0.9375 Hz-tier map shows the
  0.1–0.5 Hz band coherent only during daytime stretches, the slot
  gaps as vertical stripes, and a localized low-frequency event —
  structure the per-frequency medians average away. Follow-up the same
  day: print_params (default False, the Plot AppRes/Phase convention)
  keeps production titles clean — the provenance line (cascade step,
  window, ensemble, coherence type) is opt-in. The related
  noisy-remote WARNING idea below stays parked (separate small item). A time-frequency plot of COHERENCE instead of power — the
  diagnostic that localizes *when* a band is bad (the manual 2026-10-14
  RR investigation of south_china 630/634 needed four ad-hoc scripts to
  establish what one map would have shown at a glance: the remote
  station's 0.04-0.3 Hz noise active only during local daytime).
  **Design (settled in that investigation, keep it):**
  - Two input modes in one node: REMOTE (`fc` + `ref_fc` — inter-station
    magnetic coherence per window x bin via the existing `_pair_remote_fc`
    intersect machinery) and E-H (single `fc` — the Coherence Sort
    criterion with the time axis retained).
  - Reuse the sort's `ensemble_windows` + `smooth_bins` sliding-ensemble
    semantics so the map's numbers ARE the sort's numbers (dark region in
    the map = windows the sort down-weights). Chunk the ensemble
    accumulation per M21's memory rule.
  - Time axis = window labels (centres, the settled L11 convention);
    pcolormesh, log frequency, ConciseDateFormatter, apply_style — the
    spectrogram node (emptyui-mth5/plots.py) is the template.
  - Level handling: an FC container's levels have different window hops
    (level 4 hops 6.4 s, level 6 hops 136 s) so a single broadband map
    needs one time grid. MVP: a level selector (auto = densest level);
    a stacked broadband variant only if it earns its keep.
  - Display category, output `plot` -> Preview Plot. ~1 day with the
    plot-style tests.
  - Related smaller item (also parked, same investigation): a
    noisy-remote WARNING in Estimate TF's RR path — per-band remote/local
    magnetic power ratio + inter-station coherence are computable a few
    lines from where pairing already runs; when Pr/Pl >> 1 with collapsed
    coherence, warn via ctx.warn naming the affected periods and the
    remedy ("remote station noisy at 5-30 s; restrict the estimate
    window"). Would have flagged the 630/634 case automatically.

- [x] **L26 — user-facing documentation & migration prep (the GitHub
  milestone's doc track). DONE (2026-10-17), four phases:**
  - **Phase 0 (`1481aab` + `e9d45c5`)**: DESIGN.md/issues.md moved under
    `docs/` (111 living files re-cited; docs/history/ verbatim); the
    stray untracked `fig_src.txt` deleted; the ComfyUI-Manager-style
    plugin manager noted in §12.2 row 14. Also surfaced 3 ruff errors
    the L18 commits had shipped (gate output truncation — lesson:
    never tail -1 a gate).
  - **Phase 1 (`750d9a5`)**: scripts/serve.sh (one-port serve via
    EMPTYUI_SERVE_DIST — the backend already static-mounted the dist),
    install.sh --backend-only, scripts/test.sh (CI-mirroring gates).
  - **Pre-push data hygiene (`8fa54de`)**: the smoke archive renamed
    smoke_2h -> smoke (it spans exactly 3 h; duration-free name), 18
    references swept; the M14 entry now records the ~100 MB blob
    decision (`13215f3`).
  - **Phases 2-3 (`36ec095`, `e490713`)**: README rewritten user-facing
    (hero screenshot captured by driving the real app; quick start on
    the bundled archive; deploy section; verification anchors kept);
    docs/user-guide.md (98 lines: three recipes + plots/exports/
    troubleshooting, all labels checked against the live manifests);
    deployment.md quick start.
  - **Phase 4 — LICENSE (Apache-2.0, the user's pick; the pyprojects
    said MIT — corrected) + the fresh-clone smoke test: clone HEAD to
    an empty dir -> install.sh --backend-only (uv cache; PyPI was
    unreachable — worth re-running on real network before the push) ->
    serve.sh cold path (npm install + vite build + uvicorn) -> the
    quick start driven through the REAL UI (Load -> Run -> completed,
    3 plots rendered). Every copy-pasted command works verbatim.**

---

## Suggested fix order

**Closed:** H1–H6 (2026-09-30); M9, M10, M11, M22 (2026-10-10, commits
`105b7fa` M9, `471fe47` M10, `c4b8f7c` M11, `b908f29` M22, `adcd2c7` L11
docs); M16 Phase 0 part (`447c7bb`); M14 part 1 (`f666b3a`); M1
(2026-10-12); M2, M4, M17, M20, M21, H7, M24 (2026-10-13);
L8, L7, L10, L11-audit (2026-10-14);
L4, L5, L6, L3-backend (2026-10-14); L5-regression, M12, M15, M18, M13
(2026-10-14, Phase 3); L20 (`356b151`, 2026-10-15); M16 (2026-10-16);
L17, L18 (2026-10-17); **L19–L26 docs & migration prep (2026-10-17,
see L26)**; M14 (2026-10-18). **Open: 0** in the pre-2026-10-25
register. The plan below is organised in phases; decisions come first.

### 2026-10-25 review — open findings (ranked by impact / effort)

Fourteen new entries; H8 is already closed (2026-10-25). The rank is
impact first, then effort (a one-line fix that removes a silent wrong
result outranks a multi-day scalability item). Effort: XS = minutes,
S = ≤1 day, M = 1–2 days, L = a feature.

| Rank | ID | Finding | Impact | Effort | Why this rank |
|---|---|---|---|---|---|
| 1 | ~~H8~~ | ~~coherence map centres → ends~~ | ~~High~~ | ~~S~~ | **done 2026-10-25** (801 passed) |
| 2 | ~~H9~~ | ~~`Calibrate` mutates its input~~ | ~~High~~ | ~~S~~ | **done 2026-10-26** (804 passed) |
| 3 | ~~H10~~ | ~~`sign_convention` no-op off EDI~~ | ~~High~~ | ~~S~~ | **done 2026-10-26** (808 passed) |
| 4 | ~~M27~~ | ~~`MaskTS` 3.0 vs 8.0~~ | ~~Medium~~ | ~~XS~~ | **done 2026-10-26** (809 passed) |
| 5 | ~~M30~~ | ~~`merge_tf_levels` 1-D `Z_err`~~ | ~~Medium~~ | ~~XS~~ | **done 2026-10-26** (808 passed) |
| 6 | ~~M29~~ | ~~empty `save_path` → data dir~~ | ~~Medium~~ | ~~S~~ | **done 2026-10-27** (814 passed) |
| 7 | ~~M31~~ | ~~models/`sha256` documented, absent~~ | ~~Medium~~ | ~~S~~ | **planned 2026-10-27** (no consumer yet) |
| 8 | M28 | Phoenix ingest: list + batch buffers | Medium | M | A+B done; C parked (84 h only) |
| 9 | L27 | combobox a11y | Low | XS | cheap, L1 regression |
| 10 | L28 | settings label a11y | Low | XS | cheap |
| 11 | L30 | gitignore / `__pycache__` / hidden | Low | XS | accident prevention |
| 12 | L31 | template license MIT vs Apache-2.0 | Low | XS | propagates by design |
| 13 | L29 | docs drift sweep | Low | S | onboarding cost |
| 14 | L32 | undo / run history | Low-Med | L / M | decision first |

**Phase A — correctness (1–2 days). COMPLETE (2026-10-26).** ~~H8~~
(2026-10-25); ~~H9~~, ~~H10~~, ~~M27~~, ~~M30~~ (2026-10-26). All five
were small and each closed a silent-wrong-result or a
test/production divergence.

**Phase B — robustness and contract (2–3 days).** ~~M29~~ (done
2026-10-27) → ~~M31~~ (planned 2026-10-27) → M28 (A+B done 2026-10-27:
12 h window 903→377 MB, 24 kHz 973→359 MB at 256 blocks, full corpus
1,548→1,512 MB; C parked for 84 h corpora).

**Phase C — polish and hygiene (≈1 day).** L27, L28, L30, L31, L29,
then L32 (decision). The a11y pair and the hygiene/license items are
minutes each; the docs sweep is one commit.

**Cross-cutting rules:** one issue per commit, red test first (P0.6),
`scripts/dev/verify_index.sh` before each commit, full `pytest packages`
plus `ruff check .` per phase, and the frontend gates (`npm run check`,
Playwright) for L27/L28. The 2026-10-25 baseline is 801 collected
(727 + 74) / 38 Vitest / 36 Playwright; any new count must be recorded
in the header block.

### Ground rules (every item)

- One issue per commit; the commit message cites the issue ID.
- Red test first: write the test named in the entry's Test line, confirm it
  fails on the current code, then fix.
- Verify the staged index in isolation with `scripts/dev/verify_index.sh`
  (exports the index, runs that batch's tests and ruff) before committing;
  stage by block with `scripts/dev/stage_block.py`.
- Update the entry in the same step: mark it FIXED with the remedy and any
  measured result.
- Per phase: full `pytest packages` and `ruff check .`. Any frontend change:
  `npm run build`, lint, Vitest and the Playwright suite (31 tests), run
  locally before the commit (P0.7).

### Open decisions (answer before the affected item starts)

1. ~~**M20** — duplicate plugin type: keep failing fast and document
   it, or degrade with an error row in `GET /api/plugins`?~~
   **Resolved 2026-10-13: degrade** (user decision) — the plugin is
   skipped whole, the conflict is its error row, M-C3's file invariant
   holds via a persistence guard.
2. ~~**M19** — relative paths: relativise on save, or return root-relative
   entries from `GET /api/files`?~~
   **Resolved 2026-10-15: relativize on save** (user decision, with M23) —
   the save/run document build converts `in`-kind paths under the project
   dir/data root; the picker keeps returning absolute entries.
3. ~~**M23** — each contract item: implement it, or mark it planned in
   DESIGN?~~
   **Resolved 2026-10-15, per item** (user decision): versions recorded;
   models and X-Forwarded marked planned; project_dir implemented. See
   the M23 entry.
4. ~~**L2** — `selection` port type: hide until a producer exists, or keep
   it as a declared future surface?~~
   **Resolved 2026-10-14: hide** (see the L2 entry) — removed from
   CORE_PORT_TYPES and the frontend mirror; returns with its producer.
5. **M2** — default run deadline, and whether it can be set per node.
   **Resolved 2026-10-13**: a default deadline (600 s) editable in
   Settings, enforced per node; heavy nodes declare their own budget.
6. **M14** — resolved: ESLint and Vitest (see P0.1 and P0.2).
7. ~~**M6** — delete the dead `compute_fc`, or keep it as a delegate to the
   shipped path?~~
   **Resolved 2026-10-11: deleted** (see the M6 entry) — fixtures moved to
   `compute_fc_seconds`, which delegates to the shipped path.
8. ~~**L17** — un-ignore `docs/issues.md` so the register is version-controlled,
   or keep it local?~~
   **Resolved 2026-10-15: keep local until cleared** (user decision) — the
   original rationale stands; the L17 entry records the caveat to handle
   at un-ignore time.
9. ~~**L14** — may the ~503 MB of local build outputs be deleted?~~
   **Resolved 2026-10-14: deleted** (see the L14 entry).

### Phase 0 decisions (resolved 2026-10-10)

- **P0.1 Frontend static checks**: `tsc -b` plus ESLint (typescript-eslint and
  the react-hooks rules). No formatter for now; a formatter would reformat
  the whole frontend and gets its own decision later.
- **P0.2 Frontend unit tests**: Vitest now, for `validateConnection`,
  `connectionCreatesCycle`, `resolveInherited` and `enforceWiringRules`.
- **P0.3 Type-check scope**: a second tsconfig covers the e2e and config
  files; `tsc -b` covers all TypeScript.
- **P0.4 Branching**: push straight to `main` for now; switch to a branch and
  pull request once the code base is mature. There is no remote yet
  (`git remote -v` is empty), so nothing has been pushed.
- **P0.5 M16 split**: as in the Phase 0 section below.
- **P0.6 Red tests**: committed together with the fix, one commit per issue.
  The commit message records the red result. A red commit on `main` would
  fail CI, so red tests never go in alone.
- **P0.7 Frontend verification**: run the Playwright suite locally before each
  frontend commit (`npx playwright install chromium` once). CI is not yet
  running on our commits.
- **P0.8 Tools**: the staging and verification tools live in `scripts/dev/`
  (`stage_block.py`, `verify_index.sh`), committed as `afe96f4`.

### Phase 0 — test safety net

- **M16** (this phase, done 2026-10-10 in `447c7bb`): WebSocket delivery under
  the lifespan; contract keys
  (`cacheable`, `invalidates_inputs`, `suggest_from`); environment isolation
  (autouse `monkeypatch.delenv` for `EMPTYUI_*`); race-free status checks.
- **M16** (deferred to the fix that needs them): the `xml`, `avg` and `j` TF
  exports with L8 (Phase 1); the double-Calibrate test with M8 (done in the M8 fix); the
  concurrency test (shared cache entries, one shared `Node`) with M2 and M4
  (Phase 2).
- **M14** (part 1, done in `f666b3a`): `tsc -b` covering `src`, e2e and config (P0.3); ESLint
  (P0.1); Vitest (P0.2). Add a `web` CI job (`npm ci`, `tsc -b`, lint, Vitest,
  `vite build`) once a remote exists (P0.4). The rest of M14 (this CI job and the e2e holes) is deferred to Phase 4 by user decision.

### Phase 1 — numerical correctness (Python)

Order: M8 → M7 → M3 → M6 → L8 → L7 → L10 → L11 audit.

- **M8** (done 2026-10-10, see the M8 entry): refuse an already-calibrated payload; tests for
  Calibrate → Calibrate and Calibrate → Compute FC.
- **M7** (done 2026-10-10, see the M7 entry): express the diagnostic's band in seconds, not bins; test at 150 Hz.
- **M3**: try/finally on the collection route; test that the handle closes
  when `persist_fc_level` raises.
- **M6**: (a) DONE 2026-10-10 — align the library defaults with DESIGN §10.1 (part b of the M6 entry); (b) delete or
  delegate `compute_fc` and move its fixtures onto the shipped path. Separate
  commit — it touches about 17 test modules.
- **L8**: opt-out for the EDI→SI conversion; default unchanged.
- **L7**: make `print_params` defaults agree between manifest and method.
- **L10**: correct the DESIGN §9.4 arithmetic after checking the manifest
  defaults.
- **L11 audit**: the diurnal-prior run start and the dropped-session bounds
  read window starts; decide the spectrogram axis; test on a synthetic
  deep-level session.

### Phase 2 — executor and server safety (Python)

Order: ~~M1~~ (done 2026-10-12) → ~~M2, M4, M17, M20, M21~~ (done 2026-10-13) →
~~L4, L5, L6, L3 (backend)~~ (done 2026-10-14). **Phase 2 complete.**

- **M1** (done 2026-10-12, see the entry): param-value validation at the
  schema choke point — the trust boundary for remote deployment.
- **M2** (done 2026-10-13, see the entry): per-node watchdog with a
  settings-editable default (600 s) and per-node declared overrides.
- **M4** (done 2026-10-13, see the entry): cache-hit artifact paths
  rewritten to this run's workspace copy (shallow copy; entry untouched).
- **M17** (done 2026-10-13, see the entry): the TTL reclaim runs on a
  worker thread; failed deletions are logged.
- **M20** (done 2026-10-13, see the entry): duplicate-type plugins
  degrade to an error row; partial registrations roll back; failed
  plugins reach GET /api/plugins.
- **M21** (done 2026-10-13, see the entry): the coherence sort's
  transient is budget-bounded via haloed window chunks (5.6 GiB →
  1.1 GiB at the 12 h/150 Hz shape; the register's numbers-first rule
  honored).
- **H7** (done 2026-10-13, see the entry, user report): calibration
  absence warns on the node via the new ctx.warn → node_warning channel
  (card marker + properties-panel list); the no-ASC path is no longer
  silent. Other loaders can adopt ctx.warn the same way.
- **M24** (done 2026-10-13, see the entry, user report): Align Runs'
  inputs declare collection_ok — Extract Run's 'time range' payloads
  wire into it again (the backend always supported collections; the
  manifest lagged).
- **M4**: store a shallow copy with the rewritten path. Test with the producer
  workspace reclaimed mid-run.
- **M17**: run cleanup in a thread executor and log failures. Test that the
  event loop is not blocked.
- **L4**: when capping events, drop middle events so `run_started` survives.
- **L5**: byte budget alongside the entry count (`cache_max_gb`).
- **L6**: fsync and atomic write; map OS errors to proper HTTP responses.
- **M20**: per decision 1.
- **M21**: measure peak memory on a synthetic 12-hour 150 Hz FC first; then
  chunk over windows or add a budget warning.
- **L3** (backend): normalise the handle; stop leaking absolute paths.

### Phase 3 — frontend (packages/web)

Start with the Phase 0 gates, so every later change is type-checked.

- **M12**: keep viewers and time bars during a re-run.
- **M15**: reconnect after `offline`; surface polling errors; render
  `node_progress`.
- **L3** (frontend): DONE — `resolveInputArtifact` (graph/artifacts.ts)
  matches the edge's (source, sourceHandle, targetHandle) against the
  artifact's (nodeId, handle); the "last artifact of the source" fallback
  is gone. Writing it exposed the REAL defect: fresh runs emitted the
  store's handle ID ("h2") as `handle` — never equal to a port name — so
  the old resolver's primary branch was dead code and every viewer rode
  the fallback. Both emit paths now carry the OUTPUT PORT NAME (the
  router key SS 7.2 promises); the restore path's flaky img e2e failure
  was the same defect surfacing. Tests: artifacts.test.ts (5 pins) +
  test_artifact_handle_names_the_output_port (fresh + restored, red
  first) + the updated restore contract in test_cache.py. Full gates:
  pytest 734, vitest 30, Playwright 31.
- **M18**: subscribe to only the fields each node needs. Measure a 30-node
  graph before and after.
- **M13**: take node, category and port names from the manifests.
- **L12**, **L1**: assorted defects and accessibility, with accessibility
  assertions added to the e2e suite.
- **L2**, **M19**: per decisions 4 and 2.
- **L13** (backend): deterministic plugin ordering for `port_types`.

### Phase 4 — contract, docs and hygiene

- **M14** (deferred from Phase 0 by user decision): add the `web` CI job once a
  remote exists (P0.4); close the e2e coverage holes (accessibility assertions
  go with L1 in Phase 3).
- **M23**: per decision 3, implement each item or mark it planned in DESIGN.
- **M5**: consolidate shared constants, one duplicate group per commit; only
  once Phase 0 is in place.
- **L9**: documentation drift, including the bare `§13+` references —
  mechanical, separate commit.
- **L15**: a burst demo workflow against the tracked `smoke` archive.
- **L14**, **L16**, **L17**: cleanup, per decisions 8 and 9.

### Effort and risks

Rough effort for one developer: Phase 0 ≈ 1 day, Phase 1 ≈ 3–4 days (M6 is
the largest item), Phase 2 ≈ 5 days, Phase 3 ≈ 5–6 days, Phase 4 ≈ 3 days.

- M6 and M5 change many tests at once: do them after the safety net, each in
  its own commit.
- M18 and M21: measure before and after; make no speed or memory claim
  without numbers.
- M2 interacts with cancellation and the admission slot: test it carefully.

## Verification commands used

```bash
.venv/bin/python -m pytest packages -q            # 643 passed after H1..H5 (630 at the 2026-09-29 review, ~87 s)
.venv/bin/python -m pytest packages -q -m "not errors"   # 570 passed (happy path group)
.venv/bin/python -m pytest packages -q -m errors         # 73 passed (edge-case group)
.venv/bin/ruff check .                             # repo-wide 0 (pinned 0.16.4)
cd packages/web && npm run build                  # tsc -b && vite build, clean
cd packages/web && npx playwright test --list      # 31 tests in 10 files
cd packages/web && npx playwright test             # full-stack e2e
git clone <repo> /tmp/fresh && cd /tmp/fresh       # the ONLY honest fresh-clone sim
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e "packages/emptyui[dev]" -e … \
  -e "examples/plugin-template[dev]"
.venv/bin/python -m pytest packages -q -m "not errors"   # 544 passed, 6 corpus-guarded skips (fresh-clone run)
.venv/bin/python -m pytest packages -q -m errors         # 68 passed (fresh-clone run)
git check-ignore -v <file>                         # smoke/SDG31b trackable; day_12h/h5/sdg31-pngs ignored
```

### 2026-10-25 review

```bash
.venv/bin/python -m pytest packages -q -m "not errors"   # 727 passed, 74 deselected (~72 s)
.venv/bin/python -m pytest packages -q -m errors         # 74 passed
.venv/bin/ruff check .                                   # 0
cd packages/web && npm run typecheck                     # tsc -b, clean
cd packages/web && npm run lint                          # eslint, clean
cd packages/web && npm run test:unit                     # vitest: 38 passed (6 files)
cd packages/web && npx playwright test --list            # 36 tests in 11 files
```