"""Merge Runs node (DESIGN.md §3 'Merging multiple runs'; history §24.5 P1 rebuild).

Combines several acquisition runs of one station (e.g. SDH28 'b' + 'c') into
a single ``ts.run`` so a downstream Align Runs sees the station's full
coverage — the parity recipe's dominant short-period factor (
history §24.3 E1b: aligning against a partial remote cropped away ~29% of the local
run). The run list comes from the node parameters (like Extract Run's ids);
the dead ``selection``-payload path was removed (its producer node was
never delivered, M4-deferred).

Physical merge bridges inter-run gaps by interpolation (``gap_method``) —
fabricated samples stay limited to the gap (~0.2% of windows for typical
run gaps). The production-grade alternative — concatenating Fourier
coefficients per decimation level (aurora ``merge_stfts``; no sample
fabrication, the gap simply produces no windows) — is
``emptyui_aurora._processing.merge_fc_containers``, the engine behind
Compute FC's run-collection route (the ``Merge FC`` node was retired
with the Build FC path).
"""

from __future__ import annotations

import logging
from typing import Any

from emptyui import Input, Node, Output, Param
from emptyui_core_nodes.types import MASK_CHANNELS

from .runs import _resolve_run, _resolve_station, _resolve_survey

_log = logging.getLogger("emptyui")


def merge_runts(run_ts_list, gap_method: str = "slinear",
                new_run_id: str = "merged", logger=None):
    """Physically concatenate mth5 RunTS objects (in order) into one series.

    Runs must share the sample rate; inter-run gaps are bridged with
    ``gap_method`` interpolation (mt_timeseries ``RunTS.merge``). The merged
    run's id is set to ``new_run_id`` (station/survey metadata come from the
    first run). Returns the merged RunTS.

    §3 constraints made loud: runs sharing channel components is required
    (RunTS.merge cannot combine unequal sets) and OVERLAPPING acquisitions
    are rejected (the merge cannot combine non-monotonic time indexes, and
    the overlap would be interpolated over — fabricated samples). The
    FC-level alternative (Merge FC) concatenates without fabrication.
    """
    import logging

    _log = logger or logging.getLogger("emptyui")
    if not run_ts_list:
        raise ValueError("no runs to merge")
    rates = {float(rt.sample_rate) for rt in run_ts_list}
    if len(rates) > 1:
        raise ValueError(
            f"cannot merge runs of mixed sample rates: {sorted(rates)} — "
            "all runs of one station must share the sampling")
    # §3: channel components must be consistent — RunTS.merge cannot
    # combine runs with unequal channel sets (xarray MergeError), so this
    # is a hard error with an actionable message, not a warning
    def _rt_channels(rt):
        chans = getattr(rt, "channels", None)
        if chans is not None:
            return set(map(str, chans))
        return {str(v) for v in rt.dataset.data_vars}

    comp_sets = [_rt_channels(rt) for rt in run_ts_list]
    union = set.union(*comp_sets)
    # quality-mask aux channels (MASK_CHANNELS, M26: 'saturated' +
    # 'quality_mask') exist only on runs with flagged samples — a station's
    # sessions split into masked and unmasked runs, and merging them (the
    # documented E1b remedy: "merge the remote station's runs first") must
    # not hard-fail on QC metadata: pad the missing members with zeros
    # (no flagged samples there)
    aux = set(MASK_CHANNELS)
    for rt, comps in zip(run_ts_list, comp_sets):
        missing = sorted((union - comps) - aux)
        if missing:
            raise ValueError(
                f"cannot merge run '{_rt_id(rt)}': it lacks channel(s) "
                f"{missing} that the other runs carry — all runs of one "
                "station must share the channel components (§3)")
    run_ts_list = [_pad_aux(rt, union & aux) for rt in run_ts_list]
    # §3: OVERLAPPING acquisitions cannot be combined — RunTS.merge's
    # index combination fails on non-monotonic global time indexes (the
    # overlap would also be interpolated over, fabricating samples).
    # Reject with an actionable message instead of the raw xarray error.
    # GAPS between runs are bridged by interpolation — the provenance of
    # those fabricated stretches is recorded on the merged payload
    # (merged_gap_intervals / merged_source_runs) so the remote-reference
    # pairing can warn when a local run's windows fall inside them
    # (M4 #2: a local run spanning multiple remote runs silently pairs
    # with interpolated remote samples).
    prev_end = None
    gaps: list[list[str]] = []
    # L-M4: naive/aware spans cannot be compared (raw TypeError) — when the
    # merge set MIXES conventions, localize the naive ones to UTC (the
    # archive convention); an all-naive merge keeps its naive spans (the
    # recorded gap intervals stay offset-free, as before)
    spans: list[tuple | None] = []
    for rt in run_ts_list:
        try:
            spans.append(_rt_span(rt))
        except Exception:  # noqa: BLE001 — spans unavailable on some payloads
            spans.append(None)
    aware = any(s is not None and s[0].tzinfo is not None for s in spans)
    if aware:
        spans = [
            None if s is None else (
                s[0].tz_localize("UTC") if s[0].tzinfo is None else s[0],
                s[1].tz_localize("UTC") if s[1].tzinfo is None else s[1])
            for s in spans]
    for rt, span in zip(run_ts_list, spans):
        if span is None:
            prev_end = None
            continue
        start, end = span
        if prev_end is not None and start < prev_end:
            raise ValueError(
                f"cannot merge: run '{_rt_id(rt)}' starting {start} overlaps "
                f"the previous run's end {prev_end} — overlapping "
                "acquisitions cannot be concatenated (check the run "
                "selection; Merge FC concatenates Fourier coefficients "
                "without fabricating samples)")
        if prev_end is not None and start > prev_end:
            gaps.append([prev_end.isoformat(), start.isoformat()])
        prev_end = end
    merged = run_ts_list[0]
    for rt in run_ts_list[1:]:
        merged = merged.merge(rt, gap_method=gap_method)
    try:
        merged.run_metadata.id = new_run_id
    except Exception:  # noqa: BLE001, S110 — duck-typed payloads without run metadata
        pass
    try:
        # provenance survives on the dataset (RunTS wrappers rebuild the
        # wrapper object, not the xarray payload). Chained merges
        # (merge(merge(a, b), c)) FLATTEN the inner provenance instead of
        # overwriting it: the outer payload keeps the inner merge's source
        # list + gap intervals and appends the fresh members (L-M4 — the
        # overwrite lost the original run identities)
        sources = list(merged.dataset.attrs.get("merged_source_runs") or [])
        fresh = ([_rt_id(rt) for rt in run_ts_list]
                 if not sources else
                 [_rt_id(rt) for rt in run_ts_list[1:]])
        merged.dataset.attrs["merged_source_runs"] = sources + fresh
        if gaps:
            prior = list(merged.dataset.attrs.get("merged_gap_intervals") or [])
            method = merged.dataset.attrs.get("merged_gap_method")
            merged.dataset.attrs["merged_gap_intervals"] = prior + gaps
            if method and method != str(gap_method):
                _log.warning(
                    "chained merge mixes gap methods (%s then %s) — the "
                    "recorded method is the LAST one applied",
                    method, gap_method)
            merged.dataset.attrs["merged_gap_method"] = str(gap_method)
    except Exception as exc:  # noqa: BLE001 — duck-typed payloads without dataset attrs
        _log.debug("merge_runts: merge provenance attrs not recorded (%s)", exc)
    return merged


def _rt_id(rt):
    try:
        return str(rt.run_metadata.id)
    except Exception:  # noqa: BLE001 — duck-typed payloads
        return "<unnamed>"


def _pad_aux(rt, aux_comps):
    """A copy of ``rt`` with all-zero QC aux channels for the ones it lacks
    (physically correct: no flagged samples there) so a mixed
    masked/unmasked merge can proceed; returns ``rt`` unchanged otherwise."""
    if not aux_comps:
        return rt
    import xarray as xr
    from mt_timeseries import RunTS

    missing = [c for c in aux_comps if c not in rt.dataset.data_vars]
    if not missing:
        return rt
    ds = rt.dataset.copy()
    sample = next(iter(rt.dataset.data_vars.values()))
    for c in missing:
        # zeros_like copies the template channel's attrs (incl. `filters`
        # refs) — the fabricated mask must not carry them: downstream
        # RunTS.calibrate() divides every channel by its response chain
        ds[c] = xr.zeros_like(sample)
        ds[c].attrs = {"type": "auxiliary", "units": "1"}
    out = RunTS(ds)
    for attr in ("run_metadata", "station_metadata", "survey_metadata"):
        md = getattr(rt, attr, None)
        if md is not None:
            setattr(out, attr, md.copy() if hasattr(md, "copy") else md)
    return out


def _rt_span(rt):
    """(start, exclusive end) datetimes of a RunTS payload; None without an
    absolute time axis (overlap checks need absolute time). The exclusive
    end (last sample + 1/sr) is what gap/overlap arithmetic needs: a run
    starting exactly at another's end is contiguous, not overlapping. The
    stamps keep the payload's native convention (naive/aware) — the merge
    normalizes when the set mixes (L-M4)."""
    import numpy as np
    import pandas as pd

    ts = rt.dataset["time"].values
    if not np.issubdtype(ts.dtype, np.datetime64):
        return None
    idx = pd.DatetimeIndex(ts)
    try:
        sr = float(rt.sample_rate)
    except Exception:  # noqa: BLE001 — duck-typed payloads without a rate
        sr = 0.0
    end = idx[-1] + (pd.Timedelta(seconds=1.0 / sr) if sr > 0
                     else pd.Timedelta(seconds=1))
    # NB: stamps return in the payload's NATIVE convention — the mixed
    # naive/aware reconciliation happens in merge_runts (localizing here
    # would rewrite every recorded gap interval to +00:00)
    return idx[0], end


class MergeRuns(Node):
    type = "mth5.runs.merge"
    # raw-data/FC-scale work: a generous own budget (issues.md M2)
    timeout = 3600.0
    title = "Merge Runs"
    tooltip = "Combine several acquisition runs of one station into a single time series."
    description = (
        "Combine several acquisition runs of one station (e.g. runs 'b,c') "
        "into a single time series so a downstream Align Runs sees the "
        "station's full coverage — remote-reference alignment against a "
        "partial remote otherwise crops away local data (history §24.3 E1b). "
        "Runs are concatenated in the given order; inter-run gaps are "
        "bridged by interpolation (gap_method). Resolve ids like Extract "
        "Run: empty survey/station fall back to the first available."
    )
    category = "Processing"
    inputs = (Input("mth5", "mth5"),)
    outputs = (Output("merged_run", "ts.run", label="merged time series"),)
    params = (
        Param("survey", "string", label="Survey id", default="", optional=True),
        Param("station", "string", label="Station id", default="",
              optional=True, placeholder="auto (first available)"),
        Param("runs", "string", label="Runs", default="", optional=True,
              placeholder="e.g. b,c (empty = all runs)",
              description="Comma-separated run ids in acquisition order "
                          "(e.g. 'b,c'); empty = every run of the station "
                          "in archive order."),
        Param("gap_method", "select", label="Gap method",
              options=("slinear", "linear", "nearest", "zero"),
              default="slinear", optional=True,
              description="Bridging interpolation across inter-run gaps."),
        Param("new_run_id", "string", label="New run id", default="merged",
              optional=True),
    )

    def execute(self, ctx, mth5, survey: str = "", station: str = "",
                runs: str = "", gap_method: str = "slinear",
                new_run_id: str = "merged") -> dict[str, Any]:
        survey_obj = _resolve_survey(mth5, survey)
        station_obj = _resolve_station(survey_obj, station)
        available = [str(x) for x in station_obj.run_summary["id"]]
        wanted = ([r.strip() for r in str(runs).split(",") if r.strip()]
                  if str(runs).strip() else available)
        missing = [r for r in wanted if r not in available]
        if missing:
            raise ValueError(
                f"run(s) {missing} not in station archive {available}")
        # duplicates merge silently-wrong (the same run twice inflates its
        # weight) — dedup preserving order, with a warning
        seen: set[str] = set()
        deduped = []
        for r in wanted:
            if r in seen:
                ctx.logger.warning(
                    "run '%s' listed more than once — merging it once", r)
                continue
            seen.add(r)
            deduped.append(r)
        wanted = deduped
        run_ts_list = []
        from .runs import (
            _wrap_runts,
            read_run_channels,
            resolve_filter_registry,
        )
        for run_id in wanted:
            run_obj = _resolve_run(station_obj, run_id)
            # fast channel read (history §24.8): to_runts's mt_metadata
            # deepcopy/validation storm costs ~7-19 s PER RUN — the merge
            # of a burst archive's slots would pay it for every run
            data_vars, sample_rate, _start = read_run_channels(run_obj)
            registry = resolve_filter_registry(survey_obj, data_vars)
            run_ts_list.append(_wrap_runts(
                run_obj, station_obj, survey_obj, data_vars, sample_rate,
                registry))
        merged = merge_runts(run_ts_list, gap_method=gap_method,
                             new_run_id=new_run_id, logger=ctx.logger)
        # RunTS.merge rebuilds its result from channel arrays — the filter
        # registry does not travel; re-resolve it from the merged dataset's
        # channel attrs (history §24.12: the calibration association must survive)
        try:
            merged.filters = resolve_filter_registry(
                survey_obj, {n: merged.dataset[n]
                             for n in merged.dataset.data_vars})
        except Exception:  # noqa: BLE001 — identity is best effort
            _log.debug("merged run filter registry not re-resolved")
        ctx.progress(
            1.0,
            f"merged {len(wanted)} run(s) ({', '.join(wanted)}): "
            f"{merged.dataset.sizes['time']} samples",
        )
        return {"merged_run": merged}
