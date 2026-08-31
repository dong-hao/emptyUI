"""Time-domain quality masking — the Mask TS node (docs/DESIGN.md §9.5,
docs/issues.md M26).

The node MARKS, never touches: a per-sample binary ``quality_mask``
auxiliary channel rides the payload, and Compute FC's window-fraction
machinery (§10.1) turns the mask into window rejection at STFT time.
The mask is DATA — it rides the graph (aux channels pass the core filter
nodes untouched), survives the mth5 round trip, and is the interface a
learned-despiker plugin would write (M4 #16/#19).

Criteria are pooled (one mask for all channels, v1): per-channel flags
ORed. Repair/replacement is deliberately a separate future node — this
node never fabricates samples. Established-method basis: the Hampel
identifier (rolling median ± h·1.4826·MAD — self-calibrating, no
absolute amplitude decisions), constant-run detection (scale-free exact
equality — catches ADC clips, zero-filled gaps and stuck sensors), and
explicit UTC interval exclusion (the user's override). True time-axis
gaps carry no samples to flag — they are reported and warned instead.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from emptyui import Input, Node, Output, Param
from emptyui_core_nodes.types import (
    MASK_CHANNELS,
    TimeSeries,
    coerce_run,
    is_aux_channel,
    is_run_collection,
)

_log = logging.getLogger("emptyui")

#: seconds — the auto Hampel window cap for continuous records
_HAMPEL_AUTO_CAP_S = 60.0
#: the auto window is at most an eighth of the member's span ...
_HAMPEL_AUTO_SPAN_DIV = 8.0
#: ... and never below this many samples (a median/MAD needs enough
#: samples to be robust; a member too short for even this skips Hampel)
_HAMPEL_MIN_SAMPLES = 16


def _parse_intervals(text: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Parse ``start/end`` ISO pairs separated by ``;`` — the user's manual
    override. Raises with the offending fragment named, showing the
    expected form (2026-10-16: the user preferred the strict rule over a
    forgiving parser — 'start ; end' stays an error)."""
    example = ("2024-01-01T00:00/2024-01-01T02:00 ; "
               "2024-01-05T12:00/2024-01-05T14:00")
    out = []
    for i, frag in enumerate(str(text or "").split(";"), 1):
        frag = frag.strip()
        if not frag:
            continue
        if "/" not in frag:
            raise ValueError(
                f"exclude_intervals fragment {i} ({frag!r}) is not a "
                "'start/end' pair — write ISO datetime pairs separated by "
                f"';', e.g. '{example}'")
        a, b = (s.strip() for s in frag.split("/", 1))
        try:
            lo, hi = pd.Timestamp(a), pd.Timestamp(b)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"exclude_intervals fragment {i} has an unparseable "
                f"datetime ({frag!r}) — expected 'start/end' pairs "
                f"separated by ';', e.g. '{example}'") from exc
        if hi <= lo:
            raise ValueError(
                f"exclude_intervals fragment {i}: end {b!r} is not after "
                f"start {a!r}")
        out.append((lo, hi))
    return out


def _hampel_flags(x: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """The Hampel identifier: flag samples more than ``threshold``
    scaled-MADs from the rolling median. NaN samples are flagged
    unconditionally (pandas' rolling statistics skip them)."""
    s = pd.Series(np.asarray(x, dtype=float))
    w = int(max(3, window))
    mp = max(3, w // 2)
    med = s.rolling(w, center=True, min_periods=mp).median()
    dev = (s - med).abs()
    mad = dev.rolling(w, center=True, min_periods=mp).median()
    flags = dev > (float(threshold) * 1.4826 * mad)
    flags |= ~np.isfinite(s.values)
    return np.asarray(flags.fillna(False), dtype=bool)


def _flat_run_flags(x: np.ndarray, min_run: int) -> np.ndarray:
    """Flag runs of >= ``min_run`` IDENTICAL consecutive values — exact
    equality is scale-free: ADC clips repeat exactly, zero-filled gaps
    repeat exactly, and a natural signal never does."""
    x = np.asarray(x, dtype=float)
    flags = np.zeros(x.size, dtype=bool)
    if x.size < max(2, int(min_run)):
        return flags
    same = np.zeros(x.size, dtype=bool)
    same[1:] = (x[1:] == x[:-1])  # NaN != NaN -> False: gaps not flagged here
    idx = np.flatnonzero(same)
    if idx.size == 0:
        return flags
    breaks = np.flatnonzero(np.diff(idx) != 1)
    groups = np.split(idx, breaks + 1)
    need = int(min_run) - 1  # a run of n equal samples has n-1 "same" pairs
    for g in groups:
        if g.size >= need:
            # g holds the "same" pair positions (i where x[i]==x[i-1]) —
            # the run itself starts one earlier and ends at g[-1]
            flags[max(g[0] - 1, 0):g[-1] + 1] = True
    return flags


def _auto_hampel_window(n_samples: int, sample_rate: float) -> int | None:
    """The per-member auto window: min(60 s, span/8), floored at
    ``_HAMPEL_MIN_SAMPLES`` — None when the member is too short for a
    meaningful Hampel at all."""
    if n_samples <= 0 or sample_rate <= 0:
        return None
    w = min(_HAMPEL_AUTO_CAP_S, n_samples / sample_rate / _HAMPEL_AUTO_SPAN_DIV)
    w = max(round(w * sample_rate), _HAMPEL_MIN_SAMPLES)
    if w >= n_samples:
        return None  # the window would cover the whole member — degenerate
    return w


class MaskTS(Node):
    type = "mth5.qc.mask_ts"
    title = "Mask TS"
    tooltip = "Mark bad samples (spikes, flat runs, gaps, intervals) in a quality_mask auxiliary channel."
    description = (
        "Time-domain quality gating (§9.5): write/merge a per-sample "
        "binary 'quality_mask' auxiliary channel — 1 = bad. The mask is "
        "DATA: it rides the payload and Compute FC turns it into FC-window "
        "rejection (its window-fraction rule). Criteria are pooled across "
        "channels and OR-composed: explicit UTC interval exclusion, Hampel "
        "despiking (rolling median vs h·1.4826·MAD — self-calibrating, no "
        "absolute amplitudes), constant-run detection (ADC clips, "
        "zero-filled gaps, stuck sensors — exact equality is scale-free) "
        "and NaN/no-data flagging. True time-axis gaps carry no samples to "
        "flag — they are reported and warned instead (add them to "
        "'exclude_intervals'). The node never modifies data channels and "
        "never fabricates samples; repair is a separate node. Accepts a "
        "run collection (maps member-wise; the collection shape is "
        "preserved)."
    )
    category = "Processing"
    # O(N log N) rolling statistics on the native-rate series
    timeout: ClassVar[float | None] = 600.0
    inputs = (Input("data", "ts.run", label="time series",
                    accepts=("ts.runs",), collection_ok=True),)
    outputs = (Output("gated", "ts.run", label="masked time series"),
               Output("report", "table", label="mask report"))
    params = (
        Param("exclude_intervals", "text", label="Exclude intervals (UTC)",
              default="", optional=True,
              placeholder="2024-01-01T00:00/2024-01-01T02:00; …",
              description="'start/end' ISO datetime pairs separated by ';' — "
                          "marked unconditionally (clamped to the record; "
                          "out-of-record intervals are warned, never "
                          "rejected). The manual override — the only "
                          "criterion that cannot get it wrong."),
        Param("detect_spikes", "boolean", label="Detect spikes (Hampel)",
              default=True,
              description="Hampel identifier per channel: flag samples "
                          "farther than h scaled-MADs from the rolling "
                          "median — robust and self-calibrating (no "
                          "absolute amplitude decisions). Pooled across "
                          "channels."),
        Param("hampel_window_s", "number", label="Hampel window (s)",
              default=0.0, min=0.0, optional=True,
              description="Rolling-window length in seconds; 0 = auto — "
                          "min(60 s, span/8) per member with a 16-sample "
                          "floor, so burst slots get slot-local windows "
                          "while hour-long streams keep the 60 s default. "
                          "An explicit value is honored (clamped to the "
                          "member's span, warned)."),
        Param("hampel_threshold", "number", label="Hampel threshold (MADs)",
              default=8.0, min=0.5,
              description="Detection threshold in scaled MADs "
                          "(1.4826·MAD estimates the std of Gaussian "
                          "data); higher = only more extreme outliers. "
                          "8 is the MT-tuned default: natural-source "
                          "time series are heavy-tailed, and the "
                          "literature's Gaussian h=3 over-gates them "
                          "(measured on the smoke_2h corpus: h=3 masks "
                          "~5% of samples and collapses the FC band "
                          "coverage; h>=5 is band-neutral)."),
        Param("detect_flat", "boolean", label="Detect constant runs",
              default=True,
              description="Flag runs of identical consecutive values — "
                          "exact equality is scale-free: catches ADC "
                          "clipping, zero-filled gaps and stuck sensors "
                          "without knowing the full scale."),
        Param("flat_run_min", "integer", label="Constant-run length",
              default=8, min=2,
              description="Minimum identical consecutive samples that "
                          "constitute a constant run."),
        Param("detect_gaps", "boolean", label="Detect gaps",
              default=True,
              description="NaN values are flagged (they are samples). "
                          "True time-axis gaps carry no samples — they "
                          "are reported and warned with the remedy "
                          "(exclude them via 'Exclude intervals')."),
        Param("union_existing", "boolean", label="Union existing masks",
              default=True,
              description="OR any existing 'saturated'/'quality_mask' "
                          "channels into the new mask (layered criteria "
                          "compose); off = start fresh from this node's "
                          "criteria."),
    )

    # --- criteria ---------------------------------------------------------

    @staticmethod
    def _member_mask(channels: dict[str, np.ndarray],
                     time_index: pd.DatetimeIndex | None,
                     sample_rate: float,
                     exclude_intervals: str,
                     detect_spikes: bool, hampel_window_s: float,
                     hampel_threshold: float,
                     detect_flat: bool, flat_run_min: int,
                     detect_gaps: bool,
                     ctx) -> tuple[np.ndarray, list[dict], list[str]]:
        """One member's pooled mask + report rows + axis-gap interval
        strings. ``channels`` EXCLUDES the mask/aux channels (they are
        criteria input, not data)."""
        rows: list[dict] = []
        gap_notes: list[str] = []
        n = max((v.size for v in channels.values()), default=0)
        pooled = np.zeros(n, dtype=bool)
        deliberate = np.zeros(n, dtype=bool)  # the user's exclude_intervals

        # 1. explicit UTC intervals (clamped; out-of-record warned)
        intervals = _parse_intervals(exclude_intervals)
        if intervals and time_index is None and (
                exclude_intervals or "").strip():
            ctx.logger.warning(
                "exclude_intervals: the member has no absolute time axis — "
                "UTC intervals cannot be applied (use sample-relative "
                "means or fix the record's start time)")
        if intervals and time_index is not None and n:
            t0 = time_index[0]
            t_end = time_index[-1] + pd.to_timedelta(
                1.0 / (sample_rate or 1.0), unit="s")
            for lo, hi in intervals:
                # reconcile tz: aware record + naive bounds localize to the
                # record's zone (the archive convention is UTC)
                if lo.tzinfo is None and t0.tzinfo is not None:
                    lo, hi = lo.tz_localize(t0.tz), hi.tz_localize(t0.tz)
                elif lo.tzinfo is not None and t0.tzinfo is None:
                    lo, hi = lo.tz_localize(None), hi.tz_localize(None)
                if hi <= t0 or lo >= t_end:
                    ctx.logger.warning(
                        "exclude_intervals: %s/%s lies entirely outside "
                        "the record (%s .. %s) — ignored", lo, hi, t0, t_end)
                    continue
                clo, chi = max(lo, t0), min(hi, t_end)
                # searchsorted on the index itself — unit-agnostic (raw
                # .asi8 ticks are coordinate-precision-dependent: us vs ns)
                i0 = int(time_index.searchsorted(clo, side="left"))
                i1 = int(time_index.searchsorted(chi, side="left"))
                pooled[i0:i1] = True
                deliberate[i0:i1] = True
                rows.append({
                    "criterion": "interval", "n_flagged": int(i1 - i0),
                    "fraction": round((i1 - i0) / max(n, 1), 6),
                    "intervals": f"{clo.isoformat()}/{chi.isoformat()}",
                    "note": ""})

        # 2. Hampel despiking (per channel, pooled) with the auto window
        if detect_spikes and channels:
            if hampel_window_s and hampel_window_s > 0:
                w = round(float(hampel_window_s) * (sample_rate or 0.0))
                auto = False
                if w and w >= n:
                    ctx.logger.warning(
                        "hampel_window_s=%g covers the whole member "
                        "(%d samples) — clamping to it (a windowed "
                        "median degenerates to the global one)",
                        hampel_window_s, n)
            else:
                w = _auto_hampel_window(n, sample_rate)
                auto = True
            if w and w < n:
                total = 0
                for x in channels.values():
                    f = _hampel_flags(x, w, hampel_threshold)
                    total += int(f.sum())
                    pooled |= f
                rows.append({
                    "criterion": "hampel", "n_flagged": total,
                    "fraction": round(total / max(n * len(channels), 1), 6),
                    "intervals": "",
                    "note": (f"window {w} samples (auto)"
                             if auto else f"window {w} samples")})
            elif detect_spikes:
                rows.append({
                    "criterion": "hampel", "n_flagged": 0, "fraction": 0.0,
                    "intervals": "",
                    "note": "skipped: member too short for a meaningful "
                            f"rolling median (<{_HAMPEL_MIN_SAMPLES} "
                            "samples or window >= span)"})

        # 3. constant runs (clips, zero-filled gaps, stuck sensors)
        if detect_flat and channels:
            total = 0
            for x in channels.values():
                f = _flat_run_flags(x, flat_run_min)
                total += int(f.sum())
                pooled |= f
            rows.append({
                "criterion": "flat_run", "n_flagged": total,
                "fraction": round(total / max(n * len(channels), 1), 6),
                "intervals": "", "note": f">= {flat_run_min} identical samples"})

        # 4. NaN values are samples — flag them; axis gaps are not
        if detect_gaps and channels:
            nan_total = 0
            for x in channels.values():
                f = ~np.isfinite(np.asarray(x, dtype=float))
                nan_total += int(f.sum())
                pooled |= f
            if nan_total:
                rows.append({
                    "criterion": "nan", "n_flagged": nan_total,
                    "fraction": round(nan_total / max(n * len(channels), 1), 6),
                    "intervals": "", "note": ""})
            if time_index is not None and n > 1 and sample_rate > 0:
                # unit-agnostic: compare Timedeltas, never raw int ticks
                dts = (time_index[1:] - time_index[:-1])
                bad = np.flatnonzero(
                    np.asarray(dts) > np.timedelta64(
                        int(np.ceil(1.5 / sample_rate * 1e9)), "ns"))
                for i in bad:
                    lo = time_index[i]
                    hi = time_index[i + 1]
                    gap_notes.append(f"{lo.isoformat()}/{hi.isoformat()}")
                    ctx.logger.warning(
                        "time-axis gap %s/%s carries no samples to flag — "
                        "add it to 'exclude_intervals' so Compute FC's "
                        "windows cannot span it", lo, hi)
            if gap_notes:
                rows.append({
                    "criterion": "axis_gap", "n_flagged": 0,
                    "fraction": 0.0, "intervals": "; ".join(gap_notes),
                    "note": "no samples exist — excluded via "
                            "'exclude_intervals' by the user"})
        return pooled, rows, gap_notes, deliberate

    @staticmethod
    def _existing_mask_union(ds) -> np.ndarray | None:
        """OR of the payload's existing mask channels (union_existing)."""
        masks = []
        for name in MASK_CHANNELS:
            if name in ds.data_vars:
                masks.append(np.asarray(ds[name].values) > 0)
        if not masks:
            return None
        return np.logical_or.reduce(masks)

    def _gate_dataset(self, ctx, ds, sample_rate: float, rid: str,
                      **kw) -> tuple[Any, list[dict]]:
        """Write the pooled mask into one member's dataset copy; returns
        (new_ds, report_rows)."""
        data_channels = {
            str(k): np.asarray(ds[k].values, dtype=float)
            for k in ds.data_vars
            if not is_aux_channel(k, ds[k].attrs)}
        time_index = None
        if "time" in ds.coords and len(ds["time"]):
            t = np.asarray(ds["time"].values)
            if np.issubdtype(t.dtype, np.datetime64):
                idx = pd.DatetimeIndex(t)
                time_index = (idx.tz_localize("UTC") if idx.tz is None
                              else idx.tz_convert("UTC"))

        pooled, rows, _, deliberate = self._member_mask(
            data_channels, time_index, sample_rate,
            kw["exclude_intervals"], kw["detect_spikes"],
            kw["hampel_window_s"], kw["hampel_threshold"],
            kw["detect_flat"], kw["flat_run_min"], kw["detect_gaps"], ctx)
        if kw["union_existing"]:
            existing = self._existing_mask_union(ds)
            if existing is not None and existing.size == pooled.size:
                pooled = pooled | existing

        out = ds.copy()
        mask_attrs = {"type": "auxiliary", "units": "1"}
        existing_mask = self._existing_mask_union(ds)
        if (kw["union_existing"] and existing_mask is not None
                and existing_mask.size == pooled.size):
            out["quality_mask"] = (
                "time", (pooled | existing_mask).astype(np.uint8), mask_attrs)
        else:
            out["quality_mask"] = ("time", pooled.astype(np.uint8), mask_attrs)
        frac = float(pooled.mean()) if pooled.size else 0.0
        # the >50% misconfiguration warning judges the INVOLUNTARY flags
        # only — a range the user deliberately excluded is not a signal
        # (2026-10-16: a 1-h exclusion on a burst collection flagged
        # hundreds of 2-s members 100% and drowned the card in warnings)
        invol = float((pooled & ~deliberate).mean()) if pooled.size else 0.0
        if invol > 0.5:
            ctx.warn(
                f"mask TS: {invol:.0%} of member '{rid}' is flagged bad by "
                "the criteria — likely a dead channel or a misconfigured "
                "criterion (check the report)")
        rows = [dict(r, member=rid) for r in rows]
        rows.append({"member": rid, "criterion": "total",
                     "n_flagged": int(pooled.sum()),
                     "fraction": round(frac, 6), "intervals": "", "note": ""})
        return out, rows

    def _map_collection(self, ctx, collection, **kw) -> dict[str, Any]:
        """Element-wise map (backlog #21 Tier 2 shape): the criteria apply
        to every member; the collection shape is preserved and the report
        aggregates per member."""
        from types import SimpleNamespace

        all_rows: list[dict] = []
        new_runs = []
        for i, p in enumerate(collection.runs):
            rid = (str(collection.extents[i].get("run", i))
                   if i < len(collection.extents) else str(i))
            ds = p.dataset
            ds_out, rows = self._gate_dataset(ctx, ds, float(p.sample_rate),
                                              rid, **kw)
            all_rows.extend(rows)
            new_runs.append(SimpleNamespace(
                dataset=ds_out, sample_rate=float(p.sample_rate),
                start=getattr(p, "start", None),
                filters=getattr(p, "filters", None)))
            ctx.progress((i + 1) / len(collection.runs), f"masked {rid}")
        path = ctx.artifact_path("mask_ts", ".json")
        path.write_text(json.dumps(all_rows))
        artifact = ctx.publish_artifact(path, kind="table")
        return {"gated": SimpleNamespace(
                    runs=new_runs, extents=collection.extents),
                "report": artifact}

    def _gate_timeseries(self, ctx, data: TimeSeries, **kw) -> dict[str, Any]:
        """Core TimeSeries path: the mask becomes an aux channel."""
        aux = set(data.aux_channels) | {
            k for k in data.channels if k in MASK_CHANNELS}
        data_channels = {k: v for k, v in data.channels.items()
                         if k not in aux}
        pooled, rows, _, deliberate = self._member_mask(
            data_channels, None, data.sample_rate,
            kw["exclude_intervals"], kw["detect_spikes"],
            kw["hampel_window_s"], kw["hampel_threshold"],
            kw["detect_flat"], kw["flat_run_min"], kw["detect_gaps"], ctx)
        if kw["union_existing"]:
            for name in MASK_CHANNELS:
                if name in data.channels:
                    pooled = pooled | (np.asarray(data.channels[name]) > 0)
        channels = dict(data.channels)
        channels["quality_mask"] = pooled.astype(np.uint8)
        aux.add("quality_mask")
        out = TimeSeries(
            channels=channels, sample_rate=data.sample_rate,
            start=data.start, start_datetime=data.start_datetime,
            survey=data.survey, station=data.station, run=data.run,
            meta=dict(data.meta), units=dict(data.units),
            calibration={k: list(v) for k, v in data.calibration.items()},
            aux_channels=frozenset(aux))
        rows = [dict(r, member=data.run or "single") for r in rows]
        frac = float(pooled.mean()) if pooled.size else 0.0
        rows.append({"member": data.run or "single", "criterion": "total",
                     "n_flagged": int(pooled.sum()),
                     "fraction": round(frac, 6), "intervals": "", "note": ""})
        invol = float((pooled & ~deliberate).mean()) if pooled.size else 0.0
        if invol > 0.5:
            ctx.warn(
                f"mask TS: {invol:.0%} of the record is flagged bad by the "
                "criteria — likely a dead channel or a misconfigured "
                "criterion (check the report)")
        path = ctx.artifact_path("mask_ts", ".json")
        path.write_text(json.dumps(rows))
        artifact = ctx.publish_artifact(path, kind="table")
        return {"gated": out, "report": artifact}

    def execute(self, ctx, data, exclude_intervals: str = "",
                detect_spikes: bool = True, hampel_window_s: float = 0.0,
                hampel_threshold: float = 3.0, detect_flat: bool = True,
                flat_run_min: int = 8, detect_gaps: bool = True,
                union_existing: bool = True) -> dict[str, Any]:
        if data is None:
            raise ValueError(
                "no time series input — check the upstream node ran and the "
                "edge targets this port")
        kw = {
            "exclude_intervals": exclude_intervals,
            "detect_spikes": detect_spikes,
            "hampel_window_s": hampel_window_s,
            "hampel_threshold": hampel_threshold,
            "detect_flat": detect_flat, "flat_run_min": flat_run_min,
            "detect_gaps": detect_gaps,
            "union_existing": union_existing,
        }
        if is_run_collection(data):
            return self._map_collection(ctx, data, **kw)
        if hasattr(data, "dataset") and not isinstance(data, dict):
            # mth5 RunTS / duck payload: work on the dataset, preserve the
            # payload shape (the mask rides as an auxiliary data var)
            ds_out, rows = self._gate_dataset(
                ctx, data.dataset, float(data.sample_rate),
                str(getattr(data, "run", "") or "single"), **kw)
            from types import SimpleNamespace

            path = ctx.artifact_path("mask_ts", ".json")
            path.write_text(json.dumps(rows))
            artifact = ctx.publish_artifact(path, kind="table")
            return {"gated": SimpleNamespace(
                        dataset=ds_out,
                        sample_rate=float(data.sample_rate),
                        start=getattr(data, "start", None),
                        filters=getattr(data, "filters", None)),
                    "report": artifact}
        return self._gate_timeseries(ctx, coerce_run(data), **kw)
