"""Calibration node: remove instrument response from a time series (DESIGN §3).

Both paths delegate to mt_timeseries' ``RunTS.calibrate()``, which does the
response removal properly (detrend, zero-mean, zero-pad, windowing) instead of
a bare FFT division (which suffers circular-convolution wraparound at the
series edges):

- no ``calibration`` input: the run's own archived mth5 channel responses
  (attached by ``Load LEMI-424`` / ``attach_filters``) are applied.
- explicit ``calibration`` input: the filters are matched to the run's channel
  names (case-insensitively) and attached to the run, then ``RunTS.calibrate``
  applies them. Channels with no matching filter pass through with a unity
  filter (warning); filter keys matching no channel are skipped (warning).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from emptyui import Input, Node, Output, Param
from emptyui_core_nodes.types import (
    _iso_start,
    channel_calibration,
    collection_member_ds,
    is_run_collection,
)

from .calibration import _default_unit, attach_filters
from .nomenclature import is_electric_name, is_magnetic_name
from .runs import _CachedRunTS, _payload

_log = logging.getLogger("emptyui")


class _MappedCollection:
    """A calibrated run collection (backlog #21 Tier 2): duck-compatible
    (runs + extents) — members are the calibrated real RunTS objects, the
    extent rows are the source collection's (identity is preserved)."""

    def __init__(self, runs, extents):
        self.runs = list(runs)
        self.extents = list(extents)


def _member_runts(member, rid: str):
    """Wrap a collection member as a real RunTS WITHOUT losing its response
    chain or true start (§24.12): the duck payload's dataset carries the
    per-channel applied-filter refs and units, and the member's ``.filters``
    registry resolves them — the same wiring Extract Run's ``_wrap_runts``
    produces. The old ChannelTS rebuild (``_to_runts``) dropped both: empty
    chains made ``RunTS.calibrate`` a silent identity, and probing
    ``start_datetime`` (absent on the duck payload, which carries ``.start``)
    re-anchored the member at the mt_timeseries default 1980 epoch while the
    collection extents claimed the real UTC window.
    """

    ds, sample_rate, _start, filters = collection_member_ds(member, rid)
    runts = _CachedRunTS(
        ds.copy(),
        run_metadata={"id": rid, "sample_rate": float(sample_rate)},
    )
    runts.filters = dict(filters or {})
    return runts


def _to_runts(source):
    """Wrap a core TimeSeries as an mt_timeseries RunTS so the response
    removal can be delegated to ``RunTS.calibrate()`` for any ts.run payload.

    Note: a *relative* float-seconds start (core TimeSeries without an
    absolute time) cannot be represented in a RunTS — the wrapped run starts
    at the mt_timeseries default epoch, i.e. "no absolute time" downstream.
    """
    from mt_timeseries import ChannelTS
    from mth5.timeseries import RunTS

    channels, sample_rate = _payload(source)
    # the payload's true start: ISO strings carry ``start_datetime``, duck
    # payloads (Extract Run) carry ``.start`` (datetime64) — probing only
    # ``start_datetime`` silently re-anchored the wrapped run at the
    # mt_timeseries default 1980 epoch (§24.12)
    start_dt = getattr(source, "start_datetime", "") or ""
    start_iso = (start_dt if start_dt
                 else _iso_start(getattr(source, "start", None)))
    pieces = []
    src_units = getattr(source, "units", {}) or {}
    for name, arr in channels.items():
        ctype = ("electric" if is_electric_name(name)
                 else "magnetic" if is_magnetic_name(name) else "auxiliary")
        ch = ChannelTS(ctype, data=np.asarray(arr, dtype=float),
                       start=start_iso or None, sample_rate=float(sample_rate))
        ch.component = name
        # preserve the payload's actual unit when it carries one (§10.5:
        # "explicit A/m → leave") — the nomenclature default is only a
        # fallback; re-stamping a stamped H (A/m) payload nanoTesla fired
        # Compute FC's nT→A/m conversion a second time
        src_unit = src_units.get(name)
        ch.channel_metadata.units = src_unit or _default_unit(name)
        # ChannelTS derives its rate from the ns-quantized time axis when
        # data is present (size/(t[-1]-t[0]) drifts: 1/2400 s truncates to
        # 416666 ns → 2401) — pin the explicit rate (the getter honors it)
        ch._sample_rate = float(sample_rate)
        pieces.append(ch)
    return RunTS(array_list=pieces)


def _channel_filters(calibration, runts) -> dict:
    """Map a calibration payload onto the run's channel names.

    A dict payload is matched case-insensitively to the run's channels; keys
    matching no channel are skipped with a warning. A single filter object
    (legacy payload shape) is applied to every channel. Filters without a
    name or units get them filled from the channel, so ``RunTS.calibrate``
    can register and unit-check them.
    """
    names = [str(c) for c in runts.dataset.data_vars]
    if not isinstance(calibration, dict):
        out = {}
        for name in names:
            fap = (calibration.model_copy(deep=True)
                   if hasattr(calibration, "model_copy") else calibration)
            if not getattr(fap, "name", ""):
                fap.name = f"cal_{name}"
            unit = _default_unit(name)
            if not getattr(fap, "units_in", ""):
                fap.units_in = unit
            if not getattr(fap, "units_out", ""):
                fap.units_out = unit
            out[name] = fap
        return out
    low = {name.lower(): name for name in names}
    out: dict[str, object] = {}
    for key, fap in calibration.items():
        matched = low.get(str(key).strip().lower())
        if matched is None:
            _log.warning(
                "calibration channel '%s' matches no channel in the run %s "
                "— skipped (that channel passes through uncalibrated)",
                key, names,
            )
            continue
        if not getattr(fap, "name", ""):
            fap.name = f"cal_{matched}"
        unit = _default_unit(matched)
        if not getattr(fap, "units_in", ""):
            fap.units_in = unit
        if not getattr(fap, "units_out", ""):
            fap.units_out = unit
        out[matched] = fap
    return out


class Calibrate(Node):
    type = "mth5.calibrate"
    title = "Calibrate"
    tooltip = "Remove the instrument response (calibration) from a time series."
    description = (
        "Remove the instrument response from a time series (calibration). With "
        "no 'calibration' input, uses the run's own mth5 channel responses "
        "(mth5 RunTS.calibrate). With an explicit 'calibration' filter(s), the "
        "filters are attached to the run (case-insensitive channel match) and "
        "RunTS.calibrate applies them with proper detrend/zero-pad/windowing."
    )
    category = "Processing"
    inputs = (
        Input("data", "ts.run", collection_ok=True),
        Input("calibration", "calibration", required=False),
    )
    outputs = (Output("data", "ts.run", label="calibrated"),)
    params = (
        Param("include_decimation", "boolean", label="Include decimation", default=False),
        Param("include_delay", "boolean", label="Include delay", default=False),
    )

    def _map_collection(self, ctx, collection, calibration,
                        include_decimation: bool, include_delay: bool):
        """Element-wise calibration (backlog #21 Tier 2): each member is
        calibrated through ITS OWN response chain — the duck payloads carry
        the per-channel applied-filter refs + the resolved registry, and the
        dataset-passthrough wrapper keeps both (§24.12). The collection
        shape is preserved (extents rows unchanged — identity is not the
        recipe's business). Explicit-filter collection mapping is M4
        backlog."""
        kwargs = {"include_decimation": include_decimation,
                  "include_delay": include_delay}
        members = []
        for i, p in enumerate(collection.runs):
            rid = (str(collection.extents[i].get("run", i))
                   if i < len(collection.extents) else i)
            # validate the member's shape loudly (a degenerate collection
            # must not surface as a raw TypeError from _payload)
            collection_member_ds(p, rid)
            runts = (p if hasattr(p, "calibrate") and not isinstance(p, dict)
                     else _member_runts(p, rid))
            if not channel_calibration(runts):
                _log.warning(
                    "collection member '%s' carries no instrument response "
                    "— passing through uncalibrated (connect the run's own "
                    "archived responses or a 'Load Calibration' node)",
                    rid)
            members.append(runts.calibrate(**kwargs))
        ctx.progress(1.0, f"calibrated {len(members)} member(s) via their own responses")
        return {"data": _MappedCollection(members, collection.extents)}

    def execute(self, ctx, data, calibration=None,
                include_decimation: bool = False,
                include_delay: bool = False) -> dict[str, Any]:
        if is_run_collection(data) and calibration is None:
            return self._map_collection(ctx, data, calibration,
                                        include_decimation, include_delay)
        if is_run_collection(data):
            raise ValueError(
                "Calibrate on a collection requires each member's own "
                "responses (leave the 'calibration' input unconnected); "
                "explicit-filter collection mapping is planned (M4 backlog)")
        kwargs = {"include_decimation": include_decimation,
                  "include_delay": include_delay}
        # Idiomatic mth5 path: the run carries its own channel responses.
        if calibration is None:
            if hasattr(data, "calibrate") and not isinstance(data, dict):
                out = data.calibrate(**kwargs)
                ctx.progress(1.0, "calibrated via mth5 channel responses")
                return {"data": out}
            raise TypeError(
                "no calibration input and the time series carries no channel "
                "responses of its own — connect a 'Load Calibration' filter, "
                "or load the run from mth5 with archived responses"
            )

        # Explicit-filters path: attach, then delegate to RunTS.calibrate.
        runts = (data if hasattr(data, "dataset") and not isinstance(data, dict)
                 else _to_runts(data))
        filters = _channel_filters(calibration, runts)
        if not filters:
            raise ValueError(
                f"no calibration filter matched any run channel "
                f"{list(runts.dataset.data_vars)} — check the calibration "
                "file's channel keys"
            )
        attach_filters(runts, filters, strict=False)
        out = runts.calibrate(**kwargs)
        ctx.progress(1.0, f"calibrated {len(filters)} channel(s) via RunTS.calibrate")
        return {"data": out}


