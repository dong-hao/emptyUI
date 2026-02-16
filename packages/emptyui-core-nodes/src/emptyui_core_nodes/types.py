"""Shared data containers for bundled nodes.

M1 keeps the interchange type lightweight (numpy-backed). Migration to
mth5 RunTS/xarray happens with the mth5 plugin in M2; the port tag stays
``ts.run`` either way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimeSeries:
    """Multi-channel time series (``ts.run`` port payload).

    ``start`` is a relative start offset in seconds; ``start_datetime`` is the
    absolute start (ISO string) when known (e.g. from an mth5 RunTS), used for
    FC persistence.
    """

    channels: dict[str, np.ndarray]
    sample_rate: float
    start: float = 0.0  # seconds relative to record start
    start_datetime: str = ""
    survey: str = ""
    station: str = ""
    run: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    #: per-channel physical units (e.g. ``nanoTesla`` / ``milliVolt per
    #: kilometer``). Coerced from mth5 RunTS dataset attrs and carried
    #: through the core filter nodes so Compute FC's instrument-unit
    #: normalization survives a core node in the chain (DESIGN §19).
    units: dict[str, str] = field(default_factory=dict)
    #: per-channel instrument response chains (DESIGN §24.12):
    #: ``{channel: [(filter_object, applied: bool), ...]}`` in stage order.
    #: Compute FC divides the Fourier coefficients by the chained complex
    #: response of every *applied* filter (aurora semantics: ``applied=True``
    #: means the response is part of the recorded data and must be removed;
    #: a time-domain Calibrate flips the flags, so the two paths compose
    #: without double application). Coerced from an mth5 RunTS's per-channel
    #: ``filters`` attrs + its ``filters`` object registry.
    calibration: dict[str, list] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return max((len(v) for v in self.channels.values()), default=0)

    def copy(self) -> TimeSeries:
        return TimeSeries(
            channels={k: v.copy() for k, v in self.channels.items()},
            sample_rate=self.sample_rate,
            start=self.start,
            start_datetime=self.start_datetime,
            survey=self.survey,
            station=self.station,
            run=self.run,
            meta=dict(self.meta),
            units=dict(self.units),
            calibration={k: list(v) for k, v in self.calibration.items()},
        )


def _meta_id(value) -> str:
    """Best-effort id string from an mth5 metadata object (has ``.id``)."""
    try:
        return str(value.id) or ""
    except Exception:  # noqa: BLE001
        return ""


def _iso_start(value) -> str:
    """Best-effort ISO string from a RunTS/ChannelTS absolute start (datetime,
    pandas Timestamp, or mt_metadata MTime). Empty string if not available."""
    if value is None or isinstance(value, (int, float, np.number)):
        return ""
    for attr in ("isoformat", "iso_str"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                s = str(fn())
                if s:
                    return s
            except Exception:  # noqa: BLE001, S112
                continue
    import datetime as _dt

    if isinstance(value, _dt.datetime):
        return value.isoformat()
    return ""


def ensure_finite(channels, where: str) -> None:
    """Raise a clear error when channels contain NaN/Inf.

    scipy's ``decimate``/``filtfilt``/``stft`` propagate a single NaN into
    every downstream sample, which would silently corrupt FCs/TFs — fail at
    the processing entry point instead, naming the offending channels.
    """
    bad = []
    for name, x in channels.items():
        idx = np.flatnonzero(~np.isfinite(np.asarray(x, dtype=float)))
        if idx.size:
            bad.append(f"{name} (first at sample {int(idx[0])})")
    if bad:
        raise ValueError(
            f"{where}: channels contain NaN/Inf — fill or remove missing "
            f"samples first (e.g. interpolate gaps): {', '.join(bad)}"
        )


def channel_calibration(value) -> dict[str, list]:
    """Per-channel instrument response chains from a ts.run payload (§24.12).

    Returns ``{channel: [(filter_object, applied: bool), ...]}`` in stage
    order. Sources:

    - an mth5 RunTS: the per-channel dataset attrs ``filters`` (applied-filter
      refs) resolved against the run's ``filters`` object registry — the
      shape the archive round trip restores;
    - a core :class:`TimeSeries`: its ``calibration`` dict as-is.

    Refs whose object is missing from the registry are dropped with a
    warning (a partial chain would silently mis-calibrate); payloads with
    no response info yield ``{}``.
    """
    _log = logging.getLogger("emptyui")
    if hasattr(value, "dataset") and not isinstance(value, dict):  # mth5 RunTS
        registry = dict(getattr(value, "filters", None) or {})
        out: dict[str, list] = {}
        for name in value.dataset.data_vars:
            refs = value.dataset[name].attrs.get("filters") or []
            pairs: list = []
            for ref in refs:
                try:
                    info = ref["applied_filter"]
                    obj = registry.get(info["name"])
                except (TypeError, KeyError, IndexError):
                    continue
                if obj is None:
                    _log.warning(
                        "channel '%s' references filter '%s' that is not "
                        "registered on the payload — skipping it (FC-level "
                        "calibration would be incomplete)", name,
                        info.get("name"))
                    continue
                pairs.append((obj, bool(info.get("applied", True))))
            if pairs:
                out[str(name)] = pairs
        return out
    cal = getattr(value, "calibration", None)
    if isinstance(cal, dict):
        return {str(k): list(v) for k, v in cal.items() if v}
    return {}


def coerce_run(value):
    """Normalize any ts.run payload to a core :class:`TimeSeries`.

    Accepts an mth5 RunTS (xarray-backed) or an already-normalized TimeSeries,
    so core signal-processing/plot nodes work regardless of which plugin
    produced the time series.
    """
    if isinstance(value, TimeSeries):
        return value
    if hasattr(value, "dataset") and not isinstance(value, dict):  # mth5 RunTS
        ds = value.dataset
        sample_rate = float(value.sample_rate)
        channels = {
            str(k): np.ascontiguousarray(ds[k].values, dtype=np.float64)
            for k in ds.data_vars
        }
        units = {
            str(k): str(ds[k].attrs["units"])
            for k in ds.data_vars
            if ds[k].attrs.get("units")
        }
        return TimeSeries(
            channels=channels,
            sample_rate=sample_rate,
            start_datetime=_iso_start(getattr(value, "start", None)),
            survey=_meta_id(getattr(value, "survey_metadata", None)),
            station=_meta_id(getattr(value, "station_metadata", None)),
            run=_meta_id(getattr(value, "run_metadata", None)),
            units=units,
            calibration=channel_calibration(value),
        )
    if hasattr(value, "channels") and hasattr(value, "sample_rate"):
        return value
    raise TypeError(
        f"cannot interpret ts.run value of type {type(value).__name__} as time series"
    )


#: Placeholder epochs that mean "the start time is unknown": mt_timeseries
#: stamps unknown starts with its 1980 default epoch, and digitized paper
#: records (the pre-digital MT era — Cagniard/Tikhonov pioneers) commonly
#: carry the Unix 1970 epoch from the digitizing tool. A record starting at
#: exactly one of these is treated as having NO absolute start — with a
#: warning, never a rejection (old records must keep processing; MT data
#: from the 1950s on is legitimate).
UNKNOWN_EPOCH_STARTS = ("1980-01-01T00:00:00", "1970-01-01T00:00:00")


def is_run_collection(value) -> bool:
    """True for a run collection (Extract Run ``time range`` payload).

    Import-free duck type (core stays mth5-free): a collection carries
    ``runs`` (list of member payloads) + ``extents`` (identity/extent
    rows). NB: mt_timeseries objects convert missing-attribute
    AttributeError into NameError, hence the broad except.
    """
    try:
        extents = value.extents
    except Exception:  # noqa: BLE001
        return False
    return (isinstance(extents, list) and hasattr(value, "runs"))

_log = logging.getLogger("emptyui")


def is_unknown_start(ts) -> bool:
    """True when a timestamp equals a placeholder epoch (see
    UNKNOWN_EPOCH_STARTS) — "unknown start", not a real acquisition time.
    Timezone-aware stamps are compared in their own zone (a UTC-stamped
    1980 default from mt_timeseries must still count as unknown)."""
    import pandas as pd

    try:
        ts = pd.Timestamp(ts)
    except (TypeError, ValueError):
        return False
    return any(
        ts == pd.Timestamp(e, tz=ts.tz) for e in UNKNOWN_EPOCH_STARTS
    )


def warn_placeholder_start(source: str, epoch: str) -> None:
    """Warn that a record's start is a placeholder epoch (old/ambiguous
    metadata) — absolute-time features (alignment, persistence) are
    degraded, processing continues."""
    _log.warning(
        "%s starts at the placeholder epoch %s — treating it as 'start time "
        "unknown' (old or partially digitized record?). Absolute-time "
        "features (run alignment, FC persistence) are degraded; provide the "
        "real acquisition start if available",
        source, epoch,
    )
