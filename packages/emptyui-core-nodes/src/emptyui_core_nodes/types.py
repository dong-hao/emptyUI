"""Shared data containers for bundled nodes.

M1 keeps the interchange type lightweight (numpy-backed). Migration to
mth5 RunTS/xarray happens with the mth5 plugin in M2; the port tag stays
``ts.run`` either way.
"""

from __future__ import annotations

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
        return TimeSeries(
            channels=channels,
            sample_rate=sample_rate,
            start_datetime=_iso_start(getattr(value, "start", None)),
            survey=_meta_id(getattr(value, "survey_metadata", None)),
            station=_meta_id(getattr(value, "station_metadata", None)),
            run=_meta_id(getattr(value, "run_metadata", None)),
        )
    if hasattr(value, "channels") and hasattr(value, "sample_rate"):
        return value
    raise TypeError(
        f"cannot interpret ts.run value of type {type(value).__name__} as time series"
    )
