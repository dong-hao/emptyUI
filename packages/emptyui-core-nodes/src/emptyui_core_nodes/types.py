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
    """Multi-channel time series (``ts.run`` port payload)."""

    channels: dict[str, np.ndarray]
    sample_rate: float
    start: float = 0.0  # seconds relative to record start
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return max((len(v) for v in self.channels.values()), default=0)

    def time_axis(self) -> np.ndarray:
        return self.start + np.arange(self.n_samples) / self.sample_rate

    def copy(self) -> TimeSeries:
        return TimeSeries(
            channels={k: v.copy() for k, v in self.channels.items()},
            sample_rate=self.sample_rate,
            start=self.start,
            meta=dict(self.meta),
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
        return TimeSeries(channels=channels, sample_rate=sample_rate)
    if hasattr(value, "channels") and hasattr(value, "sample_rate"):
        return value
    raise TypeError(
        f"cannot interpret ts.run value of type {type(value).__name__} as time series"
    )
