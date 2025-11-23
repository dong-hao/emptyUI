"""Time-domain processing nodes (scipy-backed)."""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param

from .types import TimeSeries


class Bandpass(Node):
    type = "core.filter.bandpass"
    title = "Bandpass Filter"
    description = "Zero-phase Butterworth bandpass (filtfilt) on all channels."
    category = "Processing"
    inputs = (Input("data", "ts.run", label="time series"),)
    outputs = (Output("filtered", "ts.run", label="filtered time series"),)
    params = (
        Param("low_freq", "number", label="Low cutoff (Hz)", default=None,
              optional=True),
        Param("high_freq", "number", label="High cutoff (Hz)", default=None,
              optional=True),
        Param("order", "integer", label="Order", default=4, min=1, max=10),
    )

    def execute(self, ctx, data: TimeSeries, low_freq=None, high_freq=None,
                order: int = 4) -> dict[str, Any]:
        from scipy.signal import butter, filtfilt

        fs = data.sample_rate
        nyq = fs / 2.0
        low = low_freq / nyq if low_freq else None
        high = high_freq / nyq if high_freq else None

        if low is None and high is None:
            out = data.copy()
            ctx.progress(1.0, "no cutoffs set — passthrough")
            return {"filtered": out}

        if low is not None and high is not None:
            if not 0 < low < high < 1:
                raise ValueError(
                    f"need 0 < low ({low_freq}) < high ({high_freq}) < Nyquist ({nyq})"
                )
            b, a = butter(order, [low, high], btype="band")
        elif low is not None:
            if not 0 < low < 1:
                raise ValueError(f"low_freq must be < Nyquist ({nyq})")
            b, a = butter(order, low, btype="high")
        else:
            if not 0 < high < 1:
                raise ValueError(f"high_freq must be < Nyquist ({nyq})")
            b, a = butter(order, high, btype="low")

        filtered = {}
        for k, v in data.channels.items():
            filtered[k] = filtfilt(b, a, v)
            ctx.progress(len(filtered) / max(len(data.channels), 1), f"filtered {k}")
        return {"filtered": TimeSeries(
            channels=filtered, sample_rate=fs, start=data.start,
            meta={**data.meta, "filter": "bandpass"},
        )}


class Detrend(Node):
    type = "core.filter.detrend"
    title = "Detrend"
    description = "Remove linear trend from each channel."
    category = "Processing"
    inputs = (Input("data", "ts.run", label="time series"),)
    outputs = (Output("data", "ts.run", label="time series"),)
    params = ()

    def execute(self, ctx, data: TimeSeries) -> dict[str, Any]:
        from scipy.signal import detrend

        out = {k: detrend(v) for k, v in data.channels.items()}
        return {"data": TimeSeries(channels=out, sample_rate=data.sample_rate,
                                   start=data.start, meta=dict(data.meta))}


def register(registry) -> None:
    registry.add(Bandpass)
    registry.add(Detrend)
