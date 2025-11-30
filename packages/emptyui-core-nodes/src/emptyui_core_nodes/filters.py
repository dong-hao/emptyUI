"""Time-domain processing nodes (scipy-backed)."""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param

from .types import TimeSeries, coerce_run, ensure_finite


class Bandpass(Node):
    type = "core.filter.bandpass"
    title = "Bandpass Filter"
    tooltip = "Zero-phase Butterworth low/high/band-pass filter on all channels."
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

    def execute(self, ctx, data, low_freq=None, high_freq=None,
                order: int = 4) -> dict[str, Any]:
        data = coerce_run(data)
        ensure_finite(data.channels, "Bandpass Filter")
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
            start_datetime=data.start_datetime,
            survey=data.survey, station=data.station, run=data.run,
            meta={**data.meta, "filter": "bandpass"},
        )}


class Detrend(Node):
    type = "core.filter.detrend"
    title = "Detrend"
    tooltip = "Remove the DC mean or a linear trend from each channel."
    description = (
        "Remove the DC baseline / trend from each channel. 'mean' subtracts "
        "the channel mean (removes the absolute-field DC offset of fluxgate "
        "magnetometers); 'linear' fits and removes a linear trend; 'none' is "
        "a passthrough."
    )
    category = "Processing"
    inputs = (Input("data", "ts.run", label="time series"),)
    outputs = (Output("data", "ts.run", label="time series"),)
    params = (
        Param("method", "select", label="Method", default="linear",
              options=("mean", "linear", "none"),
              description="mean = subtract DC; linear = remove linear trend; none = passthrough"),
    )

    def execute(self, ctx, data, method: str = "linear") -> dict[str, Any]:
        data = coerce_run(data)
        if method != "none":
            ensure_finite(data.channels, "Detrend")

        out = {}
        for k, v in data.channels.items():
            if method == "none":
                out[k] = v
            elif method == "mean":
                out[k] = v - v.mean()
            else:  # linear
                from scipy.signal import detrend

                out[k] = detrend(v)
            ctx.progress(len(out) / max(len(data.channels), 1), f"detrended {k}")
        return {"data": TimeSeries(channels=out, sample_rate=data.sample_rate,
                                   start=data.start,
                                   start_datetime=data.start_datetime,
                                   survey=data.survey, station=data.station,
                                   run=data.run,
                                   meta=dict(data.meta))}


def register(registry) -> None:
    registry.add(Bandpass)
    registry.add(Detrend)
