"""Calibration node: remove instrument response from a time series (DESIGN §3)."""

from __future__ import annotations

from typing import Any

import numpy as np
from emptyui import Input, Node, Output, Param

from .runs import _payload


def _response_fn(calibration):
    """Return a callable frequency->complex response for a calibration payload.

    Accepts a single mt_metadata filter / ChannelResponse, or a list of filters
    (combined via ChannelResponse).
    """
    from mt_metadata.timeseries.filters import ChannelResponse

    if isinstance(calibration, (list, tuple)):
        cr = ChannelResponse()
        return lambda f: np.asarray(cr.complex_response(f, filters_list=list(calibration)))
    if hasattr(calibration, "complex_response"):
        return lambda f: np.asarray(calibration.complex_response(f))
    raise TypeError(
        f"cannot interpret calibration of type {type(calibration).__name__} "
        "(expected an mt_metadata filter / ChannelResponse)"
    )


def _remove_response(x, sample_rate, response_fn):
    """Apply frequency-domain response removal: divide FFT by the response."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    resp = response_fn(f)
    resp = np.where(np.abs(resp) < 1e-12, 1.0, resp)  # guard DC/zero response
    return np.fft.irfft(X / resp, n=len(x))


def _as_core_ts(channels, sample_rate, source):
    """Wrap calibrated numpy channels as a core TimeSeries (ts.run duck-type).

    Preserves the absolute start and survey/station/run identity, handling both
    an mth5 RunTS (``.start`` is a datetime; identity on *_metadata.id) and a
    core TimeSeries (``.start`` float seconds + ``.start_datetime`` ISO).
    """
    from emptyui_core_nodes.types import TimeSeries, _iso_start, _meta_id

    start = getattr(source, "start", 0.0)
    if isinstance(start, (int, float, np.number)):
        start_datetime = _iso_start(getattr(source, "start_datetime", ""))
    else:  # datetime-like absolute start -> move to start_datetime
        start_datetime = _iso_start(start)
        start = 0.0

    return TimeSeries(
        channels=channels,
        sample_rate=sample_rate,
        start=start,
        start_datetime=start_datetime,
        survey=_meta_id(getattr(source, "survey_metadata", None)) or getattr(source, "survey", ""),
        station=_meta_id(getattr(source, "station_metadata", None)) or getattr(source, "station", ""),
        run=_meta_id(getattr(source, "run_metadata", None)) or getattr(source, "run", ""),
    )


class Calibrate(Node):
    type = "mth5.calibrate"
    title = "Calibrate"
    description = (
        "Remove the instrument response from a time series (calibration). With "
        "no 'calibration' input, uses the run's own mth5 channel responses "
        "(mth5 RunTS.calibrate). With an explicit 'calibration' filter(s), "
        "applies frequency-domain response removal."
    )
    category = "Processing"
    inputs = (
        Input("data", "ts.run"),
        Input("calibration", "calibration", required=False),
    )
    outputs = (Output("data", "ts.run", label="calibrated"),)
    params = (
        Param("include_decimation", "boolean", label="Include decimation", default=False),
        Param("include_delay", "boolean", label="Include delay", default=False),
    )

    def execute(self, ctx, data, calibration=None,
                include_decimation: bool = False,
                include_delay: bool = False) -> dict[str, Any]:
        # Idiomatic mth5 path: the run carries its own channel responses.
        if calibration is None and hasattr(data, "calibrate") and not isinstance(data, dict):
            out = data.calibrate(include_decimation=include_decimation,
                                 include_delay=include_delay)
            ctx.progress(1.0, "calibrated via mth5 channel responses")
            return {"data": out}

        # Explicit calibration path: frequency-domain response removal.
        channels, sample_rate = _payload(data)
        calibrated = {}
        for ch, x in channels.items():
            if isinstance(calibration, dict):
                fn = _response_fn(calibration[ch]) if ch in calibration else None
            else:
                fn = _response_fn(calibration)
            calibrated[ch] = x if fn is None else _remove_response(x, sample_rate, fn)
        out = _as_core_ts(calibrated, sample_rate, data)
        ctx.progress(1.0, f"calibrated {len(calibrated)} channels")
        return {"data": out}


def register(registry) -> None:
    registry.add(Calibrate)
