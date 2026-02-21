"""Time-domain processing nodes (scipy-backed)."""

from __future__ import annotations

from typing import Any

import numpy as np

from emptyui import Input, Node, Output, Param

from .types import (TimeSeries, collection_member_ds, coerce_run,
                    ensure_finite, is_run_collection)


class Bandpass(Node):
    type = "core.filter.bandpass"
    title = "Bandpass Filter"
    tooltip = "Zero-phase Butterworth low/high/band-pass filter on all channels."
    description = (
        "Zero-phase Butterworth bandpass (filtfilt) on all channels. "
        "Accepts a run collection: the recipe maps over every member, "
        "skipping members whose sampling scope cannot support the cutoffs "
        "(loud summary; Tier 2, backlog #21) and preserving the collection "
        "shape."
    )
    category = "Processing"
    inputs = (Input("data", "ts.run", label="time series", collection_ok=True),)
    outputs = (Output("filtered", "ts.run", label="filtered time series"),)
    params = (
        Param("low_freq", "number", label="Low cutoff (Hz)", default=None,
              optional=True),
        Param("high_freq", "number", label="High cutoff (Hz)", default=None,
              optional=True),
        Param("order", "integer", label="Order", default=4, min=1, max=10),
    )

    @staticmethod
    def _design(low_freq, high_freq, order: int, nyq: float):
        """Butterworth design for one member's Nyquist (shared with the
        single path — the same design, per scope). None = passthrough."""
        from scipy.signal import butter

        low = low_freq / nyq if low_freq else None
        high = high_freq / nyq if high_freq else None
        if low is None and high is None:
            return None
        if low is not None and high is not None:
            if not 0 < low < high < 1:
                raise ValueError(
                    f"need 0 < low ({low_freq}) < high ({high_freq}) < Nyquist ({nyq})"
                )
            return butter(order, [low, high], btype="band")
        if low is not None:
            if not 0 < low < 1:
                raise ValueError(f"low_freq must be < Nyquist ({nyq})")
            return butter(order, low, btype="high")
        if not 0 < high < 1:
            raise ValueError(f"high_freq must be < Nyquist ({nyq})")
        return butter(order, high, btype="low")

    def _map_collection(self, ctx, collection, low_freq, high_freq,
                        order: int):
        """Element-wise map with per-member scope quick-reject (backlog #21
        Tier 2): a member whose Nyquist cannot support the cutoffs is
        skipped (loud summary); the whole node rejects only when NO member
        is in scope — a silently empty map would lie downstream."""
        from types import SimpleNamespace

        if not (low_freq or high_freq):
            return {"filtered": collection}  # passthrough, shape preserved
        pairs = list(zip(collection.runs, collection.extents))
        new_runs, kept, skipped = [], [], []
        for idx, (p, e) in enumerate(pairs):
            ds, sr, start, filters = collection_member_ds(
                p, str(e.get("run", idx)))
            nyq = sr / 2.0
            if ((low_freq and low_freq >= nyq)
                    or (high_freq and high_freq >= nyq)):
                skipped.append(str(e.get("run", idx)))
                continue
            from scipy.signal import filtfilt

            design = self._design(low_freq, high_freq, order, nyq)
            new_ds = Detrend._map_dataset(ds, lambda v, d=design: filtfilt(*d, v))
            new_ds.attrs["mapped_by"] = f"{self.type}:{order}"
            new_ds.attrs["filter"] = "bandpass"
            new_runs.append(SimpleNamespace(
                dataset=new_ds, sample_rate=sr, start=start,
                filters=filters))
            kept.append(e)
        if skipped:
            shown = ", ".join(skipped[:5]) + (
                f" …+{len(skipped) - 5}" if len(skipped) > 5 else "")
            ctx.logger.warning(
                "bandpass: %d/%d member(s) skipped — their sampling scope "
                "cannot support the cutoffs (%s)", len(skipped),
                len(collection.runs), shown)
        if not new_runs:
            rates = sorted({float(p.sample_rate) for p, _ in pairs})
            raise ValueError(
                f"no member's sampling scope can support the cutoffs "
                f"(low={low_freq}, high={high_freq}; rates {rates} Hz) — "
                "lower the cutoffs or extract a single run")
        return {"filtered": SimpleNamespace(
            runs=new_runs, extents=kept)}

    def execute(self, ctx, data, low_freq=None, high_freq=None,
                order: int = 4) -> dict[str, Any]:
        if is_run_collection(data):
            return self._map_collection(ctx, data, low_freq, high_freq, order)
        data = coerce_run(data)
        ensure_finite(data.channels, "Bandpass Filter")
        from scipy.signal import filtfilt

        fs = data.sample_rate
        nyq = fs / 2.0
        design = self._design(low_freq, high_freq, order, nyq)
        if design is None:
            out = data.copy()
            ctx.progress(1.0, "no cutoffs set — passthrough")
            return {"filtered": out}

        filtered = {}
        for k, v in data.channels.items():
            filtered[k] = filtfilt(*design, v)
            ctx.progress(len(filtered) / max(len(data.channels), 1), f"filtered {k}")
        return {"filtered": TimeSeries(
            channels=filtered, sample_rate=fs, start=data.start,
            start_datetime=data.start_datetime,
            survey=data.survey, station=data.station, run=data.run,
            meta={**data.meta, "filter": "bandpass"},
            units=dict(data.units),
            calibration={k: list(v) for k, v in data.calibration.items()},
        )}


class Detrend(Node):
    type = "core.filter.detrend"
    title = "Detrend"
    tooltip = "Remove the DC mean or a linear trend from each channel."
    description = (
        "Remove the DC baseline / trend from each channel. 'mean' subtracts "
        "the channel mean (removes the absolute-field DC offset of fluxgate "
        "magnetometers); 'linear' fits and removes a linear trend; 'none' is "
        "a passthrough. Accepts a run collection: the recipe maps over every "
        "member (Tier 2, backlog #21) and the collection shape is preserved."
    )
    category = "Processing"
    inputs = (Input("data", "ts.run", label="time series", collection_ok=True),)
    outputs = (Output("data", "ts.run", label="time series"),)
    params = (
        Param("method", "select", label="Method", default="linear",
              options=("mean", "linear", "none"),
              description="mean = subtract DC; linear = remove linear trend; none = passthrough"),
    )

    @staticmethod
    def _map_dataset(ds, fn):
        """Apply a per-channel function to a member's dataset, preserving
        coords/attrs (the time axis must survive exactly — fc identity)."""
        import xarray as xr

        out = xr.Dataset(
            {k: ds[k].copy(data=fn(np.asarray(ds[k].values, dtype=float)))
             for k in ds.data_vars},
            coords=ds.coords, attrs=dict(ds.attrs))
        return out

    def _map_collection(self, ctx, collection, method: str):
        """Element-wise map (backlog #21 Tier 2): the recipe applied to every
        member; the collection shape (runs + extents rows) is preserved."""
        from types import SimpleNamespace

        new_runs = []
        for i, p in enumerate(collection.runs):
            rid = (str(collection.extents[i].get("run", i))
                   if i < len(collection.extents) else i)
            ds, sr, start, filters = collection_member_ds(p, rid)
            if method == "none":
                new_ds = ds.copy()
            else:
                ensure_finite(
                    {k: np.asarray(ds[k].values, dtype=float)
                     for k in ds.data_vars}, "Detrend")
                if method == "mean":
                    fn = lambda v: v - v.mean()  # noqa: E731
                else:
                    from scipy.signal import detrend

                    fn = detrend
                new_ds = self._map_dataset(ds, fn)
            new_ds.attrs["mapped_by"] = f"{self.type}:{method}"
            new_runs.append(SimpleNamespace(
                dataset=new_ds, sample_rate=sr, start=start,
                filters=filters))
        return {"data": SimpleNamespace(
            runs=new_runs, extents=collection.extents)}

    def execute(self, ctx, data, method: str = "linear") -> dict[str, Any]:
        if is_run_collection(data):
            return self._map_collection(ctx, data, method)

        data = coerce_run(data)
        if method != "none":
            ensure_finite(data.channels, "Detrend")

        out = {}
        for k, v in data.channels.items():
            if method == "none":
                out[k] = v.copy()  # functional contract: never share arrays
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
                                   meta=dict(data.meta),
                                   units=dict(data.units),
                                   calibration={k: list(v) for k, v in
                                                data.calibration.items()})}


