"""Tests for the Calibrate node (instrument response removal). Error
surfaces live in test_errors.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes.types import TimeSeries
from emptyui_mth5.calibrate import Calibrate


def test_calibrate_explicit_coefficient_filter(tmp_path: Path):
    """A constant (coefficient) response is divided out of the signal —
    via the RunTS.calibrate delegation (detrend/zero-pad handled internally,
    exact for integer-cycle tones).

    Tolerance note: mt_timeseries' recipe replaces the DC FFT bin with the
    neighboring bin's magnitude (upstream FIXME), which injects a tiny DC
    offset (~|X[1]|/n here); 3% covers that library convention.
    """
    sr = 8.0
    n = 1024
    t = np.arange(n) / sr
    x = 3.0 * np.sin(2 * np.pi * 1.0 * t)  # integer # cycles -> clean rfft
    ts = TimeSeries(channels={"ex": x, "hx": np.sin(2 * np.pi * 0.5 * t)}, sample_rate=sr)

    from mt_metadata.timeseries.filters import CoefficientFilter

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = Calibrate().execute(ctx, ts, calibration=CoefficientFilter(gain=2.0))["data"]

    # measured = 2.0 * true -> removing the response recovers true = measured/2
    # (the explicit path now returns a RunTS for any input payload type)
    np.testing.assert_allclose(out.ex.ts, x / 2.0, rtol=0.03, atol=0.03)
    np.testing.assert_allclose(out.hx.ts, ts.channels["hx"] / 2.0, rtol=0.03, atol=0.03)
    assert out.sample_rate == sr


def test_calibrate_dict_channel_match(tmp_path: Path, caplog):
    """A per-channel dict payload: matched case-insensitively; keys matching
    no channel are skipped with a warning and that channel passes through."""
    import logging

    sr = 8.0
    n = 1024
    t = np.arange(n) / sr
    x = 3.0 * np.sin(2 * np.pi * 1.0 * t)
    ts = TimeSeries(channels={"ex": x, "hx": x.copy()}, sample_rate=sr)

    from mt_metadata.timeseries.filters import CoefficientFilter

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        out = Calibrate().execute(
            ctx, ts, calibration={"EX": CoefficientFilter(gain=2.0),
                                  "hy": CoefficientFilter(gain=5.0)})["data"]
    # EX (case-insensitive) was divided by 2 on ex; "hy" matched no channel
    # (tolerance covers mt_timeseries' DC-bin convention, see above)
    np.testing.assert_allclose(out.ex.ts, x / 2.0, rtol=0.03, atol=0.03)
    np.testing.assert_allclose(out.hx.ts, x, rtol=0.03, atol=0.03)
    assert any("hy" in r.message for r in caplog.records)


def test_calibrate_mth5_runts_path(test12rr_path: Path, tmp_path: Path):
    """RunTS.calibrate() path runs without error on an mth5 RunTS."""
    import warnings

    warnings.filterwarnings("ignore")
    from mth5.mth5 import MTH5

    m = MTH5()
    m.open_mth5(str(test12rr_path), "r")
    st = m.survey_group.stations_group.get_station("test1")
    runts = st.get_run(st.run_summary.iloc[0]["id"]).to_runts()
    m.close_mth5()

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = Calibrate().execute(ctx, runts)["data"]
    # returns a RunTS with the same channel set
    assert set(out.channels) == set(runts.channels)


def test_calibrate_collection_applies_own_responses(tmp_path: Path):
    """Collection members are calibrated through their OWN response chain and
    keep their true start (H2/history §24.12): the old ChannelTS rebuild dropped the
    per-channel applied-filter refs + registry — calibration was a silent
    identity — and probed ``start_datetime`` (absent on the duck payload),
    re-anchoring members at the mt_timeseries default 1980 epoch while the
    collection extents claimed the real UTC window.

    Tolerance note: mt_timeseries' DC-bin convention (see the explicit-filter
    test above); 3% covers it.
    """
    import warnings
    from types import SimpleNamespace

    import xarray as xr
    from mt_metadata.timeseries.filters import CoefficientFilter

    warnings.filterwarnings("ignore")

    sr = 8.0
    n = 1024
    t = np.arange(n) / sr
    x = 3.0 * np.sin(2 * np.pi * 1.0 * t)
    start = np.datetime64("2017-10-16T17:10:00", "ns")
    time_axis = start + (np.arange(n) * int(1e9 / 8)).astype("timedelta64[ns]")
    fap = CoefficientFilter(gain=2.0, units_in="mV/km", units_out="mV/km")
    fap.name = "resp_ex"
    ds = xr.Dataset({
        "ex": xr.DataArray(
            x, dims=("time",), coords={"time": time_axis},
            attrs={"type": "electric", "units": "mV/km", "sample_rate": sr,
                   "filters": [{"applied_filter": {"name": "resp_ex",
                                                   "applied": True}}]}),
    })
    member = SimpleNamespace(dataset=ds, sample_rate=sr, start=start,
                             filters={"resp_ex": fap})
    from emptyui_mth5.runs import RunCollection

    coll = RunCollection(runs=[member], extents=[{
        "survey": "svy", "station": "st", "run": "r1",
        "start": "2017-10-16T17:10:00+00:00",
        "end": "2017-10-16T17:12:08+00:00", "sample_rate": 8.0}])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = Calibrate().execute(ctx, coll)["data"]
    from emptyui_core_nodes.types import is_run_collection

    assert is_run_collection(out)
    got = out.runs[0].ex.ts
    np.testing.assert_allclose(got, x / 2.0, rtol=0.03, atol=0.03)
    # the true start survived (no 1980 re-anchor)
    assert str(out.runs[0].start).startswith("2017-10-16T17:10:00")
    # extents are the source's (identity is preserved)
    assert out.extents == coll.extents


def test_calibrate_stamps_nomenclature_default_units(tmp_path: Path):
    """Integration for the _default_unit fix: a core TimeSeries with
    standard names gets hx → nanoTesla (so ComputeFC's nT→A/m conversion
    applies downstream) and ex → mV/km; auxiliary channels are not
    mis-stamped with an electric/magnetic unit."""
    from mt_metadata.timeseries.filters import CoefficientFilter

    t = np.arange(256) / 8.0
    ts = TimeSeries(channels={"ex": t.copy(), "hx": t.copy(), "temperature": t.copy()},
                    sample_rate=8.0)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = Calibrate().execute(
        ctx, ts,
        calibration={"ex": CoefficientFilter(gain=1.0),
                     "hx": CoefficientFilter(gain=1.0)})["data"]
    assert out.hx.channel_metadata.units == "nanoTesla"
    # mt_metadata canonicalizes the abbreviation
    assert out.ex.channel_metadata.units == "milliVolt per kilometer"
    # RunTS.calibrate validates metadata: "" normalizes to "unknown" —
    # auxiliary channels are not mis-stamped with an electric/magnetic unit
    assert out.temperature.channel_metadata.units == "unknown"


def test_to_runts_preserves_stamped_units(tmp_path):
    """_to_runts preserves the payload's actual units (§10.5: "explicit
    A/m → leave") — the nomenclature default is a fallback, not an
    override: re-stamping a stamped-H (A/m) payload nanoTesla fired
    Compute FC's nT→A/m conversion a second time (Z off by mu0*1e9)."""
    from emptyui_mth5.calibrate import _to_runts

    ts = TimeSeries(channels={"ex": np.ones(64), "hx": np.ones(64)},
                    sample_rate=8.0, units={"ex": "mV/km", "hx": "A/m"})
    runts = _to_runts(ts)
    # mt_metadata canonicalizes the abbreviation
    assert runts.hx.channel_metadata.units == "Ampere per meter"
    assert runts.ex.channel_metadata.units == "milliVolt per kilometer"
    # fallback for unitless sources stays (the synthetic-fixture path)
    ts2 = TimeSeries(channels={"ex": np.ones(64), "hx": np.ones(64)},
                     sample_rate=8.0)
    runts2 = _to_runts(ts2)
    assert runts2.hx.channel_metadata.units == "nanoTesla"
    assert runts2.ex.channel_metadata.units == "milliVolt per kilometer"


def test_calibrate_does_not_mutate_the_input_runts(tmp_path: Path):
    """H9 (§7.1): the explicit-filter path attaches filters to a private
    copy. A cached/shared RunTS must come out unchanged — dataset attrs,
    filter registry and channel cache included."""
    from emptyui_mth5.calibrate import _to_runts
    from mt_metadata.timeseries.filters import CoefficientFilter

    sr = 8.0
    n = 1024
    t = np.arange(n) / sr
    x = 3.0 * np.sin(2 * np.pi * 1.0 * t)
    runts = _to_runts(TimeSeries(channels={"ex": x, "hx": x.copy()},
                                 sample_rate=sr))

    # a sentinel archived response on hx (the unmatched channel) and a
    # sentinel registry entry: the private copy must carry both, the input
    # must keep them untouched.
    sentinel = CoefficientFilter(gain=7.0)
    sentinel.name = "sentinel_hx"
    sentinel.units_in = "nanoTesla"
    sentinel.units_out = "nanoTesla"
    runts.filters = {"sentinel_hx": sentinel}
    runts.dataset["hx"].attrs["filters"] = [
        {"applied_filter": {"name": "sentinel_hx", "applied": True}}]
    before_attrs = {
        ch: dict(runts.dataset[ch].attrs) for ch in runts.dataset.data_vars
    }
    before_filters = dict(runts.filters)

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = Calibrate().execute(
        ctx, runts, calibration={"ex": CoefficientFilter(gain=2.0)})["data"]

    # the input is untouched
    for ch in runts.dataset.data_vars:
        assert dict(runts.dataset[ch].attrs) == before_attrs[ch]
    assert runts.filters.keys() == before_filters.keys()
    assert runts.filters["sentinel_hx"] is sentinel

    # the output is calibrated: ex divided by 2, hx by its sentinel (7)
    np.testing.assert_allclose(out.ex.ts, x / 2.0, rtol=0.03, atol=0.03)
    np.testing.assert_allclose(out.hx.ts, x / 7.0, rtol=0.03, atol=0.03)
    # the output's registry is rebuilt from the calibrated channels; the
    # sentinel travelled with the private copy
    assert "sentinel_hx" in out.filters


def test_calibrate_twice_on_the_same_payload_is_order_independent(tmp_path: Path):
    """H9: two branches calibrating the same cached payload with different
    responses must not see each other's attachment."""
    from emptyui_mth5.calibrate import _to_runts
    from mt_metadata.timeseries.filters import CoefficientFilter

    sr = 8.0
    n = 1024
    t = np.arange(n) / sr
    x = 3.0 * np.sin(2 * np.pi * 1.0 * t)
    runts = _to_runts(TimeSeries(channels={"ex": x}, sample_rate=sr))
    ctx = RunContext(workspace=tmp_path, store=DataStore())

    out2 = Calibrate().execute(
        ctx, runts, calibration=CoefficientFilter(gain=2.0))["data"]
    out5 = Calibrate().execute(
        ctx, runts, calibration=CoefficientFilter(gain=5.0))["data"]

    np.testing.assert_allclose(out2.ex.ts, x / 2.0, rtol=0.03, atol=0.03)
    np.testing.assert_allclose(out5.ex.ts, x / 5.0, rtol=0.03, atol=0.03)
    # the shared input never acquired a filter attachment
    assert runts.dataset["ex"].attrs["filters"] == []
    assert runts.filters == {}


def test_calibrate_does_not_mutate_the_calibration_payload(tmp_path: Path):
    """H9 (§7.1): the dict path stamps missing names/units on a COPY of
    each filter, not on the caller's calibration payload."""
    from mt_metadata.timeseries.filters import CoefficientFilter

    sr = 8.0
    n = 256
    t = np.arange(n) / sr
    ts = TimeSeries(channels={"ex": t.copy()}, sample_rate=sr)
    fap = CoefficientFilter(gain=2.0)  # no name, no units
    ctx = RunContext(workspace=tmp_path, store=DataStore())

    Calibrate().execute(ctx, ts, calibration={"ex": fap})

    assert fap.name == ""
    assert fap.units_in == ""
    assert fap.units_out == ""
