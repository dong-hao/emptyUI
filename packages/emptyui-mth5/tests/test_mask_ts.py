"""Mask TS (mth5.qc.mask_ts) — time-domain quality masking (M26, §9.5).

The node MARKS, never touches: a pooled per-sample ``quality_mask`` aux
channel; criteria OR-compose; the payload shape is preserved; the report
is published. Criteria basis: the Hampel identifier (self-calibrating —
no absolute amplitudes), constant-run detection (scale-free), explicit
intervals, NaN flagging and axis-gap reporting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes.types import TimeSeries
from emptyui_mth5.qc import (
    MaskTS,
    _auto_hampel_window,
    _flat_run_flags,
    _hampel_flags,
    _parse_intervals,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(workspace=tmp_path, store=DataStore())


def _ds(n: int = 512, seed: int = 3, spike_at: int | None = 100,
        clip_at: int | None = 300) -> SimpleNamespace:
    """A duck ts.run payload (hx/hy/ex/ey, 1 Hz, absolute time) with an
    optional injected spike and clip."""
    import xarray as xr

    rng = np.random.default_rng(seed)
    t = (pd.Timestamp("2024-01-01T00:00:00").tz_localize(None)
         + pd.to_timedelta(np.arange(n), unit="s")).values
    data = {c: rng.standard_normal(n) for c in ("hx", "hy", "ex", "ey")}
    if spike_at is not None:
        for x in data.values():
            x[spike_at] = 500.0  # an impulse on every channel
    if clip_at is not None:
        data["ex"][clip_at:clip_at + 10] = 3.0  # a constant run
    ds = xr.Dataset(
        {k: ("time", v) for k, v in data.items()}, coords={"time": t})
    return SimpleNamespace(dataset=ds, sample_rate=1.0, start=t[0],
                           filters={})


def _mask_of(out) -> np.ndarray:
    return np.asarray(out["gated"].dataset["quality_mask"].values) > 0


def _report_of(out) -> list[dict]:
    return json.loads(Path(out["report"]["path"]).read_text())


# --- helper units -------------------------------------------------------------


def test_parse_intervals():
    ivs = _parse_intervals(
        "2024-01-01T00:00:00/2024-01-01T01:00:00 ; 2024-01-02T00:00/2024-01-02T01:00")
    assert len(ivs) == 2
    assert ivs[0][1] > ivs[0][0]
    with pytest.raises(ValueError, match="not a 'start/end' pair"):
        _parse_intervals("2024-01-01T00:00:00")
    with pytest.raises(ValueError, match="not after"):
        _parse_intervals("2024-01-02T00:00:00/2024-01-01T00:00:00")

    def _msg(text):
        with pytest.raises(ValueError) as exc:
            _parse_intervals(text)
        return str(exc.value)

    # the error teaches the format: a concrete, correct example inlined
    # (2026-10-16: the user preferred the strict rule + a helpful message
    # over a forgiving parser)
    for bad in ("2024-01-01T00:00:00",                    # no pair at all
                "2024-01-01 14:00 ; 2024-01-01 23:00",    # ';' between the pair
                "2024-01-01T00:00/2024-01-01T01:00 ; oops"):
        m = _msg(bad)
        assert "start/end" in m and "e.g." in m and "2024-01-01T00:00/" in m, m


def test_hampel_flags_an_impulse_and_not_the_signal():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(512)
    x[256] = 100.0
    flags = _hampel_flags(x, 16, 3.0)
    assert flags[256]
    # the natural signal's own variation survives: a short-window MAD is
    # noisy, so a handful of Gaussian samples flag too — but the spike
    # dominates the count by orders of magnitude in amplitude terms
    assert flags.sum() <= 24


def test_flat_run_flags_clips_and_zero_gaps_but_not_noise():
    x = np.random.default_rng(2).standard_normal(256)
    x[50:60] = 3.0  # clip
    x[200:210] = 0.0  # zero-filled gap
    flags = _flat_run_flags(x, 8)
    assert flags[50:60].all() and flags[200:210].all()
    assert flags.sum() == 20  # exactly the two runs — nothing else on noise
    assert not _flat_run_flags(x, 8)[0:50].any()


def test_auto_window_scales_with_the_member_span():
    # 12 h at 1 Hz: the 60 s cap applies
    assert _auto_hampel_window(12 * 3600, 1.0) == 60
    # a 1-s slot at 2400 Hz: span/8 = 125 ms = 300 samples
    assert _auto_hampel_window(2400, 2400.0) == 300
    # a 32-sample slot at 1 Hz: span/8 = 4 -> the 16-sample floor
    assert _auto_hampel_window(32, 1.0) == 16
    # a member of 16 samples or fewer: the window would cover it all
    assert _auto_hampel_window(16, 1.0) is None
    # a 64-sample slot at 1 Hz: span/8 = 8 -> floored to 16
    assert _auto_hampel_window(64, 1.0) == 16


# --- node behavior ------------------------------------------------------------


def test_mask_ts_flags_spikes_clips_and_writes_the_mask(tmp_path):
    ctx = _ctx(tmp_path)
    payload = _ds()
    out = MaskTS().execute(ctx, payload)
    mask = _mask_of(out)
    assert mask[100]  # the spike
    assert mask[300:310].all()  # the constant run
    # the mask is an auxiliary channel; data channels are UNTOUCHED
    assert out["gated"].dataset["quality_mask"].attrs["type"] == "auxiliary"
    for c in ("hx", "hy", "ex", "ey"):
        np.testing.assert_array_equal(
            out["gated"].dataset[c].values, payload.dataset[c].values)
    rows = _report_of(out)
    criteria = {r["criterion"] for r in rows}
    assert {"hampel", "flat_run", "total"} <= criteria


def test_mask_ts_default_threshold_is_the_manifest_value(tmp_path):
    """M27: omitting hampel_threshold must gate like 8.0 (the manifest
    default), not the old 3.0 — direct/headless calls then match the
    executor. 3.0 over-gates clean Gaussian noise (the manifest's own
    rationale: ~5% of samples, collapsing FC band coverage)."""
    ctx = _ctx(tmp_path)
    payload = _ds(n=512, spike_at=None, clip_at=None)
    default = _mask_of(MaskTS().execute(ctx, payload))
    at8 = _mask_of(MaskTS().execute(ctx, payload, hampel_threshold=8.0))
    at3 = _mask_of(MaskTS().execute(ctx, payload, hampel_threshold=3.0))
    np.testing.assert_array_equal(default, at8)
    assert at3.sum() > at8.sum()  # the over-gating the manifest rejects


def test_mask_ts_unions_the_existing_saturated_mask(tmp_path):

    ctx = _ctx(tmp_path)
    payload = _ds(spike_at=None, clip_at=None)
    ds = payload.dataset
    existing = np.zeros(ds.sizes["time"], dtype=np.uint8)
    existing[7:9] = 1  # the loader's own tags
    ds["saturated"] = ("time", existing, {"type": "auxiliary", "units": "1"})
    out = MaskTS().execute(ctx, payload, union_existing=True)
    mask = _mask_of(out)
    assert mask[7:9].all()  # the loader's tags survive the union


def test_mask_ts_interval_exclusion(tmp_path):
    ctx = _ctx(tmp_path)
    payload = _ds(n=64, spike_at=None, clip_at=None)
    out = MaskTS().execute(
        ctx, payload,
        exclude_intervals="2024-01-01T00:00:20/2024-01-01T00:00:30")
    mask = _mask_of(out)
    assert mask[20:30].all()
    # the interval row counts exactly the clamped span (Hampel may flag
    # additional noise samples elsewhere — the criteria OR-compose)
    rows = _report_of(out)
    iv = next(r for r in rows if r["criterion"] == "interval")
    assert iv["n_flagged"] == 10
    # out-of-record intervals warn and are ignored, never rejected
    out2 = MaskTS().execute(
        ctx, payload,
        exclude_intervals="2030-01-01T00:00:00/2030-01-01T01:00:00")
    rows2 = _report_of(out2)
    assert not [r for r in rows2 if r["criterion"] == "interval"]


def test_mask_ts_auto_window_is_slot_local(tmp_path):
    """A 1-s 2400 Hz slot: the auto window is 300 samples (span/8), not
    the 60-s continuous cap — and the note records the effective window."""
    ctx = _ctx(tmp_path)
    n = 2400
    payload = _ds(n=n, spike_at=None, clip_at=None)
    payload.dataset = payload.dataset.isel(time=slice(0, n))
    payload.dataset["hx"].values[:] = np.random.default_rng(9).standard_normal(n)
    # rebuild the time coord at 2400 Hz
    t = (pd.Timestamp("2024-01-01T00:00:00").tz_localize(None)
         + pd.to_timedelta(np.arange(n) / 2400.0, unit="s")).values
    payload.dataset = payload.dataset.assign_coords(time=t)
    payload.sample_rate = 2400.0
    out = MaskTS().execute(ctx, payload)
    notes = [r["note"] for r in _report_of(out) if r["criterion"] == "hampel"]
    assert any("300" in n_ for n_ in notes)


def test_mask_ts_axis_gaps_are_reported_not_fabricated(tmp_path, caplog):
    ctx = _ctx(tmp_path)
    n = 128
    payload = _ds(n=n, spike_at=None, clip_at=None)
    # cut a hole: drop samples 60..67 — the axis now jumps 8 s
    keep = np.ones(n, bool)
    keep[60:68] = False
    payload.dataset = payload.dataset.isel(time=keep)
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        out = MaskTS().execute(ctx, payload, detect_gaps=True)
    assert any("gap" in r.message for r in caplog.records)
    gaps = [r for r in _report_of(out) if r["criterion"] == "axis_gap"]
    # the interval spans the hole: last sample before .. first after
    assert gaps and "00:00:59" in gaps[0]["intervals"]
    assert "00:01:08" in gaps[0]["intervals"]


def test_mask_ts_is_functional(tmp_path):
    ctx = _ctx(tmp_path)
    payload = _ds()
    before = {k: v.copy() for k, v in payload.dataset.items()}
    MaskTS().execute(ctx, payload)
    for k, v in before.items():
        np.testing.assert_array_equal(payload.dataset[k].values, v.values)


def test_mask_ts_maps_collections(tmp_path):
    ctx = _ctx(tmp_path)
    members = [_ds(seed=s, spike_at=50 + i, clip_at=None)
               for i, s in enumerate((3, 4))]
    coll = SimpleNamespace(
        runs=members,
        extents=[{"run": "r1", "sample_rate": 1.0},
                 {"run": "r2", "sample_rate": 1.0}])
    out = MaskTS().execute(ctx, coll)
    assert hasattr(out["gated"], "runs")
    assert len(out["gated"].runs) == 2
    for i, member in enumerate(out["gated"].runs):
        mask = np.asarray(member.dataset["quality_mask"].values) > 0
        assert mask[50 + i]
    rows = _report_of(out)
    assert {r["member"] for r in rows} == {"r1", "r2"}


def test_mask_ts_core_timeseries_path(tmp_path):
    ctx = _ctx(tmp_path)
    ts = TimeSeries(
        channels={"hx": np.random.default_rng(1).standard_normal(256),
                  "saturated": np.zeros(256, dtype=np.uint8)},
        sample_rate=1.0, run="a", aux_channels=frozenset({"saturated"}))
    ts.channels["hx"][40] = 500.0
    out = MaskTS().execute(ctx, ts)
    gated = out["gated"]
    assert "quality_mask" in gated.channels
    assert "quality_mask" in gated.aux_channels
    assert gated.aux_channels >= {"saturated", "quality_mask"}
    mask = np.asarray(gated.channels["quality_mask"]) > 0
    assert mask[40]
    np.testing.assert_array_equal(
        gated.channels["saturated"], ts.channels["saturated"])


def test_mask_ts_loud_on_mostly_bad(tmp_path, caplog):
    ctx = _ctx(tmp_path)
    payload = _ds(n=256, spike_at=None, clip_at=None)
    payload.dataset["ex"].values[:] = 1.0  # a dead/stuck channel: all flat
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        MaskTS().execute(ctx, payload)
    assert any("flagged bad" in r.message for r in caplog.records)


def test_mask_ts_registered_with_a_report_port():
    manifest = MaskTS().manifest()
    assert manifest["outputs"][1]["type"] == "table"
    assert manifest["inputs"][0]["collection_ok"] is True


def test_mask_ts_deliberate_exclusion_does_not_warn(tmp_path, caplog):
    """A range the USER deliberately excluded is not a misconfiguration
    signal: a member that is 'bad' only inside exclude_intervals stays
    quiet (2026-10-16: a 1-h exclusion on a burst collection flagged
    hundreds of 2-s members 100% and drowned the card in warnings)."""
    ctx = _ctx(tmp_path)
    payload = _ds(n=256, spike_at=None, clip_at=None)
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        MaskTS().execute(
            ctx, payload,
            exclude_intervals="1980-01-01T00:00:00/1980-01-01T00:05:00")
    assert not any("flagged bad" in r.message for r in caplog.records)


def test_mask_ts_still_warns_on_involuntary_badness(tmp_path, caplog):
    """Deliberate exclusions don't mask a real misconfiguration: a dead
    channel BEYOND the excluded range still warns (the involuntary
    fraction is what the >50% rule judges)."""
    ctx = _ctx(tmp_path)
    payload = _ds(n=256, spike_at=None, clip_at=None)
    payload.dataset["ex"].values[:] = 1.0  # dead across the whole record
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        MaskTS().execute(
            ctx, payload,
            exclude_intervals="1980-01-01T00:01:00/1980-01-01T00:02:00")
    assert any("flagged bad" in r.message for r in caplog.records)
    assert any("by the criteria" in r.message for r in caplog.records)
