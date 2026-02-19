"""End-to-end remote reference: two-station alignment + decimated RME_RR.
Error surfaces live in test_errors.py.

Two parts:
- Synthetic two-station test (always runs, CI-safe): a common source field
  with independent local/remote noise, overlapping but OFFSET absolute
  acquisition times. Align Runs crops both to the common interval, then
  multi-level (decimated) FCs are computed on both and RME_RR recovers the
  true impedance.
- Real-data test on the BGS LEMI sites NO46 + NX37
  (examples/data/bgs-lemi-424, gitignored): full workflow through the
  executor — Load LEMI x2 -> Extract Run x2 -> Align Runs -> Compute FC x2
  (identical settings) -> Estimate TF (Robust + remote reference) -> Export
  EDI. Skipped when the data is absent (CI without the raw data).

The remote station is magnetic-only (bx,by), as in the field, where the
remote site needs no electric channels for a remote reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _synthetic import synthetic_fc, synthetic_runts, synthetic_source
from _workflow import edge, node
from emptyui.context import RunContext
from emptyui.executor import Executor
from emptyui.registry import build_default_registry
from emptyui.schema import Workflow
from emptyui.store import DataStore
from emptyui_aurora._processing import (
    compute_fc_levels,
    estimate_tf_levels,
    merge_tf_levels,
)
from emptyui_mth5.runs import AlignRuns

DATA_DIR = Path(__file__).resolve().parents[3] / "examples" / "data" / "bgs-lemi-424"
NO46_DIR = DATA_DIR / "lemi_txt" / "NO46"
NX37_DIR = DATA_DIR / "lemi_txt" / "NX37"


class _Run:
    """Minimal core-TimeSeries-like ts.run payload (channels + time info)."""

    def __init__(self, channels, sample_rate, start_datetime=""):
        self.channels = {k: np.asarray(v, dtype=float) for k, v in channels.items()}
        self.sample_rate = sample_rate
        self.start = 0.0
        self.start_datetime = start_datetime


def _ctx(tmp_path):
    return RunContext(workspace=tmp_path, store=DataStore())


def _synthetic_stations():
    """Two stations sharing a common source field, offset acquisition times.

    Station A (local): bx,by + e1,e2 (impedance [[1,3],[-1,4]]).
    Station B (remote): magnetic-only bx,by, same sampling/setup, starting
    240 s later — the overlap is what Align Runs must find.
    """
    source = synthetic_source(21, n=16384, kind="white")
    local = synthetic_runts(source, magnetic=("bx", "by"),
                            electric=("e1", "e2"), z=(1.0, 3.0, -1.0, 4.0),
                            e_noise=0.2,
                            start_datetime="2024-01-01T00:00:00", seed=21)
    remote = synthetic_runts(source, magnetic=("bx", "by"), with_e=False,
                             start_datetime="2024-01-01T00:04:00", seed=22)
    return local, remote


def test_align_runs_finds_overlap_and_normalizes(tmp_path):
    local, remote = _synthetic_stations()
    out = AlignRuns().execute(_ctx(tmp_path), local, remote)
    a, b = out["data1"], out["data2"]
    # both cropped to the identical absolute interval
    assert a.sample_rate == b.sample_rate == 8.0
    overlap_start = pd.Timestamp("2024-01-01T00:04:00", tz="UTC")
    assert pd.Timestamp(str(a.start)) == overlap_start
    assert pd.Timestamp(str(a.start)) == pd.Timestamp(str(b.start))
    n = len(a.dataset.data_vars["bx"].values)
    assert n == len(b.dataset.data_vars["bx"].values)
    assert n == 16384 - 240 * 8  # overlap only
    assert list(a.dataset.data_vars) == ["bx", "by", "e1", "e2"]
    assert list(b.dataset.data_vars) == ["bx", "by"]


def test_remote_reference_end_to_end_decimated(tmp_path):
    """Align -> multi-level (decimated) FCs on both stations -> RME_RR
    recovers the true impedance. This is the gap being closed: a decimated
    remote FC paired with a local FC."""
    local, remote = _synthetic_stations()
    aligned = AlignRuns().execute(_ctx(tmp_path), local, remote)
    a, b = aligned["data1"], aligned["data2"]

    ins, outs = ["bx", "by"], ["e1", "e2"]
    # unit-neutral processing: treat the synthetic signals as H already
    # (magnetic_unit="h") and use a dipole length that makes the E scaling
    # identity (1e-3/1e-3), so the recovered Z matches the simulation values.
    fc_kwargs = {
        "num_decimations": 2, "factor": 4, "num_samples_window": 256,
        "overlap": 0.5, "window_type": "hamming", "magnetic_unit": "h",
        "e_dipole_length_m": 0.001,
    }
    levels_a = compute_fc_levels(a, ins, outs, **fc_kwargs)
    # remote station is magnetic-only: empty output channels resolve to the
    # magnetic channels (reference fc) in auto mode
    levels_b = compute_fc_levels(b, ins, "", **fc_kwargs)
    assert levels_b[0].attrs["output_channels"] == "bx,by"
    # identical frequency AND time grids per level (the RME_RR pairing rule)
    for lv in (0, 1, 2):
        assert np.array_equal(levels_a[lv]["frequency"].values,
                              levels_b[lv]["frequency"].values)
        assert np.array_equal(levels_a[lv]["time"].values,
                              levels_b[lv]["time"].values)

    per = estimate_tf_levels(levels_a, ins, outs, estimator_engine="Robust",
                             remote_reference=True, periods_per_octave=4,
                             ref_levels=levels_b, channel_nomenclature="lemi12")
    _, Z, _ = merge_tf_levels(per)
    assert np.all(np.isfinite(Z))
    ok = np.abs(Z[:, 0, 1]) > 1e-9
    # true impedance [[1,3],[-1,4]] (ex x hx, ex x hy, ey x hx, ey x hy)
    assert np.median(Z[ok, 0, 0].real) == pytest.approx(1.0, abs=0.5)
    assert np.median(Z[ok, 0, 1].real) == pytest.approx(3.0, abs=0.5)
    assert np.median(Z[ok, 1, 0].real) == pytest.approx(-1.0, abs=0.5)
    assert np.median(Z[ok, 1, 1].real) == pytest.approx(4.0, abs=0.5)


@pytest.mark.skipif(
    not (NO46_DIR / "NO46_lemi2d.txt").exists()
    or not (NX37_DIR / "NX37_lemi2d.txt").exists(),
    reason="BGS LEMI data not on disk (gitignored)",
)
def test_bgs_stations_coherent_after_alignment(tmp_path):
    """Regression guard for station alignment: after Align Runs the two sites
    must be strongly coherent at long periods (a data gap or clock offset
    would break this and silently degrade remote reference)."""
    from emptyui_mth5.lemi import LoadLemi
    from emptyui_mth5.runs import ExtractRun
    from scipy.signal import coherence
    ctx = _ctx(tmp_path)
    m1 = LoadLemi().execute(ctx, data_path=str(NX37_DIR), survey_id="bgs",
                            station_id="NX37", mth5_filename="nx37.mth5",
                            save_path=str(tmp_path),
                            channel_nomenclature="lemi12")["mth5"]
    m2 = LoadLemi().execute(ctx, data_path=str(NO46_DIR), survey_id="bgs",
                            station_id="NO46", mth5_filename="no46.mth5",
                            save_path=str(tmp_path),
                            channel_nomenclature="lemi12")["mth5"]
    r1 = ExtractRun().execute(ctx, m1, selection="run id", survey="bgs", station="NX37")["data"]
    r2 = ExtractRun().execute(ctx, m2, selection="run id", survey="bgs", station="NO46")["data"]
    out = AlignRuns().execute(ctx, r1, r2)
    a, b = out["data1"], out["data2"]
    assert len(a.dataset.time) == len(b.dataset.time)
    x = a.dataset["bx"].values
    y = b.dataset["bx"].values
    nperseg = min(4096, len(x) // 4)
    f, C = coherence(x, y, fs=1.0, nperseg=nperseg, noverlap=nperseg // 2)
    band = (f >= 1.0 / 1024.0) & (f <= 1.0 / 64.0)
    assert np.nanmedian(C[band]) > 0.5, (
        "stations are not coherent after alignment — check for data gaps "
        "(mt_io indexes LEMI files by row, so gaps must be filled at "
        "conversion)"
    )


@pytest.mark.skipif(
    not (NO46_DIR / "NO46_lemi2d.txt").exists()
    or not (NX37_DIR / "NX37_lemi2d.txt").exists(),
    reason="BGS LEMI data not on disk (gitignored)",
)
def test_remote_reference_real_data_workflow(tmp_path):
    """Full workflow through the executor: two BGS sites (NO46 + NX37),
    aligned to their common overlap, decimated FCs, RME_RR, EDI export."""
    out_dir = tmp_path / "rr"
    out_dir.mkdir()
    workflow = Workflow(
        nodes=[
            node("l1", "mth5.io.lemi", {
                "data_path": str(NO46_DIR), "survey_id": "bgs",
                "station_id": "NO46", "mth5_filename": "no46.mth5",
                "save_path": str(out_dir), "channel_nomenclature": "lemi12"}),
            node("x1", "mth5.io.extract_run", {"selection": "run id", "survey": "bgs", "station": "NO46"}),
            node("l2", "mth5.io.lemi", {
                "data_path": str(NX37_DIR), "survey_id": "bgs",
                "station_id": "NX37", "mth5_filename": "nx37.mth5",
                "save_path": str(out_dir), "channel_nomenclature": "lemi12"}),
            node("x2", "mth5.io.extract_run", {"selection": "run id", "survey": "bgs", "station": "NX37"}),
            node("al", "mth5.runs.align", {}),
            node("c1", "aurora.process.compute_fc", {
                "input_channels": "bx,by", "output_channels": "e1,e2,bz",
                "channel_nomenclature": "lemi12", "window_samples": 256,
                "decimation_levels": 2}),
            node("c2", "aurora.process.compute_fc", {
                "input_channels": "bx,by", "output_channels": "e1,e2,bz",
                "channel_nomenclature": "lemi12", "window_samples": 256,
                "decimation_levels": 2}),
            node("t", "aurora.process.estimate_tf", {
                "estimator_engine": "Robust", "remote_reference": True}),
            node("e", "mth5.tf.export", {
                "path": str(out_dir / "rr_tf"), "file_type": "edi"}),
        ],
        edges=[
            edge("e1", "l1", "mth5", "x1", "mth5"),
            edge("e2", "l2", "mth5", "x2", "mth5"),
            edge("e3", "x1", "data", "al", "data1"),
            edge("e4", "x2", "data", "al", "data2"),
            edge("e5", "al", "data1", "c1", "data"),
            edge("e6", "al", "data2", "c2", "data"),
            edge("e7", "c1", "fc", "t", "fc"),
            edge("e8", "c2", "fc", "t", "ref_fc"),
            edge("e9", "t", "tf", "e", "tf"),
        ],
    )
    summary = Executor(build_default_registry()).run(
        workflow, emit=lambda e: None, workspace=str(tmp_path))
    assert summary["status"] == "completed", summary
    edi = out_dir / "rr_tf.edi"
    assert edi.exists() and edi.stat().st_size > 0


def test_coherence_guard_resolves_nomenclature_names(caplog):
    """The long-period coherence diagnostic must find the horizontal
    magnetic channels through the nomenclature: LEMI data keeps bx/by names
    (the guard previously matched literal hx/hy and never fired on exactly
    the data it was built for — 2026-09-11 review fix)."""
    import logging

    import xarray as xr
    from emptyui_mth5.runs import _warn_if_uncorrelated

    sr, n = 8.0, 16384
    t = np.arange(n) / sr
    src = np.sin(2 * np.pi * 0.1 * t)  # common source, long-period enough

    def _payload(a_vals, b_vals, names=("bx", "by")):
        class P:
            pass
        p = P()
        p.dataset = xr.Dataset(
            {names[0]: ("time", a_vals), names[1]: ("time", b_vals)},
            coords={"time": t},
        )
        return p

    rng = np.random.default_rng(31)
    # strongly coherent: B = A + small independent noise
    good1 = _payload(src, src + 0.05 * rng.standard_normal(n))
    good2 = _payload(src, src + 0.05 * rng.standard_normal(n))
    # weakly coherent: independent realizations
    bad1 = _payload(rng.standard_normal(n), rng.standard_normal(n))
    bad2 = _payload(rng.standard_normal(n), rng.standard_normal(n))

    class Ctx:
        logger = logging.getLogger("emptyui.align-test")

    with caplog.at_level(logging.WARNING, logger="emptyui.align-test"):
        _warn_if_uncorrelated(Ctx(), good1, good2, sr, n)
        assert not any(r.levelname == "WARNING" for r in caplog.records)

        caplog.clear()
        _warn_if_uncorrelated(Ctx(), bad1, bad2, sr, n)
        assert any("weakly coherent" in r.message for r in caplog.records)

    # standard-named data (hx/hy) still works after the change
    std1 = _payload(src, src, names=("hx", "hy"))
    std2 = _payload(src, src + 0.05 * rng.standard_normal(n), names=("hx", "hy"))
    with caplog.at_level(logging.WARNING, logger="emptyui.align-test"):
        caplog.clear()
        _warn_if_uncorrelated(Ctx(), std1, std2, sr, n)
        assert not any(r.levelname == "WARNING" for r in caplog.records)


def test_placeholder_epoch_start_warns_and_aligns_by_relative(caplog):
    """A run stamped with a placeholder epoch (1980 mt_timeseries default /
    1970 Unix default — digitized paper records) is treated as 'start
    unknown' with a warning — never rejected — and alignment falls back to
    relative offsets (review #10 bundle)."""
    import logging

    import xarray as xr
    from emptyui_mth5.runs import _run_time_info

    sr, n = 8.0, 2048

    class P:
        sample_rate = sr
        start_datetime = ""

        def __init__(self, start):
            t = np.arange(n) / sr
            self.dataset = xr.Dataset(
                {"bx": ("time", np.sin(2 * np.pi * 0.5 * t) + 1.0),
                 "by": ("time", np.sin(2 * np.pi * 1.3 * t) + 1.0),
                 "e1": ("time", np.sin(2 * np.pi * 0.5 * t) + 2.0),
                 "e2": ("time", np.sin(2 * np.pi * 1.3 * t) + 2.0)},
                coords={"time": t},
            )
            self.start = start

    with caplog.at_level(logging.WARNING, logger="emptyui"):
        for epoch in ("1980-01-01T00:00:00", "1970-01-01T00:00:00"):
            _ch, s, a, _r = _run_time_info(P(epoch))
            assert a is None  # placeholder → no absolute start
            assert s == sr
        assert sum("placeholder epoch" in r.message
                   for r in caplog.records) == 2

    # a genuinely old (1957, Cagniard-era) digitized start is preserved —
    # normalized to timezone-aware UTC (naive stamps are assumed UTC)
    import pandas as pd

    _ch, _s, a, _r = _run_time_info(P(pd.Timestamp("1957-06-15T12:00:00")))
    assert a == pd.Timestamp("1957-06-15T12:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# M4 #2: warn when a local run spans multiple remote runs. The harmful
# variant is a remote merge that bridged acquisition gaps by interpolation
# (Merge Runs): the local windows inside the gap pair with fabricated remote
# samples while the identical-grid check still passes. Provenance rides the
# payloads (merged_gap_intervals) from Merge Runs through Align Runs /
# Compute FC to the pairing sites.


def test_merge_runts_gap_provenance_travels_through_align(tmp_path, caplog):
    """Align Runs warns when the aligned overlap includes interpolated
    gap(s) and carries the provenance onto both outputs."""
    import logging

    from emptyui_mth5.merge import merge_runts

    local = synthetic_runts(synthetic_source(31, n=8192, kind="white"),
                            start_datetime="2024-01-01T00:00:00", seed=31)
    # remote = two gapped runs physically merged (the E1b recipe): r1 ends
    # 00:12:32, r2 starts 00:14:00 — an 88-s interpolated gap inside the
    # aligned overlap [00:04:00, 00:17:04). Real RunTS objects are required
    # (merge_runts concatenates via RunTS.merge).
    from mt_timeseries import ChannelTS, RunTS

    rng = np.random.default_rng(32)
    pieces = []
    for rid, start, n in (("b", "2024-01-01T00:04:00", 4096),
                          ("c", "2024-01-01T00:14:00", 4096)):
        chs = []
        for comp in ("bx", "by"):
            c = ChannelTS("magnetic", data=rng.standard_normal(n),
                          start=start, sample_rate=8.0)
            c.component = comp
            chs.append(c)
        rt = RunTS(array_list=chs)
        rt.run_metadata.id = rid
        pieces.append(rt)
    remote = merge_runts(pieces, new_run_id="bc")
    attrs = remote.dataset.attrs
    assert attrs["merged_gap_intervals"]

    with caplog.at_level(logging.WARNING, logger="emptyui"):
        out = AlignRuns().execute(_ctx(tmp_path), local, remote)
    a, b = out["data1"], out["data2"]
    # the local run (00:00:00, 1024 s) overlaps the remote from 00:04:00 on;
    # the remote's gap (00:20:48..00:21:20 for 8192-sample r1) is inside the
    # overlap -> the warning fires and the provenance is carried
    assert any("interpolated gap" in r.message for r in caplog.records)
    # the provenance rides the merged (remote) output — not the local one
    assert "merged_gap_intervals" not in a.dataset.attrs
    assert b.dataset.attrs["merged_gap_intervals"] == attrs["merged_gap_intervals"]


def test_align_runs_warns_partial_coverage(tmp_path, caplog):
    """Aligning against a partial remote discards local data — warn with the
    discarded share (the E1b silent-loss failure mode)."""
    import logging

    local = synthetic_runts(synthetic_source(34, n=8192, kind="white"),
                            start_datetime="2024-01-01T00:00:00", seed=34)
    # remote: starts with the local but covers only half its span
    remote = synthetic_runts(synthetic_source(35, n=4096, kind="white"),
                             start_datetime="2024-01-01T00:00:00", seed=35)
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        AlignRuns().execute(_ctx(tmp_path), local, remote)
    assert any("alignment discards" in r.message and "input 1" in r.message
               for r in caplog.records)


def _aligned_pair_fcs():
    """(local_fc, ref_fc) with identical absolute time grids."""
    source = synthetic_source(36, n=8192, kind="white")
    local = synthetic_runts(source, magnetic=("hx", "hy"),
                            electric=("ex", "ey"), z=(1.0, 3.0, -1.0, 4.0),
                            start_datetime="2024-01-01T00:00:00", seed=36)
    remote = synthetic_runts(source, magnetic=("hx", "hy"), with_e=False,
                             start_datetime="2024-01-01T00:00:00", seed=37)
    fc = synthetic_fc(local, ("hx", "hy"), ("ex", "ey"))
    ref = synthetic_fc(remote, ("hx", "hy"), ("hx", "hy"))
    return fc, ref


def test_estimate_tf_warns_on_remote_gaps(caplog):
    """The RR pairing warns when the ref fc's provenance marks interpolated
    gap(s) inside the local span; silent when the gaps do not overlap."""
    import logging

    from emptyui_aurora._processing import estimate_tf

    fc, ref = _aligned_pair_fcs()
    # no provenance -> no warning even with remote reference
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="emptyui_aurora"):
        estimate_tf(fc, ("hx", "hy"), ("ex", "ey"),
                    estimator_engine="RME_RR", remote_reference=True,
                    ref_fc=ref)
    assert not any("interpolated gap" in r.message for r in caplog.records)

    # a gap interval that overlaps some local windows -> warning
    t = pd.DatetimeIndex(fc["time"].values)
    g0, g1 = t[3], t[6]
    ref.attrs["merged_gap_intervals"] = [
        [g0.isoformat(), g1.isoformat()]]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="emptyui_aurora"):
        estimate_tf(fc, ("hx", "hy"), ("ex", "ey"),
                    estimator_engine="RME_RR", remote_reference=True,
                    ref_fc=ref)
    assert any("interpolated gap" in r.message for r in caplog.records)

    # a gap interval entirely outside the local span -> silent
    outside = t[0] - pd.Timedelta(days=1)
    ref.attrs["merged_gap_intervals"] = [
        [outside.isoformat(),
         (outside + pd.Timedelta(hours=1)).isoformat()]]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="emptyui_aurora"):
        estimate_tf(fc, ("hx", "hy"), ("ex", "ey"),
                    estimator_engine="RME_RR", remote_reference=True,
                    ref_fc=ref)
    assert not any("interpolated gap" in r.message for r in caplog.records)


def test_coherence_sort_warns_on_remote_gaps(caplog):
    """The two-FC sort pairs remote rows too — the same provenance warning
    applies there."""
    import logging

    from emptyui_aurora._processing import coherence_sort_fc

    fc, ref = _aligned_pair_fcs()
    t = pd.DatetimeIndex(fc["time"].values)
    ref.attrs["merged_gap_intervals"] = [
        [t[2].isoformat(), t[5].isoformat()]]
    with caplog.at_level(logging.WARNING, logger="emptyui_aurora"):
        coherence_sort_fc(fc, ref, ("hx", "hy"), ("ex", "ey"))
    assert any("interpolated gap" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Window-grid intersection pairing (§9.3 fine layer): the time dimension
# pairs by matched window times with loud row drops (the SSMT2000 slot
# semantics); the frequency grid stays strict (duty-cycle/window mismatch
# refuses, as upstream). Default mode 'intersect'; 'strict' preserved for
# the Align Runs / parity contract.


def _paired_pair_fcs(extra_ref=2, offset_s=0.0):
    """(local_fc, ref_fc) with the ref carrying `extra_ref` extra windows
    (and, when offset_s > 0, every ref time shifted by that offset)."""
    import xarray as xr

    fc, ref = _aligned_pair_fcs()
    if extra_ref:
        step = (ref["time"].values[-1] - ref["time"].values[-2])
        extra = ref.isel(time=[-1] * extra_ref).assign_coords(
            time=("time", [ref["time"].values[-1] + step * (k + 1)
                           for k in range(extra_ref)]))
        ref = xr.concat([ref, extra], dim="time")
    if offset_s:
        t = ref["time"].values.astype("datetime64[ns]") + np.timedelta64(
            int(offset_s * 1e9), "ns")
        ref = ref.assign_coords(time=("time", t))
    return fc, ref


def test_pairing_intersect_drops_unmatched(caplog):
    """The default pairing matches window times: extra remote windows drop
    with a loud report; the estimate runs on the matched rows."""
    import logging

    from emptyui_aurora._processing import estimate_tf

    fc, ref = _paired_pair_fcs(extra_ref=2)
    with caplog.at_level(logging.WARNING, logger="emptyui_aurora"):
        periods, Z, Z_err = estimate_tf(
            fc, ("hx", "hy"), ("ex", "ey"), estimator_engine="RME_RR",
            remote_reference=True, ref_fc=ref)
    assert any("kept" in r.message and "remote window" in r.message
               for r in caplog.records)
    assert np.all(np.isfinite(Z))
    # strict mode rejects the same input with the legacy error
    with pytest.raises(ValueError, match="different time grid"):
        estimate_tf(fc, ("hx", "hy"), ("ex", "ey"),
                    estimator_engine="RME_RR", remote_reference=True,
                    ref_fc=ref, pairing="strict")


def test_pairing_intersect_tolerance_recovers(caplog):
    """GPS-stray clocks: with tolerance 0 nothing matches (actionable
    error); raising the tolerance pairs the shifted rows one-to-one."""
    import logging

    from emptyui_aurora._processing import estimate_tf

    fc, ref = _paired_pair_fcs(extra_ref=0, offset_s=0.5)
    with pytest.raises(ValueError, match="no remote-reference windows"):
        estimate_tf(fc, ("hx", "hy"), ("ex", "ey"),
                    estimator_engine="RME_RR", remote_reference=True,
                    ref_fc=ref, pairing="intersect", pair_tolerance_s=0.0)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="emptyui_aurora"):
        estimate_tf(fc, ("hx", "hy"), ("ex", "ey"),
                    estimator_engine="RME_RR", remote_reference=True,
                    ref_fc=ref, pairing="intersect", pair_tolerance_s=1.0)
    assert not any("kept" in r.message for r in caplog.records)  # all paired


def test_pairing_frequency_grid_stays_strict():
    """A window-settings (frequency) mismatch refuses in BOTH modes — the
    official SSMT2000 semantics (inconsistent duty cycles refuse to work)."""
    from emptyui_aurora._processing import estimate_tf

    fc, ref = _paired_pair_fcs(extra_ref=0)
    ref2 = ref.isel(frequency=slice(None, -1))
    with pytest.raises(ValueError, match="different frequency grid"):
        estimate_tf(fc, ("hx", "hy"), ("ex", "ey"),
                    estimator_engine="RME_RR", remote_reference=True,
                    ref_fc=ref2, pairing="intersect")
    with pytest.raises(ValueError, match="different frequency grid"):
        estimate_tf(fc, ("hx", "hy"), ("ex", "ey"),
                    estimator_engine="RME_RR", remote_reference=True,
                    ref_fc=ref2, pairing="strict")
