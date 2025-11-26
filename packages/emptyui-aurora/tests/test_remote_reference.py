"""End-to-end remote reference: two-station alignment + decimated RME_RR.

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
from emptyui.context import RunContext
from emptyui.executor import Executor
from emptyui.registry import build_default_registry
from emptyui.schema import Edge, Workflow, WorkflowNode
from emptyui.store import DataStore
from emptyui_aurora._processing import (
    compute_fc_levels,
    estimate_tf,
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
    rng = np.random.default_rng(21)
    fs = 8.0
    n = 16384
    hx_src = rng.standard_normal(n)
    hy_src = rng.standard_normal(n)
    local = _Run({
        "bx": hx_src + 0.1 * rng.standard_normal(n),
        "by": hy_src + 0.1 * rng.standard_normal(n),
        "e1": 1.0 * hx_src + 3.0 * hy_src + 0.2 * rng.standard_normal(n),
        "e2": -1.0 * hx_src + 4.0 * hy_src + 0.2 * rng.standard_normal(n),
    }, fs, start_datetime="2024-01-01T00:00:00")
    remote = _Run({
        "bx": hx_src + 0.1 * rng.standard_normal(n),
        "by": hy_src + 0.1 * rng.standard_normal(n),
    }, fs, start_datetime="2024-01-01T00:04:00")
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


def test_align_runs_no_overlap_errors(tmp_path):
    a = _Run({"bx": np.zeros(64)}, 8.0, start_datetime="2024-01-01T00:00:00")
    b = _Run({"bx": np.zeros(64)}, 8.0, start_datetime="2024-01-02T00:00:00")
    with pytest.raises(ValueError, match="do not overlap"):
        AlignRuns().execute(_ctx(tmp_path), a, b)


def test_align_runs_sample_rate_mismatch_errors(tmp_path):
    a = _Run({"bx": np.zeros(64)}, 8.0, start_datetime="2024-01-01T00:00:00")
    b = _Run({"bx": np.zeros(64)}, 1.0, start_datetime="2024-01-01T00:00:00")
    with pytest.raises(ValueError, match="sample rates"):
        AlignRuns().execute(_ctx(tmp_path), a, b)


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
    _, Z = merge_tf_levels(per)
    assert np.all(np.isfinite(Z))
    ok = np.abs(Z[:, 0, 1]) > 1e-9
    # true impedance [[1,3],[-1,4]] (ex x hx, ex x hy, ey x hx, ey x hy)
    assert np.median(Z[ok, 0, 0].real) == pytest.approx(1.0, abs=0.5)
    assert np.median(Z[ok, 0, 1].real) == pytest.approx(3.0, abs=0.5)
    assert np.median(Z[ok, 1, 0].real) == pytest.approx(-1.0, abs=0.5)
    assert np.median(Z[ok, 1, 1].real) == pytest.approx(4.0, abs=0.5)


def test_remote_reference_unaligned_time_grid_errors(tmp_path):
    """FCs with identical frequencies but different window (time) grids must
    fail with a clear error — the runs were not aligned."""
    local, _ = _synthetic_stations()
    shifted = _Run({
        "bx": local.channels["bx"][:-320],
        "by": local.channels["by"][:-320],
        "e1": local.channels["e1"][:-320],
        "e2": local.channels["e2"][:-320],
    }, local.sample_rate, start_datetime="2024-01-01T00:00:40")
    fc = compute_fc_levels(local, ["bx", "by"], ["e1", "e2"],
                           num_samples_window=256, overlap=0.5,
                           window_type="hamming")[0]
    ref = compute_fc_levels(shifted, ["bx", "by"], ["e1", "e2"],
                            num_samples_window=256, overlap=0.5,
                            window_type="hamming")[0]
    with pytest.raises(ValueError, match="time grid"):
        estimate_tf(fc, ["bx", "by"], ["e1", "e2"],
                    estimator_engine="Robust", remote_reference=True,
                    ref_fc=ref, periods_per_octave=4)


def _node(nid, typ, params):
    return WorkflowNode(id=nid, type=typ, params=params)


def _edge(eid, s, sp, t, tp):
    return Edge(id=eid, source=s, source_port=sp, target=t, target_port=tp)


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
    r1 = ExtractRun().execute(ctx, m1, survey="bgs", station="NX37")["data"]
    r2 = ExtractRun().execute(ctx, m2, survey="bgs", station="NO46")["data"]
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
            _node("l1", "mth5.io.lemi", {
                "data_path": str(NO46_DIR), "survey_id": "bgs",
                "station_id": "NO46", "mth5_filename": "no46.mth5",
                "save_path": str(out_dir), "channel_nomenclature": "lemi12"}),
            _node("x1", "mth5.io.extract_run", {"survey": "bgs", "station": "NO46"}),
            _node("l2", "mth5.io.lemi", {
                "data_path": str(NX37_DIR), "survey_id": "bgs",
                "station_id": "NX37", "mth5_filename": "nx37.mth5",
                "save_path": str(out_dir), "channel_nomenclature": "lemi12"}),
            _node("x2", "mth5.io.extract_run", {"survey": "bgs", "station": "NX37"}),
            _node("al", "mth5.runs.align", {}),
            _node("c1", "aurora.process.compute_fc", {
                "input_channels": "bx,by", "output_channels": "e1,e2,bz",
                "channel_nomenclature": "lemi12", "window_samples": 256,
                "decimation_levels": 2}),
            _node("c2", "aurora.process.compute_fc", {
                "input_channels": "bx,by", "output_channels": "e1,e2,bz",
                "channel_nomenclature": "lemi12", "window_samples": 256,
                "decimation_levels": 2}),
            _node("t", "aurora.process.estimate_tf", {
                "estimator_engine": "Robust", "remote_reference": True}),
            _node("e", "mth5.tf.export", {
                "path": str(out_dir / "rr_tf"), "file_type": "edi"}),
        ],
        edges=[
            _edge("e1", "l1", "mth5", "x1", "mth5"),
            _edge("e2", "l2", "mth5", "x2", "mth5"),
            _edge("e3", "x1", "data", "al", "data1"),
            _edge("e4", "x2", "data", "al", "data2"),
            _edge("e5", "al", "data1", "c1", "data"),
            _edge("e6", "al", "data2", "c2", "data"),
            _edge("e7", "c1", "fc", "t", "fc"),
            _edge("e8", "c2", "fc", "t", "ref_fc"),
            _edge("e9", "t", "tf", "e", "tf"),
        ],
    )
    summary = Executor(build_default_registry()).run(
        workflow, emit=lambda e: None, workspace=str(tmp_path))
    assert summary["status"] == "completed", summary
    edi = out_dir / "rr_tf.edi"
    assert edi.exists() and edi.stat().st_size > 0
