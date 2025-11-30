"""FC persistence round-trip through the workspace MTH5 (DESIGN.md §14
Phase 1). Error surfaces live in test_errors.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import compute_fc
from emptyui_mth5.persist import (
    WORKSPACE_MTH5_NAME,
    load_fc_level,
    persist_fc_level,
    runts_context,
)


def test_fc_persist_roundtrip(tmp_path: Path, fake_runts):
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    runts = fake_runts

    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    # the fc must carry an absolute (datetime64) time axis for persistence
    assert "datetime64" in str(fc.time.dtype)
    assert fc.attrs["start_datetime"]

    survey, station, run = runts_context(runts)
    assert (survey, station, run) == ("ws", "local", "run1")

    path = persist_fc_level(ctx, fc, 0, fc.attrs["sample_rate"],
                            survey=survey, station=station, run=run)
    assert path.name == WORKSPACE_MTH5_NAME
    assert path.exists()

    out = load_fc_level(ctx, 0, survey=survey, station=station, run=run)
    assert set(out.data_vars) == set(fc.data_vars)
    # mth5 stores (time, frequency); our fc is (frequency, time)
    assert out["hx"].shape == (fc.sizes["time"], fc.sizes["frequency"])
    np.testing.assert_allclose(
        np.abs(out["hx"].values.T), np.abs(fc["hx"].values), rtol=1e-6
    )


def test_persist_multiple_decimation_levels(tmp_path: Path, fake_runts):
    """ComputeFC's multi-level fc container persists each level separately."""
    from emptyui_aurora._processing import compute_fc_levels

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    runts = fake_runts
    levels = compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                               num_decimations=1, factor=4, num_samples_window=128,
                               overlap=0.5, window_type="hamming")

    survey, station, run = runts_context(runts)
    for level, fc in levels.items():
        assert "datetime64" in str(fc.time.dtype)
        persist_fc_level(ctx, fc, level, fc.attrs["sample_rate"],
                         survey=survey, station=station, run=run)

    for level in (0, 1):
        out = load_fc_level(ctx, level, survey=survey, station=station, run=run)
        assert set(out.data_vars) == set(levels[level].data_vars)
        assert out["hx"].shape == (
            levels[level].sizes["time"],
            levels[level].sizes["frequency"],
        )


def test_read_back_fc_feeds_estimate_tf(tmp_path: Path, fake_runts):
    """The full FC round-trip through an mth5 archive must produce an fc that
    EstimateTF can consume: attrs enriched (sample_rate, channel roles via the
    nomenclature) and the (time, frequency) layout normalized."""
    from emptyui_aurora._processing import estimate_tf
    from emptyui_mth5.fc import FCGroupToSpectra
    from mth5.mth5 import MTH5

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    runts = fake_runts
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")

    survey, station, run = runts_context(runts)
    path = persist_fc_level(ctx, fc, 0, fc.attrs["sample_rate"],
                            survey=survey, station=station, run=run)

    m = MTH5()
    m.open_mth5(str(path), mode="r")
    try:
        spectra = FCGroupToSpectra().execute(
            ctx, m, survey=survey, station=station, run=run,
            decimation_level=0,
        )["fc"]
    finally:
        m.close_mth5()

    # enrichment: the read-back fc carries the metadata EstimateTF needs
    assert spectra.attrs["sample_rate"] > 0
    assert spectra.attrs["channel_nomenclature"] == "default"
    assert spectra.attrs["input_channels"] == "hx,hy"
    assert spectra.attrs["output_channels"] == "ex,ey"

    periods, Z, _ = estimate_tf(spectra, ["hx", "hy"], ["ex", "ey"],
                                estimator_engine="Least Squares",
                                periods_per_octave=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))
