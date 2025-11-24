"""FC persistence round-trip through the workspace MTH5 (DESIGN.md §14 Phase 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import compute_fc
from emptyui_mth5.persist import (
    WORKSPACE_MTH5_NAME,
    load_fc_level,
    persist_fc_level,
    runts_context,
)


class _Meta:
    def __init__(self, id: str):
        self.id = id


class _FakeRunTS:
    """Duck-typed like an mth5 RunTS, with a datetime start + metadata ids."""

    sample_rate = 8.0

    def __init__(self):
        n = 8192
        t = np.arange(n) / self.sample_rate
        rng = np.random.default_rng(0)
        self.start = pd.Timestamp("2023-01-01T00:00:00")
        self.survey_metadata = _Meta("ws")
        self.station_metadata = _Meta("local")
        self.run_metadata = _Meta("run1")
        self.dataset = xr.Dataset(
            {
                "ex": ("time", rng.standard_normal(n)),
                "ey": ("time", rng.standard_normal(n)),
                "hx": ("time", rng.standard_normal(n)),
                "hy": ("time", rng.standard_normal(n)),
            },
            coords={"time": t},
        )


def test_fc_persist_roundtrip(tmp_path: Path):
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    runts = _FakeRunTS()

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


def test_persist_requires_datetime_time(tmp_path: Path):
    """Persisting an fc with a float (relative) time axis must fail clearly."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    runts = _FakeRunTS()
    # drop the datetime start so compute_fc produces a float time axis
    runts.start = 0.0
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    assert "datetime64" not in str(fc.time.dtype)

    import pytest

    with pytest.raises(ValueError):
        persist_fc_level(ctx, fc, 0, fc.attrs["sample_rate"])


def test_persist_multiple_decimation_levels(tmp_path: Path):
    """ComputeFC's multi-level fc container persists each level separately."""
    from emptyui_aurora._processing import compute_fc_levels

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    runts = _FakeRunTS()
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
