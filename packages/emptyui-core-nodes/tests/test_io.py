"""LoadCsv node — happy-path time-column handling (error cases in test_errors.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes.io import LoadCsv


@pytest.fixture()
def ctx(tmp_path: Path):
    return RunContext(workspace=tmp_path, store=DataStore())


def test_numeric_time_column_derives_rate(ctx, tmp_path: Path):
    p = tmp_path / "num.csv"
    t = np.arange(0, 10, 1 / 32.0)
    np.savetxt(p, np.column_stack([t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]),
               delimiter=",", header="time,ex,ey", comments="", fmt="%.6f")
    ts = LoadCsv().execute(ctx, path=str(p))["data"]
    assert ts.sample_rate == pytest.approx(32.0)
    assert ts.start == pytest.approx(0.0)
    assert set(ts.channels) == {"ex", "ey"}


def test_iso_datetime_time_column(ctx, tmp_path: Path):
    """An ISO datetime time column must set start_datetime + derive the rate
    (previously genfromtxt turned it into NaN and detection silently failed)."""
    p = tmp_path / "iso.csv"
    n = 64
    t = np.arange(0, n / 8.0, 1 / 8.0)
    iso = np.array([f"2024-01-01T00:00:{s:06.3f}" for s in t])
    np.savetxt(p, np.column_stack([iso, np.sin(2 * np.pi * t)]),
               delimiter=",", header="time,ex", comments="", fmt="%s")
    ts = LoadCsv().execute(ctx, path=str(p))["data"]
    assert ts.sample_rate == pytest.approx(8.0)
    assert ts.start_datetime.startswith("2024-01-01T00:00:00")
    assert set(ts.channels) == {"ex"}
