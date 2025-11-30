"""Shared fixtures (generates the synthetic test12rr dataset offline)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test12rr_path(tmp_path_factory):
    import warnings

    warnings.filterwarnings("ignore")
    from mth5.data.make_mth5_from_asc import create_test12rr_h5

    target = tmp_path_factory.mktemp("test12rr")
    path = create_test12rr_h5(target_folder=str(target))
    return path


@pytest.fixture()
def archive(test12rr_path: Path):
    """Open the synthetic archive read-only."""
    import warnings

    warnings.filterwarnings("ignore")
    from mth5.mth5 import MTH5

    m = MTH5()
    m.open_mth5(str(test12rr_path), "r")
    yield m
    m.close_mth5()


@pytest.fixture()
def fake_runts():
    """Duck-typed mth5 RunTS: datetime start, metadata ids, 4 channels."""
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import xarray as xr

    class _Meta:
        def __init__(self, id: str):
            self.id = id

    class _FakeRunTS:
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

    return _FakeRunTS()
