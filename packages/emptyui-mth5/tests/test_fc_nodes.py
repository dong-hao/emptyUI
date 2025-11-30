"""SpectraToFCGroup / MergeRuns — happy-path id resolution (error surfaces
in test_errors.py)."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.fc import SpectraToFCGroup
from mth5.mth5 import MTH5


def _fc_dataset() -> xr.Dataset:
    rng = np.random.default_rng(1)
    return xr.Dataset(
        {c: (("frequency", "time"),
             rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3)))
         for c in ("ex", "ey", "hx", "hy")},
        coords={"frequency": np.linspace(0.1, 0.9, 4),
                "time": pd.date_range("2024-01-01", periods=3, freq="s")},
    )


@pytest.fixture()
def writable_archive(test12rr_path: Path, tmp_path: Path):
    warnings.filterwarnings("ignore")
    p = tmp_path / "arch.mth5"
    shutil.copy(test12rr_path, p)
    m = MTH5()
    m.open_mth5(str(p), "r+")
    yield m
    m.close_mth5()


def test_spectra_to_fc_group_resolves_empty_ids(writable_archive, tmp_path):
    """Empty survey/station/run fall back to the first available (the old
    code used legacy mth5.get_survey/survey_group only)."""
    from emptyui_mth5.runs import _resolve_station, _resolve_survey

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    SpectraToFCGroup().execute(ctx, _fc_dataset(), mth5=writable_archive)
    survey_obj = _resolve_survey(writable_archive, "")
    st = _resolve_station(survey_obj, "")
    assert st.fourier_coefficients_group.groups_list, "no fc group written"


def test_merge_runs_physical_mode_resolves_ids(writable_archive, tmp_path):
    """Physical merge resolves survey/station/run via the dual-layout
    helpers (regression: StationGroup.runs_group does not exist in mth5
    0.6.x, so the old st.runs_group.get_run crashed with AttributeError)."""
    from emptyui_mth5.merge import MergeRuns
    from emptyui_mth5.runs import _resolve_run, _resolve_station, _resolve_survey

    survey_obj = _resolve_survey(writable_archive, "")
    station_obj = _resolve_station(survey_obj, "")
    run_id = _resolve_run(station_obj, "").metadata.id
    selection = pd.DataFrame([{
        "survey": survey_obj.metadata.id, "station": station_obj.metadata.id,
        "run": run_id, "sample_rate": 1.0, "start": None, "end": None,
    }])
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = MergeRuns().execute(ctx, selection, mth5=writable_archive, mode="physical")
    assert set(res["merged_run"].channels) >= {"ex", "ey", "hx", "hy"}

