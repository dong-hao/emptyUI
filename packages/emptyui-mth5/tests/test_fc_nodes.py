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


def test_merge_runs_resolves_empty_ids(writable_archive, tmp_path):
    """The rebuilt Merge Runs resolves survey/station via the fallback
    helpers; runs='' merges every run of the station (a single-run archive
    is a passthrough with the new run id)."""
    from emptyui_mth5.merge import MergeRuns
    from emptyui_mth5.runs import _resolve_run, _resolve_station, _resolve_survey

    survey_obj = _resolve_survey(writable_archive, "")
    station_obj = _resolve_station(survey_obj, "")
    run_id = _resolve_run(station_obj, "").metadata.id
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = MergeRuns().execute(ctx, writable_archive, runs=run_id)
    assert set(res["merged_run"].channels) >= {"ex", "ey", "hx", "hy"}
    assert res["merged_run"].run_metadata.id == "merged"



def test_spectra_to_fc_group_metadata(writable_archive, tmp_path, caplog):
    """Archive metadata hygiene (review #10 bundle): the decimation level id
    and the level's sample rate are written from the fc's metadata / params —
    never silently defaulted (the old code always wrote level "0" with rate
    1.0)."""
    from emptyui_mth5.runs import _resolve_station, _resolve_survey

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = _fc_dataset()
    fc.attrs["sample_rate"] = 8.0

    # derived from the fc's metadata when the param is left empty
    SpectraToFCGroup().execute(ctx, fc, mth5=writable_archive,
                               decimation_level=2)
    st = _resolve_station(_resolve_survey(writable_archive, ""), "")
    fcg = st.fourier_coefficients_group.get_fc_group(
        st.fourier_coefficients_group.groups_list[0])
    dec2 = fcg.get_decimation_level("2")
    assert dec2.channel_summary.sample_rate_decimation_level.to_list() == [8.0] * 4

    # an explicit param overrides the fc metadata
    SpectraToFCGroup().execute(ctx, fc, mth5=writable_archive,
                               decimation_level=1, sample_rate=2.0)
    dec1 = fcg.get_decimation_level("1")
    assert dec1.channel_summary.sample_rate_decimation_level.to_list() == [2.0] * 4

    # no attrs, no param: 1.0 fallback with a warning (never a rejection)
    import logging as _logging

    fc2 = _fc_dataset()  # no sample_rate attr
    with caplog.at_level(_logging.WARNING, logger="emptyui"):
        SpectraToFCGroup().execute(ctx, fc2, mth5=writable_archive)
    assert any("no sample_rate metadata" in r.message for r in caplog.records)
