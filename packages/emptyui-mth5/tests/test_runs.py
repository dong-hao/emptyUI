"""Extract Run node: id resolution + fallbacks."""

from __future__ import annotations

from pathlib import Path

import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.runs import (
    ExtractRun,
    _resolve_run,
    _resolve_station,
    _resolve_survey,
)
from mth5.mth5 import MTH5


@pytest.fixture()
def archive(test12rr_path: Path):
    import warnings

    warnings.filterwarnings("ignore")
    m = MTH5()
    m.open_mth5(str(test12rr_path), "r")
    yield m
    m.close_mth5()


def test_empty_params_fall_back_to_first_run(archive, tmp_path: Path):
    """With empty survey/station/run, Extract Run resolves the first available
    survey/station/run instead of erroring (regression: `surveys_list` no
    longer exists on the survey group)."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = ExtractRun().execute(ctx, archive)
    runts = res["data"]
    assert set(runts.channels) >= {"ex", "ey", "hx", "hy"}


def test_explicit_ids(archive):
    s = _resolve_survey(archive, "")
    st = _resolve_station(s, s.stations_group.groups_list[0])
    r = _resolve_run(st, st.run_summary.iloc[0]["id"])
    assert r.metadata.id


def test_bad_ids_raise_clear_errors(archive):
    with pytest.raises(ValueError, match="not in archive"):
        _resolve_survey(archive, "no_such_survey")
    s = _resolve_survey(archive, "")
    with pytest.raises(ValueError, match="not in survey"):
        _resolve_station(s, "no_such_station")
    st = _resolve_station(s, "")
    with pytest.raises(ValueError, match="not in station"):
        _resolve_run(st, "no_such_run")
