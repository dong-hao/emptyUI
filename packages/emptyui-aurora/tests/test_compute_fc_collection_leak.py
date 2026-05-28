"""M3: the collection route ('time range' mode) must close the write-open
workspace archive when persisting a member fails. Before the fix the close ran
only on the success path, so an exception leaked a write-open handle (issues.md
M3).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _archive import multi_run_archive
from emptyui.context import RunContext
from emptyui.store import DataStore

pytestmark = pytest.mark.filterwarnings("ignore")

KW = {"decimation_levels": 0, "factor": 4, "window_samples": 16, "overlap": 0.5,
      "window_type": "boxcar", "clock_zero": "epoch"}


def _two_run_archive(tmp_path: Path):
    return multi_run_archive(
        tmp_path / "two.h5",
        rids=["sr2400_0001", "sr2400_0002"],
        starts=["2024-01-01T00:00:00", "2024-01-01T00:01:36"])


def test_collection_route_closes_workspace_when_persist_raises(
        tmp_path, monkeypatch):
    from emptyui_aurora.process import ComputeFC
    from emptyui_mth5 import persist
    from emptyui_mth5._compat import is_write_open_in_process
    from emptyui_mth5.runs import ExtractRun

    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    coll = ExtractRun().execute(
        ctx, archive, selection="time range",
        start="2024-01-01 00:00:00", end="2024-01-01 00:02:08")["data"]

    def disk_full(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(persist, "persist_fc_level", disk_full)
    key = persist.workspace_mth5_path(ctx)

    with pytest.raises(RuntimeError, match="disk full"):
        ComputeFC().execute(ctx, coll, **{**KW, "persist": True})

    assert str(key) not in persist._workspace_handles
    assert not is_write_open_in_process(key)
