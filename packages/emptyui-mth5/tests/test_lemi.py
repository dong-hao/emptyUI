"""Load LEMI-424 node behavior (overwrite of stale output file)."""

from __future__ import annotations

from pathlib import Path

from emptyui.context import RunContext
from emptyui.store import DataStore


def test_load_lemi_overwrites_existing_target(tmp_path: Path, monkeypatch):
    """A leftover (e.g. force-stopped) MTH5 at the target path is removed so a
    fresh file can be written instead of failing on the stale one."""
    from mth5.clients import lemi424 as lemi424_mod
    from mth5.mth5 import MTH5

    data_dir = tmp_path / "lemi_data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from emptyui_mth5.lemi import LoadLemi

    calls: dict = {}

    class FakeClient:
        def __init__(self, **kw):
            self._save_path = Path(kw["save_path"]) / kw["mth5_filename"]

        @property
        def save_path(self):
            return self._save_path

        def make_mth5_from_lemi424(self, survey_id, station_id):
            calls["made"] = (survey_id, station_id)
            self._save_path.write_bytes(b"fresh")
            return self._save_path

    class FakeMTH5:
        def open_mth5(self, *a, **k):
            return self

    monkeypatch.setattr(lemi424_mod, "LEMI424Client", FakeClient)
    monkeypatch.setattr(MTH5, "__init__", lambda self: None)
    monkeypatch.setattr(MTH5, "open_mth5", FakeMTH5.open_mth5)

    target = out_dir / "from_lemi424.mth5"
    target.write_bytes(b"corrupt leftover from force-stopped run")
    assert target.exists()

    ctx = RunContext(workspace=tmp_path / "ws", store=DataStore())
    node = LoadLemi()
    res = node.execute(ctx, data_path=str(data_dir), survey_id="test",
                       station_id="mt001", mth5_filename="from_lemi424.mth5",
                       save_path=str(out_dir))

    assert calls["made"] == ("test", "mt001")
    assert "mth5" in res
    assert target.exists()
    assert target.read_bytes() == b"fresh"  # overwritten, not failed
