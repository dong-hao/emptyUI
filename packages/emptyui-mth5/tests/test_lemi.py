"""Load LEMI-424 node behavior (overwrite of stale output file)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
from emptyui.context import RunContext
from emptyui.store import DataStore


def test_load_lemi_overwrites_existing_target(tmp_path: Path, monkeypatch):
    """A leftover (e.g. force-stopped) MTH5 at the target path is removed so a
    fresh file can be written instead of failing on the stale one. The
    nomenclature select must reach the reader as the e_channels pair."""
    from mth5.clients import lemi424 as lemi424_mod

    data_dir = tmp_path / "lemi_data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from emptyui_mth5.lemi import LoadLemi

    read_kwargs: dict = {}

    class FakeCollection:
        survey_id: ClassVar[str] = ""
        station_id: ClassVar[str] = ""
        calibration_dict: ClassVar[dict] = {}

    class FakeClient:
        def __init__(self, **kw):
            self._save_path = Path(kw["save_path"]) / kw["mth5_filename"]
            self.collection = FakeCollection()
            self.h5_kwargs = {}
            self.mth5_file_mode = "a"

        @property
        def save_path(self):
            return self._save_path

        def get_run_dict(self):
            return {"mt001": {"sr1_0001": SimpleNamespace(
                fn=pd.Series([data_dir / "dummy.TXT"]))}}

    def fake_read_file(fn, **kwargs):
        read_kwargs.update(kwargs)
        from mt_timeseries import ChannelTS, RunTS

        ch_list = []
        for comp, typ, unit in [
            ("bx", "magnetic", "nT"), ("by", "magnetic", "nT"),
            ("bz", "magnetic", "nT"), ("e1", "electric", "mV/km"),
            ("e2", "electric", "mV/km"), ("e3", "electric", "mV/km"),
            ("e4", "electric", "mV/km"),
        ]:
            c = ChannelTS(typ)
            c.channel_metadata.units = unit
            c.sample_rate = 1.0
            c.ts = np.ones(16)
            c.component = comp
            ch_list.append(c)
        return RunTS(array_list=ch_list)

    monkeypatch.setattr(lemi424_mod, "LEMI424Client", FakeClient)
    monkeypatch.setattr("mth5.read_file", fake_read_file)

    target = out_dir / "from_lemi424.mth5"
    target.write_bytes(b"corrupt leftover from force-stopped run")
    assert target.exists()

    ctx = RunContext(workspace=tmp_path / "ws", store=DataStore())
    node = LoadLemi()
    res = node.execute(ctx, data_path=str(data_dir), survey_id="test",
                       station_id="mt001", mth5_filename="from_lemi424.mth5",
                       save_path=str(out_dir), channel_nomenclature="lemi34")

    assert "mth5" in res
    assert target.exists()
    # a real HDF5 file was written (not the stale bytes) and e3/e4 were read
    assert target.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n"
    assert read_kwargs["e_channels"] == ["e3", "e4"]
