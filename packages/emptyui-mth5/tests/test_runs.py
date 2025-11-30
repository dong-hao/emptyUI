"""Extract Run node: id resolution + fallbacks. Error surfaces live in
test_errors.py."""

from __future__ import annotations

from pathlib import Path

from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.runs import (
    ExtractRun,
    _resolve_run,
    _resolve_station,
    _resolve_survey,
)


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


def _lemi_runts(comps=("bx", "by", "bz", "e1", "e2")):
    """Build an mth5 RunTS with LEMI-style channel names."""
    import numpy as np
    from mt_timeseries import ChannelTS, RunTS

    ch_list = []
    for comp in comps:
        typ = "magnetic" if comp[0] in ("b", "h") else "electric"
        c = ChannelTS(typ)
        c.channel_metadata.units = "nT" if typ == "magnetic" else "mV/km"
        c.sample_rate = 1.0
        c.ts = np.arange(64, dtype=float)
        c.component = comp
        ch_list.append(c)
    return RunTS(array_list=ch_list)


def test_split_channels_lemi12_names(tmp_path):
    """LEMI channel names (e1/e2/bx/by/bz) must split onto the standard
    ex/ey/hx/hy/hz ports — the old getattr(ex) raised NameError."""
    from emptyui_mth5.runs import SplitChannels

    runts = _lemi_runts()
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = SplitChannels().execute(ctx, runts)
    # ex/ey ports carry the e1/e2 channels; hx/hy/hz carry bx/by/bz
    assert res["ex"].component == "e1"
    assert res["ey"].component == "e2"
    assert res["hx"].component == "bx"
    assert res["hy"].component == "by"
    assert res["hz"].component == "bz"


def test_split_channels_lemi34_four_channel(tmp_path):
    """A 4-E-channel instrument: with lemi34 selected, e3/e4 map to ex/ey."""
    from emptyui_mth5.runs import SplitChannels

    runts = _lemi_runts(("bx", "by", "bz", "e3", "e4"))
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = SplitChannels().execute(ctx, runts, channel_nomenclature="lemi34")
    assert res["ex"].component == "e3"
    assert res["ey"].component == "e4"
    assert res["hx"].component == "bx"


def test_split_merge_channels_roundtrip(tmp_path):
    """Split (LEMI names) -> Merge must rebuild a run with the actual channel
    names intact (regression: RunTS(dataset=...) was an invalid constructor)."""
    from emptyui_mth5.runs import MergeChannels, SplitChannels

    runts = _lemi_runts()
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    split = SplitChannels().execute(ctx, runts)
    merged = MergeChannels().execute(
        ctx, ex=split["ex"], ey=split["ey"], hx=split["hx"], hy=split["hy"],
        hz=split["hz"],
    )
    run = merged["data"]
    assert set(run.channels) == {"e1", "e2", "bx", "by", "bz"}
