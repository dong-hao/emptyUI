"""Tests for the rebuilt Merge Runs node (DESIGN §24.5 P1): params-based
run list, physical gap-aware concatenation. Error surfaces live in
test_errors.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.merge import MergeRuns, merge_runts

pytestmark = pytest.mark.filterwarnings("ignore")


def _synthetic_runts(start: str = "2024-01-01T00:00:00", sr: float = 1.0,
                     n: int = 256, comps=("bx", "by", "e1", "e2")):
    """A real mth5 RunTS with LEMI-style channels and an absolute start."""
    from mt_timeseries import ChannelTS, RunTS

    rng = np.random.default_rng(1)
    ch_list = []
    for comp in comps:
        c = ChannelTS("magnetic" if comp.startswith("b") else "electric",
                      data=rng.standard_normal(n), start=start,
                      sample_rate=sr)
        c.component = comp
        ch_list.append(c)
    runts = RunTS(array_list=ch_list)
    runts.run_metadata.id = "orig"
    return runts


def test_merge_runts_gap_and_length():
    """Two runs 64 s apart concatenate with the 64-s gap bridged by
    interpolation: merged length = 64 + 64 + 64 (gap) with 64-s runs."""
    sr = 1.0
    r1 = _synthetic_runts(start="2024-01-01T00:00:00", sr=sr, n=64)
    r2 = _synthetic_runts(start="2024-01-01T00:02:08", sr=sr, n=64)  # 64 s gap
    merged = merge_runts([r1, r2], gap_method="slinear", new_run_id="bc")
    n = merged.dataset.sizes["time"]
    assert n == 192
    assert float(merged.sample_rate) == sr
    assert merged.run_metadata.id == "bc"


def test_merge_runts_contiguous():
    """Back-to-back runs (no gap) concatenate exactly."""
    r1 = _synthetic_runts(start="2024-01-01T00:00:00", n=64)
    r2 = _synthetic_runts(start="2024-01-01T00:01:04", n=64)
    merged = merge_runts([r1, r2])
    assert merged.dataset.sizes["time"] == 128


def test_merge_runts_records_gap_provenance():
    """M4 #2: a gap-bridged merge records the interpolated intervals and the
    contributing runs on the payload, so the remote-reference pairing can
    warn about fabricated remote samples; a contiguous merge records none."""
    r1 = _synthetic_runts(start="2024-01-01T00:00:00", n=64)
    r1.run_metadata.id = "b"
    r2 = _synthetic_runts(start="2024-01-01T00:02:08", n=64)  # 64 s gap
    r2.run_metadata.id = "c"
    merged = merge_runts([r1, r2], new_run_id="bc")
    attrs = merged.dataset.attrs
    assert attrs["merged_source_runs"] == ["b", "c"]
    assert attrs["merged_gap_method"] == "slinear"
    assert attrs["merged_gap_intervals"] == [
        ["2024-01-01T00:01:04", "2024-01-01T00:02:08"]]

    contig = merge_runts([_synthetic_runts(start="2024-01-01T00:00:00", n=64),
                          _synthetic_runts(start="2024-01-01T00:01:04", n=64)])
    assert "merged_gap_intervals" not in contig.dataset.attrs
    assert "merged_source_runs" in contig.dataset.attrs


def _two_run_archive(tmp_path: Path):
    """A tiny mth5 archive with two 1 Hz runs 64 s apart (32-s runs, no
    overlap; the merge bridges a 32-s gap)."""
    from mth5.mth5 import MTH5

    path = tmp_path / "two_runs.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    stg = sg.stations_group.add_station("SDX")
    for i, rid in enumerate(("a", "b")):
        start = "2024-01-01T00:00:00" if i == 0 else "2024-01-01T00:01:04"
        runts = _synthetic_runts(start=start, n=32)
        runts.run_metadata.id = rid
        rg = stg.add_run(rid)
        rg.from_runts(runts)
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()
    archive = MTH5()
    archive.open_mth5(str(path), "r")
    return archive


def test_merge_runs_node_two_runs(tmp_path: Path):
    """The node resolves ids like Extract Run and concatenates the listed
    runs in acquisition order."""
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = MergeRuns().execute(ctx, archive, survey="4P", station="SDX",
                              runs="a,b")
    merged = res["merged_run"]
    assert merged.dataset.sizes["time"] == 96
    assert merged.run_metadata.id == "merged"
    # empty runs param = every run in archive order
    res2 = MergeRuns().execute(ctx, archive, runs="")
    assert res2["merged_run"].dataset.sizes["time"] == 96
    archive.close_mth5()


def test_merge_runs_node_missing_run(tmp_path: Path):
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(ValueError, match="not in station archive"):
        MergeRuns().execute(ctx, archive, runs="a,z")
    archive.close_mth5()


def test_merge_runs_rides_fast_read_and_keeps_registry(tmp_path: Path):
    """Merge Runs reads each run via the fast channel path (§24.8 — the
    per-run to_runts storm would cost ~7-19 s per run) and re-resolves the
    filter registry on the merged output (RunTS.merge rebuilds from
    channel arrays; the §24.12 calibration association must survive)."""
    from emptyui_core_nodes.types import channel_calibration
    from mt_metadata.timeseries.filters import CoefficientFilter

    # an archive whose channels reference one survey filter
    from mth5.mth5 import MTH5

    path = tmp_path / "merged_reg.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    fap = CoefficientFilter()
    fap.name = "test_response"
    fap.units_in = "nT"
    fap.units_out = "nT"
    fap.gain = 1.0
    sg.filters_group.add_filter(fap)
    stg = sg.stations_group.add_station("SDX")
    ref = {"applied_filter": {"applied": True, "name": "test_response",
                              "stage": 1}}
    for i, rid in enumerate(("a", "b")):
        start = "2024-01-01T00:00:00" if i == 0 else "2024-01-01T00:01:04"
        runts = _synthetic_runts(start=start, n=32)
        runts.run_metadata.id = rid
        rg = stg.add_run(rid)
        rg.from_runts(runts)
        rg.get_channel("e1").hdf5_dataset.attrs.modify(
            "filters", json.dumps([ref]))
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()

    archive = MTH5()
    archive.open_mth5(str(path), "r", locking=False)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    merged = MergeRuns().execute(ctx, archive, survey="4P", station="SDX",
                                 runs="a,b")["merged_run"]
    assert merged.dataset.sizes["time"] == 96
    assert merged.dataset.attrs["merged_source_runs"] == ["a", "b"]
    # registry re-resolved on the merged payload — the association survives
    pairs = channel_calibration(merged).get("e1") or []
    assert [f.name for f, _ in pairs] == ["test_response"]
    archive.close_mth5()


def test_merge_runs_node_single_run_passthrough(tmp_path: Path):
    """A single listed run is still extracted and re-id'd (no error)."""
    archive = _two_run_archive(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = MergeRuns().execute(ctx, archive, runs="b", new_run_id="only")
    assert res["merged_run"].run_metadata.id == "only"
    assert res["merged_run"].dataset.sizes["time"] == 32
    archive.close_mth5()


# --- M12: §3 constraints made loud (overlap, channel consistency, dup ids) -

def test_merge_runts_rejects_overlapping_runs():
    """Two runs with an OVERLAP: RunTS.merge fails on non-monotonic indexes
    (and would interpolate over real acquired data) — the §3 constraint is
    a hard error with an actionable message, not a silent fabrication."""
    r1 = _synthetic_runts(start="2024-01-01T00:00:00", n=64)
    r2 = _synthetic_runts(start="2024-01-01T00:00:32", n=64)  # 32 s overlap
    with pytest.raises(ValueError, match="overlaps"):
        merge_runts([r1, r2])


def test_merge_runts_errors_on_missing_channels():
    """Runs with unequal channel sets cannot be combined by RunTS.merge
    (raw xarray MergeError) — the §3 consistency constraint is a hard
    error with an actionable message."""
    r1 = _synthetic_runts(comps=("bx", "by", "e1", "e2"))
    r2 = _synthetic_runts(start="2024-01-01T00:01:04", comps=("bx", "by", "e1"))
    with pytest.raises(ValueError, match="lacks channel"):
        merge_runts([r1, r2])


def test_merge_node_dedups_duplicate_run_ids(test12rr_path, tmp_path, caplog):
    """The params run list silently merging the same run twice inflated its
    weight — duplicates are deduped with a warning."""
    import warnings

    from emptyui_mth5.runs import _resolve_run, _resolve_station, _resolve_survey

    warnings.filterwarnings("ignore")
    from mth5.mth5 import MTH5

    m = MTH5()
    m.open_mth5(str(test12rr_path), "r+")
    try:
        survey_obj = _resolve_survey(m, "")
        station_obj = _resolve_station(survey_obj, "")
        run_id = _resolve_run(station_obj, "").metadata.id
        ctx = RunContext(workspace=tmp_path, store=DataStore())
        with caplog.at_level("WARNING", logger="emptyui"):
            res = MergeRuns().execute(ctx, m, runs=f"{run_id},{run_id}")
        assert res["merged_run"] is not None
        assert any("more than once" in r.message for r in caplog.records)
    finally:
        m.close_mth5()


def test_merge_runts_pads_disagreeing_aux_mask():
    """The auto-generated 'saturated' QC mask exists only on runs with
    flagged samples — merging masked + unmasked sessions (the documented
    E1b remedy) must pad the missing members with zeros instead of
    hard-failing on QC metadata (§3's consistency check misfires on it)."""
    import numpy as np
    from mt_timeseries import ChannelTS, RunTS

    def runts(saturated, start_s=0):
        sr, n = 6.0, 12
        t = np.arange(n) / sr
        stamp = f"2024-01-01T00:00:{start_s:02d}"
        chs = []
        for comp in ("bx", "by"):
            c = ChannelTS("magnetic", data=np.sin(2 * np.pi * 1 * t),
                          start=stamp, sample_rate=sr)
            c.component = comp
            chs.append(c)
        for comp in ("e1", "e2"):
            c = ChannelTS("electric", data=t.copy(), start=stamp,
                          sample_rate=sr)
            c.component = comp
            chs.append(c)
        if saturated:
            c = ChannelTS("auxiliary", data=np.zeros(n), start=stamp,
                          sample_rate=sr)
            c.component = "saturated"
            chs.append(c)
        return RunTS(array_list=chs)

    merged = merge_runts(
        [runts(True, 0), runts(False, 2), runts(True, 4)], new_run_id="m")
    # the zero-padded members carry the mask too (merged QC stays usable)
    assert "saturated" in merged.dataset.data_vars
    assert not np.any(np.asarray(merged.saturated.ts) > 0)
    assert "merged_source_runs" in merged.dataset.attrs


def test_merge_runts_mixed_naive_aware_spans():
    """L-M4: a naive stamp (digitized record) merging against an aware one
    must not raise a raw TypeError on the span comparison — both normalize
    to UTC (the archive convention)."""
    r1 = _synthetic_runts(start="2024-01-01T00:00:00", n=64)
    r1.run_metadata.id = "b"
    r2 = _synthetic_runts(start="2024-01-01T00:01:04", n=64)
    r2.run_metadata.id = "c"
    # strip the tz from r1's time axis (naive)
    r1.dataset = r1.dataset.assign_coords(
        time=np.asarray(r1.dataset["time"].values).astype("datetime64[ns]"))
    merged = merge_runts([r1, r2], new_run_id="bc")
    assert merged.dataset.sizes["time"] == 128


def test_merge_runts_chained_provenance_flattens():
    """L-M4: merge(merge(a, b), c) must keep the ORIGINAL run identities —
    the old overwrite recorded the inner merge's new run id (and dropped
    the inner gap list)."""
    r1 = _synthetic_runts(start="2024-01-01T00:00:00", n=64)
    r1.run_metadata.id = "a"
    r2 = _synthetic_runts(start="2024-01-01T00:02:08", n=64)  # 64 s gap
    r2.run_metadata.id = "b"
    inner = merge_runts([r1, r2], new_run_id="ab")
    assert inner.dataset.attrs["merged_source_runs"] == ["a", "b"]
    r3 = _synthetic_runts(start="2024-01-01T00:04:16", n=64)  # 64 s gap
    r3.run_metadata.id = "c"
    merged = merge_runts([inner, r3], new_run_id="abc")
    attrs = merged.dataset.attrs
    assert attrs["merged_source_runs"] == ["a", "b", "c"]
    assert attrs["merged_gap_intervals"] == [
        ["2024-01-01T00:01:04", "2024-01-01T00:02:08"],
        ["2024-01-01T00:03:12", "2024-01-01T00:04:16"],
    ]
