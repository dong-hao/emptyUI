"""MTU-5A burst ingestion tests (history §24.8): per-block tag parsing
(the upstream reader's dense-concat defect fixed), maximal-contiguous
segment grouping, run-per-segment archive build, TBL scalar calibration at
ingest, quality warnings. The synthetic corpus is written from scratch
(binary TBL tags + TS blocks per the verified layout) so CI passes without
the gitignored real corpus; the real-corpus test (IAGA 1690C16C) is
skipped when the data dir is absent."""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from emptyui_mth5.phoenix_mtu import (
    _block_tag,
    _group_segments,
    _iter_blocks,
    read_phoenix_mtu,
)

pytestmark = pytest.mark.filterwarnings("ignore")

REAL_DIR = (
    Path(__file__).resolve().parents[3] / "examples" / "data" / "phoenix-mtu-test"
)


# --- synthetic MTU-5A corpus writer -------------------------------------

def _tbl_value(tag: str, value) -> bytes:
    from mt_io.phoenix.readers.mtu.mtu_table import MTUTable

    kind = MTUTable().TBL_TAG_TYPES[tag][0]
    v = bytearray(13)
    if kind == "int":
        v[0:4] = struct.pack("<i", int(value))
    elif kind == "double":
        v[0:8] = struct.pack("<d", float(value))
    elif kind == "char":
        v[0 : len(value)] = value.encode("latin-1")
    elif kind == "time":
        dt = datetime.strptime(
            value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        v[0:6] = bytes([dt.second, dt.minute, dt.hour, dt.day, dt.month,
                        dt.year - 2000])
    else:
        raise ValueError(f"unsupported TBL type {kind} for {tag}")
    return tag.encode("latin-1").ljust(12) + bytes(v)


def _write_tbl(path: Path, **overrides) -> None:
    tags = {
        "SITE": "SYNTH01", "SNUM": 1234, "SRVY": "SYNTH_SURVEY",
        "CMPY": "emptyUI", "ELEV": 100, "LATG": "5213.123,N",
        "LNGG": "01100.456,E", "STIM": "2009-01-01T00:00:00",
        "ETIM": "2011-01-01T00:00:00",  # placeholders — never data times
        "EXLN": 50.0, "EYLN": 50.0, "EAZM": 0.0, "HAZM": 90.0,
        "CHEX": 1, "CHEY": 2, "CHHX": 3, "CHHY": 4, "CHHZ": 5,
        "HSMP": 600, "L3NS": 1, "L4NS": 16,
        "SRL3": 2400, "SRL4": 150, "SRL5": 15,
        "FSCV": 6.4, "EGN": 2, "HGN": 2, "HATT": 1.0, "HNOM": 1000.0,
    }
    tags.update(overrides)
    with open(path, "wb") as fid:
        fid.writelines(_tbl_value(tag, value) for tag, value in tags.items())


def _ts_block(start: datetime, rate: int, n_scan: int, n_ch: int, *,
              clock_status: int = 4, clock_error: int = 0,
              saturated: int = 0, rng=None) -> bytes:
    tag = bytearray(32)
    tag[0:8] = bytes([start.second, start.minute, start.hour, start.day,
                      start.month, start.year - 2000, 0, 20])
    tag[8:10] = struct.pack("<H", 1234)          # serial
    tag[10:12] = struct.pack("<H", n_scan)
    tag[12] = n_ch
    tag[13] = 32                                  # tag length (MTU-5)
    tag[14] = 0                                   # status
    tag[15] = saturated
    tag[16] = 0                                   # reserved
    tag[17] = 3                                   # sample length
    tag[18:20] = struct.pack("<H", rate)
    tag[20] = 0                                   # rate units
    tag[21] = clock_status
    tag[22:26] = struct.pack("<i", clock_error)
    data = rng.integers(0, 256, n_scan * n_ch * 3, dtype=np.uint8).tobytes()
    return bytes(tag) + data


def _write_ts(path: Path, rate: int, segments: list[tuple[datetime, int]],
              n_scan: int | None = None, n_ch: int = 5, *,
              seed: int = 0) -> None:
    """segments: [(first_block_start, n_blocks)]; blocks of a segment are
    contiguous (1-s blocks)."""
    rng = np.random.default_rng(seed)
    if n_scan is None:
        n_scan = rate  # 1 s of samples per block
    with open(path, "wb") as fid:
        for start, n_blocks in segments:
            for k in range(n_blocks):
                blk = start + _second_delta(k)
                fid.write(_ts_block(blk, rate, n_scan, n_ch, rng=rng))


def _second_delta(k: int):
    from datetime import timedelta

    return timedelta(seconds=k)  # blocks are 1 s (n_scan == rate)


def _synthetic_corpus(data_path: Path) -> None:
    """One TBL + three TSn files mirroring the official MT-mode recording:
    TS3 = 2 x 1-s slots every 10 min, TS4 = 2 x 16-s slots every 10 min,
    TS5 = continuous."""
    _write_tbl(data_path / "SYNTH01.TBL")
    t0 = datetime(2009, 12, 16, 7, 55, 0, tzinfo=timezone.utc)
    _write_ts(data_path / "SYNTH01.TS3", 2400,
              [(t0, 1), (t0 + _slot_delta(10), 1)], seed=1)
    t4 = datetime(2009, 12, 16, 7, 50, 0, tzinfo=timezone.utc)
    _write_ts(data_path / "SYNTH01.TS4", 150,
              [(t4, 16), (t4 + _slot_delta(10), 16)], seed=2)
    t5 = datetime(2009, 12, 16, 7, 46, 52, tzinfo=timezone.utc)
    _write_ts(data_path / "SYNTH01.TS5", 15, [(t5, 100)], seed=3)


def _slot_delta(minutes: int):
    from datetime import timedelta

    return timedelta(minutes=minutes)


# --- unit tests ----------------------------------------------------------

def test_block_tag_decode(tmp_path):
    """The per-block 32-byte tag decodes time/rate/quality fields (the
    layout verified byte-level on the official recording, history §24.8)."""
    import pandas as pd


    rng = np.random.default_rng(0)
    t = datetime(2009, 12, 16, 8, 5, 0, tzinfo=timezone.utc)
    tag = _block_tag(_ts_block(t, 2400, 2400, 5, rng=rng))
    assert tag.start == pd.Timestamp(t)
    assert (tag.rate, tag.n_scan, tag.n_ch) == (2400, 2400, 5)
    assert tag.clock_status == 4 and tag.clock_error_s == 0.0
    assert not tag.saturated


def test_group_segments_slots_and_continuous(tmp_path):
    """Contiguous blocks regroup: slots stay separate, continuous streams
    collapse to one segment."""
    p = tmp_path / "SYNTH01.TS4"
    t4 = datetime(2009, 12, 16, 7, 50, 0, tzinfo=timezone.utc)
    _write_ts(p, 150, [(t4, 16), (t4 + _slot_delta(10), 16)], seed=4)
    segs = _group_segments(_iter_blocks(p))
    assert len(segs) == 2
    assert all(s.n_blocks == 16 for s in segs)
    assert (segs[1].start - segs[0].start).total_seconds() == 600.0

    p5 = tmp_path / "SYNTH01.TS5"
    _write_ts(p5, 15, [(datetime(2009, 12, 16, 7, 46, 52, tzinfo=timezone.utc), 100)], seed=5)
    segs5 = _group_segments(_iter_blocks(p5))
    assert len(segs5) == 1 and segs5[0].n_blocks == 100


def test_read_phoenix_mtu_archive(tmp_path):
    """Full synthetic corpus -> archive: run-per-segment ids (mth5
    convention), block-tag time periods, stamped units, TBL scalar
    calibration applied at ingest."""
    _synthetic_corpus(tmp_path)
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    ids = [r[0] for r in res["runs"]]
    assert ids == ["sr2400_0001", "sr2400_0002",
                   "sr150_0001", "sr150_0002", "sr15_0001"]
    # H7: the synthetic corpus carries no ASC tables — that fact must be
    # a loud warning, never silence (the scalar chain alone is a wrong TF)
    assert any("no ASC calibration" in w for w in res["warnings"]), (
        res["warnings"])

    from mth5.mth5 import MTH5

    with MTH5() as m:
        m.open_mth5(res["path"], "r")
        from emptyui_mth5.runs import _resolve_station, _resolve_survey

        station_obj = _resolve_station(_resolve_survey(m, ""), "")
        run_ids = [str(x) for x in station_obj.run_summary["id"]]
        # archive order is the run table's, not the ingestion order
        assert sorted(run_ids) == sorted(ids)
        rg = station_obj.get_run("sr15_0001")
        run_ts = rg.to_runts()
        assert float(run_ts.sample_rate) == 15.0
        hx = run_ts.dataset["hx"].values
        ex = run_ts.dataset["ex"].values
        assert hx.size == 100 * 15
        # units stamped from the TBL scalar calibration route
        assert run_ts.dataset["hx"].attrs.get("units") in ("nT", "nanoTesla")
        # calibration applied: channel c's first sample lives at the scan
        # bytes [3c, 3c+1, 3c+2] after the 32-byte tag (low, mid, high)
        from mt_io.phoenix.readers.mtu.mtu_table import MTUTable

        table = MTUTable(tmp_path / "SYNTH01.TBL")
        table.read_tbl()
        raw = (tmp_path / "SYNTH01.TS5").read_bytes()[32:47]  # first scan

        def first_sample(c):
            lo, mid, hi = raw[3 * c], raw[3 * c + 1], raw[3 * c + 2]
            v = int(hi) * 2**16 + int(mid) * 2**8 + int(lo)
            return v - 2**24 if v > 2**23 - 1 else v

        assert hx[0] == pytest.approx(
            first_sample(2) * table.magnetic_calibration)  # CHHX = 3
        assert ex[0] == pytest.approx(
            first_sample(0) * table.ex_calibration)  # CHEX = 1
        # tag-truth time anchors (history §24.12 regression): every archived channel
        # dataset must start at its segment's block-tag time — NOT the TBL
        # STIM placeholder. ChannelTS anchors its time index on the channel
        # metadata's time_period.start; the ingest used to set an
        # undeclared 'start_time' attr (silently ignored), so every dataset
        # was STIM-anchored, the merged FC time axes collapsed across runs,
        # and from_channel_ts propagated the wrong period into the runs.
        import pandas as pd

        slot_run = station_obj.get_run("sr2400_0002")  # the 2nd slot (t0+10 min)
        da = slot_run.get_channel("hx").to_xarray()
        # archived stamps are naive (mt_timeseries convention) — compare naive
        assert pd.Timestamp(da.coords["time"].values[0]) == \
            pd.Timestamp("2009-12-16T08:05:00")
        assert str(slot_run.metadata.time_period.start).startswith(
            "2009-12-16T08:05")


def test_read_phoenix_mtu_default_output_folder(tmp_path):
    """M29: an empty save_path writes into <data_path>/emptyui_output, never
    the data directory itself (a re-run could overwrite a source archive)."""
    _synthetic_corpus(tmp_path)
    res = read_phoenix_mtu(tmp_path, save_path=None)
    target = Path(res["path"])
    assert target.parent == tmp_path / "emptyui_output"
    assert target.exists()
    # no MTH5 beside the raw TS/TBL files
    assert not list(tmp_path.glob("*.mth5"))


def test_read_phoenix_mtu_sample_rate_filter(tmp_path):
    _synthetic_corpus(tmp_path)
    res = read_phoenix_mtu(tmp_path, sample_rates=[15], save_path=tmp_path)
    assert [r[0] for r in res["runs"]] == ["sr15_0001"]


def test_read_phoenix_mtu_quality_warnings(tmp_path):
    """Mixed clock status, saturated blocks and clock error surface as
    warnings — never rejections (history §24.8)."""
    _write_tbl(tmp_path / "SYNTH01.TBL")
    rng = np.random.default_rng(9)
    t = datetime(2009, 12, 16, 7, 46, 52, tzinfo=timezone.utc)
    with open(tmp_path / "SYNTH01.TS5", "wb") as fid:
        fid.write(_ts_block(t, 15, 15, 5, rng=rng))
        fid.write(_ts_block(t + _second_delta(1), 15, 15, 5, rng=rng,
                            clock_status=2, clock_error=3))
        fid.write(_ts_block(t + _second_delta(2), 15, 15, 5, rng=rng,
                            saturated=1))
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    text = " ".join(res["warnings"])
    assert "clock status" in text
    assert "clock error up to 3" in text
    assert "1 saturated block" in text


def test_read_phoenix_mtu_truncated_tail(tmp_path):
    """A truncated final block is ignored with a warning (short reads are
    the on-disk norm after a receiver is pulled mid-write)."""
    _write_tbl(tmp_path / "SYNTH01.TBL")
    rng = np.random.default_rng(11)
    t = datetime(2009, 12, 16, 7, 46, 52, tzinfo=timezone.utc)
    buf = _ts_block(t, 15, 15, 5, rng=rng)
    with open(tmp_path / "SYNTH01.TS5", "wb") as fid:
        fid.write(buf)
        fid.write(buf[:40])  # truncated second block
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    assert [r[0] for r in res["runs"]] == ["sr15_0001"]
    assert any("trailing byte" in w for w in res["warnings"])


def test_read_phoenix_mtu_schedule_crosscheck(tmp_path):
    """A segment length deviating from the TBL slot schedule warns (loose
    cross-check, never a rejection)."""
    _write_tbl(tmp_path / "SYNTH01.TBL")
    rng = np.random.default_rng(13)
    t = datetime(2009, 12, 16, 7, 55, 0, tzinfo=timezone.utc)
    with open(tmp_path / "SYNTH01.TS3", "wb") as fid:
        fid.write(_ts_block(t, 2400, 2400, 5, rng=rng))
    # the TBL says L3NS = 1 s; write a second 1-s slot 2 s later — fine
    # (spacing is loose), but then a 2-s continuous segment: length warns
    with open(tmp_path / "SYNTH01.TS4", "wb") as fid:
        fid.writelines(
            _ts_block(t + _second_delta(k), 150, 150, 5, rng=rng)
            for k in range(32))
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    assert any("TBL schedule says 16 s" in w for w in res["warnings"])


def test_read_phoenix_mtu_no_tbl(tmp_path):
    with pytest.raises(ValueError, match="exactly one TBL"):
        read_phoenix_mtu(tmp_path, save_path=tmp_path)


# --- ingest time window (history §24.11 night-window smoke tests) ----------------

def test_window_full_matches_unfiltered(tmp_path):
    """A window spanning the whole recording is identical to no filter."""
    _synthetic_corpus(tmp_path)
    res0 = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    res1 = read_phoenix_mtu(tmp_path, save_path=tmp_path,
                            start="2009-12-16T00:00:00",
                            end="2009-12-17T00:00:00")
    assert res0["runs"] == res1["runs"]


def test_window_crops_blocks_by_tag_time(tmp_path):
    """Blocks are filtered by per-block tag start BEFORE grouping: slots
    inside the window stay full, streams crop at the edges, outside slots
    vanish (history §24.11)."""
    _synthetic_corpus(tmp_path)
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path,
                           start="2009-12-16T07:47:00",
                           end="2009-12-16T08:03:00")
    by_rate = {}
    for rid, rate, n in res["runs"]:
        by_rate.setdefault(int(rate), []).append((rid, n))
    # TS3 (1-s slots at 07:55, 08:05): only the 07:55 slot survives, full
    assert by_rate[2400] == [("sr2400_0001", 2400)]
    # TS4 (16-s slots at 07:50, 08:00): both slots lie inside the window
    # (the 08:00 slot spans 08:00:00-08:00:16, window ends 08:03) — full
    assert by_rate[150] == [("sr150_0001", 2400), ("sr150_0002", 2400)]
    # TS5 (07:46:52 -> 07:48:32): head crop at 07:47 -> 92 s = 1380 samples
    assert by_rate[15] == [("sr15_0001", 1380)]
    # the window drop is reported
    text = " ".join(res["warnings"])
    assert "ingest window" in text and "dropped" in text
    # surviving segments are not schedule anomalies
    assert not any("TBL schedule says" in w for w in res["warnings"])


def test_window_accepts_epoch_seconds(tmp_path):
    """Epoch-seconds windows parse (naive/aware ISO both fine too)."""
    _synthetic_corpus(tmp_path)
    import pandas as pd

    t0 = pd.Timestamp("2009-12-16T07:47:00").timestamp()
    t1 = pd.Timestamp("2009-12-16T08:03:00").timestamp()
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path, start=t0, end=t1)
    assert sum(1 for r in res["runs"] if r[1] == 150) == 2


def test_window_outside_data_raises(tmp_path):
    _synthetic_corpus(tmp_path)
    with pytest.raises(ValueError, match="no MTU-5A blocks fall within"):
        read_phoenix_mtu(tmp_path, save_path=tmp_path,
                         start="2020-01-01T00:00:00",
                         end="2020-01-02T00:00:00")


def test_window_invalid_args_raise(tmp_path):
    _synthetic_corpus(tmp_path)
    with pytest.raises(ValueError, match="not an ISO"):
        read_phoenix_mtu(tmp_path, save_path=tmp_path, start="not-a-time")
    with pytest.raises(ValueError, match="end .*<= start"):
        read_phoenix_mtu(tmp_path, save_path=tmp_path,
                         start="2009-12-16T08:00:00",
                         end="2009-12-16T07:00:00")


# --- real-corpus test (user station 1690618B; history §24.11) --------------------

@pytest.mark.skipif(
    not all((REAL_DIR / f"1690618B.{ext}").exists()
            for ext in ("TS3", "TS4", "TS5")),
    reason="MTU-5A real corpus (1690618B.*) not on disk (gitignored)")
def test_real_corpus_structure(tmp_path):
    """The 1690618B recording: TS3 -> 506 x 2 s, TS4 -> 506 x 16 s,
    TS5 -> 1 x 84.4 h continuous (history §24.11 block-tag walk numbers).
    All three rates span the same 3.5 days. Parse-only (the 1013-run
    archive write costs ~3 min — its mechanics are covered by the
    synthetic tests); a single-rate archive build keeps an end-to-end
    round trip in the test without the write cost."""
    for ext, rate, n_blocks in (("TS3", 2400, 1012), ("TS4", 150, 8096),
                                ("TS5", 15, 303742)):
        segs = _group_segments(_iter_blocks(REAL_DIR / f"1690618B.{ext}"))
        assert len(segs) == (1 if ext == "TS5" else {2400: 506, 150: 506}[rate])
        assert sum(s.n_blocks for s in segs) == n_blocks
        assert all(s.rate == rate for s in segs)
        if ext == "TS4":
            assert all(s.n_blocks == 16 for s in segs)
        if ext == "TS3":
            assert all(s.n_blocks == 2 for s in segs)
    # end-to-end archive round trip at the continuous rate only (1 run)
    res = read_phoenix_mtu(REAL_DIR, sample_rates=[15], save_path=tmp_path)
    assert [r[0] for r in res["runs"]] == ["sr15_0001"]
    assert res["runs"][0][2] == 303742 * 15
    # the ingest window (history §24.11): a 2 h night sub-window crops the stream
    res_w = read_phoenix_mtu(REAL_DIR, sample_rates=[15], save_path=tmp_path,
                             start="2026-06-18T12:00:00",
                             end="2026-06-18T14:00:00")
    assert res_w["runs"][0][2] == 2 * 3600 * 15


# --- ASC calibration ingest (history §24.12) --------------------------------------

_CLB_MIN = """\
C:\\test\\new\\MTU-1234_CLB : Instrument Calibrations
MTU-1234.CLB    Normalizing  E-gain: 2  H-gain: 2

        Channel      Freq (Hz)        mV / mV        Degrees
              1         1.0            1.010             -1
              2         1.0            1.020             -1
              3         1.0            1.100              0
              4         1.0            1.110              0
              5         1.0            1.120              0
              1        10.0            1.000              0
              2        10.0            1.000              0
              3        10.0            1.000              0
              4        10.0            1.000              0
              5        10.0            1.000              0
"""

_CLC_MIN = """\
C:\\test\\new\\COIL111_CLC : Coil Calibrations
COIL111.CLC    Normalizing Factor: 1000.0 mV/nT

        Channel      Freq (Hz)        mV / mV        Degrees
              1        10.0            1.000              0
              1         1.0            1.050              2
"""


def _write_cal_files(directory, box=True, coils=("COIL111",)):
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if box:
        (directory / "MTU-1234_CLB.asc").write_text(_CLB_MIN)
    for s in coils:
        (directory / f"{s}_CLC.asc").write_text(_CLC_MIN.replace("COIL111", s))
    return directory


def test_read_phoenix_mtu_asc_calibration(tmp_path):
    """ASC box+coil tables matched by TBL serials, registered in
    /Survey/Filters, referenced per channel (box stage 1, coil stage 2),
    and restored by to_runts (objects + applied flags)."""
    _synthetic_corpus(tmp_path)
    _write_tbl(tmp_path / "SYNTH01.TBL", HXSN="COIL111", HYSN="COIL222",
               HZSN="COIL333")
    _write_cal_files(tmp_path, box=True, coils=("COIL111",))
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    assert res["calibration"] == {
        "ex": ["phoenix_clb_mtu-1234_ex"],
        "ey": ["phoenix_clb_mtu-1234_ey"],
        "hx": ["phoenix_clb_mtu-1234_hx", "phoenix_clc_coil111_hx"],
        "hy": ["phoenix_clb_mtu-1234_hy"],
        "hz": ["phoenix_clb_mtu-1234_hz"],
    }
    # only the missing-coil warnings for hy/hz; no box warning
    assert not any("_CLB" in w for w in res["warnings"])
    assert sum("coil calibration" in w for w in res["warnings"]) == 2

    from mt_metadata.timeseries.filters import FrequencyResponseTableFilter
    from mth5.mth5 import MTH5

    with MTH5() as m:
        m.open_mth5(res["path"], "r")
        from emptyui_mth5.runs import _resolve_station, _resolve_survey

        survey = _resolve_survey(m, "")
        fg = survey.filters_group
        hx_filters = [
            fg.to_filter_object(n) for n in res["calibration"]["hx"]
        ]
        assert all(isinstance(f, FrequencyResponseTableFilter)
                   for f in hx_filters)
        station = _resolve_station(survey, "")
        rg = station.get_run("sr15_0001")
        hx = rg.get_channel("hx")
        refs = [(f.name, f.applied, f.stage) for f in hx.metadata.filters]
        assert refs == [("phoenix_clb_mtu-1234_hx", True, 1),
                        ("phoenix_clc_coil111_hx", True, 2)]
        ex_refs = [f.name for f in rg.get_channel("ex").metadata.filters]
        assert ex_refs == ["phoenix_clb_mtu-1234_ex"]
        runts = rg.to_runts()
        assert set(runts.filters) >= {
            "phoenix_clb_mtu-1234_hx", "phoenix_clc_coil111_hx"}
        attrs = runts.dataset["hx"].attrs.get("filters")
        assert [d["applied_filter"]["name"] for d in attrs] == \
            res["calibration"]["hx"]
        assert all(d["applied_filter"]["applied"] for d in attrs)


def test_read_phoenix_mtu_gain_conflict_warns_tbl_wins(tmp_path):
    """CLB/CLC normalizing settings conflicting with the recording TBL
    (EGN/HGN/HNOM) warn — the TBL is the acquisition parameter set and
    wins: the scalar calibration is unchanged and the chains still build,
    but the shape mismatch is never silent."""
    _synthetic_corpus(tmp_path)
    # defaults: EGN=2 HGN=2 HNOM=1000 vs the fixtures' E-gain: 2 H-gain: 2
    # / Normalizing Factor: 1000 — force all three conflicts
    _write_tbl(tmp_path / "SYNTH01.TBL", EGN=1, HGN=1, HNOM=500.0,
               HXSN="COIL111", HYSN="COIL222", HZSN="COIL333")
    _write_cal_files(tmp_path, box=True, coils=("COIL111",))
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    conflicts = [w for w in res["warnings"] if "normalized" in w]
    assert len(conflicts) == 3
    assert any("E-gain 2" in w and "EGN=1" in w for w in conflicts)
    assert any("H-gain 2" in w and "HGN=1" in w for w in conflicts)
    assert any("Normalizing Factor (mV/nT) 1000" in w and "HNOM=500" in w
               for w in conflicts)
    # the TBL wins: the scalar calibration is unchanged and every matched
    # chain still builds (the conflict is advisory, never structural)
    assert res["calibration"]["ex"] == ["phoenix_clb_mtu-1234_ex"]
    assert res["calibration"]["hx"] == ["phoenix_clb_mtu-1234_hx",
                                        "phoenix_clc_coil111_hx"]


def test_normalizing_conflict_absent_side_is_silent():
    """A pair with either side absent (an export carrying only E-gain, a
    TBL tag not written) skips the check — no false warning from missing
    data; a matching pair is silent too."""
    from emptyui_mth5.phoenix_calibration import PhoenixCalibration
    from emptyui_mth5.phoenix_mtu import _warn_normalizing_conflict

    cal = PhoenixCalibration(kind="CLB", serial="X",
                             normalizing={"e_gain": 2.0},
                             channels={})
    warnings: list[str] = []
    _warn_normalizing_conflict(
        cal, "box calibration",
        (("EGN", "e_gain", "final E-channel gain", "E-gain"),
         ("HGN", "h_gain", "final H-channel gain", "H-gain")),
        {"EGN": 10, "HGN": 3}, warnings)
    # E-gain conflicts (2 vs 10); the HGN pair is skipped — no h_gain side
    assert len(warnings) == 1
    assert "E-gain 2" in warnings[0] and "EGN=10" in warnings[0]

    warnings.clear()
    _warn_normalizing_conflict(
        cal, "box calibration",
        (("EGN", "e_gain", "final E-channel gain", "E-gain"),),
        {"EGN": 2}, warnings)
    assert warnings == []


def test_read_phoenix_mtu_asc_calibration_missing_warns(tmp_path):
    """Missing box / mismatched coil serials warn but never fail the
    ingest — the TBL scalar calibration still applies."""
    _synthetic_corpus(tmp_path)
    _write_tbl(tmp_path / "SYNTH01.TBL", HXSN="COIL111", HYSN="COIL222",
               HZSN="COIL333")
    _write_cal_files(tmp_path, box=False, coils=("COIL999",))
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    assert res["calibration"] == {}  # nothing matched the TBL serials
    assert any("box calibration" in w for w in res["warnings"])
    assert any("COIL111" in w and "hx" in w for w in res["warnings"])
    assert any("COIL222" in w and "hy" in w for w in res["warnings"])
    assert any("COIL333" in w and "hz" in w for w in res["warnings"])


def test_read_phoenix_mtu_asc_calibration_explicit_folder(tmp_path):
    """An explicit calibration folder overrides the data-directory search."""
    _synthetic_corpus(tmp_path)
    _write_tbl(tmp_path / "SYNTH01.TBL", HXSN="COIL111", HYSN="COIL222",
               HZSN="COIL333")
    cal_dir = _write_cal_files(tmp_path / "cal", box=True,
                               coils=("COIL111", "COIL222", "COIL333"))
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path,
                           calibration_folder=cal_dir)
    assert res["calibration"]["hy"] == ["phoenix_clb_mtu-1234_hy",
                                        "phoenix_clc_coil222_hy"]
    assert res["calibration"]["hz"] == ["phoenix_clb_mtu-1234_hz",
                                        "phoenix_clc_coil333_hz"]
    assert not any("calibration" in w for w in res["warnings"])


# --- saturation-flag QC (history §24.12) ------------------------------------------

def test_read_phoenix_mtu_saturated_mask(tmp_path):
    """Saturated blocks produce a per-sample 'saturated' auxiliary channel
    aligned to the block times; clean recordings write no mask."""
    from datetime import timedelta

    from emptyui_mth5.runs import _resolve_station, _resolve_survey
    from mth5.mth5 import MTH5

    _write_tbl(tmp_path / "SYNTH01.TBL")
    t5 = datetime(2009, 12, 16, 7, 46, 52, tzinfo=timezone.utc)
    rng = np.random.default_rng(9)
    with open(tmp_path / "SYNTH01.TS5", "wb") as fid:
        fid.writelines(_ts_block(t5 + timedelta(seconds=k), 15, 15, 5,
                                saturated=1 if k == 1 else 0, rng=rng) for k in range(4))
    res = read_phoenix_mtu(tmp_path, sample_rates=[15], save_path=tmp_path)
    assert res["runs"] == [("sr15_0001", 15.0, 60)]

    with MTH5() as m:
        m.open_mth5(res["path"], "r")
        station = _resolve_station(_resolve_survey(m, ""), "")
        run = station.get_run("sr15_0001")
        assert "saturated" in run.groups_list
        da = run.get_channel("saturated").to_xarray()
        vals = np.asarray(da.values)
        assert vals.size == 60
        # block 1 (samples 15..29) flagged, others clean
        assert vals[:15].sum() == 0
        assert vals[15:30].sum() == 15
        assert vals[30:].sum() == 0
        # the mask shares the data's time axis
        import pandas as pd

        assert pd.Timestamp(da.coords["time"].values[0]) == \
            pd.Timestamp("2009-12-16T07:46:52")


def test_read_phoenix_mtu_clean_no_mask(tmp_path):
    """A recording with no saturated blocks writes no auxiliary channel."""
    from emptyui_mth5.runs import _resolve_station, _resolve_survey
    from mth5.mth5 import MTH5

    _write_tbl(tmp_path / "SYNTH01.TBL")
    t5 = datetime(2009, 12, 16, 7, 46, 52, tzinfo=timezone.utc)
    _write_ts(tmp_path / "SYNTH01.TS5", 15, [(t5, 4)], seed=10)
    res = read_phoenix_mtu(tmp_path, sample_rates=[15], save_path=tmp_path)
    with MTH5() as m:
        m.open_mth5(res["path"], "r")
        station = _resolve_station(_resolve_survey(m, ""), "")
        run = station.get_run("sr15_0001")
        assert "saturated" not in run.groups_list


def test_read_phoenix_mtu_stream_split(tmp_path):
    """history §24.13: stream_session_hours cuts a long continuous run into
    fixed-length sessions on block boundaries, with tag-truth periods and
    continued run numbering."""
    from emptyui_mth5.runs import _resolve_station, _resolve_survey
    from mth5.mth5 import MTH5

    _write_tbl(tmp_path / "SYNTH01.TBL")
    t5 = datetime(2009, 12, 16, 7, 46, 52, tzinfo=timezone.utc)
    t0 = pd.Timestamp("2009-12-16T07:46:52", tz="UTC")
    # 10 one-second blocks at 15 Hz; session = 4 s -> 4+4+2 s runs
    _write_ts(tmp_path / "SYNTH01.TS5", 15, [(t5, 10)], seed=11)
    res = read_phoenix_mtu(tmp_path, sample_rates=[15], save_path=tmp_path,
                           stream_session_hours=4.0 / 3600.0)
    ids = [r[0] for r in res["runs"]]
    ns = [r[2] for r in res["runs"]]
    assert ids == ["sr15_0001", "sr15_0002", "sr15_0003"]
    assert ns == [60, 60, 30]
    with MTH5() as m:
        m.open_mth5(res["path"], "r")
        station = _resolve_station(_resolve_survey(m, ""), "")
        for k, (rid, n) in enumerate(zip(ids, ns)):
            rg = station.get_run(rid)
            tp = rg.metadata.time_period
            assert pd.Timestamp(tp.start.isoformat()) == \
                t0 + pd.to_timedelta(4 * k, "s")
            assert pd.Timestamp(tp.end.isoformat()) == t0 + pd.to_timedelta(
                4 * k + n / 15.0 - 1 / 15.0, "s")


# --- slot schedule table (Option 1 foundation) ---------------------------

def test_slot_schedule_table(tmp_path):
    """The loader emits the integer slot grid per burst family: 0-based slot
    indices anchored at the family's first actual run start, slot_end =
    slot_start + LxNS, and one row per run. The continuous family leaves the
    slot columns empty. The synthetic corpus deliberately staggers TS3
    against TS4 (the IAGA quirk) — each family's grid is its own."""
    _synthetic_corpus(tmp_path)
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    df = res["slot_schedule"]
    assert list(df.columns) == ["run_id", "sample_rate", "slot_index",
                                "slot_start", "slot_end", "start", "end",
                                "n_samples"]
    by = {r: g.reset_index(drop=True) for r, g in df.groupby("sample_rate")}
    # block-tag times are UTC-aware ISO strings
    # TS3: slots every 600 s from 07:55:00, L3NS=1 s each
    t3 = by[2400.0]
    assert list(t3["run_id"]) == ["sr2400_0001", "sr2400_0002"]
    assert list(t3["slot_index"]) == [0.0, 1.0]
    assert t3["slot_start"][0] == "2009-12-16T07:55:00+00:00"
    assert t3["slot_start"][1] == "2009-12-16T08:05:00+00:00"
    assert t3["slot_end"][0] == "2009-12-16T07:55:01+00:00"
    assert t3["start"][0] == t3["slot_start"][0]
    # TS4: slots every 600 s from 07:50:00 (staggered vs TS3), L4NS=16 s
    t4 = by[150.0]
    assert list(t4["slot_index"]) == [0.0, 1.0]
    assert t4["slot_start"][0] == "2009-12-16T07:50:00+00:00"
    assert t4["slot_end"][0] == "2009-12-16T07:50:16+00:00"
    # TS5: continuous — no slot identity, one row per run
    t5 = by[15.0]
    assert pd.isna(t5["slot_index"]).all()
    assert t5["slot_start"][0] == ""


def test_slot_schedule_hsmp_mismatch_warns(tmp_path):
    """A TBL HSMP that disagrees with the observed slot spacing warns and
    uses the observed spacing (block timestamps are the truth) — never
    nonsense slot indices."""
    _write_tbl(tmp_path / "SYNTH01.TBL", HSMP=5)  # 5 s vs the real 600 s
    _write_ts(tmp_path / "SYNTH01.TS3", 2400,
              [(datetime(2009, 12, 16, 7, 55, 0, tzinfo=timezone.utc), 1),
               (datetime(2009, 12, 16, 8, 5, 0, tzinfo=timezone.utc), 1)], seed=5)
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    df = res["slot_schedule"]
    t3 = df[df["sample_rate"] == 2400.0]
    assert list(t3["slot_index"]) == [0.0, 1.0]
    assert t3["slot_start"][1] == "2009-12-16T08:05:00+00:00"
    assert any("HSMP says 5 s" in w for w in res["warnings"])


def test_slot_schedule_survives_missed_interior_slot(tmp_path):
    """A receiver that missed an interior slot must not corrupt the slot
    grid: the slot period is the SMALLEST observed spacing (a median over
    [600, 1200] s is 900 s — neither the period nor the grid), and the
    0-based indices anchored at the first observed start still recover the
    physical slot positions (slot 2 is simply absent)."""
    _write_tbl(tmp_path / "SYNTH01.TBL")  # defaults: HSMP=600, L3NS=1
    _write_ts(tmp_path / "SYNTH01.TS3", 2400,
              [(datetime(2009, 12, 16, 7, 55, 0, tzinfo=timezone.utc), 1),
               (datetime(2009, 12, 16, 8, 5, 0, tzinfo=timezone.utc), 1),
               (datetime(2009, 12, 16, 8, 25, 0, tzinfo=timezone.utc), 1)],  # 08:15 missed
              seed=6)
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    df = res["slot_schedule"]
    t3 = df[df["sample_rate"] == 2400.0]
    assert list(t3["slot_index"]) == [0.0, 1.0, 3.0]
    assert list(t3["slot_start"]) == [
        "2009-12-16T07:55:00+00:00", "2009-12-16T08:05:00+00:00",
        "2009-12-16T08:25:00+00:00"]
    # no HSMP mismatch warning (the base rhythm still matches HSMP=600)


# --- macro flow: Load -> Extract Run -> Align -> Compute FC --------------

def _two_station_corpora(base: Path) -> tuple[Path, Path]:
    """Two synthetic stations, identical schedule, station B's continuous
    stream starting 300 s later (Align Runs must find the overlap)."""
    dir_a, dir_b = base / "st_a", base / "st_b"
    dir_a.mkdir(), dir_b.mkdir()
    _synthetic_corpus(dir_a)
    _synthetic_corpus(dir_b)
    # station B: re-write the TBL with its own site id
    _write_tbl(dir_b / "SYNTH01.TBL", SITE="SYNTH02")
    return dir_a, dir_b


def test_macro_flow_extract_align(tmp_path):
    """The Option-1 macro flow is plausible today: Load MTU-5A -> Extract
    Run (per slot run OR the continuous stream) -> Align Runs crops two
    stations to their common overlap — the classic single-rate RR flow on
    burst archives."""
    from _fc import compute_fc_seconds
    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_mth5.phoenix_mtu import LoadPhoenixMtu
    from emptyui_mth5.runs import AlignRuns, ExtractRun

    dir_a, dir_b = _two_station_corpora(tmp_path)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    a = LoadPhoenixMtu().execute(ctx, data_path=str(dir_a),
                                 save_path=str(dir_a))["mth5"]
    b = LoadPhoenixMtu().execute(ctx, data_path=str(dir_b),
                                 save_path=str(dir_b))["mth5"]
    # slot runs expose ts.run (time-domain extension surface: e.g. NN
    # denoising consumes per-slot series)
    slot = ExtractRun().execute(ctx, a, survey="SYNTH_SURVEY",
                                station="SYNTH01", selection="run id", run="sr2400_0001")["data"]
    assert slot.dataset.sizes["time"] == 2400  # 1 s @ 2400 Hz
    # continuous streams: the RR workhorse — align crops both to the
    # common overlap (station B starts 300 s later)
    s_a = ExtractRun().execute(ctx, a, survey="SYNTH_SURVEY",
                               station="SYNTH01", selection="run id", run="sr15_0001")["data"]
    s_b = ExtractRun().execute(ctx, b, survey="SYNTH_SURVEY",
                               station="SYNTH02", selection="run id", run="sr15_0001")["data"]
    out = AlignRuns().execute(ctx, s_a, s_b)
    r_a, r_b = out["data1"], out["data2"]
    assert len(r_a.dataset.time) == len(r_b.dataset.time)
    assert str(r_a.start) == str(r_b.start)
    # and FCs computed from the aligned pair share the identical grid
    fc = compute_fc_seconds(r_a, ("hx", "hy"), ("ex", "ey"),
                    window_length_s=32.0, overlap=0.5,
                    window_type="boxcar", start_datetime=str(r_a.start))
    ref = compute_fc_seconds(r_b, ("hx", "hy"), ("hx", "hy"),
                     window_length_s=32.0, overlap=0.5,
                     window_type="boxcar", start_datetime=str(r_b.start))
    assert np.array_equal(fc["time"].values, ref["time"].values)


def test_session_by_slot(tmp_path):
    """session_by_slot cuts the continuous stream at the TBL slot grid
    (every HSMP seconds): slot-granular QC units that preserve longer
    contiguous spans than fixed-hour splitting."""
    _write_tbl(tmp_path / "SYNTH01.TBL", HSMP=20)  # 20-s slot grid
    t5 = datetime(2009, 12, 16, 7, 46, 52, tzinfo=timezone.utc)
    _write_ts(tmp_path / "SYNTH01.TS5", 15, [(t5, 100)], seed=11)  # 100 s
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path,
                           session_by_slot=True)
    ids = [r[0] for r in res["runs"]]
    ns = [r[2] for r in res["runs"]]
    assert ids == [f"sr15_{k:04d}" for k in range(1, 6)]
    assert ns == [300] * 5  # 20 s x 15 Hz per session


def test_session_by_slot_falls_back_without_hsmp(tmp_path):
    """No HSMP in the TBL -> warn and fall back to stream_session_hours."""
    _write_tbl(tmp_path / "SYNTH01.TBL")
    # strip HSMP by rewriting the TBL without it
    tags_path = tmp_path / "SYNTH01.TBL"
    raw = tags_path.read_bytes()
    tags_path.write_bytes(b"".join(
        blk for blk in (raw[i:i + 25] for i in range(0, len(raw), 25))
        if not blk.startswith(b"HSMP")))
    t5 = datetime(2009, 12, 16, 7, 46, 52, tzinfo=timezone.utc)
    _write_ts(tmp_path / "SYNTH01.TS5", 15, [(t5, 10)], seed=11)
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path,
                           session_by_slot=True, stream_session_hours=4e-4)
    assert any("no HSMP" in w for w in res["warnings"])
    # 10 one-second blocks at a 1.44-s session target -> one block each
    assert [r[0] for r in res["runs"]] == [f"sr15_{k:04d}"
                                           for k in range(1, 11)]


def test_read_phoenix_mtu_fractional_window_no_false_alarm(tmp_path):
    """L-M5: a FRACTIONAL-second window's edge crop must not false-alarm
    the schedule cross-check — a block whose tag precedes the bound is
    retained and its end spills past the bound by up to one slot; the
    exemption is bounded by the slot length."""
    _write_tbl(tmp_path / "SYNTH01.TBL")
    rng = np.random.default_rng(13)
    t4 = datetime(2009, 12, 16, 7, 50, 0, tzinfo=timezone.utc)
    with open(tmp_path / "SYNTH01.TS4", "wb") as fid:
        fid.writelines(
            _ts_block(t4 + _second_delta(k), 150, 150, 5, rng=rng)
            for k in range(16))
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path,
                           start="2009-12-16 07:49:00",
                           end="2009-12-16 07:50:10.5")
    assert not any("TBL schedule" in w for w in res["warnings"])


def test_group_segments_subsecond_tolerance_and_out_of_order(tmp_path, caplog):
    """L-M6: the tag carries whole seconds only, so a sub-second block
    step's next tag is quantized — grouping must tolerate one step of
    quantization; blocks arriving out of order fragment with a loud
    warning (sample order IS tag order — never re-sorted)."""
    from emptyui_mth5.phoenix_mtu import _BlockTag, _group_segments

    def tag(y, mo, d, h, mi, s, rate=2400, n_scan=1200):
        return _BlockTag(
            start=pd.Timestamp(
                datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)),
            rate=rate, n_scan=n_scan, n_ch=5, clock_status=4,
            clock_error_s=0.0, saturated=False)

    # step 0.5 s, round-quantized tags: :00, :01, :01, :02 — one segment
    blocks = [
        (tag(2024, 1, 1, 0, 0, 0), np.zeros((5, 1200), dtype="i4")),
        (tag(2024, 1, 1, 0, 0, 1), np.zeros((5, 1200), dtype="i4")),
        (tag(2024, 1, 1, 0, 0, 1), np.zeros((5, 1200), dtype="i4")),
        (tag(2024, 1, 1, 0, 0, 2), np.zeros((5, 1200), dtype="i4")),
    ]
    segs = _group_segments(iter(blocks))
    assert len(segs) == 1
    assert segs[0].n_blocks == 4

    # out of order: the tag precedes the running segment's end - step
    import logging as _logging
    blocks2 = [
        (tag(2024, 1, 1, 0, 0, 10), np.zeros((5, 1200), dtype="i4")),
        (tag(2024, 1, 1, 0, 0, 3), np.zeros((5, 1200), dtype="i4")),
    ]
    with caplog.at_level(_logging.WARNING, logger="emptyui"):
        segs2 = _group_segments(iter(blocks2))
    assert len(segs2) == 2  # fragmented (per block)
    assert any("out of order" in r.message for r in caplog.records)


# --- H7: calibration absence must warn, never load silently --------------

def test_read_phoenix_mtu_warns_when_no_asc(tmp_path):
    """A corpus with no *.asc anywhere must WARN: the TBL scalar chain
    applies but the frequency-dependent shape correction is missing, so
    the TF would be silently wrong. Previously this path returned {}
    without a word ('no ASC tables provided at all — stay silent')."""
    _synthetic_corpus(tmp_path)
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    assert any("no ASC calibration" in w for w in res["warnings"]), (
        res["warnings"])


def test_read_phoenix_mtu_warns_on_unmatched_serials(tmp_path):
    """ASC files present but matching no TBL serial -> a warning per
    missing chain (box + each coil with a serial)."""
    _synthetic_corpus(tmp_path)
    _write_tbl(tmp_path / "SYNTH01.TBL", HXSN="COIL111", HYSN="COIL222")
    # files exist but their names match no serial in the TBL
    (tmp_path / "MTU-9999_CLB.asc").write_text("dummy\n")
    (tmp_path / "COIL9999_CLC.asc").write_text("dummy\n")
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    joined = "\n".join(res["warnings"])
    assert "no box calibration" in joined
    assert "no coil calibration for hx" in joined
    assert "no coil calibration for hy" in joined
