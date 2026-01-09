"""MTU-5A burst ingestion tests (DESIGN §24.8): per-block tag parsing
(the upstream reader's dense-concat defect fixed), maximal-contiguous
segment grouping, run-per-segment archive build, TBL scalar calibration at
ingest, quality warnings. The synthetic corpus is written from scratch
(binary TBL tags + TS blocks per the verified layout) so CI passes without
the gitignored real corpus; the real-corpus test (IAGA 1690C16C) is
skipped when the data dir is absent."""

from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path

import numpy as np
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
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
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
        "HSMP": 5, "L3NS": 1, "L4NS": 16,
        "SRL3": 2400, "SRL4": 150, "SRL5": 15,
        "FSCV": 6.4, "EGN": 2, "HGN": 2, "HATT": 1.0, "HNOM": 1000.0,
    }
    tags.update(overrides)
    with open(path, "wb") as fid:
        for tag, value in tags.items():
            fid.write(_tbl_value(tag, value))


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
    t0 = datetime(2009, 12, 16, 7, 55, 0)
    _write_ts(data_path / "SYNTH01.TS3", 2400,
              [(t0, 1), (t0 + _slot_delta(10), 1)], seed=1)
    t4 = datetime(2009, 12, 16, 7, 50, 0)
    _write_ts(data_path / "SYNTH01.TS4", 150,
              [(t4, 16), (t4 + _slot_delta(10), 16)], seed=2)
    t5 = datetime(2009, 12, 16, 7, 46, 52)
    _write_ts(data_path / "SYNTH01.TS5", 15, [(t5, 100)], seed=3)


def _slot_delta(minutes: int):
    from datetime import timedelta

    return timedelta(minutes=minutes)


# --- unit tests ----------------------------------------------------------

def test_block_tag_decode(tmp_path):
    """The per-block 32-byte tag decodes time/rate/quality fields (the
    layout verified byte-level on the official recording, §24.8)."""
    import pandas as pd

    from emptyui_mth5.phoenix_mtu import _block_tag

    rng = np.random.default_rng(0)
    t = datetime(2009, 12, 16, 8, 5, 0)
    tag = _block_tag(_ts_block(t, 2400, 2400, 5, rng=rng))
    assert tag.start == pd.Timestamp(t, tz="UTC")
    assert (tag.rate, tag.n_scan, tag.n_ch) == (2400, 2400, 5)
    assert tag.clock_status == 4 and tag.clock_error_s == 0.0
    assert not tag.saturated


def test_group_segments_slots_and_continuous(tmp_path):
    """Contiguous blocks regroup: slots stay separate, continuous streams
    collapse to one segment."""
    p = tmp_path / "SYNTH01.TS4"
    t4 = datetime(2009, 12, 16, 7, 50, 0)
    _write_ts(p, 150, [(t4, 16), (t4 + _slot_delta(10), 16)], seed=4)
    segs = _group_segments(_iter_blocks(p))
    assert len(segs) == 2
    assert all(s.n_blocks == 16 for s in segs)
    assert (segs[1].start - segs[0].start).total_seconds() == 600.0

    p5 = tmp_path / "SYNTH01.TS5"
    _write_ts(p5, 15, [(datetime(2009, 12, 16, 7, 46, 52), 100)], seed=5)
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
    assert res["warnings"] == []  # clean synthetic recording

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


def test_read_phoenix_mtu_sample_rate_filter(tmp_path):
    _synthetic_corpus(tmp_path)
    res = read_phoenix_mtu(tmp_path, sample_rates=[15], save_path=tmp_path)
    assert [r[0] for r in res["runs"]] == ["sr15_0001"]


def test_read_phoenix_mtu_quality_warnings(tmp_path):
    """Mixed clock status, saturated blocks and clock error surface as
    warnings — never rejections (§24.8)."""
    _write_tbl(tmp_path / "SYNTH01.TBL")
    rng = np.random.default_rng(9)
    t = datetime(2009, 12, 16, 7, 46, 52)
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
    t = datetime(2009, 12, 16, 7, 46, 52)
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
    t = datetime(2009, 12, 16, 7, 55, 0)
    with open(tmp_path / "SYNTH01.TS3", "wb") as fid:
        fid.write(_ts_block(t, 2400, 2400, 5, rng=rng))
    # the TBL says L3NS = 1 s; write a second 1-s slot 2 s later — fine
    # (spacing is loose), but then a 2-s continuous segment: length warns
    with open(tmp_path / "SYNTH01.TS4", "wb") as fid:
        for k in range(32):  # 32 s continuous at 150 Hz — schedule says 16
            fid.write(_ts_block(t + _second_delta(k), 150, 150, 5, rng=rng))
    res = read_phoenix_mtu(tmp_path, save_path=tmp_path)
    assert any("TBL schedule says 16 s" in w for w in res["warnings"])


def test_read_phoenix_mtu_no_tbl(tmp_path):
    with pytest.raises(ValueError, match="exactly one TBL"):
        read_phoenix_mtu(tmp_path, save_path=tmp_path)


# --- real-corpus test (IAGA mth5-test-data phoenix_mtu) ------------------

@pytest.mark.skipif(not REAL_DIR.exists(), reason="MTU-5A corpus not on disk")
def test_real_corpus_structure(tmp_path):
    """The official 1690C16C recording: TS3 -> 121 x 1 s, TS4 -> 122 x 16 s,
    TS5 -> 1 x 73036 s (DESIGN §24.8 dry-run numbers). Parse-only (the
    244-run archive write costs ~2 min — its mechanics are covered by the
    synthetic tests); a single-rate archive build keeps an end-to-end
    round trip in the test without the write cost."""
    for ext, rate, n_blocks in (("TS3", 2400, 121), ("TS4", 150, 1952),
                                ("TS5", 15, 73036)):
        segs = _group_segments(_iter_blocks(REAL_DIR / f"1690C16C.{ext}"))
        assert len(segs) == (1 if ext == "TS5" else {2400: 121, 150: 122}[rate])
        assert sum(s.n_blocks for s in segs) == n_blocks
        assert all(s.rate == rate for s in segs)
        if ext == "TS4":
            assert all(s.n_blocks == 16 for s in segs)
        if ext == "TS3":
            assert all(s.n_blocks == 1 for s in segs)
    # end-to-end archive round trip at the continuous rate only (1 run)
    res = read_phoenix_mtu(REAL_DIR, sample_rates=[15], save_path=tmp_path)
    assert [r[0] for r in res["runs"]] == ["sr15_0001"]
    assert res["runs"][0][2] == 73036 * 15
