"""Raw-instrument loaders (legacy Phoenix MTU-5A burst recordings).

The MTU-5A writes one binary ``TSn`` file per sample-rate level (TS2..TS5;
rates come from the TBL's ``SRL3/SRL4/SRL5``) plus one ``TBL`` metadata
file. Burst-mode recordings interleave scheduled high-rate slots with
continuous low-rate streams; the per-block 32-byte tags carry the true UTC
timestamp of each block, which the upstream reader discards (DESIGN
§24.8) — here the tags are parsed and maximal contiguous block runs become
one mth5 run each (the mth5/aurora run-per-segment convention,
``sr{rate}_{NNNN:04d}``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from emptyui import Node, Output, Param

_log = logging.getLogger("emptyui")

#: Accepted time-series extensions (upstream mt_io convention).
TS_EXTENSIONS = ("TS2", "TS3", "TS4", "TS5", "TSL", "TSH")

#: TBL sample-rate tags per level (SRL3 -> TS3, ...).
_RATE_TAGS = {3: "SRL3", 4: "SRL4", 5: "SRL5"}

#: TBL seconds-recorded-per-slot tags per level.
_SLOT_TAGS = {3: "L3NS", 4: "L4NS"}


@dataclass
class _BlockTag:
    """Decoded 32-byte per-block tag (layout verified on 1690C16C, §24.8)."""

    start: pd.Timestamp  # UTC time of the block's first scan
    rate: int
    n_scan: int
    n_ch: int
    clock_status: int
    clock_error_s: float
    saturated: bool


def _sign24(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int32)
    x[x > 2**23 - 1] -= 2**24
    return x


def _block_tag(buf: bytes) -> _BlockTag:
    start = pd.Timestamp(
        datetime(2000 + buf[5], buf[4], buf[3], buf[2], buf[1], buf[0]),
        tz="UTC",
    )
    return _BlockTag(
        start=start,
        rate=int.from_bytes(buf[18:20], "little"),
        n_scan=int.from_bytes(buf[10:12], "little"),
        n_ch=buf[12],
        clock_status=buf[21],
        clock_error_s=float(int.from_bytes(buf[22:26], "little", signed=True)),
        saturated=buf[15] != 0,
    )


def _iter_blocks(
    path: Path, warnings: list[str] | None = None,
) -> Iterator[tuple[_BlockTag, np.ndarray]]:
    """Stream a TSn file block by block (tag + samples).

    Validates per-block rate/tag-length consistency and ignores a truncated
    final block with a warning (appended to ``warnings`` when given).
    """
    size = path.stat().st_size
    with open(path, "rb") as fid:
        head = fid.read(32)
        n_scan = int.from_bytes(head[10:12], "little")
        n_ch = head[12]
        tag_len = head[13]
        rate = int.from_bytes(head[18:20], "little")
        if tag_len != 32:
            raise NotImplementedError(
                f"{path.name}: tag length {tag_len} (V5-2000 lineage) is not "
                "supported — only 32-byte MTU-5 tags carry the per-block "
                "rate/time fields the burst parsing needs")
        stride = tag_len + n_scan * n_ch * 3
        n_block, tail = divmod(size, stride)
        if tail:
            msg = (f"{path.name}: {tail} trailing byte(s) after {n_block} "
                   "complete blocks — partial block ignored")
            _log.warning(msg)
            if warnings is not None:
                warnings.append(msg)
        for k in range(n_block):
            fid.seek(k * stride)
            tag = _block_tag(fid.read(32))
            if (tag.rate, tag.n_scan, tag.n_ch) != (rate, n_scan, n_ch):
                raise ValueError(
                    f"{path.name} block {k}: header fields diverge from the "
                    f"file header (rate {tag.rate} != {rate}, n_scan "
                    f"{tag.n_scan} != {n_scan}, n_ch {tag.n_ch} != {n_ch})")
            raw = np.frombuffer(fid.read(n_scan * n_ch * 3), dtype=np.uint8)
            # channel-interleaved per scan: (n_scan, n_ch*3); channel c is
            # the consecutive byte triple (low, mid, high) = [3c, 3c+1, 3c+2]
            raw = raw.reshape(n_scan, n_ch * 3).T.astype(np.int32)
            samples = np.empty((n_ch, n_scan), dtype=np.float64)
            for c in range(n_ch):
                samples[c] = _sign24(
                    raw[3 * c + 2] * 2**16 + raw[3 * c + 1] * 2**8 + raw[3 * c])
            yield tag, samples


@dataclass
class Segment:
    """A maximal contiguous block run within one TSn file."""

    start: pd.Timestamp
    end: pd.Timestamp  # exclusive
    rate: int
    n_blocks: int
    chunks: list[np.ndarray] = field(default_factory=list)
    #: per-block saturated flags (tag byte 15) — a saturated block marks all
    #: its scans (clipped ADC samples); the per-sample mask derived from this
    #: rides the archive as the "saturated" auxiliary channel (§24.12 QC)
    saturated: list[bool] = field(default_factory=list)


def _group_segments(
    blocks: Iterator[tuple[_BlockTag, np.ndarray]],
) -> list[Segment]:
    """Regroup blocks into maximal contiguous runs: a block continues the
    current segment iff its start equals the segment's exclusive end."""
    segs: list[Segment] = []
    cur: Segment | None = None
    for tag, samples in blocks:
        step = pd.to_timedelta(tag.n_scan / tag.rate, unit="s")
        end = tag.start + step
        if cur is not None and tag.start == cur.end:
            cur.end, cur.n_blocks = end, cur.n_blocks + 1
            cur.chunks.append(samples)
            cur.saturated.append(tag.saturated)
        else:
            if cur is not None:
                segs.append(cur)
            cur = Segment(tag.start, end, tag.rate, 1, [samples], [tag.saturated])
    if cur is not None:
        segs.append(cur)
    return segs


def _file_rate(path: Path) -> int:
    with open(path, "rb") as fid:
        return int.from_bytes(fid.read(32)[18:20], "little")


def _split_segments(segs: list[Segment],
                    session_seconds: float) -> list[Segment]:
    """Cut segments longer than ``session_seconds`` into consecutive
    fixed-length sessions on block boundaries (§24.13): the continuous
    stream otherwise forms one giant run whose run-level QC (Select Runs)
    could only be dropped or kept whole. Blocks are assigned greedily — a
    block starts a new session when it would push the current session past
    the target (non-integer block/session ratios crop the last session)."""
    if session_seconds <= 0:
        return segs
    out: list[Segment] = []
    for seg in segs:
        if (seg.end - seg.start).total_seconds() <= session_seconds:
            out.append(seg)
            continue
        chunks: list[np.ndarray] = []
        sat: list[bool] = []
        start = seg.start
        t = seg.start
        for chunk, s in zip(seg.chunks, seg.saturated):
            dur = pd.to_timedelta(chunk.shape[1] / seg.rate, unit="s")
            if chunks and (t + dur - start).total_seconds() > session_seconds:
                out.append(Segment(start, t, seg.rate, len(chunks),
                                   chunks, sat))
                chunks, sat, start = [], [], t
            chunks.append(chunk)
            sat.append(s)
            t = t + dur
        if chunks:
            out.append(Segment(start, t, seg.rate, len(chunks), chunks, sat))
    return out


def _window_bounds(start: str | float | None,
                   end: str | float | None) -> tuple[pd.Timestamp | None,
                                                     pd.Timestamp | None]:
    """Parse the ingest time window (§24.11): UTC ISO datetime or epoch
    seconds; naive stamps are localized to UTC (the block tags are UTC).
    Returns ``(start, end)`` as tz-aware UTC timestamps or ``None``.
    """
    def parse(value, label):
        if value in (None, ""):
            return None
        s = str(value).strip()
        ts = None
        try:
            f = float(s)
            if np.isfinite(f):
                ts = pd.Timestamp(f, unit="s")
        except ValueError:
            pass
        if ts is None:
            try:
                ts = pd.Timestamp(s)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"MTU-5A window {label}={value!r} is not an ISO "
                    "datetime or epoch seconds") from exc
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    lo, hi = parse(start, "start"), parse(end, "end")
    if lo is not None and hi is not None and hi <= lo:
        raise ValueError(
            f"MTU-5A window is empty: end ({hi}) <= start ({lo})")
    return lo, hi


def _ts_files(data_path: Path, sample_rates: list[int] | None) -> list[Path]:
    files = sorted(
        p for p in data_path.iterdir()
        if p.is_file() and p.suffix.lstrip(".").upper() in TS_EXTENSIONS
    )
    if sample_rates:
        files = [p for p in files if _file_rate(p) in sample_rates]
    return files


def _apply_tbl_calibration(
    comp: str, x: np.ndarray, table,
) -> tuple[np.ndarray, str]:
    """Multiply the TBL scalar calibration into the counts (§24.8 decision
    (c): calibration at ingest; filter/FAP handling is later work)."""
    if comp in ("ex", "ey"):
        scale = table.ex_calibration if comp == "ex" else table.ey_calibration
        return x * scale, "mV/km"
    if comp in ("hx", "hy", "hz"):
        return x * table.magnetic_calibration, "nT"
    return x, ""


def _prepare_asc_calibration(
    data_path: Path, calibration_folder: str | Path, table, warnings: list[str],
) -> dict[str, list]:
    """Resolve + parse the ASC box/coil calibration tables (§24.12).

    The TBL serials are the match keys (``SNUM`` -> ``*_CLB.asc``,
    ``HXSN``/``HYSN``/``HZSN`` -> ``*_CLC.asc``); an empty folder means the
    data directory itself (the manufacturer files usually sit next to the
    recording). Returns ``{comp: [FrequencyResponseTableFilter, ...]}`` in
    chain order (box first, coil second); unmatched tables are warnings —
    the TBL scalar calibration still applies, only the shape correction is
    missing.
    """
    from .phoenix_calibration import (
        build_phoenix_filters,
        find_asc_calibrations,
        parse_asc_calibration,
    )

    folder = (Path(str(calibration_folder)).expanduser()
              if str(calibration_folder).strip() else data_path)
    if not folder.exists():
        raise FileNotFoundError(
            f"calibration folder not found: {folder} — set the calibration "
            "parameter to the folder holding the *_CLB.asc / *_CLC.asc "
            "files, or clear it to look next to the recording")
    if not any(p.suffix.lower() == ".asc" for p in folder.iterdir()
               if p.is_file()):
        return {}  # no ASC tables provided at all — stay silent
    box_serial = str(table.tbl_dict.get("SNUM", "")).strip()
    coil_serials = {c: str(table.tbl_dict.get(t, "")).strip()
                    for c, t in (("hx", "HXSN"), ("hy", "HYSN"), ("hz", "HZSN"))}
    found = find_asc_calibrations(folder, box_serial=box_serial,
                                  coil_serials=coil_serials)
    chains: dict[str, list] = {}
    box_file = found.get("box")
    if box_file is not None:
        box = parse_asc_calibration(box_file)
        comp_map = {int(n): c for c, n in table.channel_keys.items()}
        for comp, fap in build_phoenix_filters(box, comp_map=comp_map).items():
            chains.setdefault(comp, []).append(fap)
    else:
        serial_note = (f"matching serial {box_serial!r}" if box_serial
                       else "no 'SNUM' tag in the TBL to match against")
        warnings.append(
            f"no box calibration (*_CLB.asc, {serial_note}) in {folder} — "
            "the frequency-dependent shape correction is missing (the TBL "
            "scalar calibration still applies)")
    for comp, serial in coil_serials.items():
        path = found.get(comp)
        if path is None:
            if serial:
                warnings.append(
                    f"no coil calibration for {comp} (*_CLC.asc matching "
                    f"serial {serial!r}) in {folder} — its shape correction "
                    "is missing (the TBL scalar calibration still applies)")
            continue
        coil = parse_asc_calibration(path)
        filters = build_phoenix_filters(coil, comp_map={1: comp})
        chains.setdefault(comp, []).extend(filters.values())
    return chains


def _slot_schedule(seg_info, table, warnings: list[str]):
    """Slot-grid table for a burst recording (the Option-1 foundation).

    The TBL schedule defines each rate family's burst slots: ``HSMP`` is the
    slot period in seconds (MTU-5A) and the instrument records ``L3NS``/
    ``L4NS`` seconds at the BEGINNING of every slot (mt_io's tag reference).
    Slot identity is derived from the ACTUAL block timestamps (tag truth)
    with the schedule as the cross-check: per burst family, the slot period
    is the observed median spacing of consecutive run starts (falling back
    to ``HSMP`` when a family has a single slot); a disagreement with
    ``HSMP`` is a warning, never a rejection — the block timestamps are the
    truth, and the official IAGA corpus is known to stagger TS3/TS4 against
    the TS5 slot grid (upstream documentation issue).

    Returns a DataFrame with one row per run: ``run_id, sample_rate,
    slot_index, slot_start, slot_end, start, end, n_samples`` — burst
    families carry the integer slot grid (0-based, anchored at the
    family's first actual run start); continuous families (``SRL5``, no
    slot tag) leave the slot columns empty and report one row per run
    (sessions included).
    """
    import numpy as np
    import pandas as pd

    tbl = getattr(table, "tbl_dict", {}) or {}
    hsmp = float(tbl.get("HSMP", 0) or 0)
    rate_to_slot = {}
    for lv, slot_tag in _SLOT_TAGS.items():
        rate = float(tbl.get(_RATE_TAGS[lv], 0) or 0)
        slot_len = float(tbl.get(slot_tag, 0) or 0)
        if rate > 0 and slot_len > 0:
            rate_to_slot[rate] = slot_len
    by_rate: dict[float, list] = {}
    for run_id, rate, start, end, n in seg_info:
        by_rate.setdefault(float(rate), []).append((run_id, start, end, n))
    rows = []
    for rate in sorted(by_rate, reverse=True):
        entries = sorted(by_rate[rate], key=lambda x: x[1])
        slot_len = rate_to_slot.get(rate)
        if slot_len is None:
            for run_id, start, end, n in entries:
                rows.append((run_id, rate, np.nan, "", "",
                             _iso(start), _iso(end), n))
            continue
        starts = np.array([pd.Timestamp(s).value for _, s, _, _ in entries]) / 1e9
        diffs = np.diff(starts) if starts.size > 1 else np.array([])
        period = hsmp
        if diffs.size:
            median = float(np.median(diffs))
            if hsmp > 0 and abs(median - hsmp) > max(2.0, 0.05 * hsmp):
                warnings.append(
                    f"TBL HSMP says {hsmp:g} s but the observed {rate:g} Hz "
                    f"slot spacing is {median:g} s — using the observed "
                    "spacing (actual block timestamps are the truth)")
                period = median
        elif hsmp <= 0:
            period = 0.0
        anchor = starts[0]
        for (run_id, start, end, n), k_start in zip(entries, starts):
            k = int(round((k_start - anchor) / period)) if period > 0 else 0
            grid_start = (pd.Timestamp(anchor, unit="s")
                          + pd.to_timedelta(k * period, unit="s"))
            # block-tag times are UTC (the tags carry no zone, but the
            # instrument timestamps are UTC by definition — the repo-wide
            # naive-assumed-UTC convention); the grid matches that stamp
            # convention
            if grid_start.tzinfo is None:
                grid_start = grid_start.tz_localize("UTC")
            grid_end = grid_start + pd.to_timedelta(slot_len, unit="s")
            rows.append((run_id, rate, float(k), grid_start.isoformat(),
                         grid_end.isoformat(), _iso(start), _iso(end), n))
    return pd.DataFrame(
        rows,
        columns=["run_id", "sample_rate", "slot_index", "slot_start",
                 "slot_end", "start", "end", "n_samples"],
    )


def _iso(dt) -> str:
    return pd.Timestamp(dt).isoformat()


def read_phoenix_mtu(
    data_path: str | Path,
    sample_rates: list[int] | None = None,
    survey_id: str = "",
    station_id: str = "",
    save_path: str | Path | None = None,
    mth5_filename: str = "from_phoenix_mtu.mth5",
    start: str | float = "",
    end: str | float = "",
    calibration_folder: str | Path = "",
    stream_session_hours: float = 0.0,
    session_by_slot: bool = False,
) -> dict[str, Any]:
    """Convert a legacy Phoenix MTU-5A recording (TBL + TSn files) into an
    mth5 archive with one run per maximal contiguous segment.

    Run ids follow the mth5/aurora convention ``sr{rate}_{NNNN:04d}``
    (§24.7); run time periods come from the per-block tag timestamps —
    never the TBL's placeholder STIM/ETIM. TBL scalar calibrations are
    applied at ingest and units stamped (mV/km, nT).

    ``start``/``end`` (§24.11) restrict the ingest to a time window
    (UTC ISO datetime or epoch seconds; naive stamps read as UTC): blocks
    are filtered by their per-block tag start *before* segment grouping —
    per-second granularity (every MTU-5A block is ≤ 1 s of data), so slots
    and the continuous stream crop cleanly at the window edges and partial
    slots become short runs with tag-truth time periods. Use this to
    extract a quiet night window for fast smoke tests.

    ``calibration_folder`` (§24.12): directory holding the manufacturer ASC
    tables (``*_CLB.asc`` box + ``*_CLC.asc`` coils). Empty searches the
    data directory itself. Files are matched by the TBL serials (``SNUM``,
    ``HXSN``/``HYSN``/``HZSN``) and registered in the archive's
    ``/Survey/Filters`` layer as un-applied-to-FC FAP shape corrections —
    the Compute FC node divides the Fourier coefficients by their chained
    complex response (frequency-domain calibration); unmatched tables only
    warn.

    ``stream_session_hours`` (§24.13): split runs longer than this many
    hours into consecutive fixed-length sessions on block boundaries
    (0 = off, default). The continuous stream otherwise forms one giant
    run whose run-level QC (Select Runs) could only be dropped or kept
    whole; splitting gives every hour its own run id so selection can
    screen stream data. Slots shorter than the session pass through
    unchanged.

    Returns ``{"path": archive_path, "runs": [(run_id, rate, n_samples)],
    "warnings": [str], "calibration": {comp: [filter names]}}``.
    """
    data_path = Path(data_path).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"MTU-5A data dir not found: {data_path}")
    tbls = sorted(data_path.glob("*.TBL")) + sorted(data_path.glob("*.tbl"))
    if len(tbls) != 1:
        raise ValueError(
            f"expected exactly one TBL in {data_path}, found "
            f"{[p.name for p in tbls]}")
    win_start, win_end = _window_bounds(start, end)
    from mth5.mth5 import MTH5
    from mt_io.phoenix.readers.mtu.mtu_table import MTUTable
    from mt_timeseries import ChannelTS

    table = MTUTable(tbls[0])
    table.read_tbl()
    station_id = (station_id or str(table.tbl_dict.get("SITE", "")).strip()
                  or "station")
    survey_id = (survey_id or str(table.tbl_dict.get("SRVY", "")).strip()
                 or "survey")
    channel_keys = table.channel_keys
    warnings: list[str] = []
    chains = _prepare_asc_calibration(data_path, calibration_folder, table,
                                      warnings)

    # session length: slot-grid sessions (session_by_slot) cut the
    # continuous stream every HSMP seconds — slot-granular QC preserves
    # longer contiguous spans than fixed-hour splitting (§24.13/§9.3)
    session_seconds = stream_session_hours * 3600.0
    if session_by_slot:
        hsmp = float((table.tbl_dict or {}).get("HSMP", 0) or 0)
        if hsmp > 0:
            session_seconds = (min(session_seconds, hsmp)
                               if session_seconds > 0 else hsmp)
        else:
            warnings.append(
                "session_by_slot requested but the TBL carries no HSMP — "
                "falling back to stream_session_hours")

    runs: list[tuple[str, Any, list]] = []  # (run_id, run_metadata, ChannelTS list)
    n_samples_by_run: list[int] = []
    seg_info: list[tuple] = []  # (run_id, rate, start, end, n_samples) per run
    statuses: set[int] = set()
    max_clock_error = 0.0
    n_saturated = 0

    for path in _ts_files(data_path, sample_rates):
        # single pass over the file: tags feed the quality aggregation,
        # samples feed the segment grouping (re-reading 20 h of 15 Hz data
        # per pass measurably slows ingestion)
        blocks = list(_iter_blocks(path, warnings))
        if win_start is not None or win_end is not None:
            n0 = len(blocks)
            blocks = [
                (t, s) for (t, s) in blocks
                if (win_start is None or t.start >= win_start)
                and (win_end is None or t.start < win_end)
            ]
            msg = (f"{path.name}: ingest window [{win_start}, {win_end}) "
                   f"dropped {n0 - len(blocks)}/{n0} block(s)")
            _log.info(msg)
            warnings.append(msg)
        for tag, _samples in blocks:
            statuses.add(tag.clock_status)
            max_clock_error = max(max_clock_error, abs(tag.clock_error_s))
            n_saturated += tag.saturated
        segs = _group_segments(iter(blocks))
        # schedule cross-check (loose): slot lengths vs the TBL schedule;
        # window-cropped edge segments are exempt (they legitimately span
        # less than the slot length)
        for lv, rate_tag in _RATE_TAGS.items():
            expected_rate = int(table.tbl_dict.get(rate_tag, 0) or 0)
            if expected_rate != _file_rate(path):
                continue
            slot_tag = _SLOT_TAGS.get(lv)
            expected_len = int(table.tbl_dict.get(slot_tag, 0) or 0) if slot_tag else 0
            for seg in segs:
                got = (seg.end - seg.start).total_seconds()
                if expected_len and abs(got - expected_len) > 1e-6:
                    # a window-cropped segment LIES INSIDE the window (the
                    # block filter drops whole blocks, so the first retained
                    # block's tag is >= but not necessarily == the bound) —
                    # exact-equality checks false-alarmed on those
                    inside_window = (
                        (win_start is None or seg.start >= win_start)
                        and (win_end is None or seg.end <= win_end))
                    if (win_start is not None or win_end is not None) and inside_window:
                        continue  # window crop, not a schedule anomaly
                    warnings.append(
                        f"{path.name}: segment {seg.start} spans {got:g} s, "
                        f"TBL schedule says {expected_len:g} s ({slot_tag})")
        for i, seg in enumerate(
                _split_segments(segs, session_seconds), 1):
            run_id = f"sr{seg.rate}_{i:04d}"
            run_meta = table.run_metadata.copy()
            run_meta.id = run_id
            run_meta.sample_rate = float(seg.rate)
            # actual block-tag time period — never the TBL placeholders
            # (STIM/ETIM). Run end is the last SAMPLE (mt_timeseries
            # inclusive convention); the CHANNEL metadata below uses the
            # exclusive segment end (deliberate: ChannelTS anchors its time
            # index on time_period.start, and from_channel_ts propagates
            # channel periods — changing the channel end would perturb the
            # §24.12-verified ingest behavior).
            run_meta.time_period.start = seg.start.isoformat()
            run_meta.time_period.end = (seg.end - pd.to_timedelta(
                1.0 / seg.rate, unit="s")).isoformat()
            array_list = []
            if any(seg.saturated):
                # per-sample saturation mask (§24.12 QC): a saturated block
                # marks all its scans. Rides the archive as the "saturated"
                # auxiliary channel; Compute FC rejects every FC window
                # overlapping a flagged sample (zero weight in the sort chain)
                mask = np.concatenate([
                    np.full(chunk.shape[1], float(s), dtype=float)
                    for chunk, s in zip(seg.chunks, seg.saturated)
                ])
                from mt_metadata.timeseries import Auxiliary

                aux_md = Auxiliary()
                aux_md.time_period.start = seg.start.isoformat()
                aux_md.time_period.end = seg.end.isoformat()
                aux_md.sample_rate = float(seg.rate)
                aux = ChannelTS("auxiliary", data=mask, channel_metadata=aux_md)
                aux.component = "saturated"
                array_list.append(aux)
            for comp, pos in channel_keys.items():
                meta = getattr(table, f"{comp}_metadata")
                meta.sample_rate = float(seg.rate)
                # the ChannelTS time index anchors on time_period.start —
                # setting it per segment is what gives every archived
                # dataset its tag-truth time axis (the run metadata period
                # alone is not enough: from_channel_ts propagates the
                # channel's period into the run)
                meta.time_period.start = seg.start.isoformat()
                meta.time_period.end = seg.end.isoformat()
                x = np.concatenate([c[pos - 1] for c in seg.chunks])
                x, unit = _apply_tbl_calibration(comp, x, table)
                if unit:
                    meta.units = unit
                refs = chains.get(comp) or []
                if refs:
                    from mt_metadata.timeseries.filtered import AppliedFilter

                    # applied=True: the ASC shape response IS part of the
                    # recorded data (aurora semantics — calibrate removes
                    # it); stage order = chain order (box first, coil second)
                    meta.filters = [
                        AppliedFilter(name=fap.name, applied=True, stage=i)
                        for i, fap in enumerate(refs, 1)
                    ]
                array_list.append(ChannelTS(
                    "electric" if comp.startswith("e") else "magnetic",
                    data=x,
                    channel_metadata=meta,
                ))
            # (run_id, run_metadata, channels) — written via from_channel_ts
            # below: RunGroup.from_runts costs ~5x more per run (mt_metadata
            # deepcopy/validation storm + station-metadata rebuild, DESIGN
            # §24.8 performance note)
            runs.append((run_id, run_meta, array_list))
            n_samples_by_run.append(array_list[0].ts.size)
            seg_info.append((run_id, float(seg.rate), seg.start, seg.end,
                             n_samples_by_run[-1]))
        blocks = None  # free the sample chunks

    if not runs:
        raise ValueError(
            "no MTU-5A blocks fall within the requested ingest window "
            f"[{win_start}, {win_end}) — check start/end against the "
            "recording (the per-block tags carry the true UTC times)")

    if len(statuses) > 1:
        warnings.append(
            f"per-block clock status varies across the recording: "
            f"{sorted(statuses)} — check GPS lock (DESIGN §24.5 P1 v2b)")
    if max_clock_error:
        warnings.append(
            f"per-block clock error up to {max_clock_error:g} s reported by "
            "the receiver")
    if n_saturated:
        warnings.append(
            f"{n_saturated} saturated block(s) (tag byte 15 != 0)")

    save = (Path(save_path).expanduser().resolve() if save_path
            else data_path)
    slot_schedule = _slot_schedule(seg_info, table, warnings)
    target = save / mth5_filename
    if target.exists():
        _log.warning("overwriting existing MTH5 at %s", target)
        target.unlink()

    with MTH5() as m:
        m.open_mth5(str(target), "w", locking=False)
        survey_group = m.add_survey(survey_id)
        registered: set[str] = set()
        for refs in chains.values():
            for fap in refs:
                if fap.name not in registered:
                    survey_group.filters_group.add_filter(fap)
                    registered.add(fap.name)
        station_group = survey_group.stations_group.add_station(
            station_id, station_metadata=table.station_metadata)
        station_group.metadata.id = station_id
        for run_id, run_meta, array_list in runs:
            run_group = station_group.add_run(run_id, run_metadata=run_meta)
            for ch_ts in array_list:
                run_group.from_channel_ts(ch_ts)
        station_group.update_metadata()
        survey_group.update_metadata()

    return {
        "path": str(target),
        "slot_schedule": slot_schedule,
        "runs": [(rid, float(meta.sample_rate), n)
                 for (rid, meta, _chs), n in zip(runs, n_samples_by_run)],
        "warnings": warnings,
        "calibration": {comp: [fap.name for fap in refs]
                        for comp, refs in chains.items()},
    }


class LoadPhoenixMtu(Node):
    type = "mth5.io.phoenix_mtu"
    title = "Load Phoenix MTU-5A"
    tooltip = "Convert raw legacy Phoenix MTU-5A (TBL/TSn) data into an MTH5 archive and open it."
    description = (
        "Convert a legacy Phoenix MTU-5A recording (one TBL + one TSn file "
        "per sample-rate level) into an MTH5 archive and open it as the "
        "data source. Burst-mode recordings (scheduled high-rate slots + "
        "continuous low-rate streams) become one run per maximal contiguous "
        "segment (mth5 convention sr{rate}_NNNN) — the per-block tags' true "
        "timestamps are parsed, unlike the upstream reader that densely "
        "concatenates and compresses the slots (DESIGN §24.8). TBL scalar "
        "calibrations are applied at ingest (units mV/km / nT); run time "
        "periods come from the actual block timestamps, never the TBL's "
        "placeholder schedule times. Manufacturer ASC calibration tables "
        "(box CLB + coil CLC) are auto-matched by the TBL serials and "
        "registered as FAP shape corrections that Compute FC removes "
        "frequency-domain-wise (DESIGN §24.12)."
    )
    category = "Data I/O"
    inputs = ()
    outputs = (Output("mth5", "mth5"),
               Output("slot_schedule", "table", label="slot schedule"),)
    params = (
        Param("data_path", "directory", label="MTU-5A data directory",
              path_kind="in"),
        Param("survey_id", "string", default="", optional=True,
              placeholder="auto (TBL SRVY)"),
        Param("station_id", "string", default="", optional=True,
              placeholder="auto (TBL SITE)"),
        Param("sample_rates", "string", label="Sample rates (Hz)",
              default="", optional=True,
              placeholder="e.g. 2400, 150 (empty = all TS files)"),
        Param("mth5_filename", "string", label="Output file name",
              default="from_phoenix_mtu.mth5"),
        Param("save_path", "directory", label="Output directory", default="",
              optional=True, path_kind="out"),
        Param("start", "string", label="Start (UTC)", default="", optional=True,
              placeholder="empty = recording start",
              description="Ingest window start (§24.11): UTC ISO datetime "
                          "or epoch seconds; blocks are filtered by their "
                          "per-block tag start before segment grouping — "
                          "use a quiet night window for fast smoke tests."),
        Param("end", "string", label="End (UTC)", default="", optional=True,
              placeholder="empty = recording end",
              description="Ingest window end (half-open): UTC ISO datetime "
                          "or epoch seconds."),
        Param("calibration_folder", "directory",
              label="Calibration folder (ASC)", default="", optional=True,
              path_kind="in",
              placeholder="empty = the data directory",
              description="Directory with the manufacturer ASC calibration "
                          "tables (*_CLB.asc box, *_CLC.asc coils). Files "
                          "are matched by the TBL serials (SNUM, HXSN/HYSN/"
                          "HZSN) and registered as FAP shape corrections — "
                          "Compute FC removes them frequency-domain-wise "
                          "(§24.12). Unmatched tables only warn."),
        Param("stream_session_hours", "number",
              label="Stream session length (h)", default=0.0, min=0,
              optional=True,
              description="Split runs longer than this many hours into "
                          "consecutive fixed-length sessions on block "
                          "boundaries (0 = off). The continuous stream "
                          "otherwise forms one giant run whose run-level "
                          "QC (Select Runs) could only be dropped or kept "
                          "whole (§24.13)."),
        Param("session_by_slot", "boolean", label="Session by slot",
              default=False,
              description="Cut the continuous stream into sessions at the "
                          "TBL slot grid (every HSMP seconds) instead of a "
                          "fixed length — slot-granular QC: a bad slot "
                          "interval is masked without sacrificing the "
                          "surrounding hours, so the surviving stream "
                          "stays contiguous over much longer spans than "
                          "hour-splitting allows (the deep decimation "
                          "levels cascade the KEPT contiguous span, "
                          "regardless of how sessions are labelled). "
                          "Anchored at the stream's own start; families "
                          "with staggered slot grids keep their own "
                          "rhythm. Falls back to 'Stream session length' "
                          "with a warning when the TBL carries no HSMP."),
    )

    def execute(self, ctx, data_path: str, survey_id: str = "",
                station_id: str = "", sample_rates: str = "",
                mth5_filename: str = "from_phoenix_mtu.mth5",
                save_path: str = "", start: str = "",
                end: str = "", calibration_folder: str = "",
                stream_session_hours: float = 0.0,
                session_by_slot: bool = False) -> dict[str, Any]:
        rates = ([int(x) for x in str(sample_rates).split(",") if x.strip()]
                 if str(sample_rates).strip() else None)
        res = read_phoenix_mtu(
            data_path,
            sample_rates=rates,
            survey_id=survey_id,
            station_id=station_id,
            save_path=save_path or None,
            mth5_filename=mth5_filename,
            start=start,
            end=end,
            calibration_folder=calibration_folder,
            stream_session_hours=stream_session_hours,
            session_by_slot=session_by_slot,
        )
        for w in res["warnings"]:
            ctx.logger.warning(w)
        cal = res.get("calibration") or {}
        if cal:
            ctx.logger.info(
                "ASC calibration attached: %s",
                "; ".join(f"{c}={'+'.join(ns)}" for c, ns in sorted(cal.items())),
            )
        from mth5.mth5 import MTH5

        archive = MTH5()
        archive.open_mth5(res["path"], mode="a", locking=False)
        ctx.progress(
            1.0,
            f"loaded MTU-5A -> {Path(res['path']).name}: "
            f"{len(res['runs'])} run(s)",
        )
        return {"mth5": archive, "slot_schedule": res["slot_schedule"]}
