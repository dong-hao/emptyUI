#!/usr/bin/env python3
"""Scan Phoenix MTU-5A/5C recordings (TBL + TSn files) for station time
extents and remote-reference overlap candidates.

Standalone — pure standard library, no emptyUI/mth5/mt_io imports. Copy it
anywhere and run:

    python phoenix_station_overlap.py ROOT [--output-dir DIR]
                                          [--min-overlap-seconds N]

For every ``*.TBL`` under ROOT (recursively) the sibling TSn files
(TS2..TS5, TSL, TSH) are peeked: the FIRST block's 32-byte tag is read at
the file head and the LAST block's tag near EOF (no full-file scan), giving
the station's near-exact recording extent. The TBL's own STIM/ETIM records
are deliberately NOT trusted — they are placeholders, not the data extent
(the emptyUI/aurora ingest convention: block tags are truth).

Outputs (in --output-dir, default the current directory):
  phoenix_stations.csv   one row per station (TBL): union extent + warnings
  phoenix_ts_files.csv   one row per TSn file: rate, extent, clock notes
and a console report listing remote-reference candidates: station PAIRS
whose union extents overlap by at least --min-overlap-seconds (default
3600 s = one hour), plus COMMON-INTERSECTION GROUPS — sets of >= 3
co-deployed stations sharing a common window that long (any of them can
serve as the remote for the others).

Notes:
- One TBL per directory is the MTU convention; multiple TBLs in one
  directory are processed separately (each sees all sibling TSn files)
  with a loud warning — ownership is ambiguous there.
- Block tags are UTC wall-clock (verified layout: sec/min/hour/day/month/
  year-2000 + rate/n_scan/n_ch); see emptyUI packages/emptyui-mth5/
  phoenix_mtu.py for the full record.
- The peek assumes constant (rate, n_scan, n_ch) within a file — the
  recorder guarantees this; a diverging last-block tag is warned about.
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: Accepted time-series extensions (upstream mt_io convention).
TS_EXTENSIONS = ("TS2", "TS3", "TS4", "TS5", "TSL", "TSH")

TAG_LEN = 32  # MTU-5 per-block tag length (V5-2000 lineage uses other lengths)

#: TBL tag types — ported VERBATIM from upstream mt_io
#: (MTUTable.TBL_TAG_TYPES). Tags not listed here are decoded best-effort
#: as latin-1 strings (upstream keeps raw bytes; the scanner only needs
#: readable SITE/SRVY/SNUM/SRLn/STIM/ETIM).
TBL_TAG_TYPES: dict[str, str] = {
    "SNUM": "int",
    "SITE": "char",
    "FILE": "char",
    "FLEN": "int",
    "FTIM": "time",
    "CMPY": "char",
    "SRVY": "char",
    "LATG": "char",
    "LNGG": "char",
    "ELEV": "int",
    "NREF": "int",
    "STIM": "time",
    "ETIM": "time",
    "EXLN": "double",
    "EYLN": "double",
    "HXSN": "char",
    "HYSN": "char",
    "HZSN": "char",
    "EAZM": "double",
    "HAZM": "double",
    "HTIM": "time",
    "HSMP": "int",
    "L3NS": "int",
    "L4NS": "int",
    "LTIME": "time",
    "SRL3": "int",
    "SRL4": "int",
    "SRL5": "int",
    "LFRQ": "byte",
    "EGNC": "int",
    "HGNC": "int",
    "EGN": "int",
    "HGN": "int",
    "HATT": "double",
    "HNOM": "double",
    "TCMB": "byte",
    "TALS": "byte",
    "LPFR": "byte",
    "ACDC": "byte",
    "FSCV": "double",
    "TEMP": "int",
    "TERR": "int",
    "V5SR": "int",
    "NSAT": "int",
    "DECL": "double",
    "TSTV": "double",
    "VER": "char",
    "HW": "16s",
    "EXAC": "double",
    "EXDC": "double",
    "EYAC": "double",
    "EYDC": "double",
    "HXAC": "double",
    "HXDC": "double",
    "HYAC": "double",
    "HYDC": "double",
    "HZAC": "double",
    "HZDC": "double",
    "STDE": "double",
    "STDH": "double",
    "SPTH": "char",
    "CHEX": "int",
    "CHEY": "int",
    "CHHX": "int",
    "CHHY": "int",
    "CHHZ": "int",
}


@dataclass
class BlockTag:
    """Decoded 32-byte per-block tag (layout per emptyUI phoenix_mtu.py)."""

    start: datetime  # UTC time of the block's first scan
    rate: int
    n_scan: int
    n_ch: int
    clock_status: int
    clock_error_s: int
    saturated: bool


def parse_block_tag(buf: bytes) -> BlockTag:
    start = datetime(
        2000 + buf[5],
        buf[4],
        buf[3],
        buf[2],
        buf[1],
        buf[0],
        tzinfo=timezone.utc,
    )
    return BlockTag(
        start=start,
        rate=int.from_bytes(buf[18:20], "little"),
        n_scan=int.from_bytes(buf[10:12], "little"),
        n_ch=buf[12],
        clock_status=buf[21],
        clock_error_s=int.from_bytes(buf[22:26], "little", signed=True),
        saturated=buf[15] != 0,
    )


def decode_tbl_value(value_bytes: bytes, data_type: str):
    """Decode one 13-byte TBL value (ported from upstream mt_io)."""
    if data_type == "int":
        return struct.unpack("<i", value_bytes[0:4])[0]
    if data_type == "double":
        return struct.unpack("<d", value_bytes[0:8])[0]
    if data_type == "char":
        return value_bytes.decode("latin-1").strip("\x00").strip()
    if data_type == "byte":
        return value_bytes[0]
    if data_type == "time":
        # [sec, min, hour, day, month, year-2000]
        return (
            f"20{value_bytes[5]:02}-{value_bytes[4]:02}-"
            f"{value_bytes[3]:02}-T{value_bytes[2]:02}:"
            f"{value_bytes[1]:02}:{value_bytes[0]:02}"
        )
    if data_type == "16s":
        return (
            value_bytes[:16]
            .ljust(16, b"\x00")
            .split(b"\x00", 1)[0]
            .decode("ascii", errors="ignore")
            .strip()
        )
    return value_bytes


def read_tbl_dict(tbl_path: Path) -> dict[str, object]:
    """Minimal TBL reader: 25-byte records — 12-byte null-padded tag name
    + 13-byte typed value (see TBL_TAG_TYPES). Unknown tags decode
    best-effort as latin-1 strings."""
    tags: dict[str, object] = {}
    with open(tbl_path, "rb") as fid:
        while True:
            block = fid.read(25)
            if len(block) < 25:
                break
            key = block[0:12].decode("latin-1").split("\x00")[0].strip()
            if not key:
                continue
            value_bytes = block[12:]
            data_type = TBL_TAG_TYPES.get(key)
            if data_type is None:
                data_type = "char"  # unknown tag — best-effort readable
            tags.setdefault(key, decode_tbl_value(value_bytes, data_type))
    return tags


@dataclass
class FileExtent:
    """Peeked extent of one TSn file."""

    path: Path
    rate: int
    start: datetime
    end: datetime  # exclusive: last block start + its span
    n_blocks: int
    clock_status_first: int
    clock_status_last: int
    clock_error_s_max: int
    saturated: bool
    warnings: list[str] = field(default_factory=list)


def peek_ts_file(path: Path) -> FileExtent:
    """Read the first and last block tags; extent = last.start + span.

    The block stride is constant per file (n_scan/n_ch/tag_len from the
    first tag), so the last block sits at (size // stride - 1) * stride.
    """
    warnings: list[str] = []
    size = path.stat().st_size
    with open(path, "rb") as fid:
        head = fid.read(TAG_LEN)
        if len(head) < TAG_LEN:
            raise ValueError(f"{path.name}: file smaller than one tag")
        first = parse_block_tag(head)
        if head[13] != TAG_LEN:
            raise NotImplementedError(
                f"{path.name}: tag length {head[13]} (V5-2000 lineage) is "
                "not supported — only 32-byte MTU-5 tags carry the "
                "per-block time fields"
            )
        if first.n_scan <= 0 or first.n_ch <= 0:
            raise ValueError(
                f"{path.name}: degenerate first tag (n_scan={first.n_scan}, "
                f"n_ch={first.n_ch})"
            )
        stride = TAG_LEN + first.n_scan * first.n_ch * 3
        n_blocks, tail = divmod(size, stride)
        if tail:
            warnings.append(
                f"{path.name}: {tail} trailing byte(s) after {n_blocks} "
                "complete blocks — partial final block ignored"
            )
        if n_blocks == 1:
            last = first
        else:
            fid.seek((n_blocks - 1) * stride)
            last = parse_block_tag(fid.read(TAG_LEN))
            if (last.rate, last.n_scan, last.n_ch) != (
                first.rate,
                first.n_scan,
                first.n_ch,
            ):
                warnings.append(
                    f"{path.name}: last block tag diverges from the first "
                    f"(rate {last.rate} vs {first.rate}, n_scan "
                    f"{last.n_scan} vs {first.n_scan}, n_ch {last.n_ch} "
                    f"vs {first.n_ch}) — extent may be wrong"
                )
    span_s = first.n_scan / first.rate if first.rate else 0.0
    end = last.start + timedelta(seconds=span_s)
    rate = first.rate or 0
    if not rate:
        warnings.append(f"{path.name}: first block tag carries rate 0")
    return FileExtent(
        path=path,
        rate=rate,
        start=first.start,
        end=end,
        n_blocks=n_blocks,
        clock_status_first=first.clock_status,
        clock_status_last=last.clock_status,
        clock_error_s_max=max(abs(first.clock_error_s), abs(last.clock_error_s)),
        saturated=first.saturated or last.saturated,
        warnings=warnings,
    )


@dataclass
class Station:
    """One TBL + its sibling TSn files = one station record."""

    tbl_path: Path
    tags: dict[str, object]
    files: list[FileExtent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def station_id(self) -> str:
        return self.tags.get("SITE", "") or self.tbl_path.stem

    @property
    def survey_id(self) -> str:
        return self.tags.get("SRVY", "") or ""

    @property
    def start(self) -> datetime | None:
        starts = [f.start for f in self.files]
        return min(starts) if starts else None

    @property
    def end(self) -> datetime | None:
        ends = [f.end for f in self.files]
        return max(ends) if ends else None

    def clock_note(self) -> str:
        """Clock quality from the PEEKED tags only (first/last block)."""
        statuses = sorted(
            {f.clock_status_first for f in self.files}
            | {f.clock_status_last for f in self.files}
        )
        max_err = max((f.clock_error_s_max for f in self.files), default=0)
        parts = []
        if statuses and statuses != [0]:
            parts.append(f"clock_status={','.join(map(str, statuses))}")
        if max_err:
            parts.append(f"max|clock_error|={max_err}s")
        if any(f.saturated for f in self.files):
            parts.append("saturated blocks flagged")
        return "; ".join(parts)


def scan_station(tbl_path: Path) -> Station:
    station = Station(tbl_path=tbl_path, tags=read_tbl_dict(tbl_path))
    siblings = sorted(
        p
        for p in tbl_path.parent.iterdir()
        if p.is_file() and p.suffix.lstrip(".").upper() in TS_EXTENSIONS
    )
    if not siblings:
        station.warnings.append("no TSn files next to the TBL")
    for ts_path in siblings:
        try:
            station.files.append(peek_ts_file(ts_path))
        except NotImplementedError as exc:
            station.warnings.append(str(exc))
        except (ValueError, OSError) as exc:
            station.warnings.append(f"{ts_path.name}: unreadable ({exc})")
    tbl_siblings = [
        p
        for p in tbl_path.parent.iterdir()
        if p.is_file() and p.suffix.upper() == ".TBL"
    ]
    if len(tbl_siblings) > 1:
        station.warnings.append(
            f"{len(tbl_siblings)} TBL files share this directory — TSn "
            "ownership is ambiguous; each TBL was matched against all of "
            f"them ({', '.join(p.name for p in sorted(tbl_siblings))})"
        )
    return station


def scan_root(root: Path) -> list[Station]:
    tbls = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.upper() == ".TBL"
    )
    if not tbls:
        print(f"No .TBL files found under {root}", file=sys.stderr)
        return []
    stations = [scan_station(t) for t in tbls]
    stations.sort(
        key=lambda s: (
            s.start or datetime.max.replace(tzinfo=timezone.utc),
            s.station_id,
        )
    )
    return stations


def overlap_seconds(a_end: datetime, b_start: datetime) -> float:
    return (a_end - b_start).total_seconds()


def write_csvs(stations: list[Station], out_dir: Path) -> tuple[Path, Path]:
    stations_csv = out_dir / "phoenix_stations.csv"
    files_csv = out_dir / "phoenix_ts_files.csv"
    with open(stations_csv, "w", newline="") as fid:
        writer = csv.writer(fid)
        writer.writerow(
            [
                "station",
                "survey",
                "serial",
                "folder",
                "tbl",
                "n_ts_files",
                "union_start",
                "union_end",
                "duration_s",
                "tbl_stim_placeholder",
                "tbl_etim_placeholder",
                "clock_note",
                "warnings",
            ]
        )
        for st in stations:
            union_start = st.start.isoformat() if st.start else ""
            union_end = st.end.isoformat() if st.end else ""
            duration = (
                (st.end - st.start).total_seconds() if (st.start and st.end) else ""
            )
            writer.writerow(
                [
                    st.station_id,
                    st.survey_id,
                    st.tags.get("SNUM", ""),
                    st.tbl_path.parent,
                    st.tbl_path,
                    len(st.files),
                    union_start,
                    union_end,
                    duration,
                    st.tags.get("STIM", ""),
                    st.tags.get("ETIM", ""),
                    st.clock_note(),
                    " | ".join(st.warnings),
                ]
            )
    with open(files_csv, "w", newline="") as fid:
        writer = csv.writer(fid)
        writer.writerow(
            [
                "station",
                "folder",
                "file",
                "rate_hz",
                "start",
                "end",
                "duration_s",
                "n_blocks",
                "clock_status_first",
                "clock_status_last",
                "max_abs_clock_error_s",
                "saturated",
                "warnings",
            ]
        )
        for st in stations:
            for f in st.files:
                writer.writerow(
                    [
                        st.station_id,
                        st.tbl_path.parent,
                        f.path,
                        f.rate,
                        f.start.isoformat(),
                        f.end.isoformat(),
                        (f.end - f.start).total_seconds(),
                        f.n_blocks,
                        f.clock_status_first,
                        f.clock_status_last,
                        f.clock_error_s_max,
                        f.saturated,
                        " | ".join(f.warnings),
                    ]
                )
    return stations_csv, files_csv


def find_common_groups(
    stations: list[Station], min_overlap_s: float
) -> list[tuple[list[Station], timedelta]]:
    """Groups of >= 3 stations whose extents share a common window of at
    least ``min_overlap_s``.

    Intervals on a line have the Helly property — a group overlaps at all
    iff max(starts) <= min(ends) — so every candidate common window is
    bounded by some station's start. For EACH station taken as the latest
    starter, the widest group is reported: the members are exactly the
    stations that started no later and run past the anchor's start plus
    the threshold. O(N^2), no subset enumeration. One station set can be
    reported twice with different windows when a subset has a LONGER
    common window (e.g. a short 4th deployment shrinking the window) —
    both are the answer to "who overlaps from this point onwards", so
    both are kept; identical member sets are deduplicated.
    """
    usable = [s for s in stations if s.start is not None]
    usable.sort(key=lambda s: s.start)
    seen: set[frozenset[int]] = set()
    groups: list[tuple[list[Station], timedelta]] = []
    for anchor in usable:
        anchor_start = anchor.start
        members = [
            s
            for s in usable
            if s.start <= anchor_start
            and (s.end - anchor_start).total_seconds() >= min_overlap_s
        ]
        if len(members) < 3:
            continue
        common_end = min(s.end for s in members)
        duration = common_end - anchor_start
        if duration.total_seconds() < min_overlap_s:
            continue
        key = frozenset(id(s) for s in members)
        if key in seen:
            continue
        seen.add(key)
        groups.append((members, duration))
    groups.sort(key=lambda g: (-len(g[0]), -g[1].total_seconds()))
    return groups


def print_report(stations: list[Station], min_overlap_s: float) -> None:
    print(f"\nStations ({len(stations)}) — union extents, sorted by start:")
    for st in stations:
        if st.start is None:
            print(f"  {st.station_id:<16} (no readable TSn extents)")
            continue
        dur_h = (st.end - st.start).total_seconds() / 3600.0
        rates = ", ".join(f"{f.rate} Hz" for f in st.files) or "-"
        print(
            f"  {st.station_id:<16} {st.start.isoformat()}  ->  "
            f"{st.end.isoformat()}  ({dur_h:.2f} h; {rates})"
        )
        for warning in st.warnings:
            print(f"    ! {warning}")

    print(
        f"\nRemote-reference candidate pairs (union extents, overlap "
        f">= {min_overlap_s:g} s):"
    )
    pairs = []
    for i, a in enumerate(stations):
        for b in stations[i + 1 :]:
            if a.start is None or b.start is None:
                continue
            seconds = min(a.end, b.end) - max(a.start, b.start)
            if seconds.total_seconds() >= min_overlap_s:
                pairs.append((a, b, seconds))
    pairs.sort(key=lambda p: p[2], reverse=True)
    if not pairs:
        print("  (none)")
    for a, b, seconds in pairs:
        start = max(a.start, b.start).isoformat()
        end = min(a.end, b.end).isoformat()
        hours = seconds.total_seconds() / 3600.0
        print(
            f"  {a.station_id:<16} <-> {b.station_id:<16} "
            f"overlap {hours:8.2f} h   [{start} .. {end}]"
        )
    n_shown = len(pairs)
    total = sum(1 for i, a in enumerate(stations) for b in stations[i + 1 :])
    if n_shown < total:
        print(f"  ({total - n_shown} pair(s) below the threshold omitted)")

    groups = find_common_groups(stations, min_overlap_s)
    print(
        f"\nCommon-intersection groups (>= 3 stations sharing a window "
        f">= {min_overlap_s:g} s):"
    )
    if not groups:
        print("  (none — no triplet shares a window that long)")
    for members, duration in groups:
        names = " + ".join(s.station_id for s in members)
        start = max(s.start for s in members).isoformat()
        end = min(s.end for s in members).isoformat()
        hours = duration.total_seconds() / 3600.0
        print(f"  {names}  —  {hours:.2f} h common window  [{start} .. {end}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "root", type=Path, help="folder to scan recursively for .TBL files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="where the CSV files are written (default: current directory)",
    )
    parser.add_argument(
        "--min-overlap-seconds",
        type=float,
        default=3600.0,
        help="report only overlaps of at least this length "
        "(default 3600 = one hour; smaller overlaps are rarely useful "
        "for remote referencing)",
    )
    args = parser.parse_args(argv)

    stations = scan_root(args.root)
    if not stations:
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stations_csv, files_csv = write_csvs(stations, args.output_dir)
    print(f"Wrote {stations_csv} and {files_csv}")
    print_report(stations, args.min_overlap_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
