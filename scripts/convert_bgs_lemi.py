#!/usr/bin/env python3
"""Ad-hoc conversion of a BGS long-period MT `.xyz` file to LEMI-424 `.TXT`.

The BGS Scotland `.xyz` files contain only: time, bx, by, bz, e1, e2. This
writes them into the standard LEMI-424 24-column layout (see mt_io lemi424)
with dummy values in the missing columns so the mth5 LEMI reader can parse it.

**Recording gaps**: mt_io's LEMI reader builds the time axis from the row
count (first timestamp + sample rate), NOT from the per-row timestamps. A
missing-data gap in the source therefore shifts every later sample earlier
than its true time and destroys inter-station alignment (remote reference).
This converter **fills gaps with linearly interpolated rows** so one row ==
one true second (see docs/DESIGN.md §3 "Align Runs" / the BGS note).

Usage:
    python scripts/convert_bgs_lemi.py <in.xyz> <out.txt> [--days N] [--start "YYYY-MM-DD HH:MM:SS"] [--lat 56.72884] [--lon -2.85449067] [--elev 120]

``--start`` skips everything before the given absolute time, so two sites
can be converted over the *same* window (required for remote-reference
alignment — see docs/DESIGN.md §3 "Align Runs").
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

COLS = [
    "year", "month", "day", "hour", "minute", "second",
    "bx", "by", "bz", "temperature_e", "temperature_h",
    "e1", "e2", "e3", "e4", "battery", "elevation",
    "latitude", "lat_hemisphere", "longitude", "lon_hemisphere",
    "n_satellites", "gps_fix", "time_diff",
]


def _lemi_position(deg: float) -> tuple[str, str]:
    """Decimal degrees -> (LEMI DDMM.mmmm value string, hemisphere letter)."""
    hemisphere = "N" if deg >= 0 else "S"
    if deg < 0:
        deg = -deg
    degrees = int(deg)
    minutes = (deg - degrees) * 60.0
    value = degrees * 100 + minutes
    return f"{value:.4f}", hemisphere


class _Writer:
    """Writes LEMI-424 rows; fills time gaps with interpolated values."""

    def __init__(self, fout, lat_v, lat_h, lon_v, lon_h, elev: float) -> None:
        self.fout = fout
        self.lat_v, self.lat_h = lat_v, lat_h
        self.lon_v, self.lon_h = lon_v, lon_h
        self.elev = elev
        self.prev_t: datetime | None = None
        self.prev_vals: tuple[float, ...] | None = None
        self.written = 0
        self.filled = 0

    def write(self, t: datetime, bx: float, by: float, bz: float,
              e1: float, e2: float) -> None:
        cur = (bx, by, bz, e1, e2)
        if self.prev_t is not None and self.prev_vals is not None:
            gap = round((t - self.prev_t).total_seconds())
            if gap > 1:
                for k in range(1, gap):
                    frac = k / gap
                    vals = tuple(
                        p + frac * (c - p)
                        for p, c in zip(self.prev_vals, cur)
                    )
                    self._row(self.prev_t + timedelta(seconds=k), vals)
                    self.filled += 1
        self._row(t, cur)
        self.prev_t, self.prev_vals = t, cur

    def _row(self, t: datetime, vals: tuple[float, ...]) -> None:
        bx, by, bz, e1, e2 = vals
        row = (
            f"{t.year} {t.month:02d} {t.day:02d} {t.hour:02d} {t.minute:02d} {t.second:02d} "
            f"{bx:.3f} {by:.3f} {bz:.3f} "          # bx by bz
            f"25.0 25.0 "                            # temperature_e, temperature_h (dummy)
            f"{e1:.3f} {e2:.3f} "                    # e1 e2 (real)
            f"0.0 0.0 "                              # e3 e4 (dummy)
            f"12.97 "                                # battery (dummy)
            f"{self.elev:.1f} "                      # elevation
            f"{self.lat_v} {self.lat_h} {self.lon_v} {self.lon_h} "  # coordinates
            f"10 2 0"                                # n_satellites gps_fix time_diff
        )
        self.fout.write(row + "\n")
        self.written += 1


def convert(in_xyz: str, out_txt: str, days: int, lat: float, lon: float,
            elev: float, start: str | None = None) -> None:
    n = int(days * 86400)
    start_dt = (
        datetime.strptime(start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        if start else None
    )
    lat_v, lat_h = _lemi_position(lat)
    if lon < 0:
        lon_v, lon_h = _lemi_position(lon)  # returns 'S' hemisphere, override for lon
        lon_h = "W"
        lon_v = _lemi_position(abs(lon))[0]
    else:
        lon_v, lon_h = _lemi_position(lon)
        lon_h = "E"

    skipped = 0
    with open(in_xyz) as fin, open(out_txt, "w") as fout:
        w = _Writer(fout, lat_v, lat_h, lon_v, lon_h, elev)
        for line in fin:
            if w.written >= n:
                break
            parts = line.split()
            if len(parts) < 7:
                continue
            t = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            if start_dt is not None and t < start_dt:
                skipped += 1
                continue
            bx, by, bz, e1, e2 = (float(parts[i]) for i in (2, 3, 4, 5, 6))
            w.write(t, bx, by, bz, e1, e2)

    print(
        f"wrote {w.written} samples ({w.written/3600:.1f} h, "
        f"{w.filled} gap-filled), skipped {skipped} -> {out_txt}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_xyz")
    ap.add_argument("out_txt")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--start", default=None,
                    help='skip until "YYYY-MM-DD HH:MM:SS" (inclusive)')
    ap.add_argument("--lat", type=float, default=56.72884)
    ap.add_argument("--lon", type=float, default=-2.85449067)
    ap.add_argument("--elev", type=float, default=120)
    args = ap.parse_args()
    convert(args.in_xyz, args.out_txt, args.days, args.lat, args.lon, args.elev,
            start=args.start)
