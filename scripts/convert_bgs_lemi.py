#!/usr/bin/env python3
"""Ad-hoc conversion of a BGS long-period MT `.xyz` file to LEMI-424 `.TXT`.

The BGS Scotland `.xyz` files contain only: time, bx, by, bz, e1, e2. This
writes them into the standard LEMI-424 24-column layout (see mt_io lemi424)
with dummy values in the missing columns so the mth5 LEMI reader can parse it.

Usage:
    python scripts/convert_bgs_lemi.py <in.xyz> <out.txt> [--days N] [--lat 56.72884] [--lon -2.85449067] [--elev 120]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

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


def convert(in_xyz: str, out_txt: str, days: int, lat: float, lon: float, elev: float) -> None:
    n = int(days * 86400)
    lat_v, lat_h = _lemi_position(lat)
    if lon < 0:
        lon_v, lon_h = _lemi_position(lon)  # returns 'S' hemisphere, override for lon
        lon_h = "W"
        lon_v = _lemi_position(abs(lon))[0]
    else:
        lon_v, lon_h = _lemi_position(lon)
        lon_h = "E"

    written = 0
    with open(in_xyz) as fin, open(out_txt, "w") as fout:
        for line in fin:
            if written >= n:
                break
            parts = line.split()
            if len(parts) < 7:
                continue
            t = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S")
            bx, by, bz, e1, e2 = (float(parts[i]) for i in (2, 3, 4, 5, 6))
            row = (
                f"{t.year} {t.month:02d} {t.day:02d} {t.hour:02d} {t.minute:02d} {t.second:02d} "
                f"{bx:.3f} {by:.3f} {bz:.3f} "          # bx by bz
                f"25.0 25.0 "                            # temperature_e, temperature_h (dummy)
                f"{e1:.3f} {e2:.3f} "                    # e1 e2 (real)
                f"0.0 0.0 "                              # e3 e4 (dummy)
                f"12.97 "                                # battery (dummy)
                f"{elev:.1f} "                           # elevation
                f"{lat_v} {lat_h} {lon_v} {lon_h} "      # coordinates
                f"10 2 0"                                # n_satellites gps_fix time_diff
            )
            fout.write(row + "\n")
            written += 1

    print(f"wrote {written} samples ({written/3600:.1f} h) -> {out_txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_xyz")
    ap.add_argument("out_txt")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--lat", type=float, default=56.72884)
    ap.add_argument("--lon", type=float, default=-2.85449067)
    ap.add_argument("--elev", type=float, default=120.0)
    args = ap.parse_args()
    convert(args.in_xyz, args.out_txt, args.days, args.lat, args.lon, args.elev)
