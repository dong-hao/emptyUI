"""Build MTH5 archives for the SDG31 USArray verification (history §16.3).

Downloads 1 Hz MT time series from the EarthScope FDSN dataselect service
(IRIS alias, redirected to service.earthscope.org) for the local station
4P.SDG31 and the remote reference 4P.SDH28, following the aurora
"operate-aurora" recipe (MakeMTH5.from_fdsn_client):

- network 4P (EarthScope MT array), channels LQE/LQN (100 m dipoles,
  40960 counts/mV/km) and LFE/LFN/LFZ (NIMS magnetometer, 100.9048
  counts/nT), sample rate 1 Hz.
- The channels are stored in the archive under the standard MT names
  ex/ey/hx/hy/hz (x ~ geographic north, y ~ east — fdsn_tools mapping,
  matching the official EDI layout) in RAW COUNTS; the full instrument
  response is archived with the run (response_level="response"), so a
  Calibrate step is mandatory before processing.
- Time windows are the station-inventory envelopes:
    SDG31: 2017-10-16T17:01:13 -> 2017-10-26T14:59:54 (runs a+b)
    SDH28: 2017-10-12T19:43:14 -> 2017-10-31T15:55:16

The MTH5 files land in examples/data/nims-usarray-test/ (gitignored;
regenerate with this script). The official EMTF transfer function
USArray.SDG31.2017.edi lives next to them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from mth5.clients.fdsn import FDSN

OUT_DIR = Path(__file__).resolve().parent

STATIONS = {
    "SDG31": ("2017-10-16T17:01:13", "2017-10-26T14:59:54"),
    "SDH28": ("2017-10-12T19:43:14", "2017-10-31T15:55:16"),
}

CHANNELS = ["LQE", "LQN", "LFE", "LFN", "LFZ"]


def build(station: str, start: str, end: str) -> Path:
    client = FDSN()
    rows = [["4P", station, "", ch, start, end] for ch in CHANNELS]
    df = pd.DataFrame(rows, columns=client.request_columns)
    print(f"downloading 4P.{station} {start} -> {end} ({len(CHANNELS)} channels)")
    path = client.make_mth5_from_fdsn_client(df, path=OUT_DIR, client="IRIS")
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    only = sys.argv[1:] or list(STATIONS)
    for station in only:
        start, end = STATIONS[station]
        build(station, start, end)
