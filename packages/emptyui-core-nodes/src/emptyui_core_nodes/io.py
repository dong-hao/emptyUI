"""Data I/O nodes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from emptyui import Node, Output, Param

from .types import TimeSeries


class LoadCsv(Node):
    type = "core.io.load_csv"
    title = "Load CSV"
    description = (
        "Load a multi-channel CSV time series. Auto-detects channel columns "
        "(Ex, Ey, Hx, Hy, ...); an optional time column is used to derive the "
        "sample rate."
    )
    category = "Data I/O"
    inputs = ()
    outputs = (Output("data", "ts.run", label="time series", description="Loaded time series"),)
    params = (
        Param("path", "file", label="CSV file path"),
        Param("sample_rate", "number", label="Sample rate (Hz)", default=150.0,
              min=0.000001, optional=True,
              description="Used when the file has no time column."),
        Param("time_column", "string", label="Time column name", default="",
              optional=True,
              description="Column holding time values; empty = auto-detect."),
        Param("delimiter", "string", label="Delimiter", default=","),
    )

    def execute(self, ctx, path: str, sample_rate: float = 150.0,
                time_column: str = "", delimiter: str = ",") -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        delim = delimiter.strip() or ","

        with open(p, newline="") as f:
            reader = csv.reader(f, delimiter=delim)
            header = [h.strip() for h in next(reader)]
            first_row = next(reader, None)

        # locate time column if requested/detected
        t_idx = None
        if time_column:
            matches = [i for i, h in enumerate(header) if h.lower() == time_column.lower()]
            if not matches:
                raise ValueError(f"time column '{time_column}' not in {header}")
            t_idx = matches[0]
        elif any("time" in h.lower() or h.lower() == "t" for h in header):
            t_idx = next(i for i, h in enumerate(header)
                         if "time" in h.lower() or h.lower() == "t")

        # datetime-ish time column (e.g. ISO strings): genfromtxt would turn
        # these into NaN — convert to epoch seconds via pandas instead, and
        # record the absolute start for downstream FC persistence.
        t_is_datetime = False
        if t_idx is not None and first_row is not None and t_idx < len(first_row):
            try:
                float(first_row[t_idx])
            except ValueError:
                t_is_datetime = True
        if t_idx is not None and t_is_datetime:
            import pandas as pd

            def _epoch(s) -> float:
                if isinstance(s, bytes):
                    s = s.decode("utf-8", "replace")
                return pd.Timestamp(str(s).strip()).timestamp()

            data = np.genfromtxt(p, delimiter=delim, skip_header=1,
                                 converters={t_idx: _epoch})
        else:
            data = np.genfromtxt(p, delimiter=delim, skip_header=1)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if header and len(header) != data.shape[1]:
            header = [f"col{i}" for i in range(data.shape[1])]

        start = 0.0
        start_datetime = ""
        channel_cols = list(range(data.shape[1]))
        if t_idx is not None:
            t = data[:, t_idx]
            dt = np.diff(t)
            med = float(np.median(dt)) if dt.size else 0.0
            if not np.isfinite(med) or med <= 0:
                raise ValueError(
                    f"time column '{header[t_idx]}' is not monotonically "
                    "increasing with finite steps — cannot derive a sample rate"
                )
            sample_rate = float(1.0 / med)
            if t_is_datetime:
                import pandas as pd

                start_datetime = pd.Timestamp(float(t[0]), unit="s").isoformat()
            else:
                start = float(t[0])
            channel_cols = [i for i in range(data.shape[1]) if i != t_idx]

        if sample_rate is None or sample_rate <= 0:
            raise ValueError(
                "no time column and no positive sample_rate given — cannot "
                "build a time axis"
            )
        if not channel_cols:
            raise ValueError(
                f"no channel columns found in {p.name} (only a time column?)"
            )

        channels = {
            header[i]: np.ascontiguousarray(data[:, i], dtype=np.float64)
            for i in channel_cols
        }
        ctx.progress(1.0, f"loaded {len(channels)} channels, {len(channels[next(iter(channels))])} samples")
        return {"data": TimeSeries(channels=channels, sample_rate=float(sample_rate),
                                   start=start, start_datetime=start_datetime)}


def register(registry) -> None:
    registry.add(LoadCsv)
