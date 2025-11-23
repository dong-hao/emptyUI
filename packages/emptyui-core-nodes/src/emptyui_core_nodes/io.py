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
    outputs = (Output("data", "ts.run", description="Loaded time series"),)
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

        with open(p, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter.strip() or ",")
            header = [h.strip() for h in next(reader)]

        data = np.genfromtxt(p, delimiter=delimiter.strip() or ",", skip_header=1)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if header and len(header) != data.shape[1]:
            header = [f"col{i}" for i in range(data.shape[1])]

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

        if t_idx is not None:
            t = data[:, t_idx]
            dt = np.diff(t)
            if dt.size and np.median(dt) > 0:
                sample_rate = float(1.0 / np.median(dt))
            start = float(t[0])
            channel_cols = [i for i in range(data.shape[1]) if i != t_idx]
        else:
            start = 0.0
            channel_cols = list(range(data.shape[1]))

        if sample_rate is None or sample_rate <= 0:
            raise ValueError(
                "no time column and no positive sample_rate given — cannot "
                "build a time axis"
            )

        channels = {
            header[i]: np.ascontiguousarray(data[:, i], dtype=np.float64)
            for i in channel_cols
        }
        ctx.progress(1.0, f"loaded {len(channels)} channels, {len(channels[next(iter(channels))])} samples")
        return {"data": TimeSeries(channels=channels, sample_rate=float(sample_rate), start=start)}


def register(registry) -> None:
    registry.add(LoadCsv)
