"""Time-selector window resolution (DESIGN §23.4): ISO/relative bounds,
clamped into the record (warn, never reject), slot > 1 s, legacy duration."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from emptyui_core_nodes.plots import _time_window
from emptyui_core_nodes.types import TimeSeries

SR = 10.0
N = 36000  # one hour


def _abs_data() -> TimeSeries:
    return TimeSeries(channels={"ex": np.zeros(N)}, sample_rate=SR,
                      start_datetime="2023-06-01T12:00:00")


def test_full_record_when_bounds_empty():
    n0, n1 = _time_window(_abs_data(), "", "", None)
    assert (n0, n1) == (0, N)


def test_iso_bounds():
    n0, n1 = _time_window(_abs_data(), "2023-06-01 12:01:00",
                          "2023-06-01T12:03:00", None)
    assert (n0, n1) == (600, 1800)


def test_relative_seconds_bounds():
    n0, n1 = _time_window(_abs_data(), "10", "30.5", None)
    assert (n0, n1) == (100, 305)


def test_relative_only_record_rejects_iso():
    data = TimeSeries(channels={"ex": np.zeros(100)}, sample_rate=SR)
    with pytest.raises(ValueError, match="no absolute start"):
        _time_window(data, "2023-06-01 12:00:00", "", None)
    assert _time_window(data, "5", "9", None) == (50, 90)


def test_out_of_range_clamped_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        n0, n1 = _time_window(_abs_data(), "2022-01-01", "1800", None)
    assert (n0, n1) == (0, 18000)  # start clamped to the record start
    assert any("clamped" in r.message for r in caplog.records)


def test_placeholder_epoch_degrades_to_relative():
    data = TimeSeries(channels={"ex": np.zeros(100)}, sample_rate=SR,
                      start_datetime="1980-01-01T00:00:00")
    # ISO values are ignored (degraded, with warning) — seconds work
    with pytest.raises(ValueError, match="no absolute start"):
        _time_window(data, "2023-06-01 12:00:00", "", None)
    assert _time_window(data, "2", "5", None) == (20, 50)


def test_legacy_seconds_caps_slot_from_start():
    n0, n1 = _time_window(_abs_data(), "", "1800", 60)
    assert (n0, n1) == (0, 600)


def test_minimum_slot_guard():
    for start, end in (("0", "0.5"), ("0", "1.0"), ("9999", ""), ("", "0.5")):
        with pytest.raises(ValueError, match="minimum slot"):
            _time_window(_abs_data(), start, end, None)


def test_malformed_bound_is_an_error():
    with pytest.raises(ValueError, match="start='junk'"):
        _time_window(_abs_data(), "junk", "", None)


def test_iso_bounds_are_record_relative():
    """ISO datetimes measure from the record start even when the series
    itself starts at an offset (data.start > 0)."""
    data = TimeSeries(channels={"ex": np.zeros(1000)}, sample_rate=SR,
                      start=100.0, start_datetime="2023-06-01T12:00:00")
    # record time 150 s == series time 50 s
    n0, _ = _time_window(data, "2023-06-01 12:02:30", "2023-06-01 12:04:00", None)
    assert n0 == 500  # (150 - 100) s * 10 Hz
