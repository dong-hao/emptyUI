"""M7: the inter-station coherence diagnostic must run at any station rate and
must not fail silently. Before the fix it returned without a message above
64 Hz (the 4096-sample segments are coarser than the 1/64 Hz band edge), and
every skip was silent (issues.md M7).
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes.types import TimeSeries
from emptyui_mth5.runs import _warn_if_uncorrelated

DURATION_S = 6000.0


def _ctx(tmp_path):
    return RunContext(workspace=tmp_path, store=DataStore())


def _stations(sr, *, coherent):
    """Two aligned stations: one common 1 Hz white series held at the station
    rate (a stand-in for the long-period field), or independent series."""
    rng = np.random.default_rng(7)
    n_1hz = int(DURATION_S)
    rep = round(sr)
    common = np.repeat(rng.standard_normal(n_1hz), rep)
    if coherent:
        other = common + 0.01 * rng.standard_normal(common.size)
    else:
        other = np.repeat(rng.standard_normal(n_1hz), rep)
    local = TimeSeries(channels={"hx": common, "hy": common[::-1].copy()},
                       sample_rate=sr)
    remote = TimeSeries(channels={"hx": other, "hy": other[::-1].copy()},
                        sample_rate=sr)
    return local, remote, common.size


@pytest.mark.parametrize("sr", [1.0, 8.0, 150.0])
def test_coherent_stations_report_their_coherence(tmp_path, caplog, sr):
    local, remote, n = _stations(sr, coherent=True)
    with caplog.at_level(logging.INFO, logger="emptyui"):
        _warn_if_uncorrelated(_ctx(tmp_path), local, remote, sr, n)
    msgs = [r.getMessage() for r in caplog.records]
    match = [re.search(r"station coherence \(hx, 64-1024 s\): ([0-9.]+)", m)
             for m in msgs]
    values = [float(m.group(1)) for m in match if m]
    assert values, f"no coherence report at {sr} Hz: {msgs}"
    assert values[0] > 0.9


def test_weakly_coherent_stations_warn_at_150_hz(tmp_path, caplog):
    local, remote, n = _stations(150.0, coherent=False)
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        _warn_if_uncorrelated(_ctx(tmp_path), local, remote, 150.0, n)
    assert any("weakly coherent" in r.getMessage() for r in caplog.records)


def test_short_overlap_skip_is_logged(tmp_path, caplog):
    sr = 150.0
    local, remote, _ = _stations(sr, coherent=True)
    short = int(600 * sr)  # 10 min: shorter than one longest band period
    with caplog.at_level(logging.INFO, logger="emptyui"):
        _warn_if_uncorrelated(_ctx(tmp_path), local, remote, sr, short)
    assert any("coherence check skipped" in r.getMessage()
               and "overlap" in r.getMessage() for r in caplog.records)


def test_missing_magnetic_channels_skip_is_logged(tmp_path, caplog):
    sr = 8.0
    n = int(DURATION_S * sr)
    electric = TimeSeries(channels={"ex": np.zeros(n)}, sample_rate=sr)
    remote = TimeSeries(channels={"ex": np.zeros(n)}, sample_rate=sr)
    with caplog.at_level(logging.INFO, logger="emptyui"):
        _warn_if_uncorrelated(_ctx(tmp_path), electric, remote, sr, n)
    assert any("coherence check skipped" in r.getMessage()
               and "magnetic" in r.getMessage() for r in caplog.records)
