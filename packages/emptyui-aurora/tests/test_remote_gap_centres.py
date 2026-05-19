"""M10: the remote-gap safety check must test window STARTS against the gap
intervals. The fc labels are window centres, half a window after the start
(issues.md M10; the centre convention itself is L11).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from emptyui_aurora._processing import _warn_remote_gaps, compute_fc_levels
from test_remote_reference import _Run

N = 256
OVERLAP = 0.5
FS = 1.0


def _fc(start, n=4096, seed=0):
    """Level-0 FC of a synthetic 1 Hz record starting at ``start`` (clock_zero=epoch)."""
    rng = np.random.default_rng(seed)
    chans = {c: rng.standard_normal(n) for c in ("bx", "by", "e1", "e2")}
    run = _Run(chans, FS, start_datetime=start)
    return compute_fc_levels(
        run, ["bx", "by"], ["e1", "e2"],
        num_decimations=0, num_samples_window=N, overlap=OVERLAP,
        window_type="boxcar", clock_zero="epoch")[0]


def _gapped_fc(fs, gaps, n_windows=12):
    """An fc at rate ``fs`` (window N) with centre labels on an absolute grid
    and the Merge Runs gap provenance attached. Returns ``(fc, t0)``."""
    centre = N / (2 * fs)
    step = (N - int(N * OVERLAP)) / fs
    t0 = pd.Timestamp("2024-01-01T00:00:00")
    times = t0 + pd.to_timedelta(centre + step * np.arange(n_windows), unit="s")
    fc = _fc("2024-01-01T00:00:00", n=4096).isel(time=slice(0, n_windows))
    fc = fc.assign_coords(time=times.values)
    fc.attrs = dict(fc.attrs, sample_rate=fs, num_samples_window=N,
                    merged_gap_intervals=gaps)
    return fc, t0


def _gap(t_from, t_to):
    return [[t_from.isoformat(), t_to.isoformat()]]


def test_gap_covering_only_the_first_window_start_warns(caplog):
    """The first window starts at t0 and its centre is t0 + 128 s. A gap
    covering t0 but no centre must warn (the old check compared centres and
    missed it)."""
    t0 = pd.Timestamp("2024-01-01T00:00:00")
    fc, _ = _gapped_fc(FS, _gap(t0 - pd.Timedelta(seconds=60),
                                t0 + pd.Timedelta(seconds=60)))
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        _warn_remote_gaps(fc, fc)
    assert any("interpolated gap" in r.message for r in caplog.records)


def test_gap_covering_only_a_deep_window_start_warns(caplog):
    """At a decimated level (1/16 Hz, same 256-sample window) the centre sits
    2048 s after the start, so a gap covering only the start is the case the
    old check missed."""
    t0 = pd.Timestamp("2024-01-01T00:00:00")
    fc, _ = _gapped_fc(FS / 16.0, _gap(t0 - pd.Timedelta(seconds=1000),
                                       t0 + pd.Timedelta(seconds=1000)))
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        _warn_remote_gaps(fc, fc)
    assert any("interpolated gap" in r.message for r in caplog.records)


def test_gap_between_window_starts_stays_silent(caplog):
    """Guard against false positives: a gap that contains neither a window
    start (one every 128 s here) nor a centre stays silent."""
    t0 = pd.Timestamp("2024-01-01T00:00:00")
    fc, _ = _gapped_fc(FS, _gap(t0 + pd.Timedelta(seconds=10),
                                t0 + pd.Timedelta(seconds=20)))
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        _warn_remote_gaps(fc, fc)
    assert not any("interpolated gap" in r.message for r in caplog.records)


def test_gap_covering_a_window_centre_still_warns(caplog):
    """Regression: a gap covering a window centre (here t0 + 512 s, which is
    also the next window's start) still warns."""
    t0 = pd.Timestamp("2024-01-01T00:00:00")
    fc, _ = _gapped_fc(FS, _gap(t0 + pd.Timedelta(seconds=500),
                                t0 + pd.Timedelta(seconds=520)))
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        _warn_remote_gaps(fc, fc)
    assert any("interpolated gap" in r.message for r in caplog.records)
