"""M9: clock-zero window labels are the actual sample times of the data, and the
remote-reference pairing accepts one sample of label difference.

Labels are window CENTRES (scipy segment centre = nperseg / 2 / fs past the
window start); the offset is part of the convention and is checked here
(docs/issues.md M9; the convention itself is L11).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from emptyui_aurora._processing import (
    _clock_zero_head_skip,
    _pair_remote_fc,
    compute_fc_levels,
)
from test_remote_reference import _Run

N = 256
OVERLAP = 0.5
FS = 1.0
CENTRE_S = N / (2 * FS)
EPOCH = pd.Timestamp("1970-01-01T00:00:00")


def _fc(start, n=4096, seed=0):
    """Level-0 FC of a synthetic 1 Hz record starting at ``start`` (clock_zero=epoch)."""
    rng = np.random.default_rng(seed)
    chans = {c: rng.standard_normal(n) for c in ("bx", "by", "e1", "e2")}
    run = _Run(chans, FS, start_datetime=start)
    return compute_fc_levels(
        run, ["bx", "by"], ["e1", "e2"],
        num_decimations=0, num_samples_window=N, overlap=OVERLAP,
        window_type="boxcar", clock_zero="epoch")[0]


def test_clock_zero_label_sits_on_the_record_sample_grid():
    """The first window label is a whole number of samples after the record
    start (plus the window-centre offset). A fractional-second record start
    must not move the labels off the data's sample grid."""
    start = pd.Timestamp("2024-01-01T00:00:00.4")
    fc = _fc("2024-01-01T00:00:00.4")
    label0 = pd.Timestamp(fc["time"].values[0])
    offset_samples = (label0 - start).total_seconds() * FS - CENTRE_S * FS
    assert abs(offset_samples - round(offset_samples)) < 1e-6


def test_clock_zero_label_unchanged_for_on_grid_start():
    """Regression: a record start on the sample grid keeps the grid label
    k0*step (already exact before the fix)."""
    start = pd.Timestamp("2024-01-01T00:00:00")
    dt = (start - EPOCH).total_seconds()
    step = (N - int(N * OVERLAP)) / FS
    k0 = int(np.ceil(dt / step))
    fc = _fc("2024-01-01T00:00:00")
    label0 = pd.Timestamp(fc["time"].values[0])
    assert label0 == EPOCH + pd.Timedelta(seconds=k0 * step + CENTRE_S)


def test_clock_zero_head_skip_offset_is_the_actual_sample_offset():
    """The returned offset must equal the rounded sample skip converted to
    seconds, not the unrounded grid distance."""
    start = pd.Timestamp("2024-01-01T00:00:00.4")
    n_skip, off = _clock_zero_head_skip(start, FS, N, int(N * OVERLAP), "epoch")
    assert off == pytest.approx(n_skip / FS)


def test_pairing_tolerates_one_sample_of_sub_sample_offset():
    """Two stations whose records differ by a fraction of a sample get labels
    up to one sample apart. Every remote window must still pair 1:1 with a
    local window, within one sample of label."""
    local = _fc("2024-01-01T00:00:00.0", n=4096 + 512, seed=1)
    remote = _fc("2024-01-01T00:00:00.4", n=4096, seed=2)
    lo, ro = _pair_remote_fc(local, remote)
    assert ro.sizes["time"] == remote.sizes["time"]
    assert lo.sizes["time"] == ro.sizes["time"]
    dt_s = np.abs((lo["time"].values - ro["time"].values)
                  / np.timedelta64(1, "s"))
    assert dt_s.max() <= 1.0 / FS + 1e-6


def test_pair_tolerance_floor_is_exactly_one_sample():
    """The pairing floor is one sample (1 s at 1 Hz): a 1 s label shift pairs,
    a 2 s shift (no window within tolerance) is refused."""
    local = _fc("2024-01-01T00:00:00", n=4096)
    shifted_1s = local.assign_coords(time=local["time"].values + np.timedelta64(1, "s"))
    lo, ro = _pair_remote_fc(local, shifted_1s)
    assert lo.sizes["time"] == ro.sizes["time"] == local.sizes["time"]

    shifted_2s = local.assign_coords(time=local["time"].values + np.timedelta64(2, "s"))
    with pytest.raises(ValueError, match="no remote-reference windows match"):
        _pair_remote_fc(local, shifted_2s)
