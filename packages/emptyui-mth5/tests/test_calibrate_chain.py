"""M8: a time-domain Calibrate must not run on a payload whose instrument response
has already been removed (an upstream Calibrate). The second pass divides the
same response out again: a silent double correction (issues.md M8).

The guard test at the end pins the route that must keep working: Calibrate
followed by Compute FC, which honours the applied flags.
"""

from __future__ import annotations

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import compute_fc_levels
from emptyui_core_nodes.types import TimeSeries
from emptyui_mth5.calibrate import Calibrate
from mt_metadata.timeseries.filters import CoefficientFilter

SR = 8.0
N = 4096
CHANNELS = ("ex", "ey", "hx", "hy")


def _ctx(tmp_path):
    return RunContext(workspace=tmp_path, store=DataStore())


def _series(calibration=None):
    rng = np.random.default_rng(8)
    channels = {c: rng.standard_normal(N) for c in CHANNELS}
    return TimeSeries(channels=channels, sample_rate=SR, calibration=calibration or {})


def _calibrated_once(tmp_path):
    return Calibrate().execute(_ctx(tmp_path), _series(),
                               calibration=CoefficientFilter(gain=2.0))["data"]


def test_second_time_domain_calibrate_is_refused(tmp_path):
    once = _calibrated_once(tmp_path)
    with pytest.raises(ValueError, match="already"):
        Calibrate().execute(_ctx(tmp_path), once)


def test_explicit_recalibration_of_a_calibrated_payload_is_refused(tmp_path):
    once = _calibrated_once(tmp_path)
    with pytest.raises(ValueError, match="already"):
        Calibrate().execute(_ctx(tmp_path), once,
                            calibration=CoefficientFilter(gain=5.0))


def test_calibrate_then_compute_fc_matches_the_fc_route(tmp_path):
    """Guard: the time-domain route followed by Compute FC must not divide the
    response a second time. The reference divides the same response in the
    frequency domain (the FC route); the two medians must agree."""
    once = _calibrated_once(tmp_path)
    direct = _series(calibration={
        c: [(CoefficientFilter(gain=2.0), True)] for c in CHANNELS})

    fc_time = compute_fc_levels(once, ["hx", "hy"], ["ex", "ey"],
                                num_samples_window=256, overlap=0.5,
                                window_type="boxcar")[0]
    fc_direct = compute_fc_levels(direct, ["hx", "hy"], ["ex", "ey"],
                                  num_samples_window=256, overlap=0.5,
                                  window_type="boxcar")[0]
    ratio = (np.nanmedian(np.abs(fc_time["ex"].values))
             / np.nanmedian(np.abs(fc_direct["ex"].values)))
    assert ratio == pytest.approx(1.0, rel=0.05)
