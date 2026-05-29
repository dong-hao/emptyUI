"""Validate our estimator's impedance phase convention against aurora.

Uses the official synthetic halfspace test dataset (test12rr) that the
mth5/aurora tutorial processes: aurora reports Zxy phase ~+45°, Zyx ~-135°
for a uniform halfspace. Our estimators must reproduce that convention. The
dataset is generated on the fly by the ``test12rr_runts`` fixture so the test
runs offline / in CI without gitignored raw instrument data.
"""

from __future__ import annotations

import numpy as np
import pytest
from _synthetic import compute_fc_seconds
from emptyui_aurora._processing import estimate_tf


@pytest.mark.parametrize("engine,remote_reference,use_dummy", [
    ("Least Squares", False, False),
    ("Robust", False, False),
    ("Robust", True, False),    # self-reference (no ref_fc connected)
    ("Robust", True, True),     # dummy remote station (duplicated data via ref_fc)
])
def test_phase_convention_matches_aurora_halfspace(
    test12rr_runts, engine, remote_reference, use_dummy
):
    runts = test12rr_runts
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"], 256.0, 0.5, "boxcar",
                    magnetic_unit="h")
    # model a second (dummy) remote station by duplicating the data file: a
    # copy of the same time series -> a second fc used as the remote reference
    ref_fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"], 256.0, 0.5,
                        "boxcar", magnetic_unit="h") if use_dummy else None
    _, Z, _ = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"], estimator_engine=engine,
                       remote_reference=remote_reference, periods_per_octave=8,
                       ref_fc=ref_fc)

    xy = np.degrees(np.angle(Z[:, 0, 1]))
    yx = np.degrees(np.angle(Z[:, 1, 0]))
    ok = np.abs(Z[:, 0, 1]) > 1e-12

    # uniform halfspace: Zxy phase ~ +45°, Zyx ~ -135° (aurora reference)
    assert np.allclose(np.median(xy[ok]), 45.0, atol=8.0)
    assert np.allclose(np.median(yx[ok]), -135.0, atol=8.0)


def test_multilevel_merged_phase_convention_matches_aurora_halfspace(test12rr_runts):
    """A multi-level compute -> per-level estimate -> merge must keep aurora's
    phase convention on the halfspace (this is what the user sees as the
    broadband TF in the browser)."""
    from emptyui_aurora._processing import (
        compute_fc_levels,
        estimate_tf_levels,
        merge_tf_levels,
    )

    runts = test12rr_runts
    levels = compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                               num_decimations=3, factor=2, num_samples_window=128,
                               overlap=0.5, window_type="boxcar", magnetic_unit="h")
    per = estimate_tf_levels(levels, ["hx", "hy"], ["ex", "ey"],
                             estimator_engine="Robust", periods_per_octave=8)
    _, Z, _ = merge_tf_levels(per)

    xy = np.degrees(np.angle(Z[:, 0, 1]))
    yx = np.degrees(np.angle(Z[:, 1, 0]))
    ok = np.abs(Z[:, 0, 1]) > 1e-12
    assert np.allclose(np.median(xy[ok]), 45.0, atol=8.0)
    assert np.allclose(np.median(yx[ok]), -135.0, atol=8.0)
