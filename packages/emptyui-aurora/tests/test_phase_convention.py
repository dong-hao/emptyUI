"""Validate our estimator's impedance phase convention against aurora.

Uses the official synthetic halfspace test dataset (test12rr) that the
mth5/aurora tutorial processes: aurora reports Zxy phase ~+45°, Zyx ~-135°
for a uniform halfspace. Our first-pass LSQ estimator must reproduce that
convention. Skipped when the synthetic data cannot be generated (offline).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from emptyui_aurora._processing import compute_fc, estimate_tf

SYNTH = pathlib.Path("/tmp/opencode/synth/test12rr.h5")


def _runts():
    import warnings

    warnings.filterwarnings("ignore")
    from mth5.data.make_mth5_from_asc import create_test12rr_h5
    from mth5.mth5 import MTH5

    target = SYNTH.parent
    if not SYNTH.exists():
        create_test12rr_h5(target_folder=target)
    m = MTH5()
    m = m.open_mth5(str(SYNTH), "r")
    sg = m.survey_group  # v0.1.0 synthetic file
    st = sg.stations_group.get_station("test1")
    rid = st.run_summary.iloc[0]["id"]
    runts = st.get_run(rid).to_runts()
    m.close_mth5()
    for c in runts.channels:
        runts.dataset[c] = runts.dataset[c] - runts.dataset[c].mean()
    return runts


@pytest.mark.parametrize("engine,remote_reference,use_dummy", [
    ("Least Squares", False, False),
    ("Robust", False, False),
    ("Robust", True, False),    # self-reference (no ref_fc connected)
    ("Robust", True, True),     # dummy remote station (duplicated data via ref_fc)
])
def test_phase_convention_matches_aurora_halfspace(engine, remote_reference, use_dummy):
    runts = _runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"], 256.0, 0.5, "hamming",
                    magnetic_unit="h")
    # model a second (dummy) remote station by duplicating the data file: a
    # copy of the same time series -> a second fc used as the remote reference
    ref_fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"], 256.0, 0.5,
                        "hamming", magnetic_unit="h") if use_dummy else None
    _, Z = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"], estimator_engine=engine,
                       remote_reference=remote_reference, periods_per_octave=8,
                       ref_fc=ref_fc)

    xy = np.degrees(np.angle(Z[:, 0, 1]))
    yx = np.degrees(np.angle(Z[:, 1, 0]))
    ok = np.abs(Z[:, 0, 1]) > 1e-12

    # uniform halfspace: Zxy phase ~ +45°, Zyx ~ -135° (aurora reference)
    assert np.allclose(np.median(xy[ok]), 45.0, atol=8.0)
    assert np.allclose(np.median(yx[ok]), -135.0, atol=8.0)


def test_multilevel_merged_phase_convention_matches_aurora_halfspace():
    """A multi-level compute -> per-level estimate -> merge must keep aurora's
    phase convention on the halfspace (this is what the user sees as the
    broadband TF in the browser)."""
    from emptyui_aurora._processing import (
        compute_fc_levels,
        estimate_tf_levels,
        merge_tf_levels,
    )

    runts = _runts()
    levels = compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                               num_decimations=3, factor=2, num_samples_window=128,
                               overlap=0.5, window_type="hamming", magnetic_unit="h")
    per = estimate_tf_levels(levels, ["hx", "hy"], ["ex", "ey"],
                             estimator_engine="Robust", periods_per_octave=8)
    _, Z = merge_tf_levels(per)

    xy = np.degrees(np.angle(Z[:, 0, 1]))
    yx = np.degrees(np.angle(Z[:, 1, 0]))
    ok = np.abs(Z[:, 0, 1]) > 1e-12
    assert np.allclose(np.median(xy[ok]), 45.0, atol=8.0)
    assert np.allclose(np.median(yx[ok]), -135.0, atol=8.0)
