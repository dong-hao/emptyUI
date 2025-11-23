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


@pytest.mark.skipif(not pathlib.Path("/tmp").exists(), reason="tmp unavailable")
def test_phase_convention_matches_aurora_halfspace():
    runts = _runts()
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"], 256.0, 0.5, "hamming",
                    magnetic_unit="h")
    _, Z = estimate_tf(fc, ["hx", "hy"], ["ex", "ey"], periods_per_decade=8)

    xy = np.degrees(np.angle(Z[:, 0, 1]))
    yx = np.degrees(np.angle(Z[:, 1, 0]))
    ok = np.abs(Z[:, 0, 1]) > 1e-12

    # uniform halfspace: Zxy phase ~ +45°, Zyx ~ -135° (aurora reference)
    assert np.allclose(np.median(xy[ok]), 45.0, atol=8.0)
    assert np.allclose(np.median(yx[ok]), -135.0, atol=8.0)
