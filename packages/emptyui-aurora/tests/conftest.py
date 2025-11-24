"""Shared fixtures.

test12rr_runts generates the official synthetic halfspace dataset on the fly
(offline, ~seconds) so the MT tests run in CI without any gitignored raw
instrument data on disk.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def test12rr_runts(tmp_path_factory):
    import warnings

    warnings.filterwarnings("ignore")
    from mth5.data.make_mth5_from_asc import create_test12rr_h5
    from mth5.mth5 import MTH5

    target = tmp_path_factory.mktemp("test12rr")
    path = create_test12rr_h5(target_folder=str(target))

    m = MTH5()
    m.open_mth5(str(path), "r")
    sg = m.survey_group  # v0.1.0 synthetic file
    st = sg.stations_group.get_station("test1")
    rid = st.run_summary.iloc[0]["id"]
    runts = st.get_run(rid).to_runts()
    m.close_mth5()
    for c in runts.channels:
        runts.dataset[c] = runts.dataset[c] - runts.dataset[c].mean()
    return runts
