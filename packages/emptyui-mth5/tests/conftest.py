"""Shared fixtures (generates the synthetic test12rr dataset offline)."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def test12rr_path(tmp_path_factory):
    import warnings

    warnings.filterwarnings("ignore")
    from mth5.data.make_mth5_from_asc import create_test12rr_h5

    target = tmp_path_factory.mktemp("test12rr")
    path = create_test12rr_h5(target_folder=str(target))
    return path
