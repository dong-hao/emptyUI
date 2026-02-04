"""SpectraToFCGroup / MergeRuns — happy-path id resolution (error surfaces
in test_errors.py)."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.fc import SpectraToFCGroup
from mth5.mth5 import MTH5


def _fc_dataset() -> xr.Dataset:
    rng = np.random.default_rng(1)
    return xr.Dataset(
        {c: (("frequency", "time"),
             rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3)))
         for c in ("ex", "ey", "hx", "hy")},
        coords={"frequency": np.linspace(0.1, 0.9, 4),
                "time": pd.date_range("2024-01-01", periods=3, freq="s")},
    )


@pytest.fixture()
def writable_archive(test12rr_path: Path, tmp_path: Path):
    warnings.filterwarnings("ignore")
    p = tmp_path / "arch.mth5"
    shutil.copy(test12rr_path, p)
    m = MTH5()
    m.open_mth5(str(p), "r+")
    yield m
    m.close_mth5()


def test_spectra_to_fc_group_resolves_empty_ids(writable_archive, tmp_path):
    """Empty survey/station/run fall back to the first available (the old
    code used legacy mth5.get_survey/survey_group only)."""
    from emptyui_mth5.runs import _resolve_station, _resolve_survey

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    SpectraToFCGroup().execute(ctx, _fc_dataset(), mth5=writable_archive)
    survey_obj = _resolve_survey(writable_archive, "")
    st = _resolve_station(survey_obj, "")
    assert st.fourier_coefficients_group.groups_list, "no fc group written"


def test_merge_runs_resolves_empty_ids(writable_archive, tmp_path):
    """The rebuilt Merge Runs resolves survey/station via the fallback
    helpers; runs='' merges every run of the station (a single-run archive
    is a passthrough with the new run id)."""
    from emptyui_mth5.merge import MergeRuns
    from emptyui_mth5.runs import _resolve_run, _resolve_station, _resolve_survey

    survey_obj = _resolve_survey(writable_archive, "")
    station_obj = _resolve_station(survey_obj, "")
    run_id = _resolve_run(station_obj, "").metadata.id
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    res = MergeRuns().execute(ctx, writable_archive, runs=run_id)
    assert set(res["merged_run"].channels) >= {"ex", "ey", "hx", "hy"}
    assert res["merged_run"].run_metadata.id == "merged"



def test_spectra_to_fc_group_metadata(writable_archive, tmp_path, caplog):
    """Archive metadata hygiene (review #10 bundle): the decimation level id
    and the level's sample rate are written from the fc's metadata / params —
    never silently defaulted (the old code always wrote level "0" with rate
    1.0)."""
    from emptyui_mth5.runs import _resolve_station, _resolve_survey

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = _fc_dataset()
    fc.attrs["sample_rate"] = 8.0

    # derived from the fc's metadata when the param is left empty
    SpectraToFCGroup().execute(ctx, fc, mth5=writable_archive,
                               decimation_level=2)
    st = _resolve_station(_resolve_survey(writable_archive, ""), "")
    fcg = st.fourier_coefficients_group.get_fc_group(
        st.fourier_coefficients_group.groups_list[0])
    dec2 = fcg.get_decimation_level("2")
    assert dec2.channel_summary.sample_rate_decimation_level.to_list() == [8.0] * 4

    # an explicit param overrides the fc metadata
    SpectraToFCGroup().execute(ctx, fc, mth5=writable_archive,
                               decimation_level=1, sample_rate=2.0)
    dec1 = fcg.get_decimation_level("1")
    assert dec1.channel_summary.sample_rate_decimation_level.to_list() == [2.0] * 4

    # no attrs, no param: 1.0 fallback with a warning (never a rejection)
    import logging as _logging

    fc2 = _fc_dataset()  # no sample_rate attr
    with caplog.at_level(_logging.WARNING, logger="emptyui"):
        SpectraToFCGroup().execute(ctx, fc2, mth5=writable_archive)
    assert any("no sample_rate metadata" in r.message for r in caplog.records)


# --- M8: fc geometry/identity survive the archive round trip --------------

_GEOMETRY = {
    "sample_rate": 2.0, "decimation_level": 0, "factor": 4,
    "num_samples_window": 256, "overlap": 0.5, "window_type": "hamming",
    "prewhitening": "first difference", "per_window_detrend": "linear",
}


def test_fc_provenance_round_trip(writable_archive, tmp_path):
    """Write FC -> Read FC restores the window/decimation geometry (mth5's
    native FC metadata model), the station identity (archive hierarchy) and
    the nomenclature/magnetic-unit derivations — so the merge_fc window
    warnings and the §23.3 provenance line work for read-back FCs.

    Note: attrs the FC metadata model cannot express (electric_scaling,
    calibration_applied) deliberately stay in-memory provenance — unknown
    HDF5 attributes would break mt_metadata's read_metadata for any
    foreign consumer of the archive.
    """
    from emptyui_mth5.fc import FCGroupToSpectra

    fc = _fc_dataset()
    fc.attrs.update(_GEOMETRY)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    SpectraToFCGroup().execute(ctx, fc, mth5=writable_archive)
    out = FCGroupToSpectra().execute(ctx, writable_archive)["fc"]
    # geometry from mth5's native FC metadata
    assert out.attrs["sample_rate"] == 2.0
    assert out.attrs["num_samples_window"] == 256
    assert out.attrs["window_type"] == "hamming"
    assert abs(out.attrs["overlap"] - 0.5) < 1e-9
    assert out.attrs["prewhitening"] == "first difference"
    assert out.attrs["per_window_detrend"] == "linear"
    assert out.attrs["factor"] == 4
    # identity resolved from the archive hierarchy
    assert out.attrs.get("station_id")
    assert out.attrs.get("run_id")
    # derivations from the channel names
    assert out.attrs["channel_nomenclature"] == "default"
    # magnetic unit is re-derived from the channel names (the recorded 'b'
    # vs 'h' choice is in-memory provenance; standard hx/hy derive to 'h')
    assert out.attrs["magnetic_unit"] in ("b", "h")
    assert out.attrs["input_channels"] == "hx,hy"
    assert out.attrs["output_channels"] == "ex,ey"


def test_persist_fc_level_round_trips_geometry(tmp_path):
    """The workspace persistence path (Compute FC auto-persist, Build FC
    persist) records and restores the same geometry/identity."""
    from emptyui_mth5.persist import load_fc_level, persist_fc_level

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = _fc_dataset()
    fc.attrs.update(_GEOMETRY)
    persist_fc_level(ctx, fc, 0, 2.0, survey="ws", station="st", run="r1")
    out = load_fc_level(ctx, 0, survey="ws", station="st", run="r1")
    assert out.attrs["sample_rate"] == 2.0
    assert out.attrs["num_samples_window"] == 256
    assert out.attrs["window_type"] == "hamming"
    assert abs(out.attrs["overlap"] - 0.5) < 1e-9
    assert out.attrs["station_id"] == "st"
    assert out.attrs["run_id"] == "r1"
    assert out.attrs["survey_id"] == "ws"
