"""Edge-case / error-surface tests for the mth5 plugin.

Grouped (and marked ``errors``) so the happy-path suite can run fast with
``-m "not errors"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes.types import TimeSeries
from emptyui_mth5.calibrate import Calibrate
from emptyui_mth5.calibration import (
    attach_filters,
    build_fap_filters,
    load_calibration,
    parse_fap,
    parse_json_calibration,
)
from emptyui_mth5.fc import SpectraToFCGroup
from emptyui_mth5.io import LoadMth5
from emptyui_mth5.persist import persist_fc_level
from emptyui_mth5.plots import PlotTipper
from emptyui_mth5.runs import _resolve_run, _resolve_station, _resolve_survey

pytestmark = pytest.mark.errors


def _ctx(tmp_path: Path):
    return RunContext(workspace=tmp_path, store=DataStore())


REAL_CHN = Path(__file__).resolve().parents[3] / "examples" / "data" / "bgs-lemi-424" / "lemi.chn"


# --- calibration parsing / filter building (from test_calibration.py) ---

def test_parse_fap_rejects_garbage(tmp_path: Path):
    p = tmp_path / "x.chn"
    p.write_text("not an fap file\n")
    with pytest.raises(ValueError, match="FAP header"):
        parse_fap(p)


def test_load_calibration_unsupported(tmp_path: Path):
    p = tmp_path / "x.csv"
    p.write_text("1,2\n")
    with pytest.raises(ValueError, match="unsupported calibration"):
        load_calibration(p)


def test_build_fap_filters_size_mismatch_rejected():
    rows = {"channels": {"bx": {"frequencies": [0.25, 1.0],
                                "amplitudes": [1.0],
                                "phases_deg": [0.0, 5.0]}}}
    with pytest.raises(ValueError, match="size mismatch"):
        build_fap_filters(rows)


@pytest.mark.parametrize(
    ("frequencies", "amplitudes", "phases", "message"),
    [
        ([], [], [], "empty"),
        ([0.25, np.nan], [1.0, 1.0], [0.0, 0.0], "finite and strictly positive"),
        ([0.0, 1.0], [1.0, 1.0], [0.0, 0.0], "finite and strictly positive"),
        ([1.0, 0.5], [1.0, 1.0], [0.0, 0.0], "strictly increasing"),
        ([0.5, 0.5], [1.0, 1.0], [0.0, 0.0], "strictly increasing"),
        ([0.25, 1.0], [-1.0, 1.0], [0.0, 0.0], "non-negative"),
        ([0.25, 1.0], [1.0, 1.0], [0.0, np.inf], "phases must be finite"),
    ],
)
def test_build_fap_filters_rejects_invalid_frequency_response_tables(
    frequencies, amplitudes, phases, message,
):
    rows = {
        "channels": {
            "bx": {
                "frequencies": frequencies,
                "amplitudes": amplitudes,
                "phases_deg": phases,
            }
        }
    }
    with pytest.raises(ValueError, match=message):
        build_fap_filters(rows)


def test_json_calibration_rejects_unsorted_frequency_grid(tmp_path: Path):
    p = tmp_path / "invalid.json"
    p.write_text(
        '{"bx": {"Calibration": {"Freq": [1.0, 0.5], '
        '"Re": [1.0, 1.0], "Im": [0.0, 0.0]}}}'
    )
    rows = parse_json_calibration(p)
    with pytest.raises(ValueError, match="strictly increasing"):
        build_fap_filters(rows, extend_flat=False)


def test_json_calibration_rejects_component_size_mismatch(tmp_path: Path):
    p = tmp_path / "invalid_sizes.json"
    p.write_text(
        '{"bx": {"Calibration": {"Freq": [0.25, 1.0], '
        '"Re": [1.0], "Im": [0.0, 0.0]}}}'
    )
    with pytest.raises(ValueError, match="Freq/Re/Im size mismatch"):
        parse_json_calibration(p)


def test_parse_json_calibration_per_channel_grids(tmp_path: Path):
    """Channels tabulated on different frequency grids each keep their own
    grid (previously every channel silently reused the first grid)."""
    p = tmp_path / "cal.json"
    p.write_text(
        '{"bx": {"Calibration": {"Freq": [0.25, 1.0], "Re": [1.0, 2.0], "Im": [0.0, 0.0]}},'
        '"e1": {"Calibration": {"Freq": [0.5, 2.0, 8.0], "Re": [1.0, 1.0, 1.0], "Im": [0.1, 0.0, -0.1]}}}'
    )
    rows = parse_json_calibration(p)
    np.testing.assert_allclose(rows["channels"]["bx"]["frequencies"], [0.25, 1.0])
    np.testing.assert_allclose(rows["channels"]["e1"]["frequencies"], [0.5, 2.0, 8.0])
    filters = build_fap_filters(rows, extend_flat=False)
    assert filters["bx"].frequencies.size == 2
    assert filters["e1"].frequencies.size == 3


def test_attach_filters_missing_channel():
    """A filter for a channel not in the run is rejected (strict mode)."""
    import warnings

    warnings.filterwarnings("ignore")
    from mt_timeseries import ChannelTS, RunTS

    c = ChannelTS("magnetic")
    c.sample_rate = 1.0
    c.ts = np.zeros(64)
    c.component = "bx"
    run = RunTS(array_list=[c])
    filters = build_fap_filters(parse_fap(REAL_CHN), extend_flat=True)
    with pytest.raises(ValueError, match="not present in the run"):
        attach_filters(run, filters)


def test_attach_filters_lenient_mode_skips_and_warns(caplog):
    """strict=False (Calibrate node path): a filter for an absent channel is
    skipped with a warning instead of raising."""
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    from mt_timeseries import ChannelTS, RunTS

    c = ChannelTS("magnetic")
    c.sample_rate = 1.0
    c.ts = np.zeros(64)
    c.component = "bx"
    run = RunTS(array_list=[c])
    real = build_fap_filters(parse_fap(REAL_CHN), extend_flat=True)
    filters = {"bx": real["bx"], "nope": real["by"]}
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        attach_filters(run, filters, strict=False)
    assert any("nope" in r.message for r in caplog.records)
    assert run.dataset["bx"].attrs["filters"][0]["applied_filter"]["name"] == "cal_bx"


# --- Calibrate node error surfaces (from test_calibrate.py) -------------

def test_calibrate_requires_something(tmp_path: Path):
    """A core TimeSeries with no calibration input cannot be calibrated."""
    ts = TimeSeries(channels={"ex": np.zeros(64)}, sample_rate=8.0)
    with pytest.raises(TypeError):
        Calibrate().execute(_ctx(tmp_path), ts, calibration=None)


def test_calibrate_dict_all_unmatched_raises(tmp_path: Path):
    """No filter key matches any run channel -> clear error, not silence."""
    ts = TimeSeries(channels={"ex": np.zeros(64)}, sample_rate=8.0)
    from mt_metadata.timeseries.filters import CoefficientFilter

    with pytest.raises(ValueError, match="no calibration filter matched"):
        Calibrate().execute(_ctx(tmp_path), ts,
                            calibration={"bz": CoefficientFilter(gain=1.0)})


# --- fc / merge node error surfaces (from test_fc_nodes.py) -------------

def test_spectra_to_fc_group_requires_archive(tmp_path: Path):
    fc = xr.Dataset({"hx": (("frequency", "time"), np.zeros((2, 2), dtype=complex))},
                    coords={"frequency": [0.1, 0.2],
                            "time": pd.date_range("2024-01-01", periods=2, freq="s")})
    with pytest.raises(ValueError, match="requires an archive"):
        SpectraToFCGroup().execute(_ctx(tmp_path), fc, mth5=None)


@pytest.mark.filterwarnings("ignore")
def test_merge_runts_mixed_rates_error():
    """The merge core rejects mixed sample rates with a clear error."""
    from mt_timeseries import ChannelTS, RunTS

    def _runts(sr):
        ch = ChannelTS("electric", data=np.zeros(16), start="2024-01-01",
                       sample_rate=sr)
        ch.component = "ex"
        return RunTS(array_list=[ch])

    from emptyui_mth5.merge import merge_runts

    with pytest.raises(ValueError, match="mixed sample rates"):
        merge_runts([_runts(1.0), _runts(8.0)])


# --- run id resolution / persistence / plots ----------------------------

def test_bad_ids_raise_clear_errors(archive):
    with pytest.raises(ValueError, match="not in archive"):
        _resolve_survey(archive, "no_such_survey")
    s = _resolve_survey(archive, "")
    with pytest.raises(ValueError, match="not in survey"):
        _resolve_station(s, "no_such_station")
    st = _resolve_station(s, "")
    with pytest.raises(ValueError, match="not in station"):
        _resolve_run(st, "no_such_run")


def test_persist_requires_datetime_time(tmp_path, fake_runts):
    """Persisting an fc with a float (relative) time axis must fail clearly."""
    from _fc import compute_fc_seconds

    runts = fake_runts
    # drop the datetime start so compute_fc_seconds produces a float time axis
    runts.start = 0.0
    fc = compute_fc_seconds(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="boxcar")
    assert "datetime64" not in str(fc.time.dtype)
    with pytest.raises(ValueError):
        persist_fc_level(_ctx(tmp_path), fc, 0, fc.attrs["sample_rate"])


def test_plot_tipper_requires_tipper(tmp_path: Path):
    """A 2x2 (impedance-only) TF has no tipper -> clear error."""
    from emptyui_aurora._processing import build_tf

    periods = np.array([2.0, 4.0])
    Z = np.zeros((2, 2, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j
    Z[:, 1, 1] = 2 + 0j
    tf = build_tf(periods, Z, ["bx", "by"], ["e1", "e2"],
                  channel_nomenclature="lemi12")
    assert not tf.has_tipper()
    with pytest.raises(ValueError, match="no tipper"):
        PlotTipper().execute(_ctx(tmp_path), tf)


def test_load_mth5_mode_w_rejected(tmp_path: Path):
    """L-M12: mode 'w' (silently truncating) is gone from the select; a
    legacy document carrying it fails loudly instead of truncating."""
    from mth5.mth5 import MTH5

    path = tmp_path / "a.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    m.add_survey("4P")
    m.close_mth5()

    node = LoadMth5()
    ctx = RunContext(workspace=None, store=DataStore())
    with pytest.raises(ValueError, match="TRUNCATE"):
        node.execute(ctx, path=str(path), mode="w")
    # the archive survives the refused call
    m2 = MTH5()
    m2.open_mth5(str(path), "r", locking=False)
    assert "4P" in m2.surveys_group.groups_list
    m2.close_mth5()
