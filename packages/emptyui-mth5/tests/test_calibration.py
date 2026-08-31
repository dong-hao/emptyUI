"""Tests for the calibration framework (docs/DESIGN.md §9.1). Error surfaces live
in test_errors.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui_mth5.calibration import (
    attach_filters,
    build_fap_filters,
    find_calibration_file,
    parse_fap,
    parse_json_calibration,
)

REAL_CHN = Path(__file__).resolve().parents[3] / "examples" / "data" / "bgs-lemi-424" / "lemi.chn"


def test_parse_fap_real_file():
    """The LEMI-417 .chn sample parses: 12 freqs x 5 channels, MOD/ARG."""
    rows = parse_fap(REAL_CHN)
    assert set(rows["channels"]) == {"bx", "by", "bz", "e1", "e2"}
    table = rows["channels"]["hx"] if "hx" in rows["channels"] else rows["channels"]["bx"]
    freqs = table["frequencies"]
    assert len(freqs) == 12
    assert freqs[0] == 0.25 and freqs[-1] == 1000.0
    np.testing.assert_allclose(table["amplitudes"][:2], [0.030, 0.160], rtol=1e-3)
    np.testing.assert_allclose(table["phases_deg"][:2], [211.6, 158.5], rtol=1e-3)
    # electric and magnetic rows are distinct in this file
    e_table = rows["channels"]["e1"]
    np.testing.assert_allclose(e_table["amplitudes"][:2], [0.012, 0.087], rtol=1e-3)


def test_build_fap_filters_flat_extension():
    rows = parse_fap(REAL_CHN)
    filters = build_fap_filters(rows, extend_flat=True)
    assert set(filters) == {"bx", "by", "bz", "e1", "e2"}
    fap = filters["bx"]
    assert fap.name == "cal_bx"
    assert fap.units_in == "nanoTesla" and fap.units_out == "nanoTesla"
    # two pad rows (L-M7): the second zeroes the below-pad extrapolation
    # slope so the phase stays flat below the pad point
    assert fap.frequencies.size == 14
    # flat clamp: the padded low-frequency points repeat the first row
    assert fap.amplitudes[0] == pytest.approx(fap.amplitudes[2])
    assert fap.phases[0] == 0.0 and fap.phases[1] == 0.0
    # electric channels use mV/km (canonical mt_metadata name)
    assert filters["e1"].units_in == "milliVolt per kilometer"


def test_build_fap_filters_small_phase_table_in_degrees():
    """Phases pass through as degrees: mt_metadata's constructor applies a
    max(|phase|) > 6*pi degrees->radians heuristic that silently misreads
    small-phase tables as radians (a -3 deg phase stayed -3 rad -> -171.9 deg
    response). The Phoenix ASC path carries the same guard."""
    rows = {"channels": {
        "bx": {
            "frequencies": np.array([0.1, 1.0, 10.0]),
            "amplitudes": np.array([1.0, 1.0, 1.0]),
            "phases_deg": np.array([-2.0, -3.0, -5.0]),
        }
    }}
    fap = build_fap_filters(rows, extend_flat=False)["bx"]
    resp = fap.complex_response(np.array([1.0]))
    assert np.angle(resp[0]) == pytest.approx(np.deg2rad(-3.0), abs=1e-9)
    # the real LEMI table (max |phase| > 6 pi) is unaffected
    real = build_fap_filters(parse_fap(REAL_CHN), extend_flat=False)["bx"]
    at = 1.0
    expected = np.interp(
        np.log(at), np.log(real.frequencies), real.phases,
    )
    assert np.angle(real.complex_response(np.array([at]))[0]) == pytest.approx(
        expected, abs=1e-6
    )


def test_find_calibration_file_prefers_station_pair(tmp_path: Path):
    (tmp_path / "no46.chn").write_text("")
    (tmp_path / "generic.def").write_text("")
    assert find_calibration_file(tmp_path, station_id="NO46").name == "no46.chn"
    assert find_calibration_file(tmp_path, station_id="NX37").name == "generic.def"
    assert find_calibration_file(tmp_path / "missing") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert find_calibration_file(empty) is None


def test_parse_json_calibration(tmp_path: Path):
    p = tmp_path / "cal.json"
    p.write_text(
        '{"bx": {"Calibration": {"gain": 1.0, "Freq": [0.25, 1.0], '
        '"Re": [1.0, 2.0], "Im": [0.0, 0.0]}}}'
    )
    rows = parse_json_calibration(p)
    table = rows["channels"]["bx"]
    np.testing.assert_allclose(table["frequencies"], [0.25, 1.0])
    np.testing.assert_allclose(table["amplitudes"], [1.0, 2.0], rtol=1e-9)
    np.testing.assert_allclose(table["phases_deg"], [0.0, 0.0], atol=1e-9)


def test_attach_filters_and_calibrate_round_trip():
    """Attaching the real FAP filters marks channels and calibrate applies
    the response removal (values change, units preserved)."""
    import warnings

    warnings.filterwarnings("ignore")
    from mt_timeseries import ChannelTS, RunTS

    sr = 8.0
    n = 2048
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * 1.0 * t) + 0.1
    ch_list = []
    for comp in ("bx", "by", "bz", "e1", "e2"):
        c = ChannelTS("magnetic" if comp.startswith("b") else "electric")
        c.channel_metadata.units = "nT" if comp.startswith("b") else "mV/km"
        c.sample_rate = sr
        c.ts = x.copy()
        c.component = comp
        ch_list.append(c)
    run = RunTS(array_list=ch_list)

    filters = build_fap_filters(parse_fap(REAL_CHN), extend_flat=True)
    attach_filters(run, filters)
    assert run.filters["cal_bx"].name == "cal_bx"
    assert run.dataset["bx"].attrs["filters"][0]["applied_filter"]["name"] == "cal_bx"

    cal = run.calibrate()
    # response removal changed the data (modulus < 1 at 1 Hz -> amplified)
    assert not np.allclose(cal.bx.ts, x)
    assert cal.bx.ts.size == n


def test_default_unit_by_nomenclature():
    """Default units come from the nomenclature, not a name-prefix guess:
    standard-named magnetic channels (hx/hy/hz) are nanoTesla — the old
    startswith("b") check stamped them mV/km, silently skipping aurora's
    nT→A/m conversion downstream (Z off by mu0*1e9, the history §16.3 bug class)."""
    from emptyui_mth5.calibration import _default_unit

    for ch in ("hx", "hy", "hz", "bx", "by", "bz", "h1", "H2"):
        assert _default_unit(ch) == "nanoTesla", ch
    for ch in ("ex", "ey", "e1", "e2", "e3", "E4"):
        assert _default_unit(ch) == "mV/km", ch
    # auxiliary/unknown channels have no meaningful default — never a guess
    assert _default_unit("temperature") == ""
    assert _default_unit("") == ""


def test_attach_filters_preserves_existing_chains():
    """attach_filters stamps unity only on channels WITHOUT a response:
    a channel carrying a real archived chain (e.g. a fast-path Extract
    payload's hx referencing the ASC box+coil chain) must keep it — the
    old unconditional loop replaced it with cal_<ch>_unity, silently
    dropping the instrument response from all downstream FC/TF math."""
    import warnings
    from types import SimpleNamespace

    import numpy as np
    import xarray as xr
    from mt_metadata.timeseries.filters import CoefficientFilter

    warnings.filterwarnings("ignore")

    archived = CoefficientFilter(gain=3.0, units_in="nanoTesla",
                                 units_out="nanoTesla")
    archived.name = "archived_hx"
    t = np.arange(64) / 8.0
    run = SimpleNamespace(
        dataset=xr.Dataset({
            "ex": xr.DataArray(t, dims=("time",), attrs={"units": "mV/km"}),
            "hx": xr.DataArray(t, dims=("time",),
                               attrs={"units": "nanoTesla",
                                      "filters": [{"applied_filter": {
                                          "name": "archived_hx",
                                          "applied": True}}]}),
        }),
        filters={"archived_hx": archived},
    )
    out = attach_filters(run, {"ex": CoefficientFilter(gain=1.0)},
                         strict=False)
    # ex got the explicit filter; hx kept its archived chain
    refs = out.dataset["hx"].attrs["filters"]
    assert refs[0]["applied_filter"]["name"] == "archived_hx"
    assert out.filters["archived_hx"] is archived


def test_attach_filters_invalidates_cached_channels():
    """attach_filters re-stamps the dataset attrs in place — a _CachedRunTS
    whose channel cache already resolved the response chain at first access
    must drop the cache, else a later-attached filter is silently ignored
    (the executor serves the same instance across runs/param edits)."""
    import warnings

    import numpy as np
    import xarray as xr
    from emptyui_mth5.runs import _CachedRunTS
    from mt_metadata.timeseries.filters import CoefficientFilter

    warnings.filterwarnings("ignore")

    t = np.arange(1024) / 8.0
    x = np.sin(2 * np.pi * 1.0 * t)
    t_axis = (np.datetime64("2024-01-01T00:00:00", "ns")
              + (np.arange(1024) * 125_000_000).astype("timedelta64[ns]"))
    ds = xr.Dataset({"hx": xr.DataArray(
        x, dims=("time",), coords={"time": t_axis},
        attrs={"type": "magnetic", "units": "nT"})})
    runts = _CachedRunTS(ds)
    _ = runts.hx.ts  # populate the channel cache (first access)
    fresh = CoefficientFilter(gain=2.0, units_in="nT", units_out="nT")
    fresh.name = "fresh_hx"
    attach_filters(runts, {"hx": fresh})
    # the cached channel was invalidated: the newly attached filter wins
    # (pre-fix the stale chain won and the gain was NOT divided out).
    # Tolerance covers mt_timeseries' DC-bin convention (see test_calibrate)
    cal = runts.calibrate()
    np.testing.assert_allclose(
        np.asarray(cal.hx.ts), x / 2.0, rtol=0.03, atol=0.03)


def test_fap_flat_extension_phase_flat_below_pad():
    """L-M7: the flat extension pads TWO identical rows — mt_metadata's
    extrapolation below the lowest table point follows the lowest
    interval's slope, so a single pad row ramped the phase below the pad
    point (flat in modulus, ramping in phase); two rows zero that slope."""
    from emptyui_mth5.calibration import build_fap_filters

    rows = {"channels": {"hx": {
        "frequencies": np.array([1.0, 10.0, 100.0]),
        "amplitudes": np.array([1000.0, 990.0, 980.0]),
        "phases_deg": np.array([-2.0, -2.0, -1.5]),
    }}}
    fap = build_fap_filters(rows)["hx"]
    q = fap.complex_response(np.array([1e-6, 1e-5, 1e-4]))
    phase = np.rad2deg(np.angle(q))
    assert np.allclose(phase, 0.0, atol=1e-6)  # truly flat below the pad
    assert np.allclose(np.abs(q), 1000.0, rtol=1e-9)
    # between the pad point and the measured range the ramp still applies
    q2 = fap.complex_response(np.array([1.0]))
    assert abs(np.rad2deg(np.angle(q2))[0] + 2.0) < 0.01
