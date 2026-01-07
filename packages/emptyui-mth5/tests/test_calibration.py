"""Tests for the calibration framework (DESIGN §16.1). Error surfaces live
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
    assert fap.frequencies.size == 13
    # flat clamp: the padded low-frequency point repeats the first row
    assert fap.amplitudes[0] == pytest.approx(fap.amplitudes[1])
    assert fap.phases[0] == 0.0
    # electric channels use mV/km (canonical mt_metadata name)
    assert filters["e1"].units_in == "milliVolt per kilometer"


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
    nT→A/m conversion downstream (Z off by mu0*1e9, the §16.3 bug class)."""
    from emptyui_mth5.calibration import _default_unit

    for ch in ("hx", "hy", "hz", "bx", "by", "bz", "h1", "H2"):
        assert _default_unit(ch) == "nanoTesla", ch
    for ch in ("ex", "ey", "e1", "e2", "e3", "E4"):
        assert _default_unit(ch) == "mV/km", ch
    # auxiliary/unknown channels have no meaningful default — never a guess
    assert _default_unit("temperature") == ""
    assert _default_unit("") == ""
