"""Phoenix MTU-5A ASC calibration parser tests (DESIGN §24.12).

The real manufacturer files live in examples/data/phoenix-mtu-test/
(COIL1620/2417/2418_CLC.asc + MTU-1690_CLB.asc); the synthetic fixtures
here mirror their layout so CI passes without them. The real files are
exercised when present (structure-only checks — no physics assumptions).
"""

from __future__ import annotations

import numpy as np
import pytest
from emptyui_mth5.phoenix_calibration import (
    CLC_PLACEHOLDER_CHANNEL,
    PHOENIX_CHANNEL_NUMBERS,
    PhoenixCalibration,
    build_phoenix_filters,
    find_asc_calibrations,
    parse_asc_calibration,
)

CLB_TEXT = """\
C:\\test\\new\\MTU-1690_CLB : Instrument Calibrations
MTU-1690.CLB    Normalizing  E-gain: 10  H-gain: 3

        Channel      Freq (Hz)        mV / mV        Degrees
              1       317.6471       1.033872             -2
              2       317.6471       1.031907             -3
              3       317.6471       1.150253             -2
              4       317.6471       1.147232             -2
              5       317.6471       1.14476              -2
              1       264.7059       1.019006             -2
              2       264.7059         1.0171             -2
              3       264.7059       1.133737             -2
              4       264.7059       1.130853             -2
              5       264.7059       1.128407             -2
              1           1.0            0.99               0
              2           1.0            0.98               0
              3           1.0            1.10               0
              4           1.0            1.09               0
              5           1.0            1.08               0
"""

CLC_TEXT = """\
C:\\test\\new\\COIL2417_CLC : Coil Calibrations
COIL2417.CLC    Normalizing Factor: 1000.0 mV/nT

        Channel      Freq (Hz)        mV / mV        Degrees
              1       317.6471       0.305647      -73.75193
              1       264.7059       0.260206      -60.90458
              1        1.171875       1.173319       1.036313
              1       0.2929688      0.8997241       40.65713
              1       0.000495911    0.00202397      90.5818
              1       0.000419617    0.00171259      90.5975
"""


@pytest.fixture
def cal_dir(tmp_path):
    (tmp_path / "MTU-1690_CLB.asc").write_text(CLB_TEXT)
    (tmp_path / "COIL2417_CLC.asc").write_text(CLC_TEXT)
    return tmp_path


def test_parse_clb_structure(cal_dir):
    cal = parse_asc_calibration(cal_dir / "MTU-1690_CLB.asc")
    assert isinstance(cal, PhoenixCalibration)
    assert cal.kind == "CLB" and cal.serial == "MTU-1690"
    assert cal.normalizing == {"e_gain": 10.0, "h_gain": 3.0}
    assert sorted(cal.channels) == [1, 2, 3, 4, 5]
    t = cal.channels[3]
    assert np.allclose(t.frequencies, [1.0, 264.7059, 317.6471])  # ascending
    assert np.allclose(t.amplitudes, [1.10, 1.133737, 1.150253])
    assert np.allclose(t.phases_deg, [0.0, -2.0, -2.0])
    assert cal.source == "MTU-1690_CLB.asc"


def test_parse_clc_structure(cal_dir):
    cal = parse_asc_calibration(cal_dir / "COIL2417_CLC.asc")
    assert cal.kind == "CLC" and cal.serial == "COIL2417"
    assert cal.normalizing == {"factor_mv_per_nt": 1000.0}
    assert sorted(cal.channels) == [CLC_PLACEHOLDER_CHANNEL]
    t = cal.channels[CLC_PLACEHOLDER_CHANNEL]
    # the file lists frequencies descending; the parser sorts ascending
    assert np.all(np.diff(t.frequencies) > 0)
    assert t.frequencies[0] == pytest.approx(0.000419617)
    assert t.frequencies[-1] == pytest.approx(317.6471)
    assert t.phases_deg[0] == pytest.approx(90.5975)  # dB/dt tail kept


def test_parse_rejects_malformed(tmp_path):
    (tmp_path / "x.asc").write_text("only one line\n")
    with pytest.raises(ValueError, match="too few lines"):
        parse_asc_calibration(tmp_path / "x.asc")
    (tmp_path / "b.asc").write_text(
        "x.CLB : Instrument Calibrations\nb.CLB Normalizing E-gain: 10\n"
        "h\nh\nh\n")
    with pytest.raises(ValueError, match="no calibration rows"):
        parse_asc_calibration(tmp_path / "b.asc")
    (tmp_path / "c.asc").write_text(
        "c.CLB : Instrument Calibrations\n"
        "c.CLB Normalizing Factor: 1000.0 mV/nT\n"
        "h\nh\nh\n1 1 1 1\n1 1 2 2\n")  # duplicate frequency
    with pytest.raises(ValueError, match="duplicate"):
        parse_asc_calibration(tmp_path / "c.asc")


def test_build_filters_clb(cal_dir):
    cal = parse_asc_calibration(cal_dir / "MTU-1690_CLB.asc")
    filters = build_phoenix_filters(cal)
    assert sorted(filters) == ["ex", "ey", "hx", "hy", "hz"]
    fap = filters["hx"]
    assert fap.name == "phoenix_clb_mtu-1690_hx"
    assert fap.units_in == "nanoTesla" and fap.units_out == "nanoTesla"
    assert filters["ex"].units_in == "milliVolt per kilometer"
    # the response at a table frequency reproduces amp*exp(i*phase)
    r = fap.complex_response(np.array([317.6471]))
    assert np.abs(r[0]) == pytest.approx(1.150253, rel=1e-5)
    assert np.angle(r[0]) == pytest.approx(np.radians(-2.0), abs=1e-4)


def test_build_filters_clc_linear_extension(cal_dir):
    cal = parse_asc_calibration(cal_dir / "COIL2417_CLC.asc")
    fap = build_phoenix_filters(cal, comp_map={1: "hx"})["hx"]
    assert fap.name == "phoenix_clc_coil2417_hx"
    f = np.array([cal.channels[1].frequencies[0] / 10,
                  cal.channels[1].frequencies[0]])
    r = fap.complex_response(f)
    # linear-in-f extension: amplitude scales with f, phase held at +90.6 deg
    assert np.abs(r[0]) == pytest.approx(np.abs(r[1]) / 10, rel=1e-6)
    assert np.angle(r[0]) == pytest.approx(np.radians(90.5975), abs=1e-3)


def test_build_filters_flat_extension(cal_dir):
    cal = parse_asc_calibration(cal_dir / "COIL2417_CLC.asc")
    fap = build_phoenix_filters(cal, comp_map={1: "hx"}, extend_below="flat")["hx"]
    f = np.array([cal.channels[1].frequencies[0] / 10,
                  cal.channels[1].frequencies[0]])
    r = fap.complex_response(f)
    # flat extension: constant amplitude (first row), zero phase below f_min
    assert np.abs(r[0]) == pytest.approx(0.00171259, rel=1e-5)
    assert np.abs(r[1]) == pytest.approx(0.00171259, rel=1e-5)


def test_find_asc_calibrations(cal_dir):
    found = find_asc_calibrations(
        cal_dir, box_serial="1690",
        coil_serials={"hx": "COIL2417", "hy": "COIL1620", "hz": "COIL2418"})
    assert found["box"] == cal_dir / "MTU-1690_CLB.asc"
    assert found["hx"] == cal_dir / "COIL2417_CLC.asc"
    assert found["hy"] is None and found["hz"] is None
    # no directory -> nothing found, no error
    assert find_asc_calibrations(cal_dir / "nope", box_serial="1") == {
        "box": None}


def test_real_files_structure():
    """The real manufacturer tables parse with the documented layout
    (skipped when the corpus is absent; §24.12 parsed facts)."""
    from pathlib import Path

    base = Path(__file__).resolve().parents[3] / "examples" / "data" / \
        "phoenix-mtu-test"
    if not (base / "MTU-1690_CLB.asc").exists():
        pytest.skip("real MTU-5A calibration tables not present")
    box = parse_asc_calibration(base / "MTU-1690_CLB.asc")
    assert (box.kind, box.serial) == ("CLB", "MTU-1690")
    assert box.normalizing == {"e_gain": 10.0, "h_gain": 3.0}
    assert sorted(box.channels) == [1, 2, 3, 4, 5]
    assert all(len(t.frequencies) == 99 for t in box.channels.values())
    assert box.channels[1].frequencies[0] == pytest.approx(1.3113e-05, rel=1e-3)
    assert box.channels[1].frequencies[-1] == pytest.approx(317.6471, rel=1e-4)
    for serial, n in (("COIL2417", 99), ("COIL1620", 80), ("COIL2418", 80)):
        coil = parse_asc_calibration(base / f"{serial}_CLC.asc")
        assert coil.kind == "CLC" and coil.serial == serial
        assert coil.normalizing == {"factor_mv_per_nt": 1000.0}
        assert sorted(coil.channels) == [CLC_PLACEHOLDER_CHANNEL]
        assert len(coil.channels[1].frequencies) == n
    # channel numbering matches the TBL channel_keys (§24.12)
    assert PHOENIX_CHANNEL_NUMBERS == {1: "ex", 2: "ey", 3: "hx", 4: "hy",
                                       5: "hz"}
