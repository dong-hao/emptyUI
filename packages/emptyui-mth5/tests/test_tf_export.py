"""Export TF node: nomenclature-carrying TFs must export in every format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import build_tf
from emptyui_mth5.tf import ExportTF, _to_standard_coords


def _lemi_tf():
    periods = np.array([2.0, 4.0, 8.0])
    Z = np.zeros((3, 2, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j  # ex x hx
    Z[:, 1, 1] = 2 + 0j  # ey x hy
    return build_tf(periods, Z, ["bx", "by"], ["e1", "e2"],
                    channel_nomenclature="lemi12")


def test_export_edi_from_lemi_tf(tmp_path: Path):
    """EDI export of a nomenclature-carrying TF writes standard channel names
    (mt_metadata's EDI writer standardizes on write)."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf = _lemi_tf()
    out = tmp_path / "lemi_tf.edi"
    ExportTF().execute(ctx, tf, path=str(out), file_type="edi")
    content = out.read_text()
    assert "ex" in content and "ey" in content
    assert "channels_recorded_electric=[ex, ey]" in content


def test_export_zmm_from_lemi_tf(tmp_path: Path):
    """ZMM export of a nomenclature-carrying TF must not fail: mt_metadata
    1.0.10's ZMM writer assumes standard names, so ExportTF relabels."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf = _lemi_tf()
    out = tmp_path / "lemi_tf.zmm"
    ExportTF().execute(ctx, tf, path=str(out), file_type="zmm")
    assert out.exists() and out.stat().st_size > 0


def test_to_standard_coords_preserves_impedance():
    """The ZMM-safe standard-coords copy must carry the same impedance values
    in the standard ex/ey x hx/hy positions."""
    tf = _lemi_tf()
    std = _to_standard_coords(tf)
    arr = std.dataset["transfer_function"]
    assert np.allclose(
        arr.sel(output="ex", input="hx").values,
        tf.dataset["transfer_function"].sel(output=tf.ex, input=tf.hx).values,
    )
    assert np.allclose(
        arr.sel(output="ey", input="hy").values,
        tf.dataset["transfer_function"].sel(output=tf.ey, input=tf.hy).values,
    )


def _lemi_tf_with_errors_and_station():
    periods = np.array([2.0, 4.0, 8.0])
    Z = np.zeros((3, 2, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j
    Z[:, 1, 1] = 2 + 0j
    Zerr = np.zeros((3, 2, 2), dtype=float)
    Zerr[:, 0, 0] = 0.1
    Zerr[:, 1, 1] = 0.2
    return build_tf(periods, Z, ["bx", "by"], ["e1", "e2"],
                    channel_nomenclature="lemi12", Z_err=Zerr,
                    survey_id="bgs_37", station_id="NO46",
                    latitude=56.72884, longitude=-2.85449067, elevation=120.0)


def test_export_edi_includes_errors_and_station_header(tmp_path: Path):
    """EDI export carries the TF error (as variance) and the station identity
    (DATAID/LAT/LON/SURVEY) — required for standalone archival use."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf = _lemi_tf_with_errors_and_station()
    out = tmp_path / "lemi_tf.edi"
    ExportTF().execute(ctx, tf, path=str(out), file_type="edi")
    content = out.read_text()
    assert "DATAID=NO46" in content
    assert "LAT=" in content and "LON=" in content
    assert "SURVEY=bgs_37" in content
    assert ">ZXX.VAR" in content
    assert "1.000000e-02" in content  # 0.1**2 -> variance
    assert "4.000000e-02" in content  # 0.2**2 -> variance


def test_to_standard_coords_carries_errors(tmp_path: Path):
    """The ZMM-safe standard-coords copy keeps the error field."""
    tf = _lemi_tf_with_errors_and_station()
    std = _to_standard_coords(tf)
    err = std.impedance_error
    assert np.allclose(err.sel(output="ex", input="hx").values, [0.1, 0.1, 0.1])
    assert np.allclose(err.sel(output="ey", input="hy").values, [0.2, 0.2, 0.2])
