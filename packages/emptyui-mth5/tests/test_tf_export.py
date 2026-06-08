"""Export TF node: nomenclature-carrying TFs must export in every format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import build_tf
from emptyui_mth5.tf import SI_TO_EDI, ExportTF, ImportTF, _to_standard_coords


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
    # internal TFs are SI; the EDI file stores the SEG convention
    # ([mV/km]/[nT]) — the impedance (and its error) is scaled on export:
    # std 0.1 SI -> 0.1 * SI_TO_EDI, variance = std**2
    assert f"{(0.1 * SI_TO_EDI) ** 2:.6e}" in content
    assert f"{(0.2 * SI_TO_EDI) ** 2:.6e}" in content


def test_to_standard_coords_carries_errors(tmp_path: Path):
    """The ZMM-safe standard-coords copy keeps the error field."""
    tf = _lemi_tf_with_errors_and_station()
    std = _to_standard_coords(tf)
    err = std.impedance_error
    assert np.allclose(err.sel(output="ex", input="hx").values, [0.1, 0.1, 0.1])
    assert np.allclose(err.sel(output="ey", input="hy").values, [0.2, 0.2, 0.2])


def test_export_edi_then_raw_read_gives_seg_units(tmp_path: Path):
    """Export converts SI -> SEG ([mV/km]/[nT]): a raw mt_metadata read of the
    exported file (verbatim, no conversion) yields the scaled impedance."""
    from mt_metadata.transfer_functions.core import TF

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf = _lemi_tf()
    out = tmp_path / "lemi_tf.edi"
    ExportTF().execute(ctx, tf, path=str(out), file_type="edi")
    raw = TF()
    raw.read(fn=str(out))
    raw_z = raw.dataset["transfer_function"].sel(output="ex", input="hx").values
    si_z = tf.dataset["transfer_function"].sel(output=tf.ex, input=tf.hx).values
    np.testing.assert_allclose(raw_z, si_z * SI_TO_EDI, rtol=1e-6)
    # the tipper is dimensionless (hz row) — untouched; here absent (zeros)
    raw_t = raw.dataset["transfer_function"].sel(output="hz", input="hx").values
    si_t = tf.dataset["transfer_function"].sel(output=tf.hz, input=tf.hx).values
    np.testing.assert_allclose(raw_t, si_t, rtol=1e-12, atol=0.0)


def test_import_edi_converts_to_si(tmp_path: Path):
    """Import converts the SEG impedance back to SI so downstream display
    nodes (rho = |Z|^2/(mu0*w)) see physically consistent values; the
    tipper row is untouched."""

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf_in = _lemi_tf_with_errors_and_station()
    out = tmp_path / "roundtrip.edi"
    ExportTF().execute(ctx, tf_in, path=str(out), file_type="edi")
    imported = ImportTF().execute(ctx, str(out))["tf"]
    for got_name, want_name in ((imported.ex, tf_in.ex),
                                (imported.ey, tf_in.ey)):
        for inp in (imported.hx, imported.hy):
            want_inp = tf_in.hy if inp == imported.hy else tf_in.hx
            got = imported.dataset["transfer_function"].sel(
                output=got_name, input=inp).values
            want = tf_in.dataset["transfer_function"].sel(
                output=want_name, input=want_inp).values
            np.testing.assert_allclose(got, want, rtol=1e-5)
    imp_err = imported.dataset["transfer_function_error"]
    in_err = tf_in.dataset["transfer_function_error"]
    np.testing.assert_allclose(
        imp_err.sel(output=imported.ex, input=imported.hx).values,
        in_err.sel(output=tf_in.ex, input=tf_in.hx).values, rtol=1e-5)
    # the tipper row (hz) must be untouched by the scaling
    np.testing.assert_allclose(
        imported.dataset["transfer_function"].sel(
            output=imported.hz, input=imported.hx).values,
        tf_in.dataset["transfer_function"].sel(
            output=tf_in.hz, input=tf_in.hx).values,
        rtol=1e-12, atol=0.0)


def test_import_official_edi_rho_is_physical(tmp_path: Path):
    """The bundled official Phoenix EDI imports to SI: apparent resistivity
    at the shortest period is ~17 ohm-m (SEG-correct), not ~1.1e7 (the
    verbatim-import bug, SI/SEG mixed up)."""

    edi = (Path(__file__).resolve().parents[3] / "examples" / "data"
           / "phoenix-mtu-test" / "B0940330_z.edi")
    if not edi.exists():
        pytest.skip("official Phoenix EDI not on disk (gitignored, G7)")
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf = ImportTF().execute(ctx, str(edi))["tf"]
    arr = tf.dataset["transfer_function"]
    period = arr.coords["period"].values
    zxy = arr.sel(output=tf.ex, input=tf.hy).values
    mu0 = 4e-7 * np.pi
    rho = np.abs(zxy) ** 2 / (mu0 * 2 * np.pi / period)
    # SEG-correct value at the shortest period is ~17.4 ohm-m; the verbatim
    # import bug would show 1.1e7 instead
    assert 10.0 < rho[0] < 30.0, rho[0]
    assert rho.max() < 1e4, rho.max()


def test_import_edi_si_opt_out_skips_the_conversion(tmp_path: Path):
    """L8: a file already stored in SI would be double-scaled with no way
    to say so — the EDI->SI bridge gets an explicit opt-out (default
    unchanged: every supported format stores SEG/EMTF)."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf_in = _lemi_tf_with_errors_and_station()
    out = tmp_path / "seg_stored.edi"
    ExportTF().execute(ctx, tf_in, path=str(out), file_type="edi")

    as_si = ImportTF().execute(ctx, str(out), impedance_units="SI (V/m, A/m)")
    seg_stored_z = as_si["tf"].dataset["transfer_function"].sel(
        output=as_si["tf"].ex, input=as_si["tf"].hx).values
    # verbatim: the file's SEG values pass through UNSCALED — i.e. they
    # are tf_in * SI_TO_EDI, NOT tf_in (which the default import yields)
    np.testing.assert_allclose(
        seg_stored_z,
        tf_in.dataset["transfer_function"].sel(
            output=tf_in.ex, input=tf_in.hx).values * SI_TO_EDI,
        rtol=1e-6)

    # the default import of the same file converts (existing contract)
    as_seg = ImportTF().execute(ctx, str(out))["tf"]
    np.testing.assert_allclose(
        as_seg.dataset["transfer_function"].sel(
            output=as_seg.ex, input=as_seg.hx).values,
        tf_in.dataset["transfer_function"].sel(
            output=tf_in.ex, input=tf_in.hx).values,
        rtol=1e-5)
