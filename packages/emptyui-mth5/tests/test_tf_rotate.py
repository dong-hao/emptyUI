"""Rotate TF node: frame rotation math, node behaviour, EDI declination
parsing and sign-convention handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import build_tf
from emptyui_mth5.tf import (
    ImportTF,
    RotateTF,
    _edi_declination,
    _rot_matrix,
    rotate_transfer_function,
)

#: Minimal SEG-1.0 EDI accepted by mt_metadata (2 periods, impedance + tipper).
MINI_EDI = """\
>HEAD
DATAID="TEST01"
LAT=45:22:18.34
LONG=-98:17:35.15
ELEV=394
DECL=5.7
>
>INFO
SURVEY=Test
DECLINATION=5.7 degree
>
>DEFINEMEAS
MAXCHAN=7
UNITS=M
REFTYPE=CART
>
>HMEAS ID=1001.001 CHTYPE=Hx X=0.0 Y=0.0 Z=0.0 AZM=5.7
>HMEAS ID=1002.001 CHTYPE=Hy X=0.0 Y=0.0 Z=0.0 AZM=95.7
>HMEAS ID=1003.001 CHTYPE=Hz X=0.0 Y=0.0 Z=0.0 AZM=0.0
>EMEAS ID=1004.001 CHTYPE=Ex X=-50.0 Y=0.0 X2=50.0 Y2=0.0 AZM=5.7
>EMEAS ID=1005.001 CHTYPE=Ey X=0.0 Y=-50.0 X2=0.0 Y2=50.0 AZM=95.7
>
>MTSECT
SECTID="TEST01"
NFREQ=2
HX=1001.001
HY=1002.001
HZ=1003.001
EX=1004.001
EY=1005.001
>
>FREQ //2
1.000000E-01 5.000000E-02
>
>ZROT //2
0.000000E+00 0.000000E+00
>
>ZXXR //2
1.000000E+00 1.000000E+00
>ZXXI //2
0.000000E+00 0.000000E+00
>ZXX.VAR //2
1.000000E-03 1.000000E-03
>ZXYR //2
2.000000E+00 2.000000E+00
>ZXYI //2
0.000000E+00 0.000000E+00
>ZXY.VAR //2
1.000000E-03 1.000000E-03
>ZYXR //2
3.000000E+00 3.000000E+00
>ZYXI //2
0.000000E+00 0.000000E+00
>ZYX.VAR //2
1.000000E-03 1.000000E-03
>ZYYR //2
4.000000E+00 4.000000E+00
>ZYYI //2
0.000000E+00 0.000000E+00
>ZYY.VAR //2
1.000000E-03 1.000000E-03
>
>TIPPER ROTATION ANGLES
>TROT //2
0.000000E+00 0.000000E+00
>
>TZR //2
1.000000E-01 1.000000E-01
>TZI //2
0.000000E+00 0.000000E+00
>TZ.VAR //2
1.000000E-03 1.000000E-03
>TYR //2
2.000000E-01 2.000000E-01
>TYI //2
0.000000E+00 0.000000E+00
>TY.VAR //2
1.000000E-03 1.000000E-03
>
"""


def _lemi_tf(with_tipper=True, with_errors=True):
    periods = np.array([2.0, 4.0, 8.0])
    n_out = 3 if with_tipper else 2
    Z = np.zeros((3, n_out, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j  # ex x hx
    Z[:, 1, 1] = 2 + 0j  # ey x hy
    if with_tipper:
        Z[:, 2, 0] = 0.5 + 0.1j  # hz x hx
        Z[:, 2, 1] = 0.2 - 0.3j  # hz x hy
    Z_err = None
    if with_errors:
        Z_err = np.zeros((3, n_out, 2), dtype=float)
        Z_err[:, 0, 0] = 0.1
        Z_err[:, 1, 1] = 0.2
        if with_tipper:
            Z_err[:, 2, 0] = 0.05
            Z_err[:, 2, 1] = 0.07
    outputs = ["e1", "e2", "bz"] if with_tipper else ["e1", "e2"]
    return build_tf(
        periods, Z, ["bx", "by"], outputs, channel_nomenclature="lemi12",
        Z_err=Z_err, survey_id="bgs_37", station_id="NO46",
        latitude=56.72884, longitude=-2.85449067, elevation=120.0,
    )


def _z2(tf):
    """(n,2,2) impedance block in standard order ex/ey x hx/hy."""
    a = tf.dataset["transfer_function"]
    return np.stack([
        np.stack([a.sel(output=tf.ex, input=tf.hx).values,
                  a.sel(output=tf.ex, input=tf.hy).values], axis=-1),
        np.stack([a.sel(output=tf.ey, input=tf.hx).values,
                  a.sel(output=tf.ey, input=tf.hy).values], axis=-1),
    ], axis=1)


def _tipper(tf):
    a = tf.dataset["transfer_function"]
    return np.stack([
        a.sel(output=tf.hz, input=tf.hx).values,
        a.sel(output=tf.hz, input=tf.hy).values,
    ], axis=-1)


def test_rot_matrix_convention():
    """R(angle) matches history §16.3 (positive = clockwise from true
    north): R(0) = I and R(90) maps x -> -y."""
    assert np.allclose(_rot_matrix(0.0), np.eye(2))
    assert np.allclose(_rot_matrix(90.0), [[0, 1], [-1, 0]], atol=1e-12)


def test_rotate_90_swaps_diagonal_impedance():
    """At 90° a diagonal Z maps to the swapped diagonal (independent of the
    einsum implementation)."""
    tf = _lemi_tf()
    r = rotate_transfer_function(tf, 90.0)
    z = _z2(r)
    assert np.allclose(z[:, 0, 0], 2 + 0j)
    assert np.allclose(z[:, 1, 1], 1 + 0j)
    assert np.allclose(z[:, 0, 1], 0)
    assert np.allclose(z[:, 1, 0], 0)


def test_rotate_preserves_trace_and_determinant():
    """Frame rotation is a similarity transform: trace/det are invariant."""
    tf = _lemi_tf()
    r = rotate_transfer_function(tf, 37.0)
    z0 = _z2(tf)
    z1 = _z2(r)
    assert np.allclose(np.trace(z1, axis1=1, axis2=2), np.trace(z0, axis1=1, axis2=2))
    assert np.allclose(
        np.linalg.det(z1), np.linalg.det(z0), atol=1e-12
    )


def test_rotate_roundtrip():
    """Rotate by +23° then −23° recovers the original TF."""
    tf = _lemi_tf()
    r = rotate_transfer_function(rotate_transfer_function(tf, 23.0), -23.0)
    assert np.allclose(_z2(r), _z2(tf), atol=1e-12)


def test_rotate_zero_angle_identity():
    tf = _lemi_tf()
    r = rotate_transfer_function(tf, 0.0)
    assert np.allclose(_z2(r), _z2(tf))
    assert np.allclose(_tipper(r), _tipper(tf))


def test_tipper_rotates_as_vector():
    """Tipper rotates as a vector T' = R(angle)·T (90°: (Tx,Ty) -> (Ty,-Tx))."""
    tf = _lemi_tf()
    r = rotate_transfer_function(tf, 90.0)
    t = _tipper(r)
    assert np.allclose(t[:, 0], 0.2 - 0.3j)
    assert np.allclose(t[:, 1], -(0.5 + 0.1j))


def test_errors_rotate_with_abs_R():
    """Per-element error stds rotate with |R| (no covariance): at 90° the
    diagonal stds swap."""
    tf = _lemi_tf()
    r = rotate_transfer_function(tf, 90.0)
    a = r.dataset["transfer_function_error"]
    assert np.allclose(a.sel(output=tf.ex, input=tf.hx).values, 0.2)
    assert np.allclose(a.sel(output=tf.ey, input=tf.hy).values, 0.1)
    assert np.allclose(a.sel(output=tf.hz, input=tf.hx).values, 0.07)
    assert np.allclose(a.sel(output=tf.hz, input=tf.hy).values, 0.05)


def test_rotate_lemi_tf_preserves_nomenclature():
    """The rotated TF keeps the actual channel names + nomenclature (the
    rotation is applied to the standard block, not the labels)."""
    tf = _lemi_tf()
    r = rotate_transfer_function(tf, 45.0)
    a = r.dataset["transfer_function"]
    assert r.ex == "e1" and r.ey == "e2" and r.hz == "bz"
    assert r.hx == "bx" and r.hy == "by"
    assert list(a.coords["output"].values) == ["e1", "e2", "bz", "bx", "by"]
    assert list(a.coords["input"].values) == ["e1", "e2", "bz", "bx", "by"]
    assert r.channel_nomenclature["ex"] == "e1"
    assert r.station_metadata.id == "NO46"


def test_rotate_does_not_mutate_input():
    tf = _lemi_tf()
    before = tf.dataset["transfer_function"].values.copy()
    rotate_transfer_function(tf, 45.0)
    assert np.array_equal(tf.dataset["transfer_function"].values, before)


def test_rotate_requires_standard_block():
    """A TF whose dataset lacks ex/ey outputs (a non-MT block) errors."""
    from types import SimpleNamespace

    from emptyui_mth5.tf import _rotate_arrays

    tf = _lemi_tf()
    a = tf.dataset["transfer_function"]
    stub = SimpleNamespace(
        dataset={"transfer_function": a.isel(output=[2, 3, 4])},
        inverse_channel_nomenclature=tf.inverse_channel_nomenclature,
    )
    with pytest.raises(ValueError):
        _rotate_arrays(stub, 45.0)


def test_rotate_edi_5x5_keeps_zero_rows():
    """Rotating an EDI-imported (5x5 canonical) TF touches only the 2x2
    impedance block and tipper row; the hx/hy rows stay zero. The canonical
    mt_metadata storage is always 5x5 (setter fills the 3x2 block)."""
    import xarray as xr
    from mt_metadata.transfer_functions.core import TF

    std = TF()
    std.channel_nomenclature = {s: s for s in ("ex", "ey", "hx", "hy", "hz")}
    blk = np.zeros((3, 3, 2), dtype=complex)
    blk[:, 0, 0] = 1 + 0j
    blk[:, 1, 1] = 4 + 0j
    blk[:, 2, 0] = 0.1 + 0j
    blk[:, 2, 1] = 0.2 + 0j
    std.transfer_function = xr.DataArray(
        blk, dims=("period", "output", "input"),
        coords={"period": [2.0, 4.0, 8.0], "output": ["ex", "ey", "hz"],
                "input": ["hx", "hy"]},
    )
    assert std.dataset["transfer_function"].shape == (3, 5, 5)
    r = rotate_transfer_function(std, 90.0)
    a = r.dataset["transfer_function"]
    assert np.allclose(a.sel(output="ex", input="hx").values, 4 + 0j)
    assert np.allclose(a.sel(output="ey", input="hy").values, 1 + 0j)
    for row in ("hx", "hy"):
        assert np.allclose(a.sel(output=row, input="hx").values, 0)
        assert np.allclose(a.sel(output=row, input="hy").values, 0)


def test_node_rotates_and_appends_note():
    ctx = RunContext(workspace=Path("."), store=DataStore())
    tf = _lemi_tf()
    out = RotateTF().execute(ctx, tf, angle=90.0)["tf"]
    assert np.allclose(_z2(out)[:, 0, 0], 2 + 0j)
    assert out.run_metadata.comments.value == "rotated 90 deg"


def test_node_use_declination():
    ctx = RunContext(workspace=Path("."), store=DataStore())
    tf = _lemi_tf()
    tf.station_metadata.location.declination.value = 5.7
    out = RotateTF().execute(ctx, tf, use_declination=True)["tf"]
    expected = rotate_transfer_function(tf, -5.7)
    assert np.allclose(_z2(out), _z2(expected))
    assert out.run_metadata.comments.value == "rotated -5.7 deg"


def test_node_use_declination_falls_back_to_angle():
    """No declination metadata (0.0) → warn and use the angle param."""
    ctx = RunContext(workspace=Path("."), store=DataStore())
    tf = _lemi_tf()
    out = RotateTF().execute(ctx, tf, angle=-5.7, use_declination=True)["tf"]
    expected = rotate_transfer_function(tf, -5.7)
    assert np.allclose(_z2(out), _z2(expected))


def test_node_definition():
    assert RotateTF.type == "mth5.tf.rotate"
    assert RotateTF.inputs[0].type == "tf"
    assert RotateTF.outputs[0].type == "tf"
    assert {p.name for p in RotateTF.params} == {"angle", "use_declination"}


def test_edi_declination_explicit_field():
    assert _edi_declination("DECL=5.7\n") == 5.7
    assert _edi_declination("Declination: 5.7 degree\n") == 5.7
    assert _edi_declination(">INFO\ndeclination = -3.2 deg\n") == -3.2
    assert _edi_declination("declination= 10.0 degrees\n") == 10.0


def test_edi_declination_azm_fallback():
    assert _edi_declination(
        ">HMEAS ID=1001.001 CHTYPE=Hx X=0.0 Y=0.0 Z=0.0 AZM= 5.7\n"
    ) == 5.7
    assert _edi_declination(
        ">HMEAS ID=1001.001 CHTYPE=Hx X=0.0 Y=0.0 Z=0.0 AZIMUTH=5.7\n"
    ) == 5.7


def test_edi_declination_missing():
    assert _edi_declination("nothing relevant here\n") is None
    assert _edi_declination(">HMEAS ID=1001.001 CHTYPE=Hy AZM=95.7\n") is None


def test_import_edi_parses_declination(tmp_path: Path):
    """ImportTF parses an explicit EDI declination field into station
    metadata (mt_metadata itself leaves it 0 for standard EDI files)."""
    p = tmp_path / "site.edi"
    p.write_text(MINI_EDI)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf = ImportTF().execute(ctx, str(p))["tf"]
    d = tf.station_metadata.location.declination
    assert float(d.value) == 5.7
    assert d.model == "unknown"


def test_import_edi_declination_azm_fallback(tmp_path: Path):
    """No explicit field → the Hx measurement azimuth is used (USArray-style
    EDI: the EarthScope EMTF page reports AZM as the station declination)."""
    text = MINI_EDI
    for line in ("DECL=5.7\n", "DECLINATION=5.7 degree\n"):
        text = text.replace(line, "")
    p = tmp_path / "site.edi"
    p.write_text(text)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf = ImportTF().execute(ctx, str(p))["tf"]
    assert float(tf.station_metadata.location.declination.value) == 5.7


def test_import_edi_no_declination(tmp_path: Path):
    """No declination field at all → metadata stays at the mt_metadata
    default (0.0) and the rotate node falls back to the angle param."""
    text = MINI_EDI
    for line in ("DECL=5.7\n", "DECLINATION=5.7 degree\n"):
        text = text.replace(line, "")
    text = text.replace("AZM=5.7", "AZM=0.0")
    p = tmp_path / "site.edi"
    p.write_text(text)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf = ImportTF().execute(ctx, str(p))["tf"]
    d = tf.station_metadata.location.declination
    assert float(d.value) == 0.0


# --- M5: tipper-less TF rotation -----------------------------------------

def test_rotate_tipperless_tf():
    """A tipper-less TF (zero hz row, the canonical storage for absent
    tipper) rotates: the impedance block rotates, the hz row stays zero."""
    tf = _lemi_tf(with_tipper=False)
    out = rotate_transfer_function(tf, 45.0)
    R = _rot_matrix(45.0)
    expect = np.einsum("ij,njk,lk->nil", R, _z2(tf), R)
    assert np.allclose(_z2(out), expect)
    assert np.allclose(_tipper(out), 0.0)
    assert np.allclose(_tipper(tf), 0.0)


def test_rotate_hz_less_storage_zero_fills_tipper_row():
    """A dataset genuinely lacking the hz coordinate (out-of-contract
    storage, reachable only by bypassing the TF setter) still rotates —
    mt_metadata's canonical 3x2 setter forces a zero-filled hz row."""
    from types import SimpleNamespace

    from emptyui_mth5.tf import _standard_indices

    tf = _lemi_tf(with_tipper=False)
    arr = tf.dataset["transfer_function"]
    (ex_i, ey_i, hz_i), (hx_i, hy_i) = _standard_indices(
        tf, ("ex", "ey", "hz"), ("hx", "hy"))
    assert hz_i is not None  # canonical storage always carries the hz row
    z2 = arr.isel(output=[ex_i, ey_i], input=[hx_i, hy_i]).copy()
    stub = SimpleNamespace(
        dataset={"transfer_function": z2},
        inverse_channel_nomenclature=dict(tf.inverse_channel_nomenclature),
        ex=tf.ex, ey=tf.ey, hz=tf.hz, hx=tf.hx, hy=tf.hy,
        run_metadata=SimpleNamespace(comments=SimpleNamespace(value="")),
    )
    out = rotate_transfer_function(stub, 45.0)
    blk = out.transfer_function
    assert blk.shape[1:] == (3, 2)
    assert np.allclose(blk.values[:, 2, :], 0.0)
    R = _rot_matrix(45.0)
    expect = np.einsum("ij,njk,lk->nil", R, z2.values, R)
    assert np.allclose(blk.values[:, :2, :], expect)


# --- M6: EDI sign convention (not reliably self-describing) ---------------

def _sign_edi(with_field: bool = True) -> str:
    text = MINI_EDI.replace(
        ">ZXYI //2\n0.000000E+00 0.000000E+00",
        ">ZXYI //2\n5.000000E-01 5.000000E-01")
    if with_field:
        text = text.replace(
            "REFTYPE=CART",
            "REFTYPE=CART\nSIGNCONVENTION=exp(+ i\\omega t)")
    return text


def _import(tmp_path, text, **kwargs):
    p = tmp_path / "site.edi"
    p.write_text(text)
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    return ImportTF().execute(ctx, str(p), **kwargs)["tf"]


def test_import_negate_convention(tmp_path: Path):
    base = _import(tmp_path, _sign_edi())
    neg = _import(tmp_path, _sign_edi(), sign_convention="negate (180 deg)")
    a = base.dataset["transfer_function"].values
    b = neg.dataset["transfer_function"].values
    assert np.allclose(b, -a)
    assert "negated" in neg.run_metadata.comments.value
    assert "negated" not in (base.run_metadata.comments.value or "")


def test_import_conjugate_convention(tmp_path: Path):
    base = _import(tmp_path, _sign_edi())
    con = _import(tmp_path, _sign_edi(),
                  sign_convention="conjugate (exp(+i omega t))")
    a = base.dataset["transfer_function"].values
    b = con.dataset["transfer_function"].values
    assert np.allclose(b, np.conj(a))
    assert "conjugated" in con.run_metadata.comments.value


def _mt_sample(name: str) -> Path:
    """A real sample file shipped with the pinned mt_metadata (1.0.10).

    The non-EDI formats have no corpus in this repo; the library's
    ``data/transfer_functions`` files are in the wheel RECORD, so they are
    a stable fixture for the readers (skip if a future pin drops them).
    """
    import mt_metadata

    return Path(mt_metadata.__file__).parent / "data" / "transfer_functions" / name


@pytest.mark.parametrize("name", [
    "tf_xml.xml", "tf_zmm.zmm", "tf_jfile.j", "tf_avg.avg",
])
def test_import_sign_convention_applies_to_every_format(tmp_path: Path, name: str):
    """H10: the sign conversion is format-agnostic.

    It used to live inside the ``.edi`` branch, so xml/zmm/j/avg imports
    silently ignored the param — the node reported success and the TF was
    unchanged. The samples are real files read by the real readers; the
    assertions are relative (as-is vs negate vs conjugate), so they do not
    depend on each file's own convention.
    """
    p = _mt_sample(name)
    if not p.exists():
        pytest.skip(f"mt_metadata sample {name} not installed")
    ctx = RunContext(workspace=tmp_path, store=DataStore())

    tf_base = ImportTF().execute(ctx, str(p))["tf"]
    tf_neg = ImportTF().execute(
        ctx, str(p), sign_convention="negate (180 deg)")["tf"]
    tf_con = ImportTF().execute(
        ctx, str(p), sign_convention="conjugate (exp(+i omega t))")["tf"]

    base = tf_base.dataset["transfer_function"].values
    assert np.any(base != 0), "sample carries no transfer-function data"
    np.testing.assert_allclose(
        tf_neg.dataset["transfer_function"].values, -base)
    np.testing.assert_allclose(
        tf_con.dataset["transfer_function"].values, np.conj(base))
    assert "negated" in tf_neg.run_metadata.comments.value
    assert "conjugated" in tf_con.run_metadata.comments.value
    assert "negated" not in (tf_base.run_metadata.comments.value or "")


def test_import_as_is_warns_on_declared_plus_convention(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="emptyui"):
        _import(tmp_path, _sign_edi())
    assert any("SIGNCONVENTION" in r.message for r in caplog.records)


def test_import_no_sign_field_no_warning(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="emptyui"):
        _import(tmp_path, _sign_edi(with_field=False))
    assert not any("SIGNCONVENTION" in r.message for r in caplog.records)


# --- M7: explicit-zero declination vs absent (the 'not d.value' gap) ------

def test_import_explicit_zero_declination_not_overridden(tmp_path: Path):
    """An explicit DECLINATION=0.0 must survive: the strict-match branch
    used to treat the falsy 0.0 as 'unset' and re-set it (clobbering the
    model); the Hx AZM fallback must not fire over an explicit field."""
    text = MINI_EDI.replace("DECL=5.7", "DECL=0.0")
    text = text.replace("DECLINATION=5.7 degree", "DECLINATION=0.0 degree")
    tf = _import(tmp_path, text)
    d = tf.station_metadata.location.declination
    assert float(d.value) == 0.0
    assert d.model != "unknown"


def test_import_exotic_decl_line_no_hx_override(tmp_path: Path):
    """A declination KEY line the strict regex cannot parse: mt_metadata's
    parser owns it — the Hx AZM fallback must not silently override."""
    text = MINI_EDI.replace("DECL=5.7\n", "")
    text = text.replace(
        "DECLINATION=5.7 degree\n", "DECLINATION.WMM=0.0\n")
    tf = _import(tmp_path, text)
    d = tf.station_metadata.location.declination
    assert float(d.value) == 0.0


def test_import_hx_fallback_still_fires(tmp_path: Path):
    """No declination field at all + Hx AZM=5.7 -> the USArray-style
    fallback still supplies the declination (existing behaviour)."""
    text = MINI_EDI.replace("DECL=5.7\n", "")
    text = text.replace("DECLINATION=5.7 degree\n", "")
    tf = _import(tmp_path, text)
    assert float(tf.station_metadata.location.declination.value) == 5.7
