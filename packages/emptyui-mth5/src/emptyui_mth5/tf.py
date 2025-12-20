"""Transfer-function nodes: I/O (mt_metadata TF = mth5 TF, bidirectional)
and transforms (frame rotation)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from emptyui import Input, Node, Output, Param

from .nomenclature import STANDARD_INPUTS, STANDARD_OUTPUTS


def _to_standard_coords(tf):
    """Return a copy of the TF relabelled to standard coordinates.

    TFs built by emptyUI carry the *actual* instrument channel names as
    coordinates plus the ``channel_nomenclature`` map (aurora-style). The ZMM
    writer in mt_metadata 1.0.10 assumes standard names (its channel_dict is
    built from standard-named Channel objects), so exporting a LEMI-named TF
    to ZMM fails; this helper produces a standard-labelled copy for that
    writer. Station/survey metadata is carried over.
    """
    import xarray as xr
    from mt_metadata.transfer_functions.core import TF

    arr = tf.dataset["transfer_function"]
    inv = tf.inverse_channel_nomenclature  # actual -> standard
    out = [inv.get(str(c), str(c)) for c in arr.coords["output"].values]
    inp = [inv.get(str(c), str(c)) for c in arr.coords["input"].values]
    data = np.zeros(
        (arr.sizes["period"], len(STANDARD_OUTPUTS), len(STANDARD_INPUTS)),
        dtype=complex,
    )
    for oi, o in enumerate(out):
        if o not in STANDARD_OUTPUTS:
            continue
        for ii, i in enumerate(inp):
            if i not in STANDARD_INPUTS:
                continue
            data[:, STANDARD_OUTPUTS.index(o), STANDARD_INPUTS.index(i)] = (
                arr.isel(output=oi, input=ii).values
            )
    std = TF()
    std.transfer_function = xr.DataArray(
        data,
        dims=("period", "output", "input"),
        coords={
            "period": arr.coords["period"].values,
            "output": list(STANDARD_OUTPUTS),
            "input": list(STANDARD_INPUTS),
        },
    )
    # carry the error field over (same standard-coordinate remap) so ZMM/XML
    # exports keep the error/variance information.
    if "transfer_function_error" in tf.dataset:
        err = np.zeros(
            (arr.sizes["period"], len(STANDARD_OUTPUTS), len(STANDARD_INPUTS)),
            dtype=float,
        )
        err_arr = tf.dataset["transfer_function_error"]
        for oi, o in enumerate(out):
            if o not in STANDARD_OUTPUTS:
                continue
            for ii, i in enumerate(inp):
                if i not in STANDARD_INPUTS:
                    continue
                err[:, STANDARD_OUTPUTS.index(o), STANDARD_INPUTS.index(i)] = (
                    err_arr.isel(output=oi, input=ii).values
                )
        std.transfer_function_error = xr.DataArray(
            err,
            dims=("period", "output", "input"),
            coords={
                "period": arr.coords["period"].values,
                "output": list(STANDARD_OUTPUTS),
                "input": list(STANDARD_INPUTS),
            },
        )
    std.station_metadata = tf.station_metadata
    std.survey_metadata = tf.survey_metadata
    return std


#: Explicit EDI declination fields (HEAD/INFO; EDI formats are diverse — the
#: key may be DECL/DECLINATION, separator '=' or ':', optional 'deg(rees)').
_EDI_DECL_RE = re.compile(
    r"(?im)^\s*>?\s*decl(?:ination)?\s*[=:]\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?|°)?\s*$"
)

#: Hx measurement azimuth (DEFINEMEAS) — the sensor azimuth relative to true
#: north. EarthScope EMTF products record this as the station declination
#: (e.g. USArray SDG31: Hx AZM=5.7 -> declination 5.7).
_EDI_HX_AZM_RE = re.compile(
    r"(?im)^\s*>?\s*hmeas\b[^\n]*\b(?:azm|azim(?:uth)?)\s*=\s*([-+]?\d+(?:\.\d+)?)"
)


def _edi_declination(text: str) -> float | None:
    """Best-effort station declination (deg) from an EDI file.

    EDI declination conventions are not standardized, so this is a layered
    best effort: (1) an explicit Declination/Decl field (HEAD or INFO), (2)
    the Hx measurement azimuth as a fallback. Returns None when no field is
    present — the caller falls back to the angle parameter.
    """
    m = _EDI_DECL_RE.search(text)
    if m:
        return float(m.group(1))
    for line in text.splitlines():
        if re.match(r"(?i)^\s*>?\s*hmeas\b", line) and re.search(
            r"(?i)\bchtype\s*=\s*hx\b", line
        ):
            m = _EDI_HX_AZM_RE.search(line)
            if m:
                return float(m.group(1))
    return None


class ImportTF(Node):
    type = "mth5.tf.import"
    title = "Import TF"
    tooltip = "Read a transfer function from file (edi/xml/zmm/avg/j)."
    description = "Read a transfer function from file (edi/xml/zmm/avg/j)."
    category = "Data I/O"
    inputs = ()
    outputs = (Output("tf", "tf"),)
    params = (Param("path", "file", label="TF file path", path_kind="in"),)

    def execute(self, ctx, path: str) -> dict[str, Any]:
        from mt_metadata.transfer_functions.core import TF

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"TF file not found: {p}")
        tf = TF()
        tf.read(fn=str(p))
        if p.suffix.lower() == ".edi":
            decl = _edi_declination(p.read_text(errors="replace"))
            if decl is not None:
                d = tf.station_metadata.location.declination
                if not d.value:  # never override a value mt_metadata parsed
                    d.value = float(decl)
                    d.model = "unknown"
        ctx.progress(1.0, f"imported TF from {p.name}")
        return {"tf": tf}


def _rot_matrix(angle_deg: float) -> np.ndarray:
    """Standard MT frame-rotation matrix (DESIGN.md §16.3):
    R(angle) = [[cos α, sin α], [−sin α, cos α]], positive = clockwise from
    true north; magnetic → true north ⇒ angle = −declination."""
    a = np.radians(angle_deg)
    return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def _standard_indices(tf, std_outputs, std_inputs):
    """Resolve standard output/input positions from the TF's actual channel
    coords via the nomenclature map (None when a channel is absent)."""
    arr = tf.dataset["transfer_function"]
    inv = tf.inverse_channel_nomenclature

    def find(names, std):
        for i, n in enumerate(names):
            if inv.get(str(n), str(n)) == std:
                return i
        return None

    return (
        [find(arr.coords["output"].values, s) for s in std_outputs],
        [find(arr.coords["input"].values, s) for s in std_inputs],
    )


def _rotate_arrays(tf, angle_deg: float):
    """Rotate the impedance 2×2 block and tipper row of a TF dataset.

    ``Z' = R(angle) Z R(angle)ᵀ``; the tipper rotates as a vector
    ``T' = R(angle) T``; per-element error stds rotate with ``|R|`` (the
    §16.3-validated conservative bound — the stored errors carry no
    covariance). Every other entry (e.g. the zero rows of a 5×5 EDI TF) is
    untouched. Returns ``(transfer_function, transfer_function_error)`` as
    new arrays (the error is None when the TF has none).
    """
    arr = tf.dataset["transfer_function"]
    (ex_i, ey_i, hz_i), (hx_i, hy_i) = _standard_indices(
        tf, ("ex", "ey", "hz"), ("hx", "hy")
    )
    if None in (ex_i, ey_i, hx_i, hy_i):
        raise ValueError(
            "a TF with ex/ey outputs and hx/hy inputs is required to rotate"
        )
    R = _rot_matrix(angle_deg)
    data = arr.values.copy()
    z = data[:, [ex_i, ey_i], :][:, :, [hx_i, hy_i]]
    zr = np.einsum("ij,njk,lk->nil", R, z, R)
    data[:, ex_i, hx_i] = zr[:, 0, 0]
    data[:, ex_i, hy_i] = zr[:, 0, 1]
    data[:, ey_i, hx_i] = zr[:, 1, 0]
    data[:, ey_i, hy_i] = zr[:, 1, 1]
    if hz_i is not None:
        t = data[:, hz_i, [hx_i, hy_i]]
        tr = np.einsum("ij,nj->ni", R, t)
        data[:, hz_i, hx_i] = tr[:, 0]
        data[:, hz_i, hy_i] = tr[:, 1]

    err = None
    if "transfer_function_error" in tf.dataset:
        err = tf.dataset["transfer_function_error"].values.copy()
        Rabs = np.abs(R)
        e = err[:, [ex_i, ey_i], :][:, :, [hx_i, hy_i]]
        er = np.einsum("ij,njk,lk->nil", Rabs, e, Rabs)
        err[:, ex_i, hx_i] = er[:, 0, 0]
        err[:, ex_i, hy_i] = er[:, 0, 1]
        err[:, ey_i, hx_i] = er[:, 1, 0]
        err[:, ey_i, hy_i] = er[:, 1, 1]
        if hz_i is not None:
            et = err[:, hz_i, [hx_i, hy_i]]
            etr = np.einsum("ij,nj->ni", Rabs, et)
            err[:, hz_i, hx_i] = etr[:, 0]
            err[:, hz_i, hy_i] = etr[:, 1]
    return data, err


def rotate_transfer_function(tf, angle_deg: float):
    """Return a copy of ``tf`` rotated by ``angle_deg`` (positive = clockwise
    from true north; magnetic → true north ⇒ angle = −declination).

    Standard MT convention (DESIGN.md §16.3): ``Z' = R(angle) Z R(angle)ᵀ``,
    the tipper rotates as a vector, and per-element error stds rotate with
    ``|R|`` (conservative bound — no covariance is stored). Station/survey
    metadata is carried over unchanged; a one-line processing note is
    appended to the run comments.
    """
    import copy

    import xarray as xr

    out = copy.deepcopy(tf)
    arr = out.dataset["transfer_function"]
    data, err = _rotate_arrays(out, angle_deg)
    (ex_i, ey_i, hz_i), (hx_i, hy_i) = _standard_indices(
        out, ("ex", "ey", "hz"), ("hx", "hy")
    )
    if None in (ex_i, ey_i, hz_i, hx_i, hy_i):
        raise ValueError(
            "a TF with ex/ey/hz outputs and hx/hy inputs is required to rotate"
        )
    # mt_metadata's TF setter accepts only the 3x2 impedance(+tipper) block in
    # the nomenclature-mapped channel names (the canonical storage is 5x5).
    coords = {
        "period": arr.coords["period"].values,
        "output": [out.ex, out.ey, out.hz],
        "input": [out.hx, out.hy],
    }
    blk = data[:, [ex_i, ey_i, hz_i], :][:, :, [hx_i, hy_i]]
    out.transfer_function = xr.DataArray(
        blk, dims=("period", "output", "input"), coords=coords
    )
    if err is not None:
        eblk = err[:, [ex_i, ey_i, hz_i], :][:, :, [hx_i, hy_i]]
        out.transfer_function_error = xr.DataArray(
            eblk, dims=("period", "output", "input"), coords=coords
        )
    note = f"rotated {angle_deg:g} deg"
    existing = out.run_metadata.comments.value
    out.run_metadata.comments.value = f"{existing}\n{note}" if existing else note
    return out


class RotateTF(Node):
    type = "mth5.tf.rotate"
    title = "Rotate TF"
    tooltip = "Rotate a transfer function to another frame (e.g. geomagnetic to true north)."
    description = (
        "Rotate a transfer function by an arbitrary angle (or to true north "
        "using the station declination) before comparison/inversion. Standard "
        "MT convention (DESIGN.md §16.3): Z' = R(angle) Z R(angle)ᵀ with "
        "R(angle) = [[cos α, sin α], [−sin α, cos α]], angle positive = "
        "clockwise from true north; magnetic → true north ⇒ angle = "
        "−declination. The tipper rotates as a vector; per-element error stds "
        "rotate with |R| (conservative bound — no covariance is stored). "
        "Station metadata is carried over unchanged and a one-line processing "
        "note is appended."
    )
    category = "Processing"
    inputs = (Input("tf", "tf"),)
    outputs = (Output("tf", "tf"),)
    params = (
        Param("angle", "number", label="Angle (deg)", default=0.0,
              description="Rotation angle, positive = clockwise from true "
                          "north (e.g. −5.7 for the USArray declination). "
                          "Ignored when 'Use station declination' is on."),
        Param("use_declination", "boolean", label="Use station declination",
              default=False,
              description="Angle = −declination from the TF station metadata. "
                          "Declination is parsed from EDI on import when "
                          "present (see Import TF); falls back to 'Angle' "
                          "when the TF carries none.",
              cascade={True: (("angle", 0.0),)}),
    )

    def execute(self, ctx, tf, angle: float = 0.0,
                use_declination: bool = False) -> dict[str, Any]:
        if use_declination:
            decl = tf.station_metadata.location.declination.value
            if decl:
                angle = -float(decl)
            else:
                ctx.logger.warning(
                    "TF carries no declination metadata — using angle=%s", angle
                )
        return {"tf": rotate_transfer_function(tf, angle)}


class ExportTF(Node):
    type = "mth5.tf.export"
    title = "Export TF"
    tooltip = "Write a transfer function to file (edi/xml/zmm/avg/j)."
    description = "Write a transfer function to file (edi/xml/zmm/avg/j)."
    category = "Data I/O"
    inputs = (Input("tf", "tf"),)
    outputs = ()
    params = (
        Param("path", "file", label="Output path (or basename)", default="", path_kind="out"),
        Param("file_type", "select", label="Format",
              options=("edi", "xml", "zmm", "avg", "j"), default="edi"),
    )

    def execute(self, ctx, tf, path: str = "", file_type: str = "edi") -> dict[str, Any]:
        out = Path(path) if path else Path.cwd()
        if out.suffix:  # user gave a full filename with extension
            base, ext = out.with_suffix(""), out.suffix.lstrip(".")
            file_type = ext or file_type
            out = base
        export_tf = _to_standard_coords(tf) if file_type == "zmm" else tf
        saved = export_tf.write(
            save_dir=str(out.parent),
            fn_basename=out.name,
            file_type=file_type,
        )
        ctx.progress(1.0, f"exported TF -> {saved}")
        return {}


def register(registry) -> None:
    registry.add(ImportTF)
    registry.add(RotateTF)
    registry.add(ExportTF)
