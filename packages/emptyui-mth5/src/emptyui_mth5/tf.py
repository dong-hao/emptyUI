"""Transfer-function nodes: I/O (mt_metadata TF = mth5 TF, bidirectional)
and transforms (frame rotation)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from emptyui import Input, Node, Output, Param

from .nomenclature import STANDARD_INPUTS, STANDARD_OUTPUTS

MU0 = 4e-7 * np.pi

#: Impedance unit bridge between the file formats (SEG/EMTF convention,
#: [mV/km]/[nT] — the EDI standard, EMTF XML ``_units_dict``, the Zonge avg
#: ``0.2|Z|^2/f`` formula and the EMTF j/z-files) and emptyUI's internal
#: SI convention (E in V/m, H in A/m): E mV/km -> V/m is x1e-6, H nT -> A/m
#: is x1e-9/mu0. mt_metadata's readers/writers copy impedance values
#: verbatim (verified: no conversion in edi/xml/avg/j/zmm), so emptyUI
#: converts at the format boundary — importing EDI values un-stored leaves
#: apparent resistivity inflated by (1/1.2566e-3)^2 ~= 6.33e5.
EDI_TO_SI = 1e-6 * MU0 / 1e-9
SI_TO_EDI = 1.0 / EDI_TO_SI


def _scaled_impedance_copy(tf, factor: float):
    """Return a deep copy of ``tf`` with the impedance block (ex/ey output
    rows) and its error scaled by ``factor`` — the SI <-> [mV/km]/[nT] unit
    bridge at the file-format boundary. The tipper (hz row) is dimensionless
    and untouched. Works on a copy: TF payloads are shared/cached, mutating
    an input violates the executor's immutability contract.
    """
    import copy

    out = copy.deepcopy(tf)
    arr = out.dataset["transfer_function"]
    ex, ey = out.ex, out.ey
    rows = [
        i for i, c in enumerate(arr.coords["output"].values)
        if str(c) in (ex, ey)
    ]
    if rows:
        vals = arr.values.copy()
        vals[:, rows, :] *= factor
        arr.values = vals
        if "transfer_function_error" in out.dataset:
            err = out.dataset["transfer_function_error"].values.copy()
            err[:, rows, :] *= abs(factor)
            out.dataset["transfer_function_error"].values = err
    return out


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

#: A declination KEY line in any format variant (HEAD or INFO): the line
#: merely has to START with a decl key — values may be multi-field, quoted,
#: or commented, which the strict value regex cannot parse. Presence of such
#: a line means mt_metadata's own parser already handled the field.
_EDI_DECL_KEY_RE = re.compile(r"(?im)^\s*>?\s*decl(?:ination)?\b")

#: EDI SIGNCONVENTION field (HEAD): e.g. ``SIGNCONVENTION=exp(+ i\omega t)``.
_EDI_SIGNCONVENTION_RE = re.compile(
    r"(?im)^\s*signconvention\s*=\s*(.+?)\s*$"
)


def _edi_decl_plus_family(value: str) -> bool:
    """Classify a SIGNCONVENTION value into the exp(+iωt) family ('+' or
    'exp(+ i omega t)' spellings); anything else ('-', 'exp(-iwt)', …) is
    the exp(−iωt) family."""
    return "+" in value


def _hx_azimuth(text: str) -> float | None:
    """The Hx measurement azimuth (deg) from DEFINEMEAS, or None."""
    for line in text.splitlines():
        if re.match(r"(?i)^\s*>?\s*hmeas\b", line) and re.search(
            r"(?i)\bchtype\s*=\s*hx\b", line
        ):
            m = _EDI_HX_AZM_RE.search(line)
            if m:
                return float(m.group(1))
    return None


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
    return _hx_azimuth(text)


class ImportTF(Node):
    type = "mth5.tf.import"
    title = "Import TF"
    tooltip = "Read a transfer function from file (edi/xml/zmm/avg/j)."
    description = (
        "Read a transfer function from file (edi/xml/zmm/avg/j). For EDI, "
        "the station declination is parsed (explicit Declination/Decl field, "
        "else the Hx measurement azimuth; when neither is present the "
        "'Angle' parameter of Rotate TF applies). Sign conventions in the "
        "EDI world are NOT reliably self-describing: the official USArray "
        "EMTF EDI declares exp(+i omega t) but its data already matches "
        "emptyUI's exp(-i omega t) convention as-is (history §16.3), while "
        "Phoenix MT-Editor EDIs declare nothing and need the 180-deg "
        "negation (history §24.12). 'Sign convention' = as-is imports the "
        "file untouched (default); the negation/conjugation options convert "
        "the data (a processing note records the conversion); the "
        "SIGNCONVENTION header field, when present, is checked with a "
        "warning since it may not match the data."
    )
    category = "Data I/O"
    inputs = ()
    outputs = (Output("tf", "tf"),)
    params = (
        Param("path", "file", label="TF file path", path_kind="in"),
        Param("impedance_units", "select", label="Impedance units in file",
              default="SEG/EMTF ([mV/km]/[nT])",
              options=("SEG/EMTF ([mV/km]/[nT])", "SI (V/m, A/m)"),
              description="The convention the FILE stores the impedance "
                          "in. Every supported format stores SEG/EMTF "
                          "([mV/km]/[nT]) — the default converts it to "
                          "emptyUI's internal SI on import (issues.md L8: "
                          "a file already stored in SI would otherwise be "
                          "double-scaled with no way to say so)."),
        Param("sign_convention", "select", label="Sign convention",
              default="as-is",
              options=("as-is", "negate (180 deg)",
                       "conjugate (exp(+i omega t))"),
              description="Convert the imported TF to emptyUI's exp(-i "
                          "omega t) convention. 'negate' multiplies Z and "
                          "the tipper by -1 (the empirically verified "
                          "conversion for Phoenix MT-Editor EDIs, history §24.12); "
                          "'conjugate' flips the sign of the imaginary "
                          "parts (the literal exp(+i omega t) -> exp(-i "
                          "omega t) mapping). EDI headers cannot be "
                          "trusted to distinguish the cases — see the "
                          "USArray counter-example."),
    )

    def execute(self, ctx, path: str, impedance_units: str = "SEG/EMTF ([mV/km]/[nT])",
                sign_convention: str = "as-is") -> dict[str, Any]:
        from mt_metadata.transfer_functions.core import TF

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"TF file not found: {p}")
        tf = TF()
        tf.read(fn=str(p))
        notes = []
        # all supported formats store the impedance in the SEG/EMTF
        # convention ([mV/km]/[nT]) while emptyUI's internal TFs are SI —
        # mt_metadata copies the values verbatim, so convert here (the
        # tipper is dimensionless and untouched). L8: a file already
        # stored in SI can be imported verbatim (the opt-out; the default
        # is unchanged).
        if impedance_units.startswith("SI"):
            notes.append(
                "impedance declared SI (V/m, A/m) — imported verbatim, no "
                "unit conversion")
        else:
            tf = _scaled_impedance_copy(tf, EDI_TO_SI)
            notes.append(
                "impedance units converted [mV/km]/[nT] -> SI "
                f"(x{EDI_TO_SI:.6g} on import)"
            )
        if p.suffix.lower() == ".edi":
            text = p.read_text(errors="replace")
            m = _EDI_SIGNCONVENTION_RE.search(text)
            if m and _edi_decl_plus_family(m.group(1)) and (
                    sign_convention == "as-is"):
                ctx.logger.warning(
                    "the EDI declares SIGNCONVENTION '%s' (exp(+i omega t) "
                    "family) but EDI headers are not reliably "
                    "self-describing: the official USArray EMTF EDI declares "
                    "it while its data already matches the exp(-i omega t) "
                    "convention as-is. If phases appear mirrored, set the "
                    "'Sign convention' param explicitly.", m.group(1))
            if sign_convention.startswith("negate"):
                tf.dataset["transfer_function"].values *= -1.0
                notes.append("sign convention: negated (180 deg) on import")
            elif sign_convention.startswith("conjugate"):
                tf.dataset["transfer_function"].values = np.conj(
                    tf.dataset["transfer_function"].values)
                notes.append(
                    "sign convention: conjugated (exp(+i omega t) -> "
                    "exp(-i omega t)) on import")
            m = _EDI_DECL_RE.search(text)
            d = tf.station_metadata.location.declination
            if m:
                # an explicit Decl field (strictly parsed) is authoritative —
                # including an explicit 0.0 (M7: 'not d.value' cannot
                # distinguish absent from explicit zero)
                decl = float(m.group(1))
                if decl != d.value:
                    d.value = decl
                    d.model = "unknown"
            elif _EDI_DECL_KEY_RE.search(text):
                # a declination KEY line exists in a format the strict
                # regex cannot parse — mt_metadata's own parser already
                # handled it; never override, never fall back to Hx
                pass
            elif not d.value:
                # no declination field anywhere: fall back to the Hx
                # measurement azimuth (USArray EMTF convention)
                hx = _hx_azimuth(text)
                if hx is not None:
                    d.value = float(hx)
                    d.model = "unknown"
        if notes:
            existing = tf.run_metadata.comments.value
            add = "\n".join(notes)
            tf.run_metadata.comments.value = (
                f"{existing}\n{add}" if existing else add)
        ctx.progress(1.0, f"imported TF from {p.name}")
        return {"tf": tf}


def _rot_matrix(angle_deg: float) -> np.ndarray:
    """Standard MT frame-rotation matrix (history §16.3):
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
    history §16.3-validated conservative bound — the stored errors carry no
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

    Standard MT convention (history §16.3): ``Z' = R(angle) Z R(angle)ᵀ``,
    the tipper rotates as a vector, and per-element error stds rotate with
    ``|R|`` (conservative bound — no covariance is stored). A tipper-less
    TF rotates fine (only the impedance block is rotated; mt_metadata's
    canonical storage zero-fills the hz row). Station/survey metadata is
    carried over unchanged; a one-line processing note is appended to the
    run comments.
    """
    import copy

    import xarray as xr

    out = copy.deepcopy(tf)
    arr = out.dataset["transfer_function"]
    data, err = _rotate_arrays(out, angle_deg)
    (ex_i, ey_i, hz_i), (hx_i, hy_i) = _standard_indices(
        out, ("ex", "ey", "hz"), ("hx", "hy")
    )
    if None in (ex_i, ey_i, hx_i, hy_i):
        raise ValueError(
            "a TF with ex/ey outputs and hx/hy inputs is required to rotate"
        )
    # mt_metadata's TF setter accepts only the 3x2 impedance(+tipper) block in
    # the nomenclature-mapped channel names (the canonical storage is 5x5).
    # A tipper-less TF (e.g. impedance-only formats) is stored with a
    # zero-filled hz row — mt_metadata's own representation of an absent
    # tipper; the rotation leaves it zero (nothing to rotate).
    coords = {
        "period": arr.coords["period"].values,
        "output": [out.ex, out.ey, out.hz],
        "input": [out.hx, out.hy],
    }
    if hz_i is None:
        z2 = data[:, [ex_i, ey_i], :][:, :, [hx_i, hy_i]]
        blk = np.concatenate(
            [z2, np.zeros_like(z2[:, :1, :])], axis=1)
    else:
        blk = data[:, [ex_i, ey_i, hz_i], :][:, :, [hx_i, hy_i]]
    out.transfer_function = xr.DataArray(
        blk, dims=("period", "output", "input"), coords=coords
    )
    if err is not None:
        if hz_i is None:
            e2 = err[:, [ex_i, ey_i], :][:, :, [hx_i, hy_i]]
            eblk = np.concatenate(
                [e2, np.zeros_like(e2[:, :1, :])], axis=1)
        else:
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
        "MT convention (history §16.3): Z' = R(angle) Z R(angle)ᵀ with "
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
    #: the side effect IS writing the file — never served from the
    #: structural cache (a hit would report "done (cached)" with no file
    #: written, §7.1); re-execution is cheap (the tf payload is cached)
    cacheable = False
    params = (
        Param("path", "file", label="Output path (or basename)", default="", path_kind="out"),
        Param("file_type", "select", label="Format",
              options=("edi", "xml", "zmm", "avg", "j"), default="edi"),
    )

    def execute(self, ctx, tf, path: str = "", file_type: str = "edi") -> dict[str, Any]:
        if path:
            out = Path(path)
        else:
            # empty path: write into the run workspace next to the artifacts
            # (Path.cwd() made save_dir=cwd.parent / fn_basename=cwd.name —
            # the file landed OUTSIDE the cwd; review 2026-09-24). Without a
            # workspace artifact_path yields a relative basename.
            out = Path(ctx.artifact_path("tf_export", ""))
        if out.suffix:  # user gave a full filename with extension
            base, ext = out.with_suffix(""), out.suffix.lstrip(".")
            file_type = ext or file_type
            out = base
        export_tf = _to_standard_coords(tf) if file_type == "zmm" else tf
        # file formats store the impedance in the SEG/EMTF convention
        # ([mV/km]/[nT]); internal TFs are SI — convert on a copy so the
        # input payload (shared/cached) is untouched
        export_tf = _scaled_impedance_copy(export_tf, SI_TO_EDI)
        saved = export_tf.write(
            save_dir=str(out.parent),
            fn_basename=out.name,
            file_type=file_type,
        )
        ctx.progress(1.0, f"exported TF -> {saved}")
        return {}
