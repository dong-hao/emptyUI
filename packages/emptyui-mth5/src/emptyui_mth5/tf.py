"""Transfer-function I/O nodes (mt_metadata TF = mth5 TF, bidirectional)."""

from __future__ import annotations

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
        tf = TF.read(fn=str(p))
        ctx.progress(1.0, f"imported TF from {p.name}")
        return {"tf": tf}


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
    registry.add(ExportTF)
