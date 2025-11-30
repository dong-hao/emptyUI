"""mth5 archive I/O nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emptyui import Input, Node, Output, Param


class LoadMth5(Node):
    type = "mth5.io.load"
    title = "Load MTH5"
    tooltip = "Open an MTH5 HDF5 archive as the data source."
    description = "Open an MTH5 HDF5 archive as the data source."
    category = "Data I/O"
    inputs = ()
    outputs = (Output("mth5", "mth5"),)
    params = (
        Param("path", "file", label="MTH5 file path"),
        Param("mode", "select", label="Mode",
              options=("r", "r+", "a", "w"), default="r"),
    )

    def execute(self, ctx, path: str, mode: str = "r") -> dict[str, Any]:
        from mth5 import mth5

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"MTH5 not found: {p}")
        archive = mth5.MTH5()
        archive.open_mth5(str(p), mode=mode)
        ctx.logger.info("opened %s (mode=%s)", p, mode)
        return {"mth5": archive}


class SaveMth5(Node):
    type = "mth5.io.save"
    title = "Save MTH5"
    tooltip = "Close (flush) the open MTH5 archive to disk."
    description = "Persist the archive (close/flush) and record its path."
    category = "Data I/O"
    inputs = (Input("mth5", "mth5"),)
    outputs = ()
    params = (
        Param("path", "file", label="Output MTH5 path", default="", optional=True),
    )

    def execute(self, ctx, mth5, path: str = "") -> dict[str, Any]:
        dest = Path(path) if path else None
        if dest is not None and str(dest) != mth5.filename:
            # archive is already an open mth5.MTH5; moving isn't supported yet
            raise ValueError(
                "moving an open archive is not supported yet — open with the "
                "target path via Load MTH5 in mode 'w'"
            )
        mth5.close_mth5()
        ctx.logger.info("saved MTH5 (closed)")
        return {}
