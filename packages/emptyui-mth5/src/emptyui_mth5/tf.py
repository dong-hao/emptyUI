"""Transfer-function I/O nodes (mt_metadata TF = mth5 TF, bidirectional)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emptyui import Input, Node, Output, Param


class ImportTF(Node):
    type = "mth5.tf.import"
    title = "Import TF"
    description = "Read a transfer function from file (edi/xml/zmm/avg/j)."
    category = "Data I/O"
    inputs = ()
    outputs = (Output("tf", "tf"),)
    params = (Param("path", "file", label="TF file path"),)

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
    description = "Write a transfer function to file (edi/xml/zmm/avg/j)."
    category = "Data I/O"
    inputs = (Input("tf", "tf"),)
    outputs = ()
    params = (
        Param("path", "file", label="Output path (or basename)", default=""),
        Param("file_type", "select", label="Format",
              options=("edi", "xml", "zmm", "avg", "j"), default="edi"),
    )

    def execute(self, ctx, tf, path: str = "", file_type: str = "edi") -> dict[str, Any]:
        out = Path(path) if path else Path.cwd()
        fn = tf.write(
            fn=str(out.parent / out.stem) if out.suffix else str(out),
            file_type=file_type,
        )
        ctx.progress(1.0, f"exported TF -> {fn}")
        return {}


def register(registry) -> None:
    registry.add(ImportTF)
    registry.add(ExportTF)
