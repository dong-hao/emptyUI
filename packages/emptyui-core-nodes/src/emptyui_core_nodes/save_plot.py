"""Plot export node — ComfyUI Save Image pattern (docs/DESIGN.md §3)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from emptyui import Input, Node, Param

SUPPORTED_FORMATS = ("png", "svg", "pdf")


class SavePlotNode(Node):
    type = "core.io.save_plot"
    title = "Save Plot"
    tooltip = "Export an incoming plot to a numbered PNG/SVG/PDF file."
    description = (
        "Export an incoming plot to the output directory. The file name is "
        "generated from the name prefix with an auto-incrementing counter "
        "(e.g. timeseries_001.png)."
    )
    # plots are figures, not data — Display category (ComfyUI's Save Image
    # pattern sits with the preview viewers, docs/DESIGN.md §3 Display nodes)
    category = "Display"
    terminal = True  # M13: manifest-declared run visibility
    inputs = (Input("plot", "plot"),)
    outputs = ()
    params = (
        Param("name_prefix", "string", label="Name prefix", default="plot"),
        Param("output_dir", "directory", label="Output directory",
              default="examples/output", path_kind="out"),
        Param("file_format", "select", label="Format",
              options=SUPPORTED_FORMATS, default="png"),
    )

    def execute(self, ctx, plot: dict, name_prefix: str = "plot",
                output_dir: str = "examples/output",
                file_format: str = "png") -> dict[str, Any]:
        if file_format not in SUPPORTED_FORMATS:
            raise ValueError(f"format must be one of {SUPPORTED_FORMATS}")

        src = Path(plot["path"])
        if not src.exists():
            raise FileNotFoundError(f"plot file missing: {src}")

        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        # the prefix feeds a glob scan and a filename — glob metacharacters
        # break the auto-increment scan and path separators would write
        # outside out_dir
        if re.search(r"[\[\]*?/\\]", name_prefix):
            raise ValueError(
                f"name prefix {name_prefix!r} contains glob metacharacters "
                "or path separators ([ ] * ? / \\) — use a plain name")

        # auto-increment: prefix_001.ext, prefix_002.ext, ...
        pattern = re.compile(rf"^{re.escape(name_prefix)}_(\d+)\.{file_format}$")
        next_idx = 1
        for existing in out_dir.glob(f"{name_prefix}_*.{file_format}"):
            m = pattern.match(existing.name)
            if m:
                next_idx = max(next_idx, int(m.group(1)) + 1)

        dest = out_dir / f"{name_prefix}_{next_idx:03d}.{file_format}"
        if src.suffix.lstrip(".") == file_format:
            shutil.copy(src, dest)
        else:
            # cross-format conversion needs a re-render (M2+); be explicit
            raise ValueError(
                f"source is {src.suffix!r} but requested {file_format!r}; "
                "same-format export only for now"
            )

        artifact = ctx.publish_artifact(dest, kind="image")
        ctx.progress(1.0, f"saved {dest.name}")
        return {"_saved": artifact}


