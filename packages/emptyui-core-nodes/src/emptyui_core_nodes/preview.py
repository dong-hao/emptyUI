"""Framework nodes: inline preview viewer (frontend renders the figure)."""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output


class PreviewPlotNode(Node):
    type = "core.display.preview_plot"
    title = "Preview Plot"
    tooltip = "Render an incoming plot inline in the node body."
    description = (
        "Renders an incoming plot inside this node. Drag the node corners "
        "to resize the figure."
    )
    category = "Display"
    viewer = "plot"
    inputs = (Input("plot", "plot"),)
    outputs = (Output("plot", "plot"),)
    params = ()

    def execute(self, ctx, plot: dict) -> dict[str, Any]:
        # passthrough: rendering happens in the browser from the artifact
        return {"plot": plot}


