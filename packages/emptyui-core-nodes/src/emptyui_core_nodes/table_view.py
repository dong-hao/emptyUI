"""Framework nodes: inline table viewer (frontend renders the table)."""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output


class ViewTableNode(Node):
    type = "core.display.view_table"
    title = "View Table"
    tooltip = "Render an incoming table artifact inline in the node body."
    description = (
        "Renders an incoming table inside this node. Drag the node corners "
        "to resize."
    )
    category = "Display"
    inputs = (Input("table", "table"),)
    outputs = (Output("table", "table"),)
    params = ()

    def execute(self, ctx, table: dict) -> dict[str, Any]:
        # passthrough: rendering happens in the browser from the artifact
        return {"table": table}


def register(registry) -> None:
    registry.add(ViewTableNode)
