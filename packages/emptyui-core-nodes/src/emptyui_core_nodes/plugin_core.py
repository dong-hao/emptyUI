"""Framework smoke-test node."""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param


class Echo(Node):
    type = "core.util.echo"
    title = "Echo"
    tooltip = "Framework smoke-test node that passes its input through unchanged."
    hidden = True
    description = "Passes its input through unchanged. Framework smoke test."
    category = "Processing"
    inputs = (Input("data", "any", required=False),)
    outputs = (Output("data", "any"),)
    params = (Param("label", "string", default="", optional=True),)

    def execute(self, ctx, data: Any = None, label: str = "") -> dict[str, Any]:
        if label:
            ctx.logger.info("echo: %s", label)
        return {"data": data}
