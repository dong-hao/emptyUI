"""Workflow document validation — happy path (error cases in test_errors.py)."""

from emptyui.registry import Registry
from emptyui.schema import (
    Edge,
    Workflow,
    WorkflowNode,
    validate_workflow,
)


def _registry():
    from emptyui import Input, Node, Output

    reg = Registry()

    class Source(Node):
        type = "test.source"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            return {"data": [1, 2, 3]}

    class Consumer(Node):
        type = "test.consumer"
        inputs = (Input("data", "ts.run"),)

        def execute(self, ctx, data):
            return {}

    reg.add(Source)
    reg.add(Consumer)
    return reg


def test_valid_workflow_passes():
    wf = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.source"),
            WorkflowNode(id="n2", type="test.consumer"),
        ],
        edges=[Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data")],
    )
    validate_workflow(wf, _registry())  # no exception
