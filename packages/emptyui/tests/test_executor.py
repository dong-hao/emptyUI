from emptyui.executor import ExecutionError, Executor
from emptyui.registry import Registry
from emptyui.schema import Edge, Workflow, WorkflowNode


def _registry():
    from emptyui import Input, Node, Output, Param

    reg = Registry()

    class Source(Node):
        type = "test.source"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            return {"data": [1.0, 2.0, 3.0]}

    class Scale(Node):
        type = "test.scale"
        inputs = (Input("data", "ts.run"),)
        outputs = (Output("scaled", "ts.run"),)
        params = (Param("factor", "number", default=2.0),)

        def execute(self, ctx, data, factor=2.0):
            return {"scaled": [v * factor for v in data]}

    reg.add(Source)
    reg.add(Scale)
    return reg


def test_execution_data_flows_through():
    reg = _registry()
    ex = Executor(reg)
    events = []
    wf = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.source"),
            WorkflowNode(id="n2", type="test.scale", params={"factor": 10.0}),
        ],
        edges=[Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data")],
    )
    summary = ex.run(wf, emit=events.append)
    assert summary["status"] == "completed"
    final_statuses = {}
    for e in events:
        if e["event"] == "node_status":
            final_statuses[e["node"]] = e["status"]
    assert final_statuses == {"n1": "done", "n2": "done"}
    finished = next(e for e in events if e["event"] == "run_finished")
    assert finished["status"] == "completed"


def test_execution_error_surfaced():
    from emptyui import Input, Node

    class Boom(Node):
        type = "test.boom"
        inputs = (Input("data", "ts.run"),)

        def execute(self, ctx, data):
            raise ValueError("boom")

    reg = _registry()
    reg.add(Boom)
    wf = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.source"),
            WorkflowNode(id="n2", type="test.boom"),
        ],
        edges=[Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data")],
    )
    try:
        Executor(reg).run(wf, emit=lambda e: None)
        raised = False
    except ExecutionError as exc:
        raised = True
        assert exc.node_id == "n2"
        assert isinstance(exc.original, ValueError)
    assert raised
