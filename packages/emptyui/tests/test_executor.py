from __future__ import annotations

import threading
import time

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


def test_execution_cancelled_before_start():
    reg = _registry()
    ev = threading.Event()
    ev.set()
    events = []
    wf = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.source"),
            WorkflowNode(id="n2", type="test.scale", params={"factor": 10.0}),
        ],
        edges=[Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data")],
    )
    summary = Executor(reg).run(wf, emit=events.append, cancel_event=ev)
    assert summary["status"] == "cancelled"
    finished = next(e for e in events if e["event"] == "run_finished")
    assert finished["status"] == "cancelled"
    # no node ran
    assert not any(
        e["event"] == "node_status" and e["status"] == "running" for e in events
    )


def test_execution_cancelled_mid_run():
    from emptyui import Input, Node

    class Slow(Node):
        type = "test.slow"
        inputs = (Input("data", "ts.run"),)
        outputs = ()

        def execute(self, ctx, data=None):
            time.sleep(0.3)
            return {}

    reg = _registry()
    reg.add(Slow)
    wf = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.source"),
            WorkflowNode(id="n2", type="test.slow"),
            WorkflowNode(id="n3", type="test.scale", params={"factor": 10.0}),
        ],
        edges=[
            Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data"),
            Edge(id="e2", source="n2", source_port="data", target="n3", target_port="data"),
        ],
    )
    ev = threading.Event()
    events = []
    summary_holder: dict[str, object] = {}

    def runner() -> None:
        summary_holder["summary"] = Executor(reg).run(wf, emit=events.append, cancel_event=ev)

    t = threading.Thread(target=runner)
    t.start()
    time.sleep(0.05)
    ev.set()
    t.join(timeout=5)
    assert "summary" in summary_holder
    assert summary_holder["summary"]["status"] == "cancelled"  # type: ignore[index]
    # n1+n2 completed before the cancel landed; n3 (the next node) was skipped
    statuses = {e["node"]: e["status"] for e in events if e["event"] == "node_status"}
    assert statuses.get("n1") == "done"
    assert statuses.get("n2") == "done"
    assert "n3" not in statuses
    finished = next(e for e in events if e["event"] == "run_finished")
    assert finished["status"] == "cancelled"
