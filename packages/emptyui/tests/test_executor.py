"""Executor — happy path + cancellation feature (error surfaces in test_errors.py)."""

from __future__ import annotations

import threading
import time

from emptyui.executor import Executor
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


def test_unknown_node_type_fails_cleanly():
    """Direct executor.run callers (bypassing validate_workflow) get a loud
    ExecutionError for unknown types, not a bare AttributeError from a None
    manifest (2026-09-11 review fix)."""
    import pytest

    from emptyui.executor import ExecutionError

    reg = _registry()
    workflow = Workflow(
        nodes=[WorkflowNode(id="n1", type="no.such.type")], edges=[]
    )
    with pytest.raises(ExecutionError, match="unknown node type"):
        Executor(reg).run(workflow, emit=lambda e: None)


def test_string_number_params_are_coerced():
    """Old saved workflows (or free-text combobox entries) can carry numbers
    as strings — the executor coerces to the manifest's declared type before
    execute ('10' on a number param used to reach node code as str and
    break arithmetic, review 2026-09-12)."""
    from emptyui import Node, Output, Param

    reg = Registry()
    seen = {}

    class Num(Node):
        type = "test.num"
        outputs = (Output("ok", "any"),)
        params = (
            Param("factor", "number", default=2.0),
            Param("levels", "integer", default=1),
            Param("unit", "select", options=(1, 4, 8), default=4),
        )

        def execute(self, ctx, factor=2.0, levels=1, unit=4):
            seen["factor"] = factor
            seen["levels"] = levels
            seen["unit"] = unit
            return {"ok": True}

    reg.add(Num)
    workflow = Workflow(
        nodes=[WorkflowNode(id="n1", type="test.num", params={
            "factor": "10", "levels": "3", "unit": "8", "junk": "x",
        })],
        edges=[],
    )
    Executor(reg).run(workflow, emit=lambda e: None)
    assert seen["factor"] == 10.0 and not isinstance(seen["factor"], str)
    assert seen["levels"] == 3 and isinstance(seen["levels"], int)
    assert seen["unit"] == 8  # numeric select matched back by string form


def test_string_number_param_empty_string_becomes_none():
    from emptyui import Node, Output, Param

    reg = Registry()
    seen = {}

    class Num(Node):
        type = "test.num2"
        outputs = (Output("ok", "any"),)
        params = (Param("dipole", "number", default=None, optional=True),)

        def execute(self, ctx, dipole=None):
            seen["dipole"] = dipole
            return {"ok": True}

    reg.add(Num)
    workflow = Workflow(
        nodes=[WorkflowNode(id="n1", type="test.num2", params={"dipole": ""})],
        edges=[],
    )
    Executor(reg).run(workflow, emit=lambda e: None)
    assert seen["dipole"] is None


def test_string_number_param_invalid_raises_clean_error():
    import pytest

    from emptyui import Node, Param
    from emptyui.executor import ExecutionError

    reg = Registry()

    class Num(Node):
        type = "test.num3"
        inputs = ()
        outputs = ()
        params = (Param("factor", "number", default=2.0),)

        def execute(self, ctx, factor=2.0):
            return {}

    reg.add(Num)
    workflow = Workflow(
        nodes=[WorkflowNode(id="n1", type="test.num3", params={"factor": "ten"})],
        edges=[],
    )
    with pytest.raises(ExecutionError, match="'factor'"):
        Executor(reg).run(workflow, emit=lambda e: None)
