"""Edge-case / error-surface tests for the emptyui framework.

Grouped (and marked ``errors``) so the happy-path suite can run fast with
``-m "not errors"``. A test lives here iff its primary assertion is
``pytest.raises(...)`` / an error status / degenerate-input behavior.
"""

from __future__ import annotations

import threading

import pytest

from emptyui.executor import ExecutionError, Executor
from emptyui.registry import Registry
from emptyui.schema import (
    Edge,
    ValidationError_,
    Workflow,
    WorkflowNode,
    validate_workflow,
)

pytestmark = pytest.mark.errors


def _registry():
    from emptyui import Input, Node, Output, Param

    reg = Registry()

    class Source(Node):
        type = "test.source"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            return {"data": [1.0, 2.0, 3.0]}

    class Consumer(Node):
        type = "test.consumer"
        inputs = (Input("data", "ts.run"),)

        def execute(self, ctx, data):
            return {}

    class Passthru(Node):
        type = "test.passthru"
        inputs = (Input("data", "ts.run"),)
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx, data):
            return {"data": data}

    class Boom(Node):
        type = "test.boom"
        inputs = (Input("data", "ts.run"),)

        def execute(self, ctx, data):
            raise ValueError("boom")

    class Scale(Node):
        type = "test.scale"
        inputs = (Input("data", "ts.run"),)
        outputs = (Output("scaled", "ts.run"),)
        params = (Param("factor", "number", default=2.0),)

        def execute(self, ctx, data, factor=2.0):
            return {"scaled": [v * factor for v in data]}

    for cls in (Source, Consumer, Passthru, Boom, Scale):
        reg.add(cls)
    return reg


# --- workflow validation errors (from test_schema.py) -------------------

def test_unknown_type_reported():
    wf = Workflow(nodes=[WorkflowNode(id="n1", type="nope.nope")], edges=[])
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, _registry())
    assert any("unknown type" in e for e in exc.value.errors)


def test_disabled_plugin_reported_as_disabled(monkeypatch):
    """A node from a disabled plugin gets an actionable message (§5.1) —
    distinct from "plugin not installed?"."""
    from types import SimpleNamespace

    from emptyui import Node
    from emptyui.registry import Registry, discover_plugins

    class Gone(Node):
        type = "gone.node"
        outputs = ()

        def execute(self, ctx):
            return {}

    def register(reg):
        reg.add(Gone)

    monkeypatch.setattr(
        "emptyui.registry.entry_points",
        lambda group=None: [
            SimpleNamespace(name="gone", load=lambda: register, dist=None)
        ],
    )
    reg = Registry()
    discover_plugins(reg, disabled={"gone"})
    wf = Workflow(nodes=[WorkflowNode(id="n1", type="gone.node")], edges=[])
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, reg)
    assert any("disabled" in e and "gone" in e for e in exc.value.errors)
    assert not any("plugin not installed" in e for e in exc.value.errors)


def test_type_mismatch_reported():
    from emptyui import Node, Output

    reg = _registry()

    class SpectraSrc(Node):
        type = "test.spectra_src"
        outputs = (Output("data", "spectra"),)

        def execute(self, ctx):
            return {}

    reg.add(SpectraSrc)
    wf = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.spectra_src"),
            WorkflowNode(id="n2", type="test.consumer"),
        ],
        edges=[Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data")],
    )
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, reg)
    assert any("type mismatch" in e for e in exc.value.errors)


def test_cycle_detected():
    wf = Workflow(
        nodes=[
            WorkflowNode(id="a", type="test.passthru"),
            WorkflowNode(id="b", type="test.passthru"),
        ],
        edges=[
            Edge(id="e1", source="a", source_port="data", target="b", target_port="data"),
            Edge(id="e2", source="b", source_port="data", target="a", target_port="data"),
        ],
    )
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, _registry())
    assert any("cycle" in e for e in exc.value.errors)


def test_required_input_must_be_connected():
    wf = Workflow(nodes=[WorkflowNode(id="n2", type="test.consumer")], edges=[])
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, _registry())
    assert any("required input" in e for e in exc.value.errors)


def test_fan_in_duplicate_reported():
    wf = Workflow(
        nodes=[
            WorkflowNode(id="a", type="test.source"),
            WorkflowNode(id="b", type="test.source"),
            WorkflowNode(id="c", type="test.consumer"),
        ],
        edges=[
            Edge(id="e1", source="a", source_port="data", target="c", target_port="data"),
            Edge(id="e2", source="b", source_port="data", target="c", target_port="data"),
        ],
    )
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, _registry())
    assert any("already has an incoming connection" in e for e in exc.value.errors)


def test_self_loop_reported():
    """A node's output connected to its own input gets an explicit message
    (not the generic cycle listing)."""
    wf = Workflow(
        nodes=[WorkflowNode(id="a", type="test.passthru")],
        edges=[Edge(id="e1", source="a", source_port="data", target="a", target_port="data")],
    )
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, _registry())
    assert any("cannot connect to itself" in e for e in exc.value.errors)
    # one problem -> one error (no duplicate generic cycle report)
    assert len(exc.value.errors) == 1


def test_executor_cycle_fails_loudly():
    """Direct executor.run (no validate_workflow) must fail on a cycle
    instead of silently skipping the cyclic nodes."""
    wf = Workflow(
        nodes=[
            WorkflowNode(id="a", type="test.passthru"),
            WorkflowNode(id="b", type="test.passthru"),
        ],
        edges=[
            Edge(id="e1", source="a", source_port="data", target="b", target_port="data"),
            Edge(id="e2", source="b", source_port="data", target="a", target_port="data"),
        ],
    )
    events = []
    with pytest.raises(ExecutionError, match="cycle"):
        Executor(_registry()).run(wf, emit=events.append)
    assert any(
        e["event"] == "run_finished" and e["status"] == "failed" for e in events
    )


# --- executor error surfaces (from test_executor.py) --------------------

def test_execution_error_surfaced():
    wf = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.source"),
            WorkflowNode(id="n2", type="test.boom"),
        ],
        edges=[Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data")],
    )
    try:
        Executor(_registry()).run(wf, emit=lambda e: None)
        raised = False
    except ExecutionError as exc:
        raised = True
        assert exc.node_id == "n2"
        assert isinstance(exc.original, ValueError)
    assert raised


def test_runts_payload_does_not_mask_node_error():
    """The failed-run sweep's duck-check for ``close_mth5`` must survive a
    payload whose __getattr__ converts a missing attribute into NameError
    (mt_timeseries' RunTS does exactly that): getattr(value, name, None)
    raises NameError there, escaping the sweep and MASKING the node's real
    error — a two-station remote-reference run surfaced as
    "RunTS has no attribute close_mth5" instead of its actual failure."""
    from emptyui import Input, Node, Output

    class FakeRunTS:
        def __getattr__(self, name):
            if name.startswith("_"):
                return
            raise NameError(f"RunTS has no attribute {name}")

    class TsSource(Node):
        type = "test.ts_source_runts"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            return {"data": FakeRunTS()}

    class TsBoom(Node):
        type = "test.ts_boom"
        inputs = (Input("data", "ts.run"),)
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx, data):
            raise ValueError("the real error")

    reg = _registry()
    reg.add(TsSource)
    reg.add(TsBoom)
    wf = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.ts_source_runts"),
            WorkflowNode(id="n2", type="test.ts_boom"),
        ],
        edges=[Edge(id="e1", source="n1", source_port="data",
                    target="n2", target_port="data")],
    )
    events = []
    with pytest.raises(ExecutionError, match="the real error"):
        Executor(reg).run(wf, emit=events.append)
    failed = [e for e in events
              if e["event"] == "node_status" and e.get("node") == "n2"]
    assert failed[-1]["status"] == "error"
    assert "the real error" in failed[-1]["error"]


def test_execution_cancelled_before_start():
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
    summary = Executor(_registry()).run(wf, emit=events.append, cancel_event=ev)
    assert summary["status"] == "cancelled"
    finished = next(e for e in events if e["event"] == "run_finished")
    assert finished["status"] == "cancelled"
    # no node ran
    assert not any(
        e["event"] == "node_status" and e["status"] == "running" for e in events
    )


# --- server error surfaces (from test_server.py) ------------------------

def test_invalid_workflow_returns_422(client):
    workflow = {
        "nodes": [{"id": "n1", "type": "core.util.nope"}],
        "edges": [],
    }
    resp = client.post("/api/runs", json={"workflow": workflow})
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "workflow",
    [
        {"nodes": [{"id": "n1"}], "edges": []},           # missing type
        {"nodes": "notalist", "edges": []},                # wrong nodes type
        {"nodes": [], "edges": "oops"},                    # wrong edges type
        {"nodes": [{"id": "n1", "type": "core.util.echo",
                    "position": "bad"}], "edges": []},     # wrong position type
    ],
)
def test_malformed_workflow_returns_422_not_500(client, workflow):
    """§6/§8: a structurally malformed document (pydantic envelope failure,
    not a semantic catalog mismatch) must surface as 422 with the field
    errors, not escape as an unhandled 500."""
    resp = client.post("/api/runs", json={"workflow": workflow})
    assert resp.status_code == 422
    assert resp.json()["detail"]


def test_cancel_unknown_run_404(client):
    assert client.post("/api/runs/nope/cancel").status_code == 404


def test_unknown_execution_target_returns_422(client):
    workflow = {
        "nodes": [{"id": "n1", "type": "core.util.echo"}],
        "edges": [],
    }
    resp = client.post("/api/runs", json={"workflow": workflow, "targets": ["n9"]})
    assert resp.status_code == 422


def test_executor_rejects_unknown_target():
    reg = Registry()
    from emptyui import Node

    class N(Node):
        type = "test.n"

        def execute(self, ctx):
            return {}

    reg.add(N)
    with pytest.raises(ValueError, match="unknown target"):
        Executor(reg).run(
            Workflow(nodes=[WorkflowNode(id="n1", type="test.n")]),
            emit=lambda e: None,
            targets=["ghost"],
        )


# --- pipeline error surfaces (from test_pipeline.py) --------------------

def test_load_csv_missing_file_fails_cleanly(client):
    workflow = {
        "nodes": [{"id": "n1", "type": "core.io.load_csv",
                   "params": {"path": "/nonexistent.csv"}}],
        "edges": [],
    }
    resp = client.post("/api/runs", json={"workflow": workflow})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    import time

    for _ in range(100):
        body = client.get(f"/api/runs/{run_id}").json()
        if body["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert body["status"] == "failed"
    err = next(e for e in body["events"]
               if e["event"] == "node_status" and e["status"] == "error")
    assert "CSV not found" in err["error"]
