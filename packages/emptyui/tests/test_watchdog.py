"""M2 — the per-node watchdog (run deadline).

One hung node used to wedge the whole instance: the admission slot was only
released in the worker's ``finally``, cancellation is polled between nodes
(never mid-node), and TTL cleanup refuses non-worker-done runs — with
``max_concurrent_runs=1`` a single stuck node returned 429 forever.

The contract (issues.md M2, user decision 2026-10-13): a **default
deadline, editable in Settings, enforced per node**. Each node gets the
full budget (reset at node boundaries), so a long pipeline of quick nodes
is never killed — only a single stuck node is. The deadline is enforced by
running the node on a daemon thread and abandoning it on expiry (Python
cannot kill threads; the leaked thread dies with the process). The run
ends with a ``timeout`` terminal status whose ``error`` names the node,
the budget and the remedy (H3: a terminal state must explain itself).

Per-node override: ``Node.timeout`` — ``None`` inherits the server knob,
``0`` opts out (a node that knows better), ``> 0`` is its own budget. The
heavy raw-data/FC nodes declare 3600 s so genuinely long computations are
not killed by the 600 s default.
"""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from emptyui.executor import Executor
from emptyui.registry import Registry, build_default_registry
from emptyui.schema import Workflow, WorkflowNode

_HANG = threading.Event()  # never set — the stand-in for a stuck node


def _registry(extra=()):
    from emptyui import Input, Node, Output

    reg = Registry()

    class Hang(Node):
        type = "test.hang"
        outputs = (Output("data", "any"),)

        def execute(self, ctx):
            _HANG.wait()  # blocks forever — the wedge M2 describes
            return {"data": "unreachable"}

    class Downstream(Node):
        type = "test.downstream"
        inputs = (Input("data", "any"),)
        outputs = (Output("data", "any"),)

        def execute(self, ctx, data):
            return {"data": data}

    reg.add(Hang)
    reg.add(Downstream)
    for node in extra:
        reg.add(node)
    return reg


def _wf(*types):
    ids = [f"n{i + 1}" for i in range(len(types))]
    return Workflow(
        nodes=[WorkflowNode(id=nid, type=t) for nid, t in zip(ids, types)],
        edges=[
            {"id": f"e{i}", "source": ids[i], "source_port": "data",
             "target": ids[i + 1], "target_port": "data"}
            for i in range(len(ids) - 1)
        ],
    )


def _run(workflow, registry, **kw):
    events: list[dict] = []
    summary = Executor(registry).run(
        workflow, emit=events.append, **kw)
    return summary, events


def test_hung_node_times_out_the_run():
    summary, events = _run(
        _wf("test.hang", "test.downstream"), _registry(),
        node_timeout_seconds=0.1,
    )
    assert summary["status"] == "timeout"
    assert summary["node"] == "n1"
    assert "n1" in summary["error"] and "deadline" in summary["error"]
    finished = [e for e in events if e["event"] == "run_finished"]
    assert finished and finished[-1]["status"] == "timeout"
    assert finished[-1].get("node") == "n1"
    assert finished[-1].get("error"), "H3: a terminal state must explain itself"
    # the node after the hung one never started
    assert not any(
        e["event"] == "node_status" and e["node"] == "n2"
        for e in events
    )


def test_timed_out_run_ends_promptly():
    # the wedge is bounded: the executor returns without waiting for the
    # abandoned thread (the deadline is the wait, not the node's runtime)
    hung = _registry()
    start = time.monotonic()
    _run(_wf("test.hang"), hung, node_timeout_seconds=0.2)
    assert time.monotonic() - start < 5.0


def test_node_declared_zero_timeout_opts_out():
    """A node that declares timeout=0 is never watched — its own budget
    beats the server knob (the escape hatch for genuinely long work)."""

    from emptyui import Node, Output

    class SlowButLegit(Node):
        type = "test.slow_legit"
        timeout = 0.0  # off: this node knows its workload
        outputs = (Output("data", "any"),)

        def execute(self, ctx):
            time.sleep(0.4)  # >> the 0.1 s server knob
            return {"data": "done"}

    summary, _ = _run(
        _wf("test.slow_legit"), _registry([SlowButLegit]),
        node_timeout_seconds=0.1,
    )
    assert summary["status"] == "completed"


def test_node_declared_timeout_overrides_unset_knob():
    """A node may declare its own (shorter) budget even when the server
    knob is off."""

    from emptyui import Node, Output

    class SelfLimited(Node):
        type = "test.self_limited"
        timeout = 0.2
        outputs = (Output("data", "any"),)

        def execute(self, ctx):
            _HANG.wait()
            return {"data": "unreachable"}

    summary, _ = _run(
        _wf("test.self_limited"), _registry([SelfLimited]),
        node_timeout_seconds=None,
    )
    assert summary["status"] == "timeout"


def test_unset_knob_runs_inline_no_watchdog():
    """node_timeout_seconds=None (headless/CLI default) keeps the exact
    pre-watchdog behavior: no deadline, no thread."""

    from emptyui import Node, Output

    class Quick(Node):
        type = "test.quick"
        outputs = (Output("data", "any"),)

        def execute(self, ctx):
            time.sleep(0.3)
            return {"data": "done"}

    summary, _ = _run(_wf("test.quick"), _registry([Quick]))
    assert summary["status"] == "completed"


def test_abandoned_context_ignores_late_progress():
    """The abandoned thread keeps running — its late ctx calls must not
    emit events for a run that already ended."""

    from emptyui.context import RunContext

    calls: list[tuple] = []
    ctx = RunContext(workspace=None, store=None,
                     progress_cb=lambda f, m: calls.append((f, m)),
                     node_id="n1")
    ctx.abandoned = True
    ctx.progress(0.5, "late")
    assert calls == []


def _client(node_timeout_seconds=None, **kw):
    from emptyui.server.app import create_app

    reg = build_default_registry()
    reg.add(_make_hang_node())
    return TestClient(create_app(
        reg, node_timeout_seconds=node_timeout_seconds, **kw))


def _make_hang_node():
    from emptyui import Node, Output

    class ServerHang(Node):
        type = "test.server_hang"
        outputs = (Output("data", "any"),)

        def execute(self, ctx):
            _HANG.wait()
            return {"data": "unreachable"}

    return ServerHang


def _wait_status(client, run_id, status, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body.get("status") == status:
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached {status!r}: {body}")


def test_timeout_releases_the_admission_slot():
    """The wedge, end-to-end: with max_concurrent_runs=1 a hung node used
    to 429 every later run forever. After the timeout the slot is free."""
    client = _client(node_timeout_seconds=0.2, max_concurrent_runs=1)
    first = client.post("/api/runs", json={
        "workflow": {"nodes": [{"id": "n1", "type": "test.server_hang"}],
                     "edges": []},
    })
    assert first.status_code == 200
    body = _wait_status(client, first.json()["run_id"], "timeout")
    assert "n1" in body["error"]

    second = client.post("/api/runs", json={
        "workflow": {"nodes": [{"id": "n1", "type": "test.server_hang"}],
                     "edges": []},
    })
    assert second.status_code == 200, "the slot was not released"


def test_settings_knob_default_edit_and_validation():
    client = _client()
    view = client.get("/api/settings").json()
    knob = view["live"]["node_timeout_seconds"]
    assert knob["value"] == 600.0
    assert knob["source"] == "default"
    assert knob["editable"] is True

    resp = client.put("/api/settings", json={"values": {"node_timeout_seconds": 300}})
    assert resp.status_code == 200
    knob = resp.json()["live"]["node_timeout_seconds"]
    assert knob["value"] == 300.0
    assert knob["source"] == "ui"

    # 0 = off is allowed; null is not (a knob with a non-null default)
    assert client.put(
        "/api/settings", json={"values": {"node_timeout_seconds": 0}}
    ).status_code == 200
    null_resp = client.put(
        "/api/settings", json={"values": {"node_timeout_seconds": None}}
    )
    assert null_resp.status_code == 422
    assert "does not accept null" in null_resp.json()["detail"]

    bad = client.put(
        "/api/settings", json={"values": {"node_timeout_seconds": -5}}
    )
    assert bad.status_code == 422


def test_platform_baselined_knob_is_not_ui_editable():
    client = _client(node_timeout_seconds=120.0)
    view = client.get("/api/settings").json()
    assert view["live"]["node_timeout_seconds"]["value"] == 120.0
    assert view["live"]["node_timeout_seconds"]["source"] == "platform"
    resp = client.put(
        "/api/settings", json={"values": {"node_timeout_seconds": 300}}
    )
    assert resp.status_code == 409
    assert "node_timeout_seconds" in resp.json()["detail"]
