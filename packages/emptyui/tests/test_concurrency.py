"""M16: concurrency + multi-client WebSocket delivery (design approved
2026-10-16, options 1+3).

Three hazards the pre-existing suite could not see:

- H-A  one shared ``Node`` instance per type (``registry.py`` -- the
  table entry IS the instance, ``manifest.execute`` runs on it): two
  concurrent runs executing the same node must not cross instance state;
- H-B  the cache's signature->entry mapping under concurrent and
  repeated runs;
- H-C/H-D  WS fan-out must reach EVERY connected client, and a
  mid-run disconnect must not kill the hub or starve the survivors
  (the multi-user case: one tab closing must not degrade another).

Deterministic throughout: no sleeps for correctness (only the test
nodes' own durations), bounded queue timeouts, ``_wait_status``-style
polling. The barrier stress variant was rejected as timing-sensitive.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time

import pytest
from fastapi.testclient import TestClient

from emptyui import Node, Output, Param
from emptyui.cache import NodeCache
from emptyui.executor import Executor
from emptyui.registry import Registry, build_default_registry
from emptyui.schema import Workflow, WorkflowNode
from emptyui.server.app import create_app

pytestmark = pytest.mark.filterwarnings("ignore")


# --- shared helpers -------------------------------------------------------

def _pump_ws(ws, sink: queue.Queue) -> None:
    """Forward every WebSocket message to ``sink`` until the socket closes
    (same shape as test_server's helper -- tests are not a package, so the
    8 lines are duplicated rather than imported)."""
    try:
        while True:
            sink.put(ws.receive_json())
    except Exception:  # noqa: BLE001 -- the session closed; nothing more
        return


def _drain_until(sink: queue.Queue, pred, timeout: float = 5.0) -> list[dict]:
    """Collect messages until ``pred`` holds for the stream, bounded."""
    seen: list[dict] = []
    deadline = time.monotonic() + timeout
    while not pred(seen):
        try:
            seen.append(sink.get(timeout=max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            raise AssertionError(
                "the client missed its event stream") from None
    return seen


def _finished(seen: list[dict]) -> bool:
    return any(e.get("event") == "run_finished" for e in seen)


# --- H-A/H-B: the shared-instance canary (executor level) ------------------

def _canary_registry() -> Registry:
    reg = Registry()

    class Canary(Node):
        # the canary is STACK-LOCAL: every value it emits comes from its
        # own execute frame. Measured during the design review: a node
        # that writes an instance attribute and reads it back after a
        # yield point DOES cross under concurrent runs -- the executor
        # runs the shared instance thread-per-run by design, so nodes
        # must be stateless (recorded in the M16 register entry).
        type = "test.canary"
        params = (Param("value", "number", default=0.0),)
        outputs = (Output("value", "any"),)

        def execute(self, ctx, value: float = 0.0):
            time.sleep(0.05)  # widen the interleaving window
            ctx.warn(f"canary saw {value}")
            return {"value": value}

    reg.add(Canary)
    return reg


def _canary_wf(value: float) -> Workflow:
    return Workflow(nodes=[WorkflowNode(
        id="n1", type="test.canary", params={"value": value})])


def _saw(events: list[dict]) -> list[str]:
    return [e["msg"] for e in events
            if e["event"] == "node_warning"
            and "canary saw" in e.get("msg", "")]


def test_concurrent_runs_through_one_shared_node_stay_correct(tmp_path):
    """H-A: two concurrent runs execute the SAME shared node instance
    (the registry's singleton) -- each run's warnings and outputs carry
    only its own value. The mutant: an executor that mixed runs' frames
    (or a node leaning on instance state) emits the other run's value."""
    reg = _canary_registry()
    cache = NodeCache(max_entries=8)
    ex = Executor(reg)  # ONE instance, like the server's
    results: dict[str, tuple[str, list[str]]] = {}

    def run(tag: str, value: float) -> None:
        events: list[dict] = []
        summary = ex.run(_canary_wf(value), emit=events.append,
                         workspace=tmp_path / tag, cache=cache)
        results[tag] = (summary["status"], _saw(events))

    threads = [threading.Thread(target=run, args=("a", 1.0)),
               threading.Thread(target=run, args=("b", 2.0))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert all(not t.is_alive() for t in threads), "a run hung"
    assert results["a"][0] == results["b"][0] == "completed"
    assert results["a"][1] and all(m == "canary saw 1.0" for m in results["a"][1]), results["a"][1]
    assert results["b"][1] and all(m == "canary saw 2.0" for m in results["b"][1]), results["b"][1]


def test_identical_rerun_hits_the_cache(tmp_path):
    """H-B: the second identical run restores its node from the cache
    (``cached`` on the node_status event) with the same outcome."""
    reg = _canary_registry()
    cache = NodeCache(max_entries=8)
    ex = Executor(reg)
    ev1: list[dict] = []
    ev2: list[dict] = []
    s1 = ex.run(_canary_wf(7.0), emit=ev1.append, workspace=tmp_path,
                cache=cache)
    s2 = ex.run(_canary_wf(7.0), emit=ev2.append, workspace=tmp_path,
                cache=cache)
    assert s1["status"] == s2["status"] == "completed"
    cached = [e for e in ev2 if e["event"] == "node_status"
              and e.get("cached")]
    assert cached and cached[0]["node"] == "n1"
    assert _saw(ev1) == ["canary saw 7.0"]
    # the cached replay does not re-execute, so it does not re-warn
    assert _saw(ev2) == []


def test_concurrent_identical_runs_stay_correct(tmp_path):
    """H-B: two concurrent runs of the SAME workflow -- correctness only
    (the cache does not dedupe in-flight runs; one or both may compute).
    Neither run may observe the other's intermediate state."""
    reg = _canary_registry()
    cache = NodeCache(max_entries=8)
    ex = Executor(reg)
    results: dict[str, tuple[str, list[str]]] = {}

    def run(tag: str) -> None:
        events: list[dict] = []
        summary = ex.run(_canary_wf(7.0), emit=events.append,
                         workspace=tmp_path / tag, cache=cache)
        results[tag] = (summary["status"], _saw(events))

    threads = [threading.Thread(target=run, args=("a",)),
               threading.Thread(target=run, args=("b",))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert all(not t.is_alive() for t in threads), "a run hung"
    assert results["a"][0] == results["b"][0] == "completed"
    for tag in ("a", "b"):
        assert results[tag][1], f"run {tag} emitted no canary warning"
        assert all(m == "canary saw 7.0" for m in results[tag][1]), \
            results[tag][1]


# --- H-C/H-D: multi-client WebSocket delivery (server level) ---------------

def test_ws_fan_out_reaches_every_client():
    """H-C: three connected clients each receive the identical ordered
    stream (the mutant: a hub that broadcast to one arbitrary client)."""
    workflow = {"nodes": [{"id": "n1", "type": "core.util.echo",
                           "params": {"label": "hi"}}],
                "edges": []}
    sinks = [queue.Queue() for _ in range(3)]
    with TestClient(create_app(build_default_registry())) as client, \
            contextlib.ExitStack() as stack:
        for sink in sinks:
            ws = stack.enter_context(client.websocket_connect("/api/ws"))
            threading.Thread(target=_pump_ws, args=(ws, sink),
                             daemon=True).start()
        run_id = client.post(
            "/api/runs", json={"workflow": workflow}).json()["run_id"]
        deadline = time.monotonic() + 5.0
        while client.get(f"/api/runs/{run_id}").json()["status"] \
                not in ("completed", "failed", "cancelled", "timeout"):
            assert time.monotonic() < deadline, "the run did not finish"
            time.sleep(0.02)
        streams = [_drain_until(s, _finished) for s in sinks]
    for i, seen in enumerate(streams):
        assert seen[0]["event"] == "run_started", i
        statuses = {e["node"]: e["status"]
                    for e in seen if e["event"] == "node_status"}
        assert statuses == {"n1": "done"}, i
        finished = [e for e in seen if e["event"] == "run_finished"]
        assert finished[-1]["status"] == "completed", i


def test_mid_run_disconnect_does_not_starve_survivors():
    """H-D: user 1 closes the tab while a slow run streams; the run still
    completes, user 2 still receives it, and a SUBSEQUENT run still
    broadcasts to user 2 (the hub survived the dead-client send)."""
    class Slow(Node):
        type = "test.slow_m16"
        outputs = ()

        def execute(self, ctx):
            time.sleep(0.4)
            return {}

    reg = build_default_registry()
    reg.add(Slow)
    fast = {"nodes": [{"id": "n1", "type": "core.util.echo"}], "edges": []}
    slow = {"nodes": [{"id": "n1", "type": "test.slow_m16"}], "edges": []}
    sink1: queue.Queue = queue.Queue()
    sink2: queue.Queue = queue.Queue()
    with TestClient(create_app(reg)) as client, \
            client.websocket_connect("/api/ws") as ws1, \
            client.websocket_connect("/api/ws") as ws2:
        threading.Thread(target=_pump_ws, args=(ws1, sink1),
                         daemon=True).start()
        threading.Thread(target=_pump_ws, args=(ws2, sink2),
                         daemon=True).start()
        run_id = client.post(
            "/api/runs", json={"workflow": slow}).json()["run_id"]
        time.sleep(0.1)  # the run is inside the 0.4 s node now
        ws1.close()  # user 1 closes the tab mid-run
        deadline = time.monotonic() + 5.0
        while client.get(f"/api/runs/{run_id}").json()["status"] \
                not in ("completed", "failed", "cancelled", "timeout"):
            assert time.monotonic() < deadline, "the run did not finish"
            time.sleep(0.02)
        assert client.get(f"/api/runs/{run_id}").json()["status"] \
            == "completed"
        seen2 = _drain_until(sink2, _finished)
        finished2 = [e for e in seen2 if e["event"] == "run_finished"]
        assert finished2[-1]["status"] == "completed"
        assert client.get("/api/health").json() == {"status": "ok"}
        # a subsequent run still reaches the surviving client
        run2 = client.post("/api/runs", json={"workflow": fast}).json()["run_id"]
        deadline = time.monotonic() + 5.0
        while len([e for e in seen2 if e["event"] == "run_finished"]) < 2:
            try:
                seen2.append(sink2.get(
                    timeout=max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                raise AssertionError(
                    "the hub stopped broadcasting after the disconnect"
                ) from None
        _ = run2  # the second stream is counted, not attributed (events
        # ride the socket without a run_id)
        assert client.get(f"/api/runs/{run2}").json()["status"] == "completed"
