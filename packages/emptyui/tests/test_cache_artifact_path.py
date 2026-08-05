"""M4 — a cache hit must not hand downstream a path into the *producer*
run's workspace.

On a cache hit the executor copies the artifact files into the new run's
workspace and advertises the fresh paths in the ``artifact`` events — but
re-registered the producer's ORIGINAL output dicts in the new run's store.
A consumer reading ``value["path"]`` (Save Plot, or any node dereferencing
an artifact) resolved run A's file while the event advertised run B's; if
TTL reclaimed A mid-run the read failed although a fresh copy existed.

The contract: the stored value must point at THIS run's copy, as a
**shallow copy** (rewriting the cache entry's dict in place would violate
the §7.1 immutability contract), and the event path must equal the stored
value path. Headless restores (no workspace, no copy) keep the cache path;
same-workspace restores are a no-op.

Test-fixture note: deleting the producer's workspace *between* runs is NOT
a discriminating scenario — the vanished-src guard re-executes the
producer. The real hazard window is after the restore, before the consumer
runs, which ``ReclaimAndRead`` simulates deterministically.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from emptyui.cache import NodeCache
from emptyui.executor import Executor
from emptyui.registry import Registry
from emptyui.schema import Workflow, WorkflowNode


def _registry():
    from emptyui import Input, Node, Output, Param

    reg = Registry()

    class PlotSource(Node):
        type = "test.plot_source"
        outputs = (Output("plot", "plot"),)

        def execute(self, ctx):
            p = ctx.artifact_path("fig", ".txt")
            p.write_text("figure-data")
            return {"plot": ctx.publish_artifact(p, "image")}

    class ReclaimAndRead(Node):
        """Reads the incoming artifact's path — optionally reclaiming a
        directory first (the test's simulation of a mid-run TTL reclaim of
        the producer's workspace)."""

        type = "test.reclaim_read"
        inputs = (Input("plot", "plot"),)
        outputs = (Output("text", "any"),)
        params = (
            Param("reclaim", "string", default="", optional=True),
            Param("suffix", "string", default="", optional=True),
        )

        def execute(self, ctx, plot, reclaim="", suffix=""):
            if reclaim:
                shutil.rmtree(reclaim, ignore_errors=True)
            return {"text": Path(plot["path"]).read_text() + suffix}

    reg.add(PlotSource)
    reg.add(ReclaimAndRead)
    return reg


def _run(registry, workflow, cache, workspace=None, **kw):
    events: list[dict] = []
    summary = Executor(registry).run(
        workflow, emit=events.append, cache=cache,
        workspace=workspace, **kw)
    return summary, events


def _two_source_runs(cache, include_store=False):
    """Two identical runs of the single-node workflow in different
    workspaces sharing one cache — run 2 hits for the producer."""
    registry = _registry()
    wf = Workflow(nodes=[WorkflowNode(id="n1", type="test.plot_source")])
    ws1 = tempfile.mkdtemp(prefix="m4-a-")
    ws2 = tempfile.mkdtemp(prefix="m4-b-")
    _run(registry, wf, cache, workspace=ws1)
    return ws1, ws2, _run(
        registry, wf, cache, workspace=ws2, include_store=include_store)


def test_cached_artifact_output_points_at_this_runs_copy():
    _ws1, ws2, (summary, _) = _two_source_runs(
        NodeCache(), include_store=True)
    assert summary["status"] == "completed", summary
    store = summary["store"]
    value = store.get(summary["outputs"]["n1"]["plot"]).value
    assert value["path"].startswith(str(Path(ws2))), (
        "the store value points into the producer run's workspace: "
        f"{value['path']}")


def test_artifact_event_path_matches_the_stored_value():
    _ws1, _ws2, (summary, events) = _two_source_runs(
        NodeCache(), include_store=True)
    assert summary["status"] == "completed", summary
    store = summary["store"]
    value = store.get(summary["outputs"]["n1"]["plot"]).value
    artifact_events = [e for e in events if e["event"] == "artifact"]
    assert artifact_events[-1]["path"] == value["path"], (
        "the event advertises a different file than the store value carries")


def test_cache_entry_is_not_mutated():
    cache = NodeCache()
    ws1, _ws2, (summary, _) = _two_source_runs(cache)
    assert summary["status"] == "completed"
    entry = next(iter(cache._entries.values()))
    assert str(Path(ws1)) in entry.outputs["plot"]["path"], (
        "the cache entry's output dict was mutated in place")


def test_consumer_survives_ttl_reclaim_mid_run():
    """The honest end-to-end: run 2's consumer re-executes (its param
    changed — a producer cache hit, a consumer miss) and reclaims the
    producer's workspace BEFORE reading — the mid-run TTL window. It must
    read this run's fresh copy, not the producer's vanished file."""
    registry = _registry()
    wf1 = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.plot_source"),
            WorkflowNode(id="n2", type="test.reclaim_read"),
        ],
        edges=[{"id": "e1", "source": "n1", "source_port": "plot",
                "target": "n2", "target_port": "plot"}],
    )
    # run 2: the reclaim param changes the consumer's structural key, so
    # the consumer re-executes while the producer is served from cache
    wf2 = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.plot_source"),
            WorkflowNode(id="n2", type="test.reclaim_read",
                         params={"suffix": "-r2"}),
        ],
        edges=[{"id": "e1", "source": "n1", "source_port": "plot",
                "target": "n2", "target_port": "plot"}],
    )
    ws1 = tempfile.mkdtemp(prefix="m4-a-")
    ws2 = tempfile.mkdtemp(prefix="m4-b-")
    cache = NodeCache()
    _run(registry, wf1, cache, workspace=ws1)
    wf2.nodes[1].params["reclaim"] = ws1  # the dir run 2's consumer reclaims
    summary, _ = _run(registry, wf2, cache, workspace=ws2, include_store=True)
    assert summary["status"] == "completed", summary
    store = summary["store"]
    text = store.get(summary["outputs"]["n2"]["text"]).value
    assert text == "figure-data-r2"


def test_artifact_handle_names_the_output_port():
    """L3 (frontend half): the artifact event's ``handle`` must be the
    OUTPUT PORT NAME — the router key the frontend matches against an
    edge's sourceHandle (SS 7.2). The fresh-run path emitted the store's
    handle id ("h2"), which never matches a port name, so every viewer
    silently fell back to "the source's last artifact" — the wrong-figure
    bug for multi-output sources. Cache restores already emitted the port
    name; this pins the fresh path to the same contract."""
    reg = _registry()
    wf = Workflow(nodes=[
        WorkflowNode(id="src", type="test.plot_source", params={}),
    ], edges=[])
    cache = NodeCache()
    _summary, fresh = _run(reg, wf, cache)
    _summary2, restored = _run(reg, wf, cache)  # same signature -> cache hit
    for tag, events in (("fresh", fresh), ("restored", restored)):
        arts = [e for e in events if e.get("event") == "artifact"]
        assert arts, f"no artifact event emitted ({tag})"
        assert arts[0]["handle"] == "plot", (tag, arts[0])
