"""Cross-run node cache + partial execution ("execute to here") — DESIGN.md §7.

Feature tests: structural signatures, hit/miss invalidation, artifact
restore, ancestor-closure targets. Error surfaces (unknown targets) live in
test_errors.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from emptyui.cache import NodeCache
from emptyui.executor import Executor
from emptyui.registry import Registry
from emptyui.schema import Edge, Workflow, WorkflowNode


def _registry():
    from emptyui import Input, Node, Output, Param

    reg = Registry()

    class Source(Node):
        type = "test.source"
        outputs = (Output("data", "ts.run"),)
        params = (Param("offset", "number", default=0.0),)
        calls = 0

        def execute(self, ctx, offset=0.0):
            Source.calls += 1
            return {"data": [v + offset for v in [1.0, 2.0, 3.0]]}

    class Scale(Node):
        type = "test.scale"
        inputs = (Input("data", "ts.run"),)
        outputs = (Output("scaled", "ts.run"),)
        params = (Param("factor", "number", default=2.0),)
        calls = 0
        seen_inputs: ClassVar[list[list[float]]] = []

        def execute(self, ctx, data, factor=2.0):
            Scale.calls += 1
            Scale.seen_inputs.append(data)
            return {"scaled": [v * factor for v in data]}

    reg.add(Source)
    reg.add(Scale)
    return reg


def _workflow(factor=2.0, offset=0.0):
    return Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.source", params={"offset": offset}),
            WorkflowNode(id="n2", type="test.scale", params={"factor": factor}),
        ],
        edges=[Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data")],
    )


def test_second_run_hits_cache_and_shares_instances():
    reg = _registry()
    cache = NodeCache()
    ex = Executor(reg)

    ex.run(_workflow(), emit=lambda e: None, cache=cache)
    assert (reg.get("test.source").calls, reg.get("test.scale").calls) == (1, 1)

    # change only the downstream param: source hits the cache, scale re-runs
    # and receives the CACHED payload instance — the immutability contract
    # (same object across runs, no defensive copies)
    events: list[dict] = []
    summary = ex.run(_workflow(factor=3.0), emit=events.append, cache=cache)
    assert summary["status"] == "completed"
    assert (reg.get("test.source").calls, reg.get("test.scale").calls) == (1, 2)
    statuses = {e["node"]: e for e in events if e["event"] == "node_status"}
    assert statuses["n1"]["cached"] is True
    assert "cached" not in statuses["n2"]
    assert reg.get("test.scale").seen_inputs[1] is reg.get("test.scale").seen_inputs[0]


def test_param_change_invalidates_only_from_that_node():
    reg = _registry()
    cache = NodeCache()
    ex = Executor(reg)
    ex.run(_workflow(), emit=lambda e: None, cache=cache)

    ex.run(_workflow(factor=5.0), emit=lambda e: None, cache=cache)
    # source unchanged ⇒ cached; scale's param changed ⇒ re-executed
    assert reg.get("test.source").calls == 1
    assert reg.get("test.scale").calls == 2


def test_upstream_change_propagates_to_all_descendants():
    from emptyui import Input, Node

    reg = _registry()

    class Sink(Node):
        type = "test.sink"
        inputs = (Input("scaled", "ts.run"),)
        outputs = ()

        def execute(self, ctx, scaled=None):
            return {}

    reg.add(Sink)

    def chain(offset=0.0):
        return Workflow(
            nodes=[
                WorkflowNode(id="n1", type="test.source", params={"offset": offset}),
                WorkflowNode(id="n2", type="test.scale"),
                WorkflowNode(id="n3", type="test.sink"),
            ],
            edges=[
                Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data"),
                Edge(id="e2", source="n2", source_port="scaled", target="n3", target_port="scaled"),
            ],
        )

    cache = NodeCache()
    ex = Executor(reg)
    ex.run(chain(), emit=lambda e: None, cache=cache)
    ex.run(chain(), emit=lambda e: None, cache=cache)
    assert (reg.get("test.source").calls, reg.get("test.scale").calls) == (1, 1)

    # change only n1's param — n2 and n3 must re-run even though their own
    # structure is untouched (signature recursion)
    ex.run(chain(offset=1.0), emit=lambda e: None, cache=cache)
    assert (reg.get("test.source").calls, reg.get("test.scale").calls) == (2, 2)


def test_structurally_identical_nodes_share_one_entry():
    reg = _registry()
    wf = Workflow(
        nodes=[
            WorkflowNode(id="a", type="test.source"),
            WorkflowNode(id="b", type="test.source"),
        ],
        edges=[],
    )
    ex = Executor(reg)
    events: list[dict] = []
    ex.run(wf, emit=events.append, cache=NodeCache())
    # node ids are excluded from the signature: the second identical node
    # hits the entry the first one just created — within the same run
    statuses = {e["node"]: e for e in events if e["event"] == "node_status"}
    assert "cached" not in statuses["a"]
    assert statuses["b"]["cached"] is True


def test_targets_run_target_and_ancestors_only():
    from emptyui import Input, Node

    reg = _registry()

    class Sink(Node):
        type = "test.sink"
        inputs = (Input("scaled", "ts.run"),)
        outputs = ()

        def execute(self, ctx, scaled=None):
            return {}

    reg.add(Sink)
    wf = Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.source"),
            WorkflowNode(id="n2", type="test.scale"),
            WorkflowNode(id="n3", type="test.sink"),
        ],
        edges=[
            Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data"),
            Edge(id="e2", source="n2", source_port="scaled", target="n3", target_port="scaled"),
        ],
    )
    events: list[dict] = []
    summary = Executor(reg).run(wf, emit=events.append, targets=["n2"])
    assert summary["status"] == "completed"
    statuses = {e["node"]: e["status"] for e in events if e["event"] == "node_status"}
    assert statuses == {"n1": "done", "n2": "done"}
    assert "n3" not in statuses


def test_file_param_signature_detects_external_modification(tmp_path: Path):
    from emptyui import Node, Output, Param

    data_file = tmp_path / "data.txt"
    data_file.write_text("1 2 3")

    class ReadFile(Node):
        type = "test.readfile"
        outputs = (Output("text", "any"),)
        params = (Param("path", "string", default=""),)
        calls = 0

        def execute(self, ctx, path=""):
            ReadFile.calls += 1
            return {"text": Path(path).read_text()}

    reg = Registry()
    reg.add(ReadFile)
    wf = Workflow(
        nodes=[WorkflowNode(id="n1", type="test.readfile", params={"path": str(data_file)})],
        edges=[],
    )
    ex = Executor(reg)
    ex.run(wf, emit=lambda e: None, cache=NodeCache())
    assert ReadFile.calls == 1

    cache = NodeCache()
    ex.run(wf, emit=lambda e: None, cache=cache)
    assert ReadFile.calls == 2  # fresh cache ⇒ cold run (baseline)

    ex.run(wf, emit=lambda e: None, cache=cache)
    assert ReadFile.calls == 2  # untouched file ⇒ cached

    os.utime(data_file, ns=(1_000_000_000, 1_000_000_000))
    ex.run(wf, emit=lambda e: None, cache=cache)
    assert ReadFile.calls == 3  # mtime changed ⇒ re-executed


def test_artifacts_restored_into_new_workspace(tmp_path: Path):
    from emptyui import Node, Output

    class Plot(Node):
        type = "test.plot"
        outputs = (Output("plot", "any"),)
        calls = 0

        def execute(self, ctx):
            Plot.calls += 1
            path = ctx.artifact_path("plot_test", ".png")
            path.write_text("fake png")
            artifact = ctx.publish_artifact(path, kind="image")
            return {"plot": artifact}

    reg = Registry()
    reg.add(Plot)
    wf = Workflow(nodes=[WorkflowNode(id="n1", type="test.plot")], edges=[])

    ws1, ws2 = tmp_path / "run1", tmp_path / "run2"
    events: list[dict] = []
    cache = NodeCache()
    Executor(reg).run(wf, emit=events.append, workspace=ws1, cache=cache)
    assert Plot.calls == 1

    Executor(reg).run(wf, emit=events.append, workspace=ws1, cache=cache)
    Executor(reg).run(wf, emit=events.append, workspace=ws2, cache=cache)
    assert Plot.calls == 1

    artifacts = [e for e in events if e["event"] == "artifact" and e["path"].startswith(str(ws2))]
    assert len(artifacts) == 1
    restored = Path(artifacts[0]["path"])
    assert restored.exists()
    assert restored.name == "plot_test_n1.png"  # renamed for this run's instance
    # the restored artifact event carries the fresh store handle of the port
    # whose value is the artifact dict (viewer edge resolution, §7)
    assert artifacts[0]["handle"].startswith("h")
    cached_status = [e for e in events if e["event"] == "node_status" and e.get("cached")]
    assert cached_status and cached_status[-1]["node"] == "n1"


def test_missing_cached_artifact_file_forces_reexecution(tmp_path: Path):
    from emptyui import Node, Output

    class Plot(Node):
        type = "test.plot2"
        outputs = (Output("plot", "any"),)
        calls = 0

        def execute(self, ctx):
            Plot.calls += 1
            path = ctx.artifact_path("plot_test", ".png")
            path.write_text("fake png")
            artifact = ctx.publish_artifact(path, kind="image")
            return {"plot": artifact}

    reg = Registry()
    reg.add(Plot)
    wf = Workflow(nodes=[WorkflowNode(id="n1", type="test.plot2")], edges=[])
    ws = tmp_path / "run"
    events: list[dict] = []
    cache = NodeCache()
    Executor(reg).run(wf, emit=events.append, workspace=ws, cache=cache)
    assert Plot.calls == 1

    for stale in ws.glob("plot_test_n1.png"):
        stale.unlink()

    events2: list[dict] = []
    Executor(reg).run(wf, emit=events2.append, workspace=ws, cache=cache)
    assert Plot.calls == 2  # artifact file gone ⇒ re-executed, never half-served
    assert not any(e.get("cached") for e in events2 if e["event"] == "node_status")


def test_cache_lru_eviction_and_clear():
    from emptyui.cache import CacheEntry

    cache = NodeCache(max_entries=2)
    cache.put("k1", CacheEntry(outputs={}))
    cache.put("k2", CacheEntry(outputs={}))
    cache.put("k3", CacheEntry(outputs={}))
    assert len(cache) == 2
    assert cache.get("k1") is None  # oldest evicted
    assert cache.get("k3") is not None

    assert cache.clear() == 2
    assert len(cache) == 0
