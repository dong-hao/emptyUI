"""Cross-run node cache + partial execution ("execute to here") — DESIGN.md §7.

Feature tests: structural signatures, hit/miss invalidation, artifact
restore, ancestor-closure targets. Error surfaces (unknown targets) live in
test_errors.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from emptyui.cache import CacheEntry, NodeCache, _approx_size
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
        params = (Param("path", "file", default=""),)
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


def test_string_and_number_params_share_cache_entry():
    """§7 hashes the EFFECTIVE params — the combobox string "4" and the
    number 4 on a number param produce one cache entry (coercion happens
    at key time, matching execution)."""
    from emptyui import Input, Node, Output, Param
    from emptyui.schema import Edge, Workflow, WorkflowNode

    calls = {"n": 0}

    class Src(Node):
        type = "test.sig_src"
        outputs = (Output("data", "any"),)

        def execute(self, ctx):
            return {"data": 1}

    class Scale(Node):
        type = "test.sig_scale"
        inputs = (Input("data", "any"),)
        outputs = (Output("out", "any"),)
        params = (Param("factor", "number", default=2.0),)

        def execute(self, ctx, data, factor=2.0):
            calls["n"] += 1
            return {"out": data * factor}

    reg = Registry()
    reg.add(Src)
    reg.add(Scale)
    cache = NodeCache()
    ex = Executor(reg)

    def wf(factor):
        return Workflow(
            nodes=[
                WorkflowNode(id="n1", type="test.sig_src"),
                WorkflowNode(id="n2", type="test.sig_scale",
                             params={"factor": factor}),
            ],
            edges=[Edge(id="e1", source="n1", source_port="data",
                        target="n2", target_port="data")],
        )

    ex.run(wf(4), emit=lambda e: None, cache=cache)
    ex.run(wf("4"), emit=lambda e: None, cache=cache)
    assert calls["n"] == 1  # second run hit the cache


def test_stat_signature_requires_path_widget(tmp_path: Path):
    """§7.1: only ``file``/``directory`` widgets get stat signatures — a
    plain string param whose value coincidentally names an existing path
    stays a plain string (no CWD-dependent stat quirk)."""
    real = tmp_path / "NO46.txt"
    real.write_text("x")
    ex = Executor(Registry())
    assert ex._value_signature(str(real), widget="string") == str(real)
    assert ex._value_signature("NO46.txt", widget="string") == "NO46.txt"
    sig = ex._value_signature(str(real), widget="file")
    assert sig == [str(real), real.stat().st_size, real.stat().st_mtime_ns]
    # missing file ⇒ value as-is (unchanged)
    assert ex._value_signature(str(tmp_path / "absent.txt"),
                               widget="file") == str(tmp_path / "absent.txt")


def test_file_signature_survives_data_root_relocation(tmp_path: Path):
    """§7.1/§12.3: the path part of a file-param signature is taken
    data-root-relative (POSIX form), so relocating the data root with an
    identical relative layout keeps the cache key stable. A file outside
    any root keeps its absolute-path signature."""
    import shutil

    root_a, root_b = tmp_path / "root_a", tmp_path / "root_b"
    for root in (root_a, root_b):
        (root / "site").mkdir(parents=True)
    file_a = root_a / "site" / "data.txt"
    file_a.write_text("payload")
    shutil.copy2(file_a, root_b / "site" / "data.txt")  # mtime preserved

    sig_a = Executor._value_signature("site/data.txt", data_root=root_a,
                                      path_kind="in", widget="file")
    sig_b = Executor._value_signature("site/data.txt", data_root=root_b,
                                      path_kind="in", widget="file")
    assert sig_a == sig_b
    assert sig_a[0] == "site/data.txt"

    outside = tmp_path / "outside.txt"
    outside.write_text("y")
    sig = Executor._value_signature(str(outside), widget="file")
    assert sig[0] == str(outside)

    # project dir shares the same relative rule (§12.3 scaffolding)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "state.bin").write_text("z")
    sig_p = Executor._value_signature(str(proj / "state.bin"),
                                      widget="file", project_dir=proj)
    assert sig_p[0] == "state.bin"


def test_relocated_data_root_hits_cache(tmp_path: Path):
    """End-to-end §7.1: the same relative workflow on a relocated data
    root (mtime-preserving copy) hits the cache; editing the relocated
    file's content still invalidates (stat half of the signature)."""
    import shutil

    from emptyui import Node, Output, Param

    class ReadFile(Node):
        type = "test.reloc_read"
        outputs = (Output("text", "any"),)
        params = (Param("path", "file", path_kind="in"),)
        calls = 0

        def execute(self, ctx, path=""):
            ReadFile.calls += 1
            return {"text": Path(path).read_text()}

    reg = Registry()
    reg.add(ReadFile)

    root_a, root_b = tmp_path / "root_a", tmp_path / "root_b"
    for root in (root_a, root_b):
        (root / "site").mkdir(parents=True)
    src = root_a / "site" / "data.txt"
    src.write_text("payload")
    shutil.copy2(src, root_b / "site" / "data.txt")

    def wf():
        return Workflow(
            nodes=[WorkflowNode(id="n1", type="test.reloc_read",
                                params={"path": "site/data.txt"})],
            edges=[],
        )

    cache = NodeCache()
    ex = Executor(reg)
    ex.run(wf(), emit=lambda e: None, cache=cache, data_root=root_a)
    ex.run(wf(), emit=lambda e: None, cache=cache, data_root=root_a)
    assert ReadFile.calls == 1  # second run cached

    ex.run(wf(), emit=lambda e: None, cache=cache, data_root=root_b)
    assert ReadFile.calls == 1  # relocated root ⇒ same key ⇒ cache hit

    (root_b / "site" / "data.txt").write_text("changed-content")
    ex.run(wf(), emit=lambda e: None, cache=cache, data_root=root_b)
    assert ReadFile.calls == 2  # external modification still invalidates


def test_non_cacheable_node_reexecutes_every_run(tmp_path: Path):
    """§7.1: save/export nodes' side effect IS writing the file — a
    structural cache hit that skips them reports "done (cached)" with no
    file written (and an externally deleted destination stays gone). They
    re-execute every run; the inputs come from the cache either way."""
    from emptyui import Input, Node, Output

    class Saver(Node):
        type = "test.saver"
        inputs = (Input("data", "ts.run"),)
        outputs = ()
        cacheable = False
        calls = 0

        def execute(self, ctx, data):
            Saver.calls += 1
            (tmp_path / "out.txt").write_text(str(data))

    class Source(Node):
        type = "test.src"
        outputs = (Output("data", "ts.run"),)
        calls = 0

        def execute(self, ctx):
            Source.calls += 1
            return {"data": [1.0]}

    reg = Registry()
    reg.add(Source)
    reg.add(Saver)

    def wf():
        return Workflow(
            nodes=[WorkflowNode(id="n1", type="test.src"),
                   WorkflowNode(id="n2", type="test.saver")],
            edges=[Edge(id="e1", source="n1", source_port="data",
                        target="n2", target_port="data")],
        )

    cache = NodeCache()
    ex = Executor(reg)
    ex.run(wf(), emit=lambda e: None, cache=cache)
    ex.run(wf(), emit=lambda e: None, cache=cache)
    assert Saver.calls == 2  # never cached — always re-written
    # the source upstream is still cached (one execution)
    assert reg.get("test.src").calls == 1
    # a cached rerun event surface: statuses must not claim "cached" for n2
    events: list[dict] = []
    ex.run(wf(), emit=events.append, cache=cache)
    statuses = {e["node"]: e for e in events if e["event"] == "node_status"}
    assert "cached" not in statuses["n2"]


def test_invalidates_inputs_evicts_upstream_entry():
    """§7.1 immutability: a node that closes/mutates its input (Save MTH5's
    close-in-place) must not leave the producer's cache entry serving the
    poisoned payload to later structurally-identical runs — the executor
    evicts the direct-input producers' entries after it executes."""
    from emptyui import Input, Node, Output

    class Archive:
        closed = False

        def close_mth5(self):
            self.closed = True

    class Closer(Node):
        type = "test.closer"
        inputs = (Input("mth5", "mth5"),)
        outputs = ()
        invalidates_inputs = True

        def execute(self, ctx, mth5):
            mth5.close_mth5()
            return {}

    class Opener(Node):
        type = "test.opener"
        outputs = (Output("mth5", "mth5"),)
        calls = 0

        def execute(self, ctx):
            Opener.calls += 1
            return {"mth5": Archive()}

    reg = Registry()
    reg.add(Opener)
    reg.add(Closer)

    def wf():
        return Workflow(
            nodes=[WorkflowNode(id="n1", type="test.opener"),
                   WorkflowNode(id="n2", type="test.closer")],
            edges=[Edge(id="e1", source="n1", source_port="mth5",
                        target="n2", target_port="mth5")],
        )

    cache = NodeCache()
    ex = Executor(reg)
    s1 = ex.run(wf(), emit=lambda e: None, cache=cache, include_store=True)
    arch1 = s1["store"].get(s1["outputs"]["n1"]["mth5"]).value
    assert arch1.closed  # the closer ran and closed it (same instance)

    # run 2: the opener's entry was evicted by the closer ⇒ re-executed, and
    # the served payload is a FRESH archive (not the closed instance)
    events: list[dict] = []
    s2 = ex.run(wf(), emit=events.append, cache=cache, include_store=True)
    assert reg.get("test.opener").calls == 2
    statuses = {e["node"]: e for e in events if e["event"] == "node_status"}
    assert "cached" not in statuses["n1"]
    arch2 = s2["store"].get(s2["outputs"]["n1"]["mth5"]).value
    assert arch2 is not arch1 and not arch2.closed


def _fake_run_summary():
    class _Row:
        def get(self, k, default=""):
            return {"survey": "sv", "station": "st", "run": "r1",
                    "start": "2017-10-16T17:10:00", "end": "2017-10-16T18:00:00",
                    "sample_rate": 1.0}.get(k, default)

        def __getitem__(self, k):
            return self.get(k)

    class _Summary:
        def __len__(self):
            return 1

        def iterrows(self):
            yield None, _Row()

    return _Summary()


def test_restore_failure_surfaces_as_node_error():
    """A poisoned cached payload (e.g. a closed archive behind the cache —
    the §7.1 immutability violation) must fail the run through the node's
    error event, never as a raw crash escaping the executor (which left the
    node "running" forever in the UI)."""
    from emptyui import Input, Node, Output

    class Poison:
        closed = False

        @property
        def run_summary(self):
            if self.closed:
                raise KeyError("Unable to synchronously open object")
            return _fake_run_summary()

    class Source(Node):
        type = "test.poison_src"
        outputs = (Output("mth5", "mth5"),)

        def execute(self, ctx):
            return {"mth5": Poison()}

    class Sink(Node):
        type = "test.poison_sink"
        inputs = (Input("mth5", "mth5"),)
        outputs = ()

        def execute(self, ctx, mth5):
            return {}

    reg = Registry()
    reg.add(Source)
    reg.add(Sink)

    def wf():
        return Workflow(
            nodes=[WorkflowNode(id="n1", type="test.poison_src"),
                   WorkflowNode(id="n2", type="test.poison_sink")],
            edges=[Edge(id="e1", source="n1", source_port="mth5",
                        target="n2", target_port="mth5")],
        )

    cache = NodeCache()
    ex = Executor(reg)
    ex.run(wf(), emit=lambda e: None, cache=cache)
    # simulate the original bug: the shared payload is closed but the cache
    # entry still serves it (no invalidates_inputs declaration)
    ex.registry.get("test.poison_src").execute.__func__  # noqa: B018
    src_payload = None
    for entry in cache._entries.values():
        if entry.outputs.get("mth5") is not None:
            src_payload = entry.outputs["mth5"]
    src_payload.closed = True

    import pytest as _pytest

    from emptyui.executor import ExecutionError

    events: list[dict] = []
    with _pytest.raises(ExecutionError):
        ex.run(wf(), emit=events.append, cache=cache)
    statuses = [e for e in events if e["event"] == "node_status"]
    assert any(e.get("status") == "error" for e in statuses)
    assert any(e.get("event") == "run_finished" and e.get("status") == "failed"
               for e in events)


def test_failed_run_sweep_closes_uncached_payloads():
    """Failed-run hygiene (§9.3): a failed run closes open MTH5 payloads left
    in its store — except cached ones (shared with later runs; the sweep
    never poisons the cache)."""
    from emptyui import Input, Node, Output
    from emptyui.executor import ExecutionError

    class Archive:
        closed = False

        def close_mth5(self):
            self.closed = True

    class Opener(Node):
        type = "test.sweep_opener"
        outputs = (Output("mth5", "mth5"),)

        def execute(self, ctx):
            return {"mth5": Archive()}

    class Failing(Node):
        type = "test.sweep_failing"
        inputs = (Input("mth5", "mth5"),)
        outputs = ()

        def execute(self, ctx, mth5):
            raise ValueError("boom")

    reg = Registry()
    reg.add(Opener)
    reg.add(Failing)

    def wf():
        return Workflow(
            nodes=[WorkflowNode(id="n1", type="test.sweep_opener"),
                   WorkflowNode(id="n2", type="test.sweep_failing")],
            edges=[Edge(id="e1", source="n1", source_port="mth5",
                        target="n2", target_port="mth5")],
        )

    ex = Executor(reg)
    events: list[dict] = []
    # no cache: nothing is shared — the sweep closes the payload
    with pytest.raises(ExecutionError):
        ex.run(wf(), emit=events.append, cache=None)
    assert any(e.get("event") == "run_finished" and e.get("status") == "failed"
               for e in events)

    # with a cache: the opener's payload is cached — NOT closed
    cache = NodeCache()
    ex2 = Executor(reg)
    with pytest.raises(ExecutionError):
        ex2.run(wf(), emit=lambda e: None, cache=cache)
    assert not cache.cached_values()[0].closed


# --- L5: the cache budget (bytes, not just entries) -----------------------

def test_budget_evicts_oldest_when_exceeded():
    """A byte budget bounds the cache like the entry count: the oldest
    entries go first; when the budget shrinks live, the trim keeps the
    newest."""
    cache = NodeCache(max_entries=8, max_bytes=2000)
    cache.put("a", CacheEntry(outputs={"v": np.zeros(100)}))  # 800 B data
    cache.put("b", CacheEntry(outputs={"v": np.zeros(100)}))
    cache.put("c", CacheEntry(outputs={"v": np.zeros(100)}))
    assert len(cache) == 2  # 3 x 800 B > 2000 B: the oldest went
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
    cache.set_budget(900)  # live shrink: only one fits
    assert len(cache) == 1
    assert cache.get("c") is not None


def test_budget_none_is_unbounded():
    cache = NodeCache(max_entries=8)  # default: no byte budget
    cache.put("a", CacheEntry(outputs={"v": np.zeros(100_000)}))
    assert cache.get("a") is not None


def test_oversized_entry_never_empties_the_cache():
    """At put time a single entry larger than the budget keeps its slot
    (the just-computed payload is never evicted) — but a live budget
    shrink enforces against everything (see set_budget's docstring)."""
    cache = NodeCache(max_entries=8, max_bytes=100)
    cache.put("big", CacheEntry(outputs={"v": np.zeros(10_000)}))
    cache.put("small", CacheEntry(outputs={"v": np.zeros(1)}))
    assert cache.get("big") is not None or cache.get("small") is not None
    assert len(cache) >= 1


def test_approx_size_covers_the_payload_kinds():
    """The estimator sees through the containers the executor actually
    caches: numpy arrays, xarray datasets, (duck-typed) RunTS-ish
    payloads, dicts and lists thereof."""
    import xarray as xr

    arr = np.zeros((100, 100))  # 80_000 B
    assert _approx_size(arr) >= 80_000
    ds = xr.Dataset({"a": (("x",), np.zeros(1000, dtype="float64"))})
    assert _approx_size(ds) >= 8_000
    nested = {"a": [arr, np.zeros(10)]}
    assert _approx_size(nested) >= 80_000


def test_approx_size_survives_the_mtimeseries_getattr_quirk():
    """L5 regression (caught by the pipeline e2e): mt_timeseries converts a
    missing-attribute AttributeError into NameError — defeating getattr's
    default (the landmine documented at the executor's cleanup). A payload
    class with that __getattr__ must not blow up the size probe; the cache
    put would otherwise fail the NODE with "RunTS has no attribute nbytes"."""
    import xarray as xr

    class QuirkyRunTS:
        """Duck-typed RunTS: has .dataset, __getattr__ raises NameError."""

        def __getattr__(self, name):
            raise NameError(f"RunTS has no attribute {name}")

        def __init__(self):
            self.dataset = xr.Dataset(
                {"a": (("x",), np.zeros(1000, dtype="float64"))})

    q = QuirkyRunTS()
    assert _approx_size(q) >= 8_000  # seen through .dataset, no raise
    assert _approx_size(object()) >= 64  # plain object: the floor
