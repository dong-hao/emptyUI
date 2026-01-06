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


def test_payload_summary_events_for_ts_outputs():
    """ts.run outputs stream a lightweight extent summary (DESIGN §23.4)."""
    from emptyui_core_nodes.types import TimeSeries

    from emptyui import Node, Output

    reg = _registry()

    class TsSource(Node):
        type = "test.tssource"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            n = 100
            return {"data": TimeSeries(
                channels={"ex": [0.0] * n}, sample_rate=10.0,
                start_datetime="2023-06-01T12:00:00")}

    reg.add(TsSource)
    ex = Executor(reg)
    events = []
    wf = Workflow(nodes=[WorkflowNode(id="n1", type="test.tssource")], edges=[])
    ex.run(wf, emit=events.append)
    ev = next(e for e in events if e["event"] == "payload")
    assert ev["node"] == "n1" and ev["port"] == "data" and ev["type"] == "ts.run"
    s = ev["summary"]
    assert s["absolute"] is True
    assert s["start"] == "2023-06-01T12:00:00"
    assert s["end"] == "2023-06-01T12:00:10"  # 100 samples @ 10 Hz
    assert s["sample_rate"] == 10.0
    assert s["channels"] == ["ex"]

    # relative-only record: relative-second bounds, absolute False
    from emptyui import Node, Output

    class RelSource(Node):
        type = "test.relsource"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            return {"data": TimeSeries(
                channels={"ex": [0.0] * 50}, sample_rate=10.0, start=5.0)}

    reg.add(RelSource)
    events = []
    Executor(reg).run(
        Workflow(nodes=[WorkflowNode(id="n1", type="test.relsource")], edges=[]),
        emit=events.append)
    s = next(e for e in events if e["event"] == "payload")["summary"]
    assert s["absolute"] is False and s["start"] == 5.0 and s["end"] == 10.0

    # non-ts outputs emit nothing
    events = []
    Executor(_registry()).run(
        Workflow(nodes=[WorkflowNode(id="n1", type="test.source")], edges=[]),
        emit=events.append)
    assert not [e for e in events if e["event"] == "payload"]


def test_payload_summary_placeholder_epoch_degrades_to_relative():
    """1980/1970 placeholder starts mean 'start unknown' — relative bounds."""
    from emptyui_core_nodes.types import TimeSeries

    from emptyui import Node, Output

    class PSource(Node):
        type = "test.psource"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            return {"data": TimeSeries(
                channels={"ex": [0.0] * 10}, sample_rate=10.0,
                start_datetime="1980-01-01T00:00:00")}

    reg = _registry()
    reg.add(PSource)
    events = []
    Executor(reg).run(
        Workflow(nodes=[WorkflowNode(id="n1", type="test.psource")], edges=[]),
        emit=events.append)
    s = next(e for e in events if e["event"] == "payload")["summary"]
    assert s["absolute"] is False and s["start"] == 0.0 and s["end"] == 1.0


def test_payload_summary_mth5_runts_shape():
    """Duck-typed mth5 RunTS payloads (`.dataset` + `.start`) summarize too."""
    from types import SimpleNamespace

    class RunTSStub:
        sample_rate = 8.0
        n_samples = 16
        start = "2023-06-01T12:30:05"
        dataset = SimpleNamespace(data_vars={"ex": None, "ey": None})

    from emptyui import Node, Output

    class MSource(Node):
        type = "test.msource"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            return {"data": RunTSStub()}

    reg = _registry()
    reg.add(MSource)
    events = []
    Executor(reg).run(
        Workflow(nodes=[WorkflowNode(id="n1", type="test.msource")], edges=[]),
        emit=events.append)
    s = next(e for e in events if e["event"] == "payload")["summary"]
    assert s["absolute"] is True
    assert s["start"] == "2023-06-01T12:30:05"
    assert s["end"] == "2023-06-01T12:30:07"
    assert s["channels"] == ["ex", "ey"]


def test_payload_summary_on_cache_hit():
    """A cached node's restored ts.run outputs stream the summary too."""
    from emptyui_core_nodes.types import TimeSeries

    from emptyui.cache import NodeCache

    calls = {"n": 0}

    from emptyui import Node, Output

    class Counting(Node):
        type = "test.counting"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            calls["n"] += 1
            return {"data": TimeSeries(
                channels={"ex": [0.0] * 20}, sample_rate=10.0,
                start_datetime="2023-06-01T00:00:00")}

    reg = _registry()
    reg.add(Counting)
    wf = Workflow(nodes=[WorkflowNode(id="n1", type="test.counting")], edges=[])
    cache = NodeCache()
    first, second = [], []
    Executor(reg).run(wf, emit=first.append, cache=cache)
    Executor(reg).run(wf, emit=second.append, cache=cache)
    assert calls["n"] == 1  # second run was a cache hit
    hit = next(e for e in second if e.get("cached") is True)
    assert hit["event"] == "node_status"
    ev = next(e for e in second if e["event"] == "payload")
    assert ev["summary"]["start"] == "2023-06-01T00:00:00"
    assert ev["summary"]["end"] == "2023-06-01T00:00:02"


def test_payload_summary_mth5_archive_per_run_extents():
    """An open archive (`.run_summary`) streams per-run extents for the
    Extract Run bar (DESIGN §23.4 phase 2)."""
    import pandas as pd

    from emptyui import Node, Output

    class ArchiveStub:
        filename = "x.h5"

        @property
        def run_summary(self):
            return pd.DataFrame([{
                "survey": "4P", "station": "SDG31", "run": "a",
                "start": pd.Timestamp("2017-10-16 17:01:13+00:00"),
                "end": pd.Timestamp("2017-10-16 17:32:06+00:00"),
                "sample_rate": 1.0,
            }])

    class ASrc(Node):
        type = "test.asrc"
        outputs = (Output("mth5", "mth5"),)

        def execute(self, ctx):
            return {"mth5": ArchiveStub()}

    reg = _registry()
    reg.add(ASrc)
    events = []
    Executor(reg).run(
        Workflow(nodes=[WorkflowNode(id="n1", type="test.asrc")], edges=[]),
        emit=events.append)
    ev = next(e for e in events if e["event"] == "payload")
    assert ev["type"] == "mth5"
    s = ev["summary"]
    assert s["kind"] == "mth5"
    assert s["runs"][0]["run"] == "a"
    assert s["runs"][0]["start"] == "2017-10-16 17:01:13"
    assert s["runs"][0]["end"] == "2017-10-16 17:32:06"


def test_payload_summary_channel_ts_single_channel():
    """A ts.channel payload (ChannelTS with .component) summarizes too."""
    from mt_timeseries import ChannelTS

    from emptyui import Node, Output

    class CSrc(Node):
        type = "test.csrc"
        outputs = (Output("hx", "ts.channel"),)

        def execute(self, ctx):
            ch = ChannelTS("magnetic", data=[0.0] * 40, sample_rate=10.0,
                           start="2023-06-01T12:00:00")
            ch.component = "hx"
            return {"hx": ch}

    reg = _registry()
    reg.add(CSrc)
    events = []
    Executor(reg).run(
        Workflow(nodes=[WorkflowNode(id="n1", type="test.csrc")], edges=[]),
        emit=events.append)
    s = next(e for e in events if e["event"] == "payload")["summary"]
    assert s["kind"] == "ts"
    assert s["absolute"] is True
    assert s["channels"] == ["hx"]
    assert s["end"] == "2023-06-01T12:00:04"


def test_ts_summary_kind_field_and_mt_getattr_safety():
    """kind is always present; missing attrs on mt_timeseries-style objects
    (NameError instead of AttributeError) do not crash the summarizer."""
    from emptyui.executor import _ts_summary

    class Tricky:
        def __getattr__(self, name):
            raise NameError(f"no {name}")

    assert _ts_summary(Tricky()) is None
    assert _ts_summary(42) is None
