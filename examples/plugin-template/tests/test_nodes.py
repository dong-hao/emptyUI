"""Self-tests for the example plugin (required — plugin guide §"Self-tests").

Run from the package root: ``pytest``. These tests are the trust contract of
the ecosystem: they execute the nodes against the framework registry the way
the real app does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from emptyui.executor import Executor
from emptyui.registry import Registry
from emptyui.schema import Edge, Workflow, WorkflowNode, validate_workflow


@dataclass
class FakeSeries:
    """Minimal stand-in for a ``ts.run`` payload.

    Plugins duck-type the documented interchange attributes (``channels``,
    ``sample_rate``, ``copy()``) instead of importing any bundled plugin's
    container classes.
    """

    channels: dict[str, np.ndarray]
    sample_rate: float
    meta: dict = field(default_factory=dict)

    def copy(self) -> FakeSeries:
        return FakeSeries(
            {k: v.copy() for k, v in self.channels.items()}, self.sample_rate, dict(self.meta)
        )


@pytest.fixture()
def registry() -> Registry:
    from emptyui_example.plugin import register

    reg = Registry()
    register(reg)
    return reg


def _series() -> FakeSeries:
    return FakeSeries(
        channels={
            "ex": np.array([1.0, -1.0, 2.0, -2.0]),
            "hy": np.array([0.5, 0.5, -0.5, -0.5]),
        },
        sample_rate=150.0,
    )


def _execute(registry: Registry, wf: Workflow) -> dict:
    """Full-app path: validate, then execute through the framework executor."""
    validate_workflow(wf, registry)
    events: list[dict] = []
    summary = Executor(registry).run(wf, emit=events.append)
    assert summary["status"] == "completed"
    return summary["outputs"]


def test_manifest_protocol(registry):
    """§4 protocol: namespaced types, tooltip present, metadata declared."""
    for node in ("example.ts.gain", "example.demo.rms"):
        m = registry.get(node).manifest()
        assert m["type"].startswith("example.")
        assert m["tooltip"]  # one-sentence hover tip
        assert m["description"]
    catalog = registry.manifests()
    # new port type declared with UI metadata (plugin guide §"Port types")
    assert catalog["port_types"]["example.signal"]["color"] == "#16a085"


def test_gain_scales_all_channels_without_mutating_input():
    data = _series()
    original_ex = data.channels["ex"].copy()
    from emptyui_example.nodes import Gain

    out = Gain().execute(None, data, gain=10.0)["scaled"]
    # output is a copy — the input must be untouched (immutability contract)
    np.testing.assert_array_equal(data.channels["ex"], original_ex)
    np.testing.assert_allclose(out.channels["ex"], original_ex * 10.0)
    assert out.sample_rate == data.sample_rate


def test_rms_meter_outputs_declared_type():
    from emptyui_example.nodes import RmsMeter

    handle = RmsMeter().execute(None, _series())["signal"]
    assert handle["rms"]["ex"] == pytest.approx(np.sqrt(np.mean(np.array([1, -1, 2, -2]) ** 2)))
    assert handle["sample_rate"] == 150.0


def test_gain_interoperates_with_core_ts_run_producers(registry):
    """A core ``ts.run`` producer connects to the example consumer — the
    full-app path (validation + executor) with the ports wired."""
    from emptyui import Input, Node, Output

    class FakeSource(Node):
        type = "test.fake_source"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            return {"data": _series()}

    class Capture(Node):
        type = "test.capture"
        inputs = (Input("scaled", "ts.run"),)
        outputs = ()
        captured = None

        def execute(self, ctx, scaled=None):
            Capture.captured = scaled
            return {}

    registry.add(FakeSource)
    registry.add(Capture)
    _execute(
        registry,
        Workflow(
            nodes=[
                WorkflowNode(id="n1", type="test.fake_source"),
                WorkflowNode(id="n2", type="example.ts.gain", params={"gain": 3.0}),
                WorkflowNode(id="n3", type="test.capture"),
            ],
            edges=[
                Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data"),
                Edge(id="e2", source="n2", source_port="scaled", target="n3", target_port="scaled"),
            ],
        ),
    )
    np.testing.assert_allclose(
        Capture.captured.channels["ex"], np.array([3.0, -3.0, 6.0, -6.0])
    )
