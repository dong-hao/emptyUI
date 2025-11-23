import pytest

from emptyui.registry import Registry
from emptyui.schema import (
    Edge,
    ValidationError_,
    Workflow,
    WorkflowNode,
    validate_workflow,
)


class Echo:
    pass


def _registry():
    from emptyui import Input, Node, Output

    reg = Registry()

    class Source(Node):
        type = "test.source"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            return {"data": [1, 2, 3]}

    class Consumer(Node):
        type = "test.consumer"
        inputs = (Input("data", "ts.run"),)

        def execute(self, ctx, data):
            return {}

    reg.add(Source)
    reg.add(Consumer)
    return reg


def _wf(nodes, edges):
    return Workflow(nodes=nodes, edges=edges)


def test_valid_workflow_passes():
    wf = _wf(
        [
            WorkflowNode(id="n1", type="test.source"),
            WorkflowNode(id="n2", type="test.consumer"),
        ],
        [Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data")],
    )
    validate_workflow(wf, _registry())  # no exception


def test_unknown_type_reported():
    wf = _wf([WorkflowNode(id="n1", type="nope.nope")], [])
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, _registry())
    assert any("unknown type" in e for e in exc.value.errors)


def test_type_mismatch_reported():
    reg = _registry()
    from emptyui import Node, Output

    class SpectraSrc(Node):
        type = "test.spectra_src"
        outputs = (Output("data", "spectra"),)

        def execute(self, ctx):
            return {}

    reg.add(SpectraSrc)
    wf = _wf(
        [
            WorkflowNode(id="n1", type="test.spectra_src"),
            WorkflowNode(id="n2", type="test.consumer"),
        ],
        [Edge(id="e1", source="n1", source_port="data", target="n2", target_port="data")],
    )
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, reg)
    assert any("type mismatch" in e for e in exc.value.errors)


def test_cycle_detected():
    reg = _registry()
    from emptyui import Input, Node, Output

    class Passthru(Node):
        type = "test.passthru"
        inputs = (Input("data", "ts.run"),)
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx, data):
            return {"data": data}

    reg.add(Passthru)
    wf = _wf(
        [
            WorkflowNode(id="a", type="test.passthru"),
            WorkflowNode(id="b", type="test.passthru"),
        ],
        [
            Edge(id="e1", source="a", source_port="data", target="b", target_port="data"),
            Edge(id="e2", source="b", source_port="data", target="a", target_port="data"),
        ],
    )
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, reg)
    assert any("cycle" in e for e in exc.value.errors)


def test_required_input_must_be_connected():
    wf = _wf([WorkflowNode(id="n2", type="test.consumer")], [])
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(wf, _registry())
    assert any("required input" in e for e in exc.value.errors)
