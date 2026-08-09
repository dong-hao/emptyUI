"""M1 — server-side validation of param *values* at the schema choke point.

``validate_workflow`` must reject values that violate the manifest's own
declarations — ``min``/``max`` bounds, ``select``/``radio`` ``options``, and
required (non-optional, default-less) params left unprovided — with the same
actionable ``node (...)`` messages as the structural checks. Widgets guard
the UI, but DESIGN §11 makes the API the trust boundary in remote mode:
a workflow POSTed with ``{"order": -5}`` on a ``min=1, max=10`` param must
422, not run.

The coercion parity cases (string-carried numbers, bool strings, numeric
select options) pin that validation accepts exactly what the executor's
``_coerce_param`` would pass to node code — validation coerces with the
shared helper, so the two can never diverge.
"""

from __future__ import annotations

import pytest

from emptyui.registry import Registry
from emptyui.schema import (
    ValidationError_,
    Workflow,
    WorkflowNode,
    validate_workflow,
)


def _registry():
    from emptyui import Input, Node, Output, Param

    reg = Registry()

    class Bounded(Node):
        type = "test.bounded"
        inputs = (Input("data", "ts.run"),)
        outputs = (Output("data", "ts.run"),)
        params = (
            Param("order", "integer", default=4, min=1, max=10),
            Param("factor", "number", default=1.0, min=0.5),
            Param("taper", "select", default="boxcar",
                  options=("boxcar", "hamming", "hanning")),
            Param("mode", "radio", default="a", options=("a", "b")),
            Param("path", "file"),
            Param("low_freq", "number", default=None, optional=True),
        )

        def execute(self, ctx, data, **kw):
            return {"data": data}

    class Source(Node):
        type = "test.source"
        outputs = (Output("data", "ts.run"),)

        def execute(self, ctx):
            return {"data": [1.0, 2.0, 3.0]}

    reg.add(Bounded)
    reg.add(Source)
    return reg


def _wf(**params):
    return Workflow(
        nodes=[
            WorkflowNode(id="n1", type="test.source"),
            WorkflowNode(id="n2", type="test.bounded", params=params),
        ],
        edges=[
            {"id": "e1", "source": "n1", "source_port": "data",
             "target": "n2", "target_port": "data"},
        ],
    )


def test_out_of_range_integer_is_rejected():
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(_wf(order=-5), _registry())
    assert any(
        "n2" in e and "order" in e and "-5" in e for e in exc.value.errors)


def test_out_of_range_number_is_rejected():
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(_wf(factor=0.1), _registry())
    assert any("factor" in e and "0.1" in e for e in exc.value.errors)


def test_above_max_is_rejected():
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(_wf(order=11), _registry())
    assert any("order" in e for e in exc.value.errors)


def test_non_numeric_value_on_number_param_is_rejected():
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(_wf(order="fast"), _registry())
    assert any("order" in e and "not a number" in e
               for e in exc.value.errors)


def test_select_option_violation_is_rejected():
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(_wf(taper="blackman"), _registry())
    assert any(
        "taper" in e and "blackman" in e for e in exc.value.errors)


def test_radio_option_violation_is_rejected():
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(_wf(mode="c"), _registry())
    assert any("mode" in e and "c" in e for e in exc.value.errors)


def test_missing_required_param_is_rejected():
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(_wf(), _registry())
    assert any(
        "path" in e and "required" in e for e in exc.value.errors)


def test_empty_path_value_counts_as_missing():
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(_wf(path=""), _registry())
    assert any("path" in e and "required" in e for e in exc.value.errors)


def test_none_required_param_counts_as_missing():
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(_wf(path=None), _registry())
    assert any("path" in e and "required" in e for e in exc.value.errors)


# --- coercion parity: validation accepts what execution coerces ----------


def test_string_integer_within_bounds_passes():
    validate_workflow(_wf(order="4", path="x.h5"), _registry())  # no exception


def test_string_number_coerced_before_bounds_check_passes():
    validate_workflow(_wf(factor="2.5", path="x.h5"), _registry())  # no exception


def test_boundary_values_pass():
    validate_workflow(_wf(order=1, factor=0.5, path="x.h5"), _registry())  # no exception


def test_numeric_string_matching_a_select_option_passes():
    reg = _registry()

    from emptyui import Param

    bounded = reg.get("test.bounded")
    bounded.params = bounded.params + (
        Param("levels", "select", default=4, options=(4, 8, 16)),
    )
    validate_workflow(_wf(levels="8", path="x.h5"), reg)  # no exception


def test_boolean_string_passes_through():
    validate_workflow(_wf(order=4, path="x.h5"), _registry())  # sanity: no unrelated failure


# --- documented policies that must survive ------------------------------


def test_optional_param_absent_passes():
    validate_workflow(_wf(order=4, path="x.h5"), _registry())  # no exception


def test_optional_param_none_passes():
    validate_workflow(_wf(order=4, path="x.h5", low_freq=None),
                      _registry())  # no exception


def test_stale_params_still_tolerated():
    # history §24.9 policy: params a node no longer declares are warned about by
    # the executor, never hard-rejected by validation.
    validate_workflow(_wf(removed_param=3, order=4, path="x.h5"),
                      _registry())  # no exception


def test_all_errors_aggregated_not_first_only():
    with pytest.raises(ValidationError_) as exc:
        validate_workflow(_wf(order=-5, taper="blackman", path=None),
                          _registry())
    msgs = exc.value.errors
    assert any("order" in e for e in msgs)
    assert any("taper" in e for e in msgs)
    assert any("path" in e for e in msgs)
