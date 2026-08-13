"""Node-shell smoke tests (issues F1): manifest/param wiring for the
aurora processing nodes whose underlying functions were already covered
directly — Merge FC and Merge TF Levels."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import estimate_tf_levels
from emptyui_aurora.process import MergeTFLevels
from test_sort_drop_grammar import _container

pytestmark = pytest.mark.filterwarnings("ignore")


def _assert_shell(node, *, viewer=None) -> None:
    """Manifest and execute() wiring agree: unique names, and every
    declared input/param is an execute() kwarg (the executor passes both
    as **kwargs — a renamed parameter would only fail at run time)."""
    m = node.manifest()
    assert m["type"] == node.type
    assert m["title"] and m["category"]
    assert m["viewer"] is viewer
    for key in ("inputs", "outputs"):
        names = [p["name"] for p in m[key]]
        assert len(names) == len(set(names)), f"duplicate {key} names"
    params = [p["name"] for p in m["params"]]
    assert len(params) == len(set(params)), "duplicate param names"
    sig = inspect.signature(node.execute)
    assert next(iter(sig.parameters)) == "ctx"
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        for name in [p["name"] for p in m["inputs"]] + params:
            assert name in sig.parameters, f"execute() ignores {name!r}"
    for p in m["params"]:
        if p.get("options") and p.get("default") is not None:
            assert p["default"] in p["options"], p["name"]


def test_shells():
    _assert_shell(MergeTFLevels(), viewer=None)


def test_merge_tf_levels_node_builds_the_broadband_tf(tmp_path):
    """The full Estimate TF → Merge TF Levels shell: per-level entries in,
    one sorted broadband TF out (duplicate periods resolved)."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    levels = _container()
    est = estimate_tf_levels(
        levels, ["hx", "hy"], ["ex", "ey"], estimator_engine="OLS",
        periods_per_octave=3, band_style="uniform")
    tls = {"tf_levels": est}  # executor-style: keyed by port NAME
    out = MergeTFLevels().execute(ctx, **tls)
    assert set(out) == {"tf"}
    period = out["tf"].dataset["transfer_function"].coords["period"].values
    assert len(period) >= 1
    assert len(np.unique(period)) == len(period)


def test_merge_tf_levels_empty_input_names_the_remedy(tmp_path):
    """The L-A9 regression: an empty tf-levels payload is a loud,
    actionable error instead of a raw StopIteration."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    # the dict-literal key IS the point: payloads are keyed by port NAME
    with pytest.raises(ValueError, match="Estimate TF"):
        MergeTFLevels().execute(ctx, **{"tf_levels": {}})  # noqa: PIE804