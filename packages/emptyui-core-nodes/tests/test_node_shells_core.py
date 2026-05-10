"""Node-shell smoke tests (issues F1): manifest/param wiring and artifact
paths for the display nodes whose underlying behavior was already covered —
Preview Plot, Save Plot, View Table. Also pins the manifest `viewer` flag
(D1: the frontend must never switch on node-type literals, L-F12)."""

from __future__ import annotations

import inspect

import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes.preview import PreviewPlotNode
from emptyui_core_nodes.save_plot import SavePlotNode
from emptyui_core_nodes.table_view import ViewTableNode


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


def test_shells_and_viewer_flags():
    _assert_shell(PreviewPlotNode(), viewer="plot")
    _assert_shell(ViewTableNode(), viewer="table")
    _assert_shell(SavePlotNode(), viewer=None)


def test_viewer_flag_reaches_the_catalog():
    """D1 end-to-end: the flag survives registry serialization and every
    bundled manifest declares a viewer kind the frontend can render."""
    from emptyui.registry import build_default_registry

    manifests = {m["type"]: m
                 for m in build_default_registry().manifests()["nodes"]}
    assert manifests["core.display.preview_plot"]["viewer"] == "plot"
    assert manifests["core.display.view_table"]["viewer"] == "table"
    assert manifests["core.io.save_plot"]["viewer"] is None
    assert {m["viewer"] for m in manifests.values()} <= {None, "plot", "table"}


def test_viewer_bodies_pass_through(tmp_path):
    """Viewer bodies render in the browser: execute() forwards the payload
    on the same-named output port (the executor contract, not the figure)."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    plot = {"path": "fig.png", "kind": "image"}
    plot_in = {"plot": plot}  # executor-style: payloads keyed by port NAME
    assert PreviewPlotNode().execute(ctx, **plot_in) == {"plot": plot}
    table = {"path": "table.json", "kind": "table"}
    table_in = {"table": table}
    assert ViewTableNode().execute(ctx, **table_in) == {"table": table}


def test_save_plot_writes_numbered_artifacts(tmp_path):
    src = tmp_path / "figure.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    out = tmp_path / "out"
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    node = SavePlotNode()
    first = node.execute(ctx, plot={"path": str(src)}, output_dir=str(out))
    second = node.execute(ctx, plot={"path": str(src)}, output_dir=str(out))
    assert (out / "plot_001.png").exists()
    assert (out / "plot_002.png").exists()
    assert first["_saved"]["kind"] == "image"
    assert first["_saved"]["path"].endswith("plot_001.png")
    assert second["_saved"]["path"].endswith("plot_002.png")
    assert len(ctx.published) == 2


def test_save_plot_rejects_cross_format_export(tmp_path):
    src = tmp_path / "figure.png"
    src.write_bytes(b"fake")
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    with pytest.raises(ValueError, match="same-format export"):
        SavePlotNode().execute(ctx, plot={"path": str(src)},
                                output_dir=str(tmp_path), file_format="svg")