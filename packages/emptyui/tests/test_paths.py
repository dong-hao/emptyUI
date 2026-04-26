"""Path confinement (DESIGN.md §11.2) — remote-mode data-root policy.

Feature tests: PathPolicy resolution/validation and the manifest
``path_kind`` serialization. Error surfaces (outside-root rejection) are
covered here as they are the feature's primary behavior; server-side
clamping lives in test_server.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from emptyui import Node, Param
from emptyui.executor import Executor
from emptyui.paths import PathPolicy
from emptyui.registry import Registry
from emptyui.schema import Workflow, WorkflowNode


def _manifest():
    from emptyui import Input, Output

    class Io(Node):
        type = "test.io"
        inputs = (Input("data", "any", required=False),)
        outputs = (Output("ok", "any"),)
        params = (
            Param("input_path", "file", path_kind="in", default=""),
            Param("output_dir", "directory", path_kind="out", default=""),
            Param("plain", "string", default="x"),
        )

        def execute(self, ctx, input_path="", output_dir="", plain="x", data=None):
            return {"ok": {"input": input_path, "output": output_dir, "plain": plain}}

    return Io


def test_manifest_serializes_path_kind():
    assert _manifest()().manifest()["params"][0]["path_kind"] == "in"


def test_settings_from_env(monkeypatch):
    from emptyui.config import Settings

    monkeypatch.setenv("EMPTYUI_DATA_ROOT", "~/data")
    monkeypatch.setenv("EMPTYUI_SERVE_DIST", "dist")
    monkeypatch.setenv("EMPTYUI_CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("EMPTYUI_WORKSPACE_TTL", "30.5")
    monkeypatch.setenv("EMPTYUI_MAX_CONCURRENT_RUNS", "2")
    s = Settings.from_env()
    assert s.data_root == Path("~/data").expanduser()
    assert s.serve_dist == "dist"
    assert s.cors_origins == ("https://a.example", "https://b.example")
    assert s.workspace_ttl_seconds == 30.5
    assert s.max_concurrent_runs == 2

    monkeypatch.delenv("EMPTYUI_DATA_ROOT")
    monkeypatch.setenv("EMPTYUI_CORS_ORIGINS", "")
    assert Settings.from_env().data_root is None
    assert Settings.from_env().cors_origins == ()

    monkeypatch.delenv("EMPTYUI_CORS_ORIGINS")
    assert Settings.from_env().cors_origins == ("*",)  # unset stays permissive
    monkeypatch.delenv("EMPTYUI_WORKSPACE_TTL")
    monkeypatch.delenv("EMPTYUI_MAX_CONCURRENT_RUNS")
    s = Settings.from_env()
    assert s.workspace_ttl_seconds is None
    assert s.max_concurrent_runs is None


def test_no_data_root_is_a_noop(tmp_path: Path):
    policy = PathPolicy(None, tmp_path)
    params = {"input_path": "/abs/in.csv", "output_dir": "out", "plain": "x"}
    assert policy.resolve_params(_manifest(), params) == params


def test_in_params_resolve_under_data_root(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    policy = PathPolicy(root, tmp_path / "ws")
    resolved = policy.resolve_params(_manifest(), {"input_path": "station.csv"})
    assert resolved["input_path"] == str(root / "station.csv")


def test_in_param_absolute_inside_root_allowed(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    f = root / "a.csv"
    f.write_text("x")
    policy = PathPolicy(root, tmp_path / "ws")
    resolved = policy.resolve_params(_manifest(), {"input_path": str(f)})
    assert resolved["input_path"] == str(f)


def test_in_param_outside_root_rejected(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "a.csv"
    policy = PathPolicy(root, tmp_path / "ws")
    with pytest.raises(ValueError, match="outside the data root"):
        policy.resolve_params(_manifest(), {"input_path": str(outside)})


def test_out_params_resolve_into_workspace(tmp_path: Path):
    ws = tmp_path / "ws"
    policy = PathPolicy(tmp_path / "data", ws)
    resolved = policy.resolve_params(_manifest(), {"output_dir": "results"})
    assert resolved["output_dir"] == str(ws / "results")
    # empty output param defaults to the workspace itself
    resolved = policy.resolve_params(_manifest(), {"output_dir": ""})
    assert resolved["output_dir"] == str(ws)


def test_out_param_absolute_is_rejected_in_remote_mode(tmp_path: Path):
    outside = tmp_path / "abs"
    policy = PathPolicy(tmp_path / "data", tmp_path / "ws")
    with pytest.raises(ValueError, match="absolute output paths"):
        policy.resolve_params(_manifest(), {"output_dir": str(outside)})


def test_out_param_dotdot_escape_is_rejected(tmp_path: Path):
    policy = PathPolicy(tmp_path / "data", tmp_path / "ws")
    with pytest.raises(ValueError, match="escapes the run workspace"):
        policy.resolve_params(_manifest(), {"output_dir": "../evil"})
    with pytest.raises(ValueError, match="escapes the run workspace"):
        policy.resolve_params(_manifest(), {"output_dir": "sub/../../evil"})


def test_executor_applies_policy_with_data_root(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "in.csv").write_text("x")
    seen: dict[str, str] = {}

    class Io(_manifest()):
        def execute(self, ctx, input_path="", output_dir="", plain="x", data=None):
            seen["input"] = input_path
            seen["output"] = output_dir
            return {"ok": True}

    reg = Registry()
    reg.add(Io)
    events: list[dict] = []
    summary = Executor(reg).run(
        Workflow(
            nodes=[
                WorkflowNode(
                    id="n1",
                    type="test.io",
                    params={"input_path": "in.csv", "output_dir": "res"},
                )
            ],
            edges=[],
        ),
        emit=events.append,
        data_root=root,
        workspace=str(tmp_path / "ws"),
    )
    assert summary["status"] == "completed"
    assert seen["input"] == str(root / "in.csv")
    assert seen["output"] == str(tmp_path / "ws" / "res")
