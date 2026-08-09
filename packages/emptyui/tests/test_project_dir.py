"""M23 — the project directory: portable relative paths + recorded plugin
versions (user decisions 2026-10-15).

Four dispositions landed:

1. **Plugin versions recorded**: each catalog manifest carries its plugin's
   distribution version (``plugin_version``); the serializer records it per
   node; the executor compares it at run time and warns on the node
   (``node_warning`` — the H7 channel) and in the log. The run ALWAYS uses
   the installed codebase; the record is informational.
2. **models**: marked planned in DESIGN (no implementation — nothing uses
   model weights yet). Not tested here (a doc disposition).
3. **X-Forwarded-\\***: marked planned in DESIGN (waits for a real web
   server). Not tested here (a doc disposition).
4. **project_dir**: live — a relative ``in`` path anchors to it (data root
   takes precedence), cache signatures are project-relative (relocation
   stable), and the save-time relativization makes workflows portable.
   Unset = the server CWD (status quo).
"""

from __future__ import annotations

import logging
from pathlib import Path

from emptyui import Node, Output, Param
from emptyui.executor import Executor
from emptyui.paths import PathPolicy
from emptyui.registry import PluginInfo, Registry
from emptyui.schema import Workflow, WorkflowNode


def _manifest():
    from emptyui import Input, Output

    class Io(Node):
        type = "test.io"
        inputs = (Input("data", "any", required=False),)
        outputs = (Output("ok", "any"),)
        params = (
            Param("input_path", "file", path_kind="in", default=""),
            Param("plain", "string", default="x"),
        )

        def execute(self, ctx, input_path="", plain="x", data=None):
            return {"ok": {"input": input_path, "plain": plain}}

    return Io


def _reader():
    """A node that actually touches the file, for end-to-end anchoring."""

    class Reader(Node):
        type = "test.reader"
        outputs = (Output("ok", "any"),)
        params = (Param("path", "file", path_kind="in", default=""),)

        def execute(self, ctx, path=""):
            p = Path(path)
            return {"ok": {"path": str(p), "exists": p.is_file()}}

    return Reader


def _plugin_registry(version: str = "1.2.3") -> Registry:
    reg = Registry()
    reg.add(_reader())
    # simulate a discovered plugin's attribution (discover_plugins fills
    # these two tables; a manual `add` carries none)
    reg._plugins["testplug"] = PluginInfo(
        name="testplug", version=version, bundled=False,
        node_types=["test.reader"])
    reg._plugin_of["test.reader"] = "testplug"
    return reg


# --- 1. plugin_version in the manifest + drift warning ----------------------


def test_manifest_carries_plugin_version():
    manifest = _plugin_registry().manifests()["nodes"][0]
    assert manifest["plugin_version"] == "1.2.3"


def test_manual_node_manifest_has_no_plugin_version():
    reg = Registry()
    reg.add(_reader())
    assert "plugin_version" not in reg.manifests()["nodes"][0]


def _wf(version: str | None = None) -> Workflow:
    node = WorkflowNode(
        id="n1", type="test.reader", params={"path": "data/x.csv"},
        plugin_version=version)
    return Workflow(nodes=[node])


def test_version_drift_warns_on_the_node_and_in_the_log(caplog):
    events: list[dict] = []
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        summary = Executor(_plugin_registry("9.9.9")).run(
            _wf("1.2.3"), emit=events.append)
    assert summary["status"] == "completed"
    warnings = [e for e in events if e["event"] == "node_warning"]
    assert len(warnings) == 1
    assert warnings[0]["node"] == "n1"
    assert "1.2.3" in warnings[0]["msg"] and "9.9.9" in warnings[0]["msg"]
    assert "re-save" in warnings[0]["msg"]
    assert any("9.9.9" in r.getMessage() for r in caplog.records)


def test_matching_or_absent_or_unattributed_version_does_not_warn():
    events: list[dict] = []
    Executor(_plugin_registry("1.2.3")).run(_wf("1.2.3"), emit=events.append)
    Executor(_plugin_registry("1.2.3")).run(_wf(None), emit=events.append)
    # a manual/test node registered outside any plugin: nothing to compare
    bare = Registry()
    bare.add(_reader())
    Executor(bare).run(_wf("1.2.3"), emit=events.append)
    assert not [e for e in events if e["event"] == "node_warning"]


def test_version_drift_warns_on_cache_hits_too():
    """The record is a document fact, not a computation result — a cache
    hit must not swallow it (and the frontend keeps cached-node warnings:
    no `running` event clears them)."""
    from emptyui.cache import NodeCache

    events: list[dict] = []
    ex = Executor(_plugin_registry("9.9.9"))
    cache = NodeCache(max_entries=16)
    ex.run(_wf("1.2.3"), emit=events.append, cache=cache)  # populates it
    events.clear()
    ex.run(_wf("1.2.3"), emit=events.append, cache=cache)
    cached = [e for e in events if e.get("cached")]
    assert cached, "second run must hit the cache"
    assert [e for e in events if e["event"] == "node_warning"]


# --- 4. project_dir: anchoring, signatures, end-to-end ----------------------


def test_pathpolicy_anchors_relative_inputs_to_project_dir(tmp_path):
    manifest = _manifest()()
    out = PathPolicy(None, None, project_dir=tmp_path).resolve_params(
        manifest, {"input_path": "data/x.csv", "plain": "y"})
    assert out["input_path"] == str(tmp_path / "data" / "x.csv")
    assert out["plain"] == "y"  # non-path params untouched


def test_pathpolicy_absolute_input_untouched_without_data_root(tmp_path):
    manifest = _manifest()()
    absolute = tmp_path / "elsewhere.csv"
    out = PathPolicy(None, None, project_dir=tmp_path).resolve_params(
        manifest, {"input_path": str(absolute)})
    assert out["input_path"] == str(absolute)


def test_pathpolicy_data_root_takes_precedence(tmp_path):
    data_root = tmp_path / "data_root"
    project = tmp_path / "project"
    manifest = _manifest()()
    out = PathPolicy(data_root, None, project_dir=project).resolve_params(
        manifest, {"input_path": "x.csv"})
    assert out["input_path"] == str(data_root / "x.csv")


def test_signature_is_project_relative_and_relocation_stable(tmp_path):
    """The absolute and the project-relative form of the SAME file produce
    the SAME cache signature (§7.1/§12.3 relocation promise) — a workflow
    that moved from absolute to relative paths (or across machines) stays
    a cache hit."""
    proj = tmp_path / "proj"
    f = proj / "data"
    f.mkdir(parents=True)
    f = f / "x.csv"
    f.write_text("t,v\n1,2\n")
    sig_rel = Executor._value_signature(
        "data/x.csv", data_root=None, path_kind="in",
        widget="file", project_dir=proj)
    sig_abs = Executor._value_signature(
        str(f), data_root=None, path_kind="in",
        widget="file", project_dir=proj)
    assert sig_rel == sig_abs
    assert sig_rel[0] == "data/x.csv"  # the POSIX relative form
    # a relative input stats under the project dir, not the CWD
    assert sig_rel[1:] == [f.stat().st_size, f.stat().st_mtime_ns]


def test_relative_input_resolves_and_runs_end_to_end(tmp_path, monkeypatch):
    """A project-relative workflow runs from any CWD: the executor hands
    the node the project-dir-anchored absolute path."""
    proj = tmp_path / "proj"
    d = proj / "data"
    d.mkdir(parents=True)
    (d / "x.csv").write_text("t,v\n1,2\n")
    monkeypatch.chdir(tmp_path)  # the CWD deliberately lacks data/x.csv
    events: list[dict] = []
    summary = Executor(_plugin_registry()).run(
        _wf(), emit=events.append, project_dir=proj)
    assert summary["status"] == "completed"
    # the node saw the anchored path and the file existed
    node_status = [e for e in events if e["event"] == "node_status"]
    assert node_status[-1]["status"] == "done"


def test_relocation_across_project_dirs_stays_a_cache_hit(tmp_path):
    """The §12.3 relocation scenario: the signature's PATH half is the
    project-relative form under either root, so an mtime-preserving move
    (shutil.copy2) to a relocated project keeps the SAME signature — a
    fresh write (new mtime) correctly misses, the safe direction."""
    import shutil

    sigs = []
    for name in ("proj_a", "proj_b"):
        proj = tmp_path / name
        d = proj / "data"
        d.mkdir(parents=True)
        f = d / "x.csv"
        f.write_text("t,v\n1,2\n")
        sigs.append(Executor._value_signature(
            "data/x.csv", data_root=None, path_kind="in",
            widget="file", project_dir=proj))
    # the path half is the relative form under both roots
    assert sigs[0][0] == sigs[1][0] == "data/x.csv"
    # an mtime-preserving copy (copy2) of proj_a's file over proj_b's
    # equalizes the stat half too — the relocated project stays a hit
    shutil.copy2(tmp_path / "proj_a/data/x.csv", tmp_path / "proj_b/data/x.csv")
    sig_b2 = Executor._value_signature(
        "data/x.csv", data_root=None, path_kind="in",
        widget="file", project_dir=tmp_path / "proj_b")
    assert sig_b2 == sigs[0]


def test_schema_tolerates_plugin_version():
    wf = _wf("0.6.8")
    assert wf.nodes[0].plugin_version == "0.6.8"
    assert _wf(None).nodes[0].plugin_version is None


# --- config / settings surface ----------------------------------------------


def test_settings_from_env_project_dir(monkeypatch):
    from emptyui.config import Settings

    monkeypatch.setenv("EMPTYUI_PROJECT_DIR", "~/survey")
    assert Settings.from_env().project_dir == Path("~/survey").expanduser()
    monkeypatch.delenv("EMPTYUI_PROJECT_DIR")
    assert Settings.from_env().project_dir is None


def test_deployment_view_shows_effective_project_dir(tmp_path):
    from fastapi.testclient import TestClient

    from emptyui.server.app import create_app

    client = TestClient(create_app(_plugin_registry()))
    view = client.get("/api/settings").json()
    # unset → the effective anchor is the server CWD (absolute)
    assert Path(view["deployment"]["project_dir"]).is_absolute()

    client2 = TestClient(create_app(_plugin_registry(), project_dir=tmp_path))
    view2 = client2.get("/api/settings").json()
    assert view2["deployment"]["project_dir"] == str(tmp_path)
