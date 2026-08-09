"""Plugin enable/disable (DESIGN.md §5.1) — happy path.

Error surfaces (disabled-plugin validation message, unknown-plugin 404)
live in test_errors.py. The suite builds fake entry points so it never
depends on which plugins happen to be installed in the dev/CI environment.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from emptyui import Node
from emptyui.config import Settings
from emptyui.registry import Registry, build_default_registry, discover_plugins
from emptyui.server.app import create_app


class FakeCore(Node):
    type = "fake.core"
    outputs = ()
    runs = 0

    def execute(self, ctx):
        type(self).runs += 1
        return {}


class FakeSlow(Node):
    type = "fake.slow"
    outputs = ()
    started = threading.Event()

    def execute(self, ctx):
        self.started.set()
        time.sleep(0.3)
        return {}


class FakeFast(Node):
    type = "fake.fast"
    outputs = ()

    def execute(self, ctx):
        return {}


def _register(*classes):
    def register(reg):
        for cls in classes:
            reg.add(cls)

    return register


def _fake_eps():
    core = _register(FakeCore)
    core.bundled = True
    return [
        SimpleNamespace(name="core", load=lambda r=core: r, dist=None),
        SimpleNamespace(
            name="third", load=lambda: _register(FakeSlow, FakeFast), dist=None
        ),
    ]


def _patch_plugins(monkeypatch):
    monkeypatch.setattr("emptyui.registry.entry_points", lambda group=None: _fake_eps())


@pytest.fixture()
def plugin_app(tmp_path, monkeypatch):
    _patch_plugins(monkeypatch)
    return create_app(
        build_default_registry(),
        settings=Settings(settings_path=tmp_path / "settings.json"),
    )


def _wait_status(client, run_id, timeout=5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = client.get(f"/api/runs/{run_id}").json()
        if detail["status"] in ("completed", "failed", "cancelled"):
            return detail
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish in {timeout}s")


# --- discovery / registry --------------------------------------------------


def test_discovery_records_plugins_and_disabled_state(monkeypatch):
    _patch_plugins(monkeypatch)
    reg = Registry()
    discover_plugins(reg, disabled={"third"})
    infos = {p["name"]: p for p in reg.plugins()}
    assert set(infos) == {"core", "third"}
    assert infos["core"]["bundled"] is True
    assert infos["core"]["enabled"] is True
    assert infos["third"]["enabled"] is False
    assert infos["third"]["node_count"] == 2
    assert reg.plugin("third").node_types == ["fake.fast", "fake.slow"]
    # disabled nodes are not registered; status stays actionable
    assert reg.get("fake.slow") is None
    assert reg.status_of("fake.slow") == "disabled"
    assert reg.status_of("fake.core") == "active"
    assert reg.status_of("nope.nope") == "unknown"
    assert reg.plugin_name_of("fake.slow") == "third"


def test_rebuild_reenables_plugin_nodes(monkeypatch):
    _patch_plugins(monkeypatch)
    reg = Registry()
    discover_plugins(reg, disabled={"third"})
    assert reg.get("fake.fast") is None
    reg.rebuild(disabled=set())
    assert reg.get("fake.fast") is not None
    assert reg.status_of("fake.fast") == "active"


def test_unknown_disabled_names_silently_ignored(monkeypatch):
    _patch_plugins(monkeypatch)
    reg = build_default_registry(disabled_plugins=("nope", "missing"))
    assert reg.get("fake.core") is not None
    assert all(p["enabled"] for p in reg.plugins())


def test_manual_nodes_survive_rebuild(monkeypatch):
    _patch_plugins(monkeypatch)

    class Manual(Node):
        type = "manual.node"
        outputs = ()

        def execute(self, ctx):
            return {}

    reg = Registry()
    reg.add(Manual)
    discover_plugins(reg)
    reg.rebuild(disabled={"third"})
    assert reg.get("manual.node") is not None
    assert reg.get("fake.core") is not None
    assert reg.get("fake.slow") is None


# --- server API ------------------------------------------------------------


def test_toggles_work_without_settings_file(monkeypatch):
    """Default is memory-only (§5.1): no settings path configured, toggles
    still work and nothing is written to disk."""
    _patch_plugins(monkeypatch)
    client = TestClient(create_app(build_default_registry()))
    assert client.post("/api/plugins/third/disable").json()["enabled"] is False
    assert client.post("/api/plugins/third/enable").json()["enabled"] is True


def test_api_disable_enable_roundtrip(plugin_app, tmp_path):
    client = TestClient(plugin_app)
    third = next(p for p in client.get("/api/plugins").json() if p["name"] == "third")
    assert third["enabled"] is True

    resp = client.post("/api/plugins/third/disable")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
    types = [m["type"] for m in client.get("/api/nodes").json()["nodes"]]
    assert "fake.slow" not in types
    assert "fake.core" in types
    stored = json.loads((tmp_path / "settings.json").read_text())
    assert stored["disabled_plugins"] == ["third"]

    resp = client.post("/api/plugins/third/enable")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    types = [m["type"] for m in client.get("/api/nodes").json()["nodes"]]
    assert "fake.slow" in types
    stored = json.loads((tmp_path / "settings.json").read_text())
    assert stored["disabled_plugins"] == []


def test_bundled_toggle_is_noop(plugin_app, tmp_path):
    client = TestClient(plugin_app)
    for action in ("disable", "enable"):
        resp = client.post(f"/api/plugins/core/{action}")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is True
    # bundled plugins are always on: no settings file entry is written
    assert not (tmp_path / "settings.json").exists()


def test_unknown_plugin_404(plugin_app):
    client = TestClient(plugin_app)
    resp = client.post("/api/plugins/nope/disable")
    assert resp.status_code == 404
    assert "unknown plugin" in resp.json()["detail"]


def test_settings_file_baseline_at_startup(monkeypatch, tmp_path):
    _patch_plugins(monkeypatch)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"disabled_plugins": ["third"]}))
    app = create_app(settings=Settings(settings_path=settings_file))
    client = TestClient(app)
    third = next(p for p in client.get("/api/plugins").json() if p["name"] == "third")
    assert third["enabled"] is False


def test_env_baseline_wins_over_settings_file(monkeypatch, tmp_path):
    _patch_plugins(monkeypatch)
    settings_file = tmp_path / "settings.json"
    # the settings file enables third, but the platform baseline disables it
    settings_file.write_text(json.dumps({"disabled_plugins": []}))
    app = create_app(
        settings=Settings(disabled_plugins=("third",), settings_path=settings_file),
    )
    client = TestClient(app)
    third = next(p for p in client.get("/api/plugins").json() if p["name"] == "third")
    assert third["enabled"] is False
    # a UI enable toggle cannot override the platform baseline
    resp = client.post("/api/plugins/third/enable")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
    # the toggle still persisted (file entry removed); env keeps it off
    stored = json.loads(settings_file.read_text())
    assert stored["disabled_plugins"] == []


def test_disable_clears_cross_run_cache(plugin_app):
    client = TestClient(plugin_app)
    FakeCore.runs = 0
    workflow = {"nodes": [{"id": "n1", "type": "fake.core"}], "edges": []}
    run_id = client.post("/api/runs", json={"workflow": workflow}).json()["run_id"]
    assert _wait_status(client, run_id)["status"] == "completed"
    assert FakeCore.runs == 1
    run_id = client.post("/api/runs", json={"workflow": workflow}).json()["run_id"]
    assert _wait_status(client, run_id)["status"] == "completed"
    assert FakeCore.runs == 1  # second run was served from the cache
    client.post("/api/plugins/third/disable")
    run_id = client.post("/api/runs", json={"workflow": workflow}).json()["run_id"]
    assert _wait_status(client, run_id)["status"] == "completed"
    assert FakeCore.runs == 2  # the toggle cleared the cache


def test_executor_snapshot_survives_mid_run_toggle(plugin_app):
    client = TestClient(plugin_app)
    FakeSlow.started.clear()
    workflow = {
        "nodes": [
            {"id": "n1", "type": "fake.slow"},
            {"id": "n2", "type": "fake.fast"},
        ],
        "edges": [],
    }
    run_id = client.post("/api/runs", json={"workflow": workflow}).json()["run_id"]
    assert FakeSlow.started.wait(timeout=2)
    # disable the plugin while n1 executes: n2 must still run under the
    # manifest snapshot taken at run start (§5.1)
    assert client.post("/api/plugins/third/disable").status_code == 200
    detail = _wait_status(client, run_id)
    assert detail["status"] == "completed"
    statuses = {
        e["node"]: e["status"]
        for e in detail["events"]
        if e["event"] == "node_status"
    }
    assert statuses == {"n1": "done", "n2": "done"}


class FakeDup(Node):
    """Declares the same type as FakeCore — a cross-plugin collision."""
    type = "fake.core"
    outputs = ()

    def execute(self, ctx):
        return {}


class FakeOk(Node):
    """A non-conflicting type registered BEFORE the duplicate."""
    type = "fake.ok"
    outputs = ()

    def execute(self, ctx):
        return {}


class FakeBrokenImport:
    """An entry point whose load() itself fails."""


def test_disabled_plugin_type_collision_does_not_crash_discovery(monkeypatch):
    """A disabled plugin whose node type collides with an enabled plugin must
    not crash discovery (previously register() raised duplicate-type before
    the disabled pop — outcome depended on entry-point order)."""
    core = _register(FakeCore)
    core.bundled = True
    eps = [
        SimpleNamespace(name="core", load=lambda r=core: r, dist=None),
        SimpleNamespace(name="third", load=lambda: _register(FakeDup), dist=None),
    ]
    monkeypatch.setattr("emptyui.registry.entry_points", lambda group=None: eps)

    registry = build_default_registry(disabled_plugins={"third"})
    assert registry.get("fake.core").__class__ is FakeCore  # enabled plugin wins
    assert registry.plugin("third").enabled is False
    assert "fake.core" in registry.plugin("third").node_types  # learned


def test_enabled_duplicate_plugin_degrades_with_error_row(monkeypatch):
    """M20 (decision 2026-10-13): an ENABLED plugin declaring a duplicate
    node type degrades like any other registration failure — the instance
    starts, the plugin is skipped with an error, and the first
    registration wins the type. Previously the ValueError escaped
    discovery and bricked startup (and made every toggle 409)."""
    core = _register(FakeCore)
    core.bundled = True
    eps = [
        SimpleNamespace(name="core", load=lambda r=core: r, dist=None),
        SimpleNamespace(name="third", load=lambda: _register(FakeDup), dist=None),
    ]
    monkeypatch.setattr("emptyui.registry.entry_points", lambda group=None: eps)

    registry = build_default_registry()  # does not raise
    assert registry.get("fake.core").__class__ is FakeCore  # first wins
    info = registry.plugin("third")
    assert info is not None
    assert info.enabled is False
    assert info.node_types == []
    assert "duplicate node type" in info.error
    assert "fake.core" in info.error


def test_duplicate_plugin_registration_is_all_or_nothing(monkeypatch):
    """A plugin that registers a good node and THEN hits the duplicate must
    not leave the good node half-registered (its other types are rolled
    back — the plugin is skipped whole, like any other failure)."""
    core = _register(FakeCore)
    core.bundled = True
    eps = [
        SimpleNamespace(name="core", load=lambda r=core: r, dist=None),
        SimpleNamespace(
            name="third", load=lambda: _register(FakeOk, FakeDup), dist=None
        ),
    ]
    monkeypatch.setattr("emptyui.registry.entry_points", lambda group=None: eps)

    registry = build_default_registry()
    assert registry.get("fake.core").__class__ is FakeCore
    assert registry.get("fake.ok") is None  # rolled back with the plugin
    assert registry.plugin("third").error != ""
    assert registry.plugin("third").node_types == []


def test_failed_plugins_reach_the_plugins_api(monkeypatch):
    """A failed plugin (import error OR duplicate type) is recorded in the
    registry so GET /api/plugins shows the error row — previously the
    info existed only in discovery's local list and the API never saw it
    (the 'shows up disabled with a note in /api/plugins' comment was
    aspirational)."""
    core = _register(FakeCore)
    core.bundled = True

    def broken_register():
        raise RuntimeError("no heavy dep")

    eps = [
        SimpleNamespace(name="core", load=lambda r=core: r, dist=None),
        SimpleNamespace(
            name="third", load=lambda: _register(FakeDup), dist=None
        ),
        SimpleNamespace(
            name="broken",
            load=lambda: (_ for _ in ()).throw(
                ImportError("missing heavy dep")),
            dist=None,
        ),
    ]
    monkeypatch.setattr("emptyui.registry.entry_points", lambda group=None: eps)
    registry = build_default_registry()
    rows = {p["name"]: p for p in registry.plugins()}
    assert "duplicate node type" in rows["third"]["error"]
    assert rows["third"]["enabled"] is False
    assert "missing heavy dep" in rows["broken"]["error"]
    assert rows["core"]["error"] == ""


def test_duplicate_plugin_api_surface(monkeypatch):
    """End-to-end over HTTP: the app starts with the conflicting plugin,
    the catalog and runs keep working, and re-enabling the plugin surfaces
    the conflict as an error row — never a 500/409."""
    core = _register(FakeCore)
    core.bundled = True
    eps = [
        SimpleNamespace(name="core", load=lambda r=core: r, dist=None),
        SimpleNamespace(name="third", load=lambda: _register(FakeDup), dist=None),
    ]
    monkeypatch.setattr("emptyui.registry.entry_points", lambda group=None: eps)
    client = TestClient(create_app(build_default_registry()))
    plugins = client.get("/api/plugins").json()
    third = next(p for p in plugins if p["name"] == "third")
    assert third["enabled"] is False and "duplicate node type" in third["error"]
    # the catalog is intact (first registration wins)
    assert any(n["type"] == "fake.core" for n in client.get("/api/nodes").json()["nodes"])
    # re-enabling surfaces the conflict in the row, not as a 500/409
    resp = client.post("/api/plugins/third/enable")
    assert resp.status_code == 200
    third = next(p for p in resp.json()["plugins"] if p["name"] == "third")
    assert "duplicate node type" in third["error"]


def test_rebuild_swaps_node_table_atomically(monkeypatch):
    """rebuild() builds into a scratch registry and swaps: the live table is
    never observed empty/partial (a run starting mid-toggle sees the old or
    the new table, not a broken one)."""
    eps = _fake_eps()
    monkeypatch.setattr("emptyui.registry.entry_points", lambda group=None: eps)

    registry = build_default_registry()

    # rebuild with third disabled: fake.core must remain continuously visible
    registry.rebuild({"third"})
    assert registry.get("fake.core") is not None
    assert registry.get("fake.slow") is None

    registry.rebuild(set())
    assert registry.get("fake.core") is not None
    assert registry.get("fake.slow") is not None


# --- low-list: a broken third-party plugin must not crash discovery -------

def test_broken_plugin_degrades_gracefully():
    """A plugin whose import or register() raises must be skipped with a
    recorded error, not take down discovery/startup (§5.1 robustness)."""
    from types import SimpleNamespace

    import emptyui.registry as R

    def good_register(r):
        r.add(_BrokenOk)

    class _BrokenOk(Node):
        type = "test.broken_ok"

    class BoomEntryPoint:
        name = "broken_plugin"

        def load(self):
            raise ImportError("missing heavy dep")

    ok_ep = SimpleNamespace(name="ok_plugin", group="emptyui.nodes",
                            value="mod:register", load=lambda: good_register)
    real = R.entry_points

    class _FakeSource:
        def __call__(self, group=None):
            return [ok_ep, BoomEntryPoint()]

    R.entry_points = _FakeSource()
    try:
        reg = Registry()
        infos = R.discover_plugins(reg, set())
    finally:
        R.entry_points = real
    by_name = {i.name: i for i in infos}
    assert by_name["ok_plugin"].enabled
    assert "test.broken_ok" in reg.types()
    assert not by_name["broken_plugin"].enabled
    assert "ImportError" in by_name["broken_plugin"].error
    assert by_name["broken_plugin"].to_dict()["error"]
    assert by_name["broken_plugin"].to_dict()["node_count"] == 0


def test_port_type_declaration_is_install_order_independent():
    """L13: two plugins declaring the same port-type tag must resolve to the
    SAME winner regardless of entry-point iteration order — the first
    declaration used to flip with dict insertion order (sorted node types
    now iterate)."""
    from emptyui import Output

    class Aaa(Node):
        type = "aaa.node"
        outputs = (Output("o", "x.kind"),)
        port_types: ClassVar[dict] = {
            "x.kind": {"color": "#111111", "label": "A"}}
        def execute(self, ctx):
            return {}

    class Bbb(Node):
        type = "bbb.node"
        outputs = (Output("o", "x.kind"),)
        port_types: ClassVar[dict] = {
            "x.kind": {"color": "#222222", "label": "B"}}
        def execute(self, ctx):
            return {}

    winner = "#111111"  # aaa.node sorts first -> its declaration wins
    for order in ((Aaa, Bbb), (Bbb, Aaa)):
        reg = Registry()
        for cls in order:
            reg.add(cls)
        assert reg.manifests()["port_types"]["x.kind"]["color"] == winner
