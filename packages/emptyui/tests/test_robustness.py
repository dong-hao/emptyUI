"""Phase 6 robustness & API hygiene regressions (issues #30-35).

Each test targets one latent defect from the review; they were all
uncovered by the pre-existing suite.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from emptyui.registry import Registry, discover_plugins
from emptyui.server.app import create_app
from emptyui.settings import SettingsStore

# --- M-C1/L-C11: settings PUT validation ------------------------------------


def test_settings_put_rejects_non_finite():
    """NaN slipped through the `< 0` check (poisoning the TTL runtime),
    ±inf exploded `round()` with OverflowError (500 instead of 422). NB the
    browser's JSON.stringify maps NaN→null (already rejected), but Python's
    json decoder ACCEPTS NaN/Infinity literals — a raw API client sends
    them; exercise that decode path directly."""
    client = TestClient(create_app(Registry()))
    for literal in ("NaN", "Infinity", "-Infinity"):
        body = json.dumps(
            {"values": {"workspace_ttl_seconds": literal}}).replace(
            f'"{literal}"', literal)
        resp = client.put("/api/settings", content=body,
                          headers={"content-type": "application/json"})
        assert resp.status_code == 422, literal
        # int knob: was a 500 (OverflowError) before the finite guard
        body = json.dumps(
            {"values": {"cache_max_entries": literal}}).replace(
            f'"{literal}"', literal)
        resp = client.put("/api/settings", content=body,
                          headers={"content-type": "application/json"})
        assert resp.status_code == 422, literal
    # the poisoned NaN never reached the runtime: TTL stays off
    view = client.get("/api/settings").json()
    assert view["live"]["workspace_ttl_seconds"]["value"] is None


def test_settings_put_rejects_non_integer():
    """An int knob silently rounded 256.7 → 257 (banker's rounding on top:
    2.5 → 2) — a non-integral value must be rejected, not coerced."""
    client = TestClient(create_app(Registry()))
    resp = client.put(
        "/api/settings", json={"values": {"cache_max_entries": 256.7}})
    assert resp.status_code == 422
    assert "integer" in resp.json()["detail"]


def test_settings_store_rolls_back_failed_save(tmp_path):
    """L-C8: a failed settings-file write must not leave memory claiming a
    value the file never received (the live knobs stay consistent with
    disk; the caller surfaces the error)."""
    store = SettingsStore(path=tmp_path / "s.json")

    class FailingSave(SettingsStore):
        def save(self):
            raise OSError("disk full")

    store.__class__ = FailingSave
    with pytest.raises(OSError):
        store.set_settings({"cache_max_entries": 7})
    assert "cache_max_entries" not in store._settings

    with pytest.raises(OSError):
        store.set_disabled("some-plugin", True)
    assert "some-plugin" not in store._disabled


# --- M-C2: disabled-plugin register crash degrades ---------------------------


def test_disabled_plugin_register_crash_degrades(monkeypatch):
    """A disabled plugin whose register() crashes must not kill discovery
    (startup crash + 500 on every toggle): degrade with PluginInfo.error
    like the enabled path does."""
    def register(reg):
        raise RuntimeError("boom at import side effect")

    eps = [SimpleNamespace(name="brk", load=lambda: register, dist=None)]
    monkeypatch.setattr("emptyui.registry.entry_points", lambda group=None: eps)
    reg = Registry()
    infos = discover_plugins(reg, {"brk"})
    assert infos[0].enabled is False
    assert "RuntimeError" in (infos[0].error or "")
    assert reg.get("brk.node") is None  # live table untouched


def test_toggle_failure_keeps_settings_file_consistent(tmp_path, monkeypatch):
    """M-C3: a failed rebuild (duplicate-type conflict on re-enable) must
    not leave the settings file claiming an enable state the live registry
    doesn't have — rebuild first, persist only on success."""
    from emptyui import Node

    class Shared(Node):
        type = "dup.type"
        outputs = ()

        def execute(self, ctx):
            return {}

    def make_register():
        def register(reg):
            reg.add(Shared)

        return register

    # two plugins claiming the same type: re-enabling the second collides
    eps = [
        SimpleNamespace(name="dupa", load=lambda: make_register(), dist=None),
        SimpleNamespace(name="dupb", load=lambda: make_register(), dist=None),
    ]
    monkeypatch.setattr("emptyui.registry.entry_points", lambda group=None: eps)

    settings_file = tmp_path / "settings.json"
    # disable dupb via the settings FILE (the UI tier): an env-baselined
    # disable would make the re-enable a documented no-op (platform wins)
    settings_file.write_text(json.dumps({"disabled_plugins": ["dupb"]}))
    client = TestClient(create_app(
        settings_path=settings_file,
    ))
    assert json.loads(settings_file.read_text())["disabled_plugins"] == ["dupb"]

    # re-enable fails (duplicate type against the still-enabled dupa) —
    # a clean 409, and the file must still say disabled (persist happens
    # AFTER a successful rebuild only)
    resp = client.post("/api/plugins/dupb/enable")
    assert resp.status_code == 409
    assert "duplicate node type" in resp.json()["detail"]
    assert json.loads(settings_file.read_text())["disabled_plugins"] == ["dupb"]


# --- M-C5: registry table snapshot -------------------------------------------


def test_registry_table_snapshot_is_consistent():
    """table() returns a snapshot; a rebuild REPLACES the live dict (never
    mutates in place), so a single fetch is a consistent snapshot even if a
    toggle lands mid-snapshot — and add()'s in-place mutation can't reach
    the snapshot either (shallow copy)."""
    from emptyui import Node

    class A(Node):
        type = "snap.a"
        outputs = ()

        def execute(self, ctx):
            return {}

    class B(Node):
        type = "snap.b"
        outputs = ()

        def execute(self, ctx):
            return {}

    reg = Registry()
    reg.add(A)
    snapshot = reg.table()
    reg.add(B)
    assert "snap.a" in snapshot and "snap.b" not in snapshot
    assert reg.get("snap.b") is not None  # live table has it
