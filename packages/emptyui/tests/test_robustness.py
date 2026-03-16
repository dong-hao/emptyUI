"""Phase 6 robustness & API hygiene regressions (issues #30-35).

Each test targets one latent defect from the review; they were all
uncovered by the pre-existing suite.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from emptyui.cache import CacheEntry, NodeCache
from emptyui.executor import ExecutionError, Executor, _ts_summary
from emptyui.registry import Registry, discover_plugins
from emptyui.schema import ValidationError_, Workflow, validate_workflow
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
    calls = {"n": 0}

    def register(reg):
        calls["n"] += 1
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


# --- M-C4: payload summarizer hardening --------------------------------------


class _FakeRuns:
    """Duck payload: a run collection whose extents all lack starts."""

    def __init__(self, extents):
        self.runs = ["r1"]
        self.extents = extents


def test_ts_summary_all_empty_starts_does_not_crash():
    extents = [
        {"survey": "s", "station": "st", "run": f"r{i}",
         "start": "", "end": "", "sample_rate": 1.0}
        for i in range(3)
    ]
    summary = _ts_summary(_FakeRuns(extents))
    assert summary is not None
    assert summary["kind"] == "ts.runs"
    assert summary["runs"]  # per-member extents retained (provenance)
    assert "start" not in summary and "end" not in summary
    # the frontend degrades: no bar, no crash


def test_ts_summary_non_numeric_sample_rate_degrades():
    extents = [
        {"survey": "s", "station": "st", "run": "r1",
         "start": "2023-01-01T00:00:00", "end": "2023-01-01T01:00:00",
         "sample_rate": "not-a-number"},
    ]
    summary = _ts_summary(_FakeRuns(extents))
    assert summary["runs"][0]["sample_rate"] == 0.0
    assert summary["start"] == "2023-01-01T00:00:00"


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


# --- M-C8: TTL GC invalidates cache entries in the doomed workspace ----------


def test_invalidate_workspace_cache_drops_referencing_entries(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    archive = SimpleNamespace(filename=str(ws / "a.h5"))

    class _P:
        pass

    cache = NodeCache(max_entries=10)
    cache.put("k1", CacheEntry(outputs={"mth5": archive}))
    cache.put("k2", CacheEntry(outputs={"ts": _P()}))
    dropped = Executor.invalidate_workspace_cache(
        cache, str(ws), __import__("logging").getLogger("test"))
    assert dropped == 1
    assert cache.get("k1") is None
    assert cache.get("k2") is not None


# --- M-C10: malformed edges fail with the node-failure contract --------------


def test_dangling_edge_raises_execution_error():
    from emptyui.schema import Edge as E
    from emptyui.schema import WorkflowNode as WN

    wf = Workflow(nodes=[WN(id="n1", type="core.io.load_csv")],
                  edges=[E(id="e1", source="n1", source_port="data",
                           target="ghost", target_port="data")])
    with pytest.raises(ExecutionError) as ei:
        Executor(Registry()).run(wf)
    assert "unknown node 'ghost'" in str(ei.value)


# --- L-C1: boolean param coercion --------------------------------------------


def test_boolean_param_coercion():
    from emptyui.api import Param

    p = Param("flag", "boolean")
    coerce = Executor._coerce_param
    assert coerce(p, "false") is False
    assert coerce(p, "False") is False
    assert coerce(p, "true") is True
    assert coerce(p, "1") is True
    assert coerce(p, "0") is False
    assert coerce(p, "") is False
    assert coerce(p, False) is False  # real bool passes through
    with pytest.raises(ValueError):
        coerce(p, "maybe")


# --- L-C2: document envelope validation --------------------------------------


def test_workflow_envelope_rejected():
    from emptyui import Node

    class N(Node):
        type = "env.node"
        outputs = ()

        def execute(self, ctx):
            return {}

    reg = Registry()
    reg.add(N)
    wf = Workflow(nodes=[{"id": "n1", "type": "env.node"}], edges=[])
    wf = wf.model_copy(update={"format": "other/tool"})
    with pytest.raises(ValidationError_) as ei:
        validate_workflow(wf, reg)
    assert "not an emptyUI workflow" in str(ei.value)

    wf2 = wf.model_copy(update={"format": "emptyui/workflow", "version": 2})
    with pytest.raises(ValidationError_) as ei:
        validate_workflow(wf2, reg)
    assert "unsupported workflow version" in str(ei.value)


# --- L-C3: empty out-paths do not stat the CWD into cache keys ---------------


def test_out_path_signature_has_no_cwd_stat(tmp_path, monkeypatch):
    """An empty out-path param must contribute its plain value — the old
    stat() of Path('') == stat('.') tied every cache key to the server
    CWD's mtime."""
    import os


    before = Executor._value_signature("", data_root=None, path_kind="out",
                                       widget="directory")
    assert before == ""  # plain value, no [path, size, mtime] signature
    # and an out-path that DOES exist still isn't statted
    f = tmp_path / "dest.h5"
    f.write_text("x")
    sig = Executor._value_signature(str(f), data_root=None, path_kind="out",
                                    widget="file")
    assert sig == str(f)
    monkeypatch.chdir(tmp_path)  # keep the test hermetic about CWD assumptions
    assert os.getcwd() == str(tmp_path)


# --- L-C5/L-C6/L-C9: artifact endpoints --------------------------------------


