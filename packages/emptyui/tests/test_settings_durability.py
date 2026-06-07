"""L6 — settings durability + HTTP polish.

Three defects: (a) ``save()`` did not ``fsync`` before ``os.replace`` — a
crash could leave a corrupt file that is then silently defaulted away;
(b) an ``OSError`` from ``set_disabled``/``set_settings`` escaped as a
bare 500 (no useful detail); (c) ``cache_max_entries: 0`` — documented as
"off" (§7.4: null/0 = off) — clamped to 1 and the view reported the
clamp, so the UI could not actually turn the cache off.
"""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from emptyui.registry import Registry
from emptyui.server.app import create_app


def test_settings_save_fsyncs_before_replace(tmp_path, monkeypatch):
    import emptyui.settings as settings_module

    calls: list[int] = []
    real_fsync = settings_module.os.fsync
    monkeypatch.setattr(
        settings_module.os, "fsync",
        lambda fd: (calls.append(fd), real_fsync(fd)))
    path = tmp_path / "settings.json"
    client = TestClient(create_app(settings_path=str(path)))
    resp = client.put(
        "/api/settings", json={"values": {"cache_max_entries": 32}})
    assert resp.status_code == 200
    assert calls, "the settings file was replaced without an fsync"
    assert json.loads(path.read_text())["settings"]["cache_max_entries"] == 32


def test_settings_write_failure_is_a_clean_500(tmp_path, monkeypatch):
    """A failing settings write surfaces as a 500 NAMING the failure —
    not a bare traceback (the memory state already rolled back, L-C8;
    the client must know the file disagrees)."""
    from types import SimpleNamespace

    from emptyui import Node

    class Fake(Node):
        type = "fake.l6"
        outputs = ()

        def execute(self, ctx):
            return {}

    def register(reg):
        reg.add(Fake)

    # a third-party plugin so the toggle reaches the persistence
    monkeypatch.setattr(
        "emptyui.registry.entry_points", lambda group=None: [
            SimpleNamespace(name="fake6", load=lambda: register, dist=None)])
    # a DIRECTORY at the settings path: save() fails at os.replace time
    # with IsADirectoryError (an OSError) — deterministic, no spy needed
    client = TestClient(
        create_app(settings_path=str(tmp_path)),
        raise_server_exceptions=False)
    resp = client.put("/api/settings",
                      json={"values": {"cache_max_entries": 32}})
    assert resp.status_code == 500
    assert "settings" in str(resp.json()["detail"]).lower()

    toggle = client.post("/api/plugins/fake6/disable")
    assert toggle.status_code == 500
    assert "settings" in str(toggle.json()["detail"]).lower()


def test_cache_max_entries_zero_truly_disables_the_cache():
    """§7.4 policy: 0 = off. The view must report 0 (not the internal
    clamp to 1) and identical runs must re-execute."""
    from emptyui import Node, Output

    class Counted(Node):
        type = "test.counted_l6"
        outputs = (Output("data", "any"),)
        calls = 0

        def execute(self, ctx):
            type(self).calls += 1
            return {"data": 1}

    reg = Registry()
    reg.add(Counted)
    workflow = {"nodes": [{"id": "n1", "type": "test.counted_l6"}],
                "edges": []}
    client = TestClient(create_app(reg))

    def _run_to_done():
        run_id = client.post("/api/runs",
                             json={"workflow": workflow}).json()["run_id"]
        for _ in range(300):
            body = client.get(f"/api/runs/{run_id}").json()
            if body.get("status") in ("completed", "failed", "timeout"):
                return body
            time.sleep(0.02)
        raise AssertionError("run never finished")

    # 0 = off: every run executes
    assert client.put(
        "/api/settings", json={"values": {"cache_max_entries": 0}}
    ).status_code == 200
    assert client.get("/api/settings").json()["live"][
        "cache_max_entries"]["value"] == 0
    _run_to_done()
    _run_to_done()
    assert Counted.calls == 2

    # and back on: the same shape hits again
    assert client.put(
        "/api/settings", json={"values": {"cache_max_entries": 8}}
    ).status_code == 200
    _run_to_done()
    _run_to_done()
    assert Counted.calls == 3


# --- L3 (backend half): artifact event hygiene ----------------------------

def test_artifact_events_leak_no_paths_and_normalize_handles():
    """Artifact events carried the ABSOLUTE server-side path to every WS
    client and emitted `handle: null` on the fresh path vs "" on the
    cache path — the preview URL is the router (§7.2), the path is
    internal, and the handle is normalized to ""."""
    from emptyui import Node

    class Publisher(Node):
        type = "test.pub_l3"
        outputs = ()  # the artifact is not an output value -> handle ""

        def execute(self, ctx):
            p = ctx.artifact_path("fig", ".txt")
            p.write_text("figure")
            ctx.publish_artifact(p, "image")
            return {}

    reg = Registry()
    reg.add(Publisher)
    client = TestClient(create_app(reg))
    run_id = client.post("/api/runs", json={
        "workflow": {"nodes": [{"id": "n1", "type": "test.pub_l3"}],
                     "edges": []}}).json()["run_id"]
    for _ in range(300):
        detail = client.get(f"/api/runs/{run_id}").json()
        if detail.get("status") in ("completed", "failed", "timeout"):
            break
        time.sleep(0.02)
    assert detail["status"] == "completed"
    arts = [e for e in detail["events"] if e["event"] == "artifact"]
    assert arts, "no artifact event"
    assert all("path" not in e for e in arts), (
        "the absolute server-side path leaked to clients")
    assert all(e["handle"] == "" for e in arts), (
        [e["handle"] for e in arts])
    # the server kept the path internally — the preview still resolves
    preview = client.get(f"/api/artifacts/{arts[0]['id']}/preview")
    assert preview.status_code == 200
