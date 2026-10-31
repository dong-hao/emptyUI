"""Session-only run history (docs/issues.md L32).

The history is the in-memory run table: the exact submitted workflow
snapshot, a data-derived label and a compact result snapshot. A restart
clears it (ComfyUI parity). These tests pin the label rules, the
list/detail/delete/clear endpoints and the active-run guards.
"""

from __future__ import annotations

import time
from threading import Event

import pytest
from fastapi.testclient import TestClient

from emptyui import Node, Param
from emptyui.registry import Registry
from emptyui.server.app import _history_label, create_app


class Source(Node):
    """A data source with an input path, a calibration path and an artifact."""

    type = "test.source"
    outputs = ()
    params = (
        Param("path", "file", path_kind="in"),
        Param("calibration_path", "file", path_kind="in", optional=True),
        Param("save_path", "directory", path_kind="out", optional=True),
    )

    def execute(self, ctx, path="", calibration_path="", save_path=""):
        p = ctx.artifact_path("out", ".txt")
        p.write_text("artifact")
        ctx.publish_artifact(p, "text")
        return {}


class Blocking(Node):
    type = "test.blocking"
    outputs = ()
    started = Event()
    release = Event()

    def execute(self, ctx):
        type(self).started.set()
        type(self).release.wait(timeout=5)
        return {}


class Boom(Node):
    type = "test.boom"
    outputs = ()

    def execute(self, ctx):
        raise RuntimeError("boom")


def _wait_status(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = client.get(f"/api/runs/{run_id}").json()
        finished = any(e["event"] == "run_finished" for e in detail["events"])
        if detail["status"] in ("completed", "failed", "cancelled") and finished:
            return detail
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish in {timeout}s")


@pytest.fixture()
def history_client():
    reg = Registry()
    reg.add(Source)
    return TestClient(create_app(reg))


def _source_workflow(path: str = "/data/station-01/1690618B.TS5") -> dict:
    return {
        "nodes": [
            {
                "id": "n1",
                "type": "test.source",
                "params": {"path": path, "calibration_path": "/cal/box.asc"},
            }
        ],
        "edges": [],
    }


def _run(client: TestClient, workflow: dict) -> str:
    resp = client.post("/api/runs", json={"workflow": workflow})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    _wait_status(client, run_id)
    return run_id


# --- label derivation ------------------------------------------------------


def test_history_label_uses_input_data_basename_not_calibration():
    reg = Registry()
    reg.add(Source)
    wf = _source_workflow()
    assert _history_label(wf, reg) == "1690618B.TS5"


def test_history_label_joins_multiple_sources_and_caps():
    reg = Registry()
    reg.add(Source)
    wf = {
        "nodes": [
            {"id": "a", "type": "test.source", "params": {"path": "/d/one.mth5"}},
            {"id": "b", "type": "test.source", "params": {"path": "/d/two.mth5"}},
            {"id": "c", "type": "test.source", "params": {"path": "/d/three.mth5"}},
        ]
    }
    assert _history_label(wf, reg) == "one.mth5 + two.mth5 +1"


def test_history_label_falls_back_to_document_name_then_untitled():
    reg = Registry()
    reg.add(Source)
    assert _history_label({"nodes": [], "meta": {"name": "my-flow"}}, reg) == "my-flow"
    assert _history_label({"nodes": [], "meta": {"name": "untitled"}}, reg) == "Untitled"
    assert _history_label({"nodes": []}, reg) == "Untitled"


def test_history_label_skips_calibration_only_sources():
    class CalibrationSource(Node):
        type = "test.calibration.source"
        outputs = ()
        params = (Param("path", "file", path_kind="in"),)

        def execute(self, ctx, path=""):
            return {}

    reg = Registry()
    reg.add(CalibrationSource)
    wf = {
        "nodes": [
            {"id": "c", "type": "test.calibration.source",
             "params": {"path": "/cal/box.asc"}},
        ],
        "meta": {"name": "cal-only"},
    }
    assert _history_label(wf, reg) == "cal-only"


# --- endpoints -------------------------------------------------------------


def test_history_lists_finished_run_with_label_and_artifact(history_client):
    run_id = _run(history_client, _source_workflow())

    entries = history_client.get("/api/history").json()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["run_id"] == run_id
    assert entry["label"] == "1690618B.TS5"
    assert entry["status"] == "completed"
    assert entry["node_count"] == 1
    assert entry["artifact_count"] == 1
    assert entry["duration_s"] is not None and entry["duration_s"] >= 0

    detail = history_client.get(f"/api/history/{run_id}").json()
    assert detail["workflow"] == _source_workflow()
    assert detail["results"]["nodes"]["n1"]["status"] == "done"
    assert detail["results"]["artifacts"][0]["node"] == "n1"
    assert detail["results"]["artifacts"][0]["url"].startswith("/api/artifacts/")


def test_history_delete_removes_entry_artifact_and_workspace(history_client):
    run_id = _run(history_client, _source_workflow())
    artifact_id = history_client.get(
        f"/api/history/{run_id}").json()["results"]["artifacts"][0]["id"]
    assert history_client.get(
        f"/api/artifacts/{artifact_id}/preview").status_code == 200

    resp = history_client.delete(f"/api/history/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert history_client.get("/api/history").json() == []
    assert history_client.get(f"/api/history/{run_id}").status_code == 404
    # the artifact registry entry and its workspace file are gone
    assert history_client.get(
        f"/api/artifacts/{artifact_id}/preview").status_code == 404


def test_history_clear_all_removes_finished_entries(history_client):
    _run(history_client, _source_workflow("/d/one.mth5"))
    _run(history_client, _source_workflow("/d/two.mth5"))
    assert len(history_client.get("/api/history").json()) == 2

    resp = history_client.delete("/api/history")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    assert history_client.get("/api/history").json() == []


def test_history_excludes_active_run_and_rejects_delete():
    Blocking.started.clear()
    Blocking.release.clear()
    reg = Registry()
    reg.add(Blocking)
    client = TestClient(create_app(reg))
    run_id = client.post("/api/runs", json={
        "workflow": {"nodes": [{"id": "n1", "type": "test.blocking"}], "edges": []},
    }).json()["run_id"]
    assert Blocking.started.wait(timeout=2)

    assert client.get("/api/history").json() == []
    assert client.delete(f"/api/history/{run_id}").status_code == 409

    Blocking.release.set()
    _wait_status(client, run_id)
    assert len(client.get("/api/history").json()) == 1


def test_history_failed_run_reports_failed_status():
    reg = Registry()
    reg.add(Boom)
    client = TestClient(create_app(reg))
    run_id = client.post("/api/runs", json={
        "workflow": {"nodes": [{"id": "n1", "type": "test.boom"}], "edges": []},
    }).json()["run_id"]
    _wait_status(client, run_id)

    entry = client.get("/api/history").json()[0]
    assert entry["status"] == "failed"
    assert entry["label"] == "Untitled"
    detail = client.get(f"/api/history/{run_id}").json()
    assert detail["results"]["status"] == "failed"
    assert "boom" in (detail["summary"].get("error") or "")


def test_history_is_session_only():
    reg = Registry()
    reg.add(Source)
    first = TestClient(create_app(reg))
    second = TestClient(create_app(reg))
    _run(first, _source_workflow())
    assert len(first.get("/api/history").json()) == 1
    assert second.get("/api/history").json() == []
