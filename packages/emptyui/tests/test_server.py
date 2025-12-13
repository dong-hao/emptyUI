"""Server API — happy path (error surfaces in test_errors.py)."""

import time
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient

from emptyui import Node
from emptyui.registry import Registry
from emptyui.server.app import create_app


def _wait_status(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    """Wait for a terminal summary *and* the worker's run_finished event."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = client.get(f"/api/runs/{run_id}").json()
        finished = any(e["event"] == "run_finished" for e in detail["events"])
        if detail["status"] in ("completed", "failed", "cancelled") and finished:
            return detail
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish in {timeout}s")


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_nodes_catalog_includes_core_plugin(client):
    catalog = client.get("/api/nodes").json()
    types = [m["type"] for m in catalog["nodes"]]
    assert "core.util.echo" in types


def test_clear_cache_endpoint(client):
    resp = client.delete("/api/cache")
    assert resp.status_code == 200
    assert "cleared" in resp.json()


def test_files_endpoint_confined_to_data_root(tmp_path):
    from emptyui.config import Settings

    root = tmp_path / "data"
    root.mkdir()
    (root / "station.csv").write_text("x")
    confined = TestClient(create_app(settings=Settings(data_root=root)))

    listing = confined.get("/api/files").json()
    assert listing["path"] == str(root)
    assert any(e["name"] == "station.csv" for e in listing["entries"])

    # a path outside the data root clamps back to the root instead of 404
    resp = confined.get("/api/files", params={"path": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json()["path"] == str(root)


def test_files_endpoint_unrestricted_without_data_root(client):
    from pathlib import Path

    listing = client.get("/api/files").json()
    assert listing["path"] == str(Path.cwd())


def test_cors_middleware_toggle_from_settings():
    from emptyui.config import Settings

    app_default = create_app(settings=Settings())
    assert any(m.cls.__name__ == "CORSMiddleware" for m in app_default.user_middleware)

    app_no_cors = create_app(settings=Settings(cors_origins=()))
    assert not any(m.cls.__name__ == "CORSMiddleware" for m in app_no_cors.user_middleware)


def test_run_echo_workflow(client):
    workflow = {
        "nodes": [{"id": "n1", "type": "core.util.echo", "params": {"label": "hi"}}],
        "edges": [],
    }
    resp = client.post("/api/runs", json={"workflow": workflow})
    assert resp.status_code == 200
    body = resp.json()
    run_id = body["run_id"]

    detail = client.get(f"/api/runs/{run_id}").json()
    statuses = {e["node"]: e["status"] for e in detail["events"] if e["event"] == "node_status"}
    assert statuses == {"n1": "done"}


def test_cancel_run():
    class Slow(Node):
        type = "test.slow"
        outputs = ()

        def execute(self, ctx):
            time.sleep(0.5)
            return {}

    reg = Registry()
    reg.add(Slow)
    client = TestClient(create_app(reg))

    workflow = {"nodes": [{"id": "n1", "type": "test.slow"}], "edges": []}
    run_id = client.post("/api/runs", json={"workflow": workflow}).json()["run_id"]

    resp = client.post(f"/api/runs/{run_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("cancelled", "running", "queued")

    detail = _wait_status(client, run_id)
    assert detail["status"] == "cancelled"
    finished = [e for e in detail["events"] if e["event"] == "run_finished"]
    assert finished and finished[-1]["status"] == "cancelled"


def test_run_status_transitions_to_running():
    started = Event()
    release = Event()

    class Blocking(Node):
        type = "test.blocking"
        outputs = ()

        def execute(self, ctx):
            started.set()
            release.wait(timeout=5)
            return {}

    reg = Registry()
    reg.add(Blocking)
    client = TestClient(create_app(reg))
    workflow = {"nodes": [{"id": "n1", "type": "test.blocking"}], "edges": []}
    run_id = client.post("/api/runs", json={"workflow": workflow}).json()["run_id"]
    assert started.wait(timeout=2)
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "running"
    release.set()
    assert _wait_status(client, run_id)["status"] == "completed"


def test_max_concurrent_runs_rejects_until_slot_released():
    started = Event()
    release = Event()

    class Blocking(Node):
        type = "test.limited"
        outputs = ()

        def execute(self, ctx):
            started.set()
            release.wait(timeout=5)
            return {}

    reg = Registry()
    reg.add(Blocking)
    client = TestClient(create_app(reg, max_concurrent_runs=1))
    workflow = {"nodes": [{"id": "n1", "type": "test.limited"}], "edges": []}
    first = client.post("/api/runs", json={"workflow": workflow})
    assert first.status_code == 200
    assert started.wait(timeout=2)

    second = client.post("/api/runs", json={"workflow": workflow})
    assert second.status_code == 429
    assert "maximum concurrent runs" in second.json()["detail"]

    release.set()
    assert _wait_status(client, first.json()["run_id"])["status"] == "completed"
    third = client.post("/api/runs", json={"workflow": workflow})
    assert third.status_code == 200
    assert _wait_status(client, third.json()["run_id"])["status"] == "completed"


def test_workspace_ttl_cleanup_removes_artifacts(tmp_path):
    class Snapshot(Node):
        type = "test.expiring_snapshot"
        outputs = ()

        def execute(self, ctx):
            p = Path(ctx.workspace) / "snap.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
            ctx.publish_artifact(p, kind="image")
            return {}

    reg = Registry()
    reg.add(Snapshot)
    with TestClient(create_app(reg, workspace_ttl_seconds=10)) as client:
        workflow = {"nodes": [{"id": "n1", "type": "test.expiring_snapshot"}], "edges": []}
        run_id = client.post("/api/runs", json={"workflow": workflow}).json()["run_id"]
        detail = _wait_status(client, run_id)
        artifact_id = next(e["id"] for e in detail["events"] if e["event"] == "artifact")

        # The cleanup helper is exposed on app.state so expiry can be tested
        # without sleeping for a production-sized TTL.
        app = client.app
        info = app.state.cleanup_expired_runs
        assert client.get(f"/api/artifacts/{artifact_id}/preview").status_code == 200
        # The run's completion timestamp is internal; force expiry through the
        # helper's injected clock by using a timestamp beyond the TTL.
        assert info(time.time() + 11) == 1
        assert client.get(f"/api/runs/{run_id}").status_code == 404
        assert client.get(f"/api/artifacts/{artifact_id}/preview").status_code == 404


def test_artifact_download_and_preview():
    class Snapshot(Node):
        type = "test.snapshot"
        outputs = ()

        def execute(self, ctx):
            p = Path(ctx.workspace) / "snap.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
            ctx.publish_artifact(p, kind="image")
            return {}

    reg = Registry()
    reg.add(Snapshot)
    client = TestClient(create_app(reg))

    workflow = {"nodes": [{"id": "n1", "type": "test.snapshot"}], "edges": []}
    run_id = client.post("/api/runs", json={"workflow": workflow}).json()["run_id"]
    detail = _wait_status(client, run_id)
    artifact_ids = [e["id"] for e in detail["events"] if e["event"] == "artifact"]
    assert len(artifact_ids) == 1
    artifact_id = artifact_ids[0]

    dl = client.get(f"/api/artifacts/{artifact_id}")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("image/png")
    disposition = dl.headers.get("content-disposition", "")
    assert "attachment" in disposition
    assert dl.content.startswith(b"\x89PNG")

    prev = client.get(f"/api/artifacts/{artifact_id}/preview")
    assert prev.status_code == 200
    assert prev.headers["content-type"].startswith("image/png")
    assert "attachment" not in prev.headers.get("content-disposition", "")

    assert client.get("/api/artifacts/unknown").status_code == 404
