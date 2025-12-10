"""Server API — happy path (error surfaces in test_errors.py)."""

import time
from pathlib import Path

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
