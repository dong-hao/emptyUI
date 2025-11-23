import pytest
from fastapi.testclient import TestClient

from emptyui.registry import build_default_registry
from emptyui.server.app import create_app


@pytest.fixture()
def client():
    return TestClient(create_app(build_default_registry()))


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_nodes_catalog_includes_core_plugin(client):
    catalog = client.get("/api/nodes").json()
    types = [m["type"] for m in catalog["nodes"]]
    assert "core.util.echo" in types


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


def test_invalid_workflow_returns_422(client):
    workflow = {
        "nodes": [{"id": "n1", "type": "core.util.nope"}],
        "edges": [],
    }
    resp = client.post("/api/runs", json={"workflow": workflow})
    assert resp.status_code == 422
