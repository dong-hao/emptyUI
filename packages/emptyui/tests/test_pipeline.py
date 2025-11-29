"""End-to-end pipeline tests: load_csv → bandpass → plots (M1)."""

from __future__ import annotations

import numpy as np
import pytest

from emptyui.registry import build_default_registry


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory):
    """Synthetic 150 Hz, 4-channel CSV resembling the repo sample."""
    rng = np.random.default_rng(7)
    n = 3000
    t = np.arange(n) / 150.0
    ex = 10 * np.sin(2 * np.pi * 5 * t) + 0.5 * rng.standard_normal(n)
    ey = 10 * np.sin(2 * np.pi * 7 * t) + 0.5 * rng.standard_normal(n)
    hx = 100 * np.sin(2 * np.pi * 0.5 * t) + rng.standard_normal(n)
    hy = 100 * np.sin(2 * np.pi * 2 * t) + rng.standard_normal(n)
    p = tmp_path_factory.mktemp("data") / "synthetic.csv"
    np.savetxt(p, np.column_stack([ex, ey, hx, hy]), delimiter=",",
               header="Ex, Ey, Hx, Hy", comments="", fmt="%.6f")
    return str(p)


def _run(client, workflow):
    resp = client.post("/api/runs", json={"workflow": workflow})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    # async execution: poll until finished
    import time

    for _ in range(100):
        body = client.get(f"/api/runs/{run_id}").json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


def test_full_pipeline_produces_artifacts(client, synthetic_csv):
    workflow = {
        "nodes": [
            {"id": "n1", "type": "core.io.load_csv",
             "params": {"path": synthetic_csv, "sample_rate": 150.0}},
            {"id": "n2", "type": "core.filter.bandpass",
             "params": {"low_freq": 1.0, "high_freq": 40.0}},
            {"id": "n3", "type": "core.plot.timeseries", "params": {}},
            {"id": "n4", "type": "core.plot.spectrogram",
             "params": {"channel": "Ex"}},
        ],
        "edges": [
            {"id": "e1", "source": "n1", "source_port": "data",
             "target": "n2", "target_port": "data"},
            {"id": "e2", "source": "n2", "source_port": "filtered",
             "target": "n3", "target_port": "data"},
            {"id": "e3", "source": "n2", "source_port": "filtered",
             "target": "n4", "target_port": "data"},
        ],
    }
    body = _run(client, workflow)
    assert body["status"] == "completed", body.get("error")

    artifacts = [e for e in body["events"] if e["event"] == "artifact"]
    assert len(artifacts) == 2
    for a in artifacts:
        img = client.get(f"/api/artifacts/{a['id']}/preview")
        assert img.status_code == 200
        assert img.headers["content-type"] == "image/png"
        assert len(img.content) > 1000


def test_bandpass_attenuates_low_frequency(client, synthetic_csv):
    from emptyui.context import RunContext
    from emptyui.store import DataStore

    reg = build_default_registry()
    store = DataStore()
    ctx = RunContext(workspace=None, store=store)

    load = reg.get("core.io.load_csv")
    ts = load.execute(ctx, path=synthetic_csv, sample_rate=150.0)["data"]

    bp = reg.get("core.filter.bandpass")
    out = bp.execute(ctx, ts, low_freq=1.0, high_freq=40.0)["filtered"]

    # 5 Hz signal survives, DC drift / <1 Hz content is attenuated
    raw = ts.channels["Ex"]
    filt = out.channels["Ex"]
    assert abs(filt.mean()) < abs(raw.mean()) / 10
    assert filt.shape == raw.shape


def test_load_csv_missing_file_fails_cleanly(client):
    workflow = {
        "nodes": [{"id": "n1", "type": "core.io.load_csv",
                   "params": {"path": "/nonexistent.csv"}}],
        "edges": [],
    }
    body = _run(client, workflow)
    assert body["status"] == "failed"
    err = next(e for e in body["events"]
               if e["event"] == "node_status" and e["status"] == "error")
    assert "CSV not found" in err["error"]
