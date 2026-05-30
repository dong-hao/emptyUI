"""M17 — workspace TTL cleanup must not run on the event loop.

``cleanup_loop`` awaited the plain sync ``cleanup_expired_runs`` — path
resolution, cache scans and ``shutil.rmtree`` inline — so reclaiming a
workspace of MTH5 archives stalled the same loop that drives every WS
broadcast and HTTP request. And ``rmtree(..., ignore_errors=True)`` leaked
failed deletions silently.

The contract: the reclaim runs on a worker thread (the loop stays
responsive during a multi-GB rmtree), and a failed delete is logged
naming the workspace — best-effort, never silent.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time

from fastapi.testclient import TestClient

from emptyui.registry import build_default_registry
from emptyui.server.app import create_app

TTL = 0.05  # expiry window; the loop polls at max(0.1, ttl/4) = 0.1 s

ECHO_WORKFLOW = {
    "nodes": [{"id": "n1", "type": "core.util.echo"}],
    "edges": [],
}


def _client():
    return TestClient(create_app(
        build_default_registry(), workspace_ttl_seconds=TTL))


def _run_to_completion(client) -> str:
    resp = client.post("/api/runs", json={"workflow": ECHO_WORKFLOW})
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    deadline = time.monotonic() + 10.0
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body.get("status") == "completed":
            return run_id
        time.sleep(0.05)
    raise AssertionError(f"run never completed: {body}")


def _wait(condition, timeout=5.0, msg="condition never became true"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(msg)


def test_cleanup_does_not_block_the_event_loop(monkeypatch):
    """While a workspace reclaim is in flight, the loop still serves
    requests. The fake rmtree sleeps long enough to represent a
    multi-GB MTH5 workspace deletion."""
    real_rmtree = shutil.rmtree  # captured BEFORE the patch (tidy-up below)
    entered = threading.Event()
    doomed: list[str] = []

    def slow_fake_rmtree(path, *args, **kwargs):
        doomed.append(str(path))
        entered.set()
        time.sleep(3.0)  # the stand-in for a huge rmtree

    monkeypatch.setattr(
        "emptyui.server.app.shutil.rmtree", slow_fake_rmtree)
    client = _client()
    with client:
        _run_to_completion(client)
        # the next cleanup pass reclaims the completed run's workspace
        _wait(entered.is_set, msg="the reclaim never started")
        assert doomed, "no workspace was handed to rmtree"

        latency: list[float] = []

        def probe():
            start = time.monotonic()
            resp = client.get("/api/health")
            latency.append(time.monotonic() - start)
            assert resp.status_code == 200

        probe_thread = threading.Thread(target=probe)
        probe_thread.start()
        probe_thread.join(timeout=5.0)
        assert not probe_thread.is_alive()
        # generous margin: pre-fix the probe waited out the full 3 s fake
        # sleep (the loop was blocked); post-fix it is served in
        # microseconds — a 1 s ceiling still discriminates under load
        assert latency[0] < 1.0, (
            f"the event loop was blocked for {latency[0]:.2f} s by the "
            "workspace reclaim")
    # the fake never deleted (and still being patched, it would append to
    # `doomed` forever — iterate a snapshot with the real rmtree)
    for path in doomed:
        real_rmtree(path, ignore_errors=True)


def test_failed_reclaim_is_logged_not_silent(monkeypatch, caplog):
    def raising_fake_rmtree(path, *args, **kwargs):
        raise OSError(f"[Errno 1] mocked deletion failure: {path}")

    monkeypatch.setattr(
        "emptyui.server.app.shutil.rmtree", raising_fake_rmtree)
    client = _client()
    with client:
        _run_to_completion(client)

        def _logged():
            return any(
                "reclaim failed" in r.getMessage()
                and r.levelno >= logging.WARNING
                for r in caplog.records
            )

        # the run entry is popped BEFORE the rmtree raises and the log is
        # written — wait for the record itself, not for a 404 proxy (the
        # poll could otherwise observe the 404 in that gap and race the
        # assertion ahead of the record)
        _wait(_logged, msg="the failed deletion was swallowed silently")


def test_workspace_actually_removed_on_success(monkeypatch):
    """Guard: the thread offload still deletes (the fake above skips the
    real work — this one uses the real rmtree)."""
    real_rmtree = shutil.rmtree
    monkeypatch.setattr(
        "emptyui.server.app.shutil.rmtree", real_rmtree)
    client = _client()
    with client:
        run_id = _run_to_completion(client)
        _wait(
            lambda: client.get(f"/api/runs/{run_id}").status_code == 404,
            msg="the expired run was never reclaimed")
