"""L20 — the session-QC/gating demo workflow runs end-to-end.

Executes the SHIPPED example document through the real executor against
the tracked smoke archive (corpus-guarded like the other data tests):
the QC story must actually work — sessions scored, the noisy tail
dropped at the production defaults, and a TF estimated from the
survivors. Skipped when the corpus is absent (fresh clones without
examples/data).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")

DEMO = Path("examples/workflow/phoenix-mtu5a-session-qc.emptyui.json")
ARCHIVE = Path("examples/data/phoenix-mtu-test/smoke_phoenix_mtu.mth5")

requires_corpus = pytest.mark.skipif(
    not (DEMO.exists() and ARCHIVE.exists()),
    reason="demo workflow or smoke corpus not present (fresh clone)")


@requires_corpus
def test_session_qc_demo_runs_end_to_end(tmp_path):
    from emptyui.executor import Executor
    from emptyui.registry import build_default_registry
    from emptyui.schema import Workflow, validate_workflow

    doc = json.loads(DEMO.read_text())
    reg = build_default_registry()
    wf = Workflow.model_validate(doc)
    validate_workflow(wf, reg)  # the shipped document must validate clean

    events: list[dict] = []
    summary = Executor(reg).run(
        wf, emit=events.append, workspace=tmp_path, cache=None)

    assert summary["status"] == "completed", (
        [e for e in events if e["event"] == "node_status"
         and e.get("status") == "error"])

    # the mask report and the select-runs diagnostics were published
    tables = [e for e in events if e["event"] == "artifact"
              and e.get("kind") == "table"]
    assert len(tables) >= 2  # mask report + run diagnostics

    # the QC demonstrably acted: the noisy second-half sessions are
    # dropped at the production defaults (measured on this corpus with
    # the mask stage in the chain: s0009–s0012, 21:15–23:00 UTC)
    select_rows = json.loads(
        Path(_artifact_path(events, "select_runs")).read_text())
    dropped = {r["run_id"] for r in select_rows if r["dropped"]}
    assert {"sr15_0001_s0009", "sr15_0001_s0010",
            "sr15_0001_s0011", "sr15_0001_s0012"} <= dropped
    scored = [r for r in select_rows if r["deviation"] is not None]
    assert len(scored) == 12  # 3 h / 0.25 h sessions, all scored

    # the mask criteria found real artifacts but did not swamp the record
    mask_rows = json.loads(
        Path(_artifact_path(events, "mask_ts")).read_text())
    total = next(r for r in mask_rows if r["criterion"] == "total")
    assert 0.001 < total["fraction"] < 0.2

    # a TF was estimated and plotted from the survivors
    done = {e["node"] for e in events
            if e["event"] == "node_status" and e.get("status") == "done"}
    assert {"est", "plot", "select"} <= done


def _artifact_path(events: list[dict], stem: str) -> str:
    """The artifact event whose server path carries the stem (the executor
    emits `path` only to the emit callback, never to clients)."""
    for e in events:
        if e["event"] == "artifact" and stem in Path(
                e.get("path", "")).stem:
            return e["path"]
    raise AssertionError(f"no artifact with stem {stem!r} in {events}")
