"""End-to-end LEMI424 test through the emptyUI workflow engine.

Requires the raw LEMI data at examples/data/lemi-424-test (gitignored, on
disk). Skipped when absent so CI without the data still passes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from _workflow import edge, node
from emptyui.executor import Executor
from emptyui.registry import build_default_registry
from emptyui.schema import Workflow

LEMI_DIR = Path(__file__).resolve().parents[3] / "examples" / "data" / "lemi-424-test"


def _lemi_workflow(lemi_path, out_dir):
    return {
        "nodes": [
            node("l", "mth5.io.lemi", {
                "data_path": lemi_path, "survey_id": "test", "station_id": "mt001",
                "mth5_filename": "lemi_e2e.mth5", "save_path": str(out_dir)}),
            node("x", "mth5.io.extract_run",
                  {"survey": "test", "station": "mt001", "run": "sr1_0001"}),
            node("d", "core.filter.detrend", {"method": "mean"}),
            node("c", "aurora.process.compute_fc",
                  {"input_channels": "bx,by", "output_channels": "e1,e2",
                   "channel_nomenclature": "lemi12", "window_samples": 4096}),
            node("t", "aurora.process.estimate_tf", {}),
            node("e", "mth5.tf.export",
                  {"path": str(out_dir / "lemi_tf"), "file_type": "edi"}),
        ],
        "edges": [
            edge("e1", "l", "mth5", "x", "mth5"),
            edge("e2", "x", "data", "d", "data"),
            edge("e3", "d", "data", "c", "data"),
            edge("e4", "c", "fc", "t", "fc"),
            edge("e5", "t", "tf", "e", "tf"),
        ],
    }


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("lemi_e2e")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.skipif(not LEMI_DIR.exists(), reason="LEMI data not on disk")
def test_lemi424_pipeline_end_to_end(run_dir):
    registry = build_default_registry()
    wf = Workflow.model_validate(_lemi_workflow(str(LEMI_DIR), run_dir))
    events = []
    summary = Executor(registry).run(wf, emit=events.append, workspace=run_dir / "ws")

    assert summary["status"] == "completed", [e for e in events if e["event"] == "node_status"]
    assert not any(e["event"] == "node_status" and e["status"] == "error" for e in events)

    edi = run_dir / "lemi_tf.edi"
    assert edi.exists()
    assert edi.stat().st_size > 1000
    # a TF was actually estimated with a sensible period range
    progress = {e["node"]: e["msg"] for e in events if e["event"] == "node_progress"}
    assert "27" in progress.get("t", "") or "periods" in progress.get("t", "")

    mth5 = run_dir / "lemi_e2e.mth5"
    assert mth5.exists()
