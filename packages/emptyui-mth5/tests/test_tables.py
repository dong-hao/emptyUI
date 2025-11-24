"""Tests for the table-producing Display node."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_mth5.tables import ViewSelection


def test_view_selection_emits_table_artifact(tmp_path: Path):
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    selection = pd.DataFrame(
        {
            "survey": ["test", "test"],
            "station": ["mt001", "mt001"],
            "run": ["sr1_0001", "sr2_0001"],
            "sample_rate": [1.0, 1.0],
        }
    )

    result = ViewSelection().execute(ctx, selection)

    # the table port carries a published artifact
    artifact = result["table"]
    assert artifact["kind"] == "table"
    assert len(ctx.published) == 1

    # the artifact file is JSON-serializable records
    path = Path(artifact["path"])
    assert path.exists()
    records = json.loads(path.read_text())
    assert len(records) == 2
    assert records[0]["run"] == "sr1_0001"
