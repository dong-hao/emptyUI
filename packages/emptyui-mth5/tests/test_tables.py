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


# --- M13: View Selection is palette-hidden and its fallback is loud -------

def test_view_selection_hidden_without_producer():
    """No bundled node produces a 'selection' payload (producer
    M4-deferred, §3) — the viewer must not be palette-reachable."""
    from emptyui.registry import build_default_registry

    reg = build_default_registry()
    node = reg.get("mth5.table.selection")
    assert node is not None  # still registered (saved workflows execute)
    assert node.manifest()["hidden"] is True


def test_as_dataframe_rejects_broken_fallback():
    """The old duck-typed fallback returned a plain dict whose to_dict()
    takes no orient argument — a crash-on-use; anything without a real
    '.df' is a loud error."""
    import pytest as _pytest
    from emptyui_mth5.tables import _as_dataframe

    with _pytest.raises(TypeError, match="expected a pandas DataFrame"):
        _as_dataframe({"a": 1})
