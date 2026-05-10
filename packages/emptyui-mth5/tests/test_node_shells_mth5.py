"""Node-shell smoke tests (issues F1): manifest/param wiring for the mth5
Display/I-O nodes whose underlying functions were already covered
directly — View TF and Load Calibration."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import build_tf
from emptyui_mth5.calibration import LoadCalibration
from emptyui_mth5.tables import ViewTF

#: tracked LEMI-417 FAP sample (same file test_calibration parses)
REAL_CHN = Path(__file__).resolve().parents[3] / "examples" / "data" / "bgs-lemi-424" / "lemi.chn"


def _assert_shell(node, *, viewer=None) -> None:
    """Manifest and execute() wiring agree: unique names, and every
    declared input/param is an execute() kwarg (the executor passes both
    as **kwargs — a renamed parameter would only fail at run time)."""
    m = node.manifest()
    assert m["type"] == node.type
    assert m["title"] and m["category"]
    assert m["viewer"] is viewer
    for key in ("inputs", "outputs"):
        names = [p["name"] for p in m[key]]
        assert len(names) == len(set(names)), f"duplicate {key} names"
    params = [p["name"] for p in m["params"]]
    assert len(params) == len(set(params)), "duplicate param names"
    sig = inspect.signature(node.execute)
    assert next(iter(sig.parameters)) == "ctx"
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        for name in [p["name"] for p in m["inputs"]] + params:
            assert name in sig.parameters, f"execute() ignores {name!r}"
    for p in m["params"]:
        if p.get("options") and p.get("default") is not None:
            assert p["default"] in p["options"], p["name"]


def test_shells():
    _assert_shell(ViewTF(), viewer=None)
    _assert_shell(LoadCalibration(), viewer=None)


def test_view_tf_emits_table_artifact(tmp_path):
    """View TF's shell: the table output carries a published JSON artifact
    with one record per period (rho/phase for the off-diagonals)."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    periods = np.array([1.0, 10.0])
    Z = np.zeros((2, 2, 2), dtype=complex)
    Z[:, 0, 1] = 100 + 0j  # ex x hy (xy)
    Z[:, 1, 0] = 90 + 0j   # ey x hx (yx)
    tf = build_tf(periods, Z, ["hx", "hy"], ["ex", "ey"])
    out = ViewTF().execute(ctx, tf)
    artifact = out["table"]
    assert artifact["kind"] == "table"
    assert len(ctx.published) == 1
    records = json.loads(Path(artifact["path"]).read_text())
    assert [r["period_s"] for r in records] == [1.0, 10.0]
    assert set(records[0]) >= {
        "rho_xy_ohm_m", "rho_yx_ohm_m", "phase_xy_deg", "phase_yx_deg"}
    assert records[0]["rho_xy_ohm_m"] > 0


def test_load_calibration_parses_the_chn_sample(tmp_path):
    """Load Calibration's shell: the tracked LEMI .chn sample becomes a
    per-channel calibration payload (same parse path test_calibration
    exercises on the functions directly)."""
    assert REAL_CHN.exists(), "tracked lemi.chn missing from checkout"
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    out = LoadCalibration().execute(ctx, path=str(REAL_CHN))
    assert set(out) == {"calibration"}
    filters = out["calibration"]
    # the LEMI sample's five channels arrive under their instrument names
    assert sorted(filters) == ["bx", "by", "bz", "e1", "e2"]
    assert filters["bx"].frequencies.size == 14
    assert filters["bx"].amplitudes[0] == filters["bx"].amplitudes[2]