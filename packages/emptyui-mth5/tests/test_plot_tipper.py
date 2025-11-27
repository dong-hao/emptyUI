"""Plot Tipper node: renders the geomagnetic transfer function (tipper)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import build_tf
from emptyui_mth5.plots import PlotTipper


def _tipper_tf():
    """A TF carrying a vertical magnetic output (bz): 3x2 -> impedance + tipper."""
    periods = np.array([2.0, 4.0, 8.0])
    Z = np.zeros((3, 3, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j  # ex x bx
    Z[:, 1, 1] = 2 + 0j  # ey x by
    Z[:, 2, 0] = 0.3 + 0.1j  # Tzx
    Z[:, 2, 1] = -0.2 + 0.05j  # Tzy
    return build_tf(periods, Z, ["bx", "by"], ["e1", "e2", "bz"],
                    channel_nomenclature="lemi12")


def test_plot_tipper_renders(tmp_path: Path):
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    tf = _tipper_tf()
    assert tf.has_tipper()
    out = PlotTipper().execute(ctx, tf)
    path = Path(out["plot"]["path"])
    assert path.exists() and path.stat().st_size > 0
    assert path.name == "plot_tipper.png"
    assert path.read_bytes().startswith(b"\x89PNG")


def test_plot_tipper_requires_tipper(tmp_path: Path):
    """A 2x2 (impedance-only) TF has no tipper -> clear error."""
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    periods = np.array([2.0, 4.0])
    Z = np.zeros((2, 2, 2), dtype=complex)
    Z[:, 0, 0] = 1 + 0j
    Z[:, 1, 1] = 2 + 0j
    tf = build_tf(periods, Z, ["bx", "by"], ["e1", "e2"],
                  channel_nomenclature="lemi12")
    assert not tf.has_tipper()
    with pytest.raises(ValueError, match="no tipper"):
        PlotTipper().execute(ctx, tf)
