"""Phase 7 aurora-group LOW regressions (L-A3..L-A12)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from emptyui_aurora._processing import (
    _container_geometry,
    _pair_remote_times,
    _resolve_cross_level_claims,
    coherence_sort_fc,
    compute_fc,
)


def _fake_level(fs: float, n_win: int = 4, N: int = 256) -> xr.Dataset:
    ds = xr.Dataset(
        data_vars={
            "hx": (("frequency", "time"), np.ones((N // 2 + 1, n_win), complex)),
            "hy": (("frequency", "time"), np.ones((N // 2 + 1, n_win), complex)),
            "ex": (("frequency", "time"), np.ones((N // 2 + 1, n_win), complex)),
            "ey": (("frequency", "time"), np.ones((N // 2 + 1, n_win), complex)),
        },
        coords={"frequency": np.arange(N // 2 + 1) * fs / N,
                "time": np.arange(n_win)},
        attrs={"sample_rate": fs, "num_samples_window": N},
    )
    return ds


# --- L-A5: single-level compute_fc overlap guard -----------------------------


def test_compute_fc_rejects_overlap_ge_one():
    from mt_timeseries import ChannelTS, RunTS

    rng = np.random.default_rng(1)
    chs = [ChannelTS("magnetic", data=rng.standard_normal(64), start="2024-01-01",
                    sample_rate=1.0)]
    chs[0].component = "hx"
    runts = RunTS(array_list=chs)
    with pytest.raises(ValueError, match=r"overlap must be in \[0, 1\)"):
        compute_fc(runts, ["hx"], ["hx"], 32.0, 1.0, "hamming")


# --- L-A6: the dead cascade factor field is gone ------------------------------


def test_container_geometry_has_no_factor():
    levels = {lv: _fake_level(150.0 / 4 ** lv) for lv in range(3)}
    g = _container_geometry(levels)
    assert g is not None and "factor" not in g


# --- L-A8: cross-level claims keep every unclaimed piece ----------------------


def test_cross_level_claims_keep_all_pieces():
    """A shallower band cut by two deeper claims leaves TWO unclaimed
    pieces — both are kept (the old keep-the-center-piece rule left a
    permanent hole estimated by nobody)."""
    levels = {0: _fake_level(10.0), 1: _fake_level(1.0)}
    out = {
        1: [(1.0 / 0.20, 1.0 / 0.25)],   # deep claims periods [0.20, 0.25]
        0: [(1.0 / 0.125, 1.0 / 0.40)],  # shallow band spans [0.125, 0.40]
    }
    kept = _resolve_cross_level_claims(out, levels)
    # the shallow level keeps BOTH pieces: [0.125, 0.20] and [0.25, 0.40]
    spans = sorted((1.0 / hi, 1.0 / lo) for hi, lo in kept[0])
    assert len(spans) == 2
    assert spans[0][0] == pytest.approx(0.125, rel=1e-6)
    assert spans[0][1] == pytest.approx(0.20, rel=1e-6)
    assert spans[1][0] == pytest.approx(0.25, rel=1e-6)
    assert spans[1][1] == pytest.approx(0.40, rel=1e-6)




# --- L-A12: one-to-one pairing (no remote reuse) ------------------------------


def test_pair_remote_times_dedupes_duplicate_local_times():
    t = np.array([0, 1, 2], dtype="datetime64[s]")
    li = np.array([t[0], t[0], t[1]])  # duplicated start
    ri = np.array([t[0], t[1], t[2]])
    li_idx, ri_idx = _pair_remote_times(li, ri, 0.0)
    assert len(set(ri_idx)) == len(ri_idx)  # bijection: no remote reuse
    assert li_idx.tolist() == [0, 2]        # the first duplicate wins


