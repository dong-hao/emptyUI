"""Residual Sort node tests (DESIGN.md §10.4): residual-based two-pass
selection. The score asks 'does this window fit the first-pass TF model'
rather than 'is the floor high' — the orthogonal complement of power and
coherence screening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora.process import ResidualSort

pytestmark = pytest.mark.filterwarnings("ignore")

FREQS = np.array([0.05, 0.12, 0.2, 0.3, 0.5, 0.7, 1.5])  # band 0.1-1.0 Hz


def _ds_for_run(rid: str, z: float, start: str, n_win: int = 8,
                noise_scale: float = 0.0):
    """One run's FC level: unit magnetics, outputs = z * magnetics plus
    optional model-INCONSISTENT noise (uncorrelated with the magnetics —
    the residual test's target signature)."""
    import xarray as xr

    rng = np.random.default_rng(sum(ord(c) for c in rid) % 2**32)
    h = np.ones((FREQS.size, n_win), dtype=complex)
    e = z * h
    if noise_scale:
        e = e + noise_scale * (
            rng.standard_normal((FREQS.size, n_win))
            + 1j * rng.standard_normal((FREQS.size, n_win)))
    times = (np.datetime64(start) + np.arange(n_win) * np.timedelta64(600, "s"))
    ds = xr.Dataset(
        data_vars={"hx": (("frequency", "time"), h),
                   "hy": (("frequency", "time"), h),
                   "ex": (("frequency", "time"), e),
                   "ey": (("frequency", "time"), e)},
        coords={"frequency": FREQS, "time": times,
                "run_id": ("time", np.array([rid] * n_win))},
        attrs={"input_channels": "hx,hy", "output_channels": "ex,ey",
               "sample_rate": 1.0},
    )
    return ds


def _payload():
    """5 clean runs (pure Z*H signal, tiny consistent noise) + one run with
    large model-inconsistent noise. Runs concatenated along time."""
    import xarray as xr

    parts = [_ds_for_run(f"clean_{k}", 1.0, "2024-01-01T00:00:00",
                         noise_scale=0.01) for k in range(5)]
    parts.append(_ds_for_run("noisy", 1.0, "2024-01-01T08:00:00",
                             noise_scale=0.5))
    merged = xr.concat(parts, dim="time")
    merged.attrs.update(parts[0].attrs)
    return {0: merged}


def _tf_levels(z=1.0):
    """First-pass tf levels matching the payload: one band covering the
    in-band bins, Z = z * I."""
    return {0: {
        "periods": np.array([5.0]),
        "Z": (z * np.eye(2, dtype=complex))[np.newaxis, :, :],
        "Z_err": np.zeros((1, 2, 2)),
        "input_channels": ["hx", "hy"],
        "output_channels": ["ex", "ey"],
        "band_intervals": [(1.0, 0.1)],
        "periods_per_octave": 3,
    }}


@pytest.fixture()
def ctx(tmp_path):
    return RunContext(workspace=tmp_path, store=DataStore())


def test_residual_sort_drops_model_inconsistent_run(ctx):
    res = ResidualSort().execute(ctx, _payload(), _tf_levels(),
                                 unexplained_max=2.0, use_prior=False)
    out = res["fc"][0]
    kept = set(map(str, np.unique(out["run_id"].values)))
    assert "noisy" not in kept
    assert kept == {f"clean_{k}" for k in range(5)}
    import json

    rows = json.loads(Path(res["table"]["path"]).read_text())
    by_id = {r["run_id"]: r for r in rows}
    assert by_id["noisy"]["unexplained"] > 2.0
    assert by_id["clean_0"]["unexplained"] <= 2.0


def test_residual_sort_flags_tight_but_wrong_run_with_bounded_score(ctx):
    """A coherent-but-wrong run (tight cluster at Z=2 while the model says
    Z=1) DOES drop — it disagrees with the model — but with a bounded,
    structured score (~0.25 = (2/1-1)^2/(2/1)^2) rather than the explosive
    score of incoherent noise. The residual lever separates 'model-
    inconsistent' from 'floor-high'; the impedance-level lever (Select
    Runs) handles the scale question."""
    import xarray as xr

    parts = [_ds_for_run(f"clean_{k}", 1.0, "2024-01-01T00:00:00",
                         noise_scale=0.01) for k in range(5)]
    parts.append(_ds_for_run("wrong", 2.0, "2024-01-01T08:00:00",
                             noise_scale=0.01))
    merged = xr.concat(parts, dim="time")
    merged.attrs.update(parts[0].attrs)
    res = ResidualSort().execute(ctx, {0: merged}, _tf_levels(),
                                 unexplained_max=2.0, use_prior=False)
    import json

    rows = json.loads(Path(res["table"]["path"]).read_text())
    by_id = {r["run_id"]: r for r in rows}
    assert by_id["wrong"]["dropped"]
    # bounded, structured model-inconsistency: raw unexplained fraction
    # ~0.25 = (2/1-1)^2 / (2/1)^2, vs the explosive incoherent-noise score
    assert by_id["wrong"]["unexplained_frac"] == pytest.approx(0.25, rel=0.2)


def test_residual_sort_requires_run_id(ctx):
    ds = _ds_for_run("a", 1.0, "2024-01-01T00:00:00").drop_vars("run_id")
    with pytest.raises(ValueError, match="run_id"):
        ResidualSort().execute(ctx, {0: ds}, _tf_levels())


def test_residual_sort_requires_band_intervals(ctx):
    tf = _tf_levels()
    tf[0]["band_intervals"] = None
    with pytest.raises(ValueError, match="band_intervals|test could be formed"):
        ResidualSort().execute(ctx, _payload(), tf, use_prior=False)


def test_residual_sort_z_row_mismatch_raises(ctx):
    tf = _tf_levels()
    tf[0]["Z"] = np.zeros((3, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="do not match"):
        ResidualSort().execute(ctx, _payload(), tf, use_prior=False)


def test_residual_sort_empty_scope_warns_and_passes(ctx):
    """Bands outside the period scope: warn and return the payload
    unfiltered (residual_sort_tests == 0)."""
    res = ResidualSort().execute(ctx, _payload(), _tf_levels(),
                                 period_min=100.0, use_prior=False)
    assert res["fc"][0].attrs["residual_sort_tests"] == 0
    kept = set(map(str, np.unique(res["fc"][0]["run_id"].values)))
    assert "noisy" in kept  # nothing was filtered


def test_band_intervals_populated_for_uniform_style(ctx):
    """M10: band_intervals were only populated on the l2u primary path —
    for band_style='uniform' (and the l2u metadata-less fallback) the
    entry carried None and Residual Sort silently skipped the level.
    estimate_tf_levels now resolves the same table estimate_tf uses and
    records it, so the pooling key always exists."""
    from emptyui_aurora._processing import compute_fc_levels, estimate_tf_levels
    from test_magnetic_units import _field, _runts

    rng = np.random.default_rng(11)
    hx, hy, ex, ey = _field(rng, n=16384, sr=8.0)
    runts = _runts(hx, hy, ex, ey)
    levels = compute_fc_levels(runts, "hx,hy", "ex,ey", num_decimations=2,
                               factor=4, num_samples_window=256,
                               overlap=0.5, window_type="hamming")
    # Residual Sort's contract: per-window run_id (Build FC stamps it)
    for lv in levels:
        levels[lv] = levels[lv].assign_coords(
            run_id=("time", np.full(int(levels[lv].sizes["time"]), "r1")))
    result = estimate_tf_levels(
        levels, ["hx", "hy"], ["ex", "ey"], estimator_engine="OLS",
        periods_per_octave=3, band_style="uniform")
    tested = 0
    for lv, entry in result.items():
        assert entry["band_intervals"], f"level {lv} lost its band geometry"
        assert len(entry["band_intervals"]) == entry["Z"].shape[0], (
            "intervals must align 1:1 with the Z rows")
        # and Residual Sort actually tests the level now
        res = ResidualSort().execute(
            ctx, {lv: levels[lv]}, {lv: entry}, period_min=0, period_max=0)
        assert "fc" in res
        tested += 1
    assert tested == len(result)
