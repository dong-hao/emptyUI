"""Select Runs node tests (DESIGN §24.13): run-level QC by per-window
impedance-level deviation in a reference band, with the soft diurnal
prior. Weckmann et al. (2005, GJI 161:635): power and coherence alone
mislead; here the score is the run's stable impedance level vs the
station-wide median — self-calibrating, both tails checked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora.process import SelectRuns

pytestmark = pytest.mark.filterwarnings("ignore")

FREQS = np.array([0.05, 0.12, 0.2, 0.3, 0.5, 0.7, 1.5])  # band 0.1-1.0 Hz


def _ds_for_run(rid: str, dev: float, start: str, n_win: int = 8):
    """One run's FC-level dataset: unit magnetic channels, electric channels
    scaled so the per-window |Z| = dev in every bin."""
    import xarray as xr

    nt = n_win
    h = np.ones((FREQS.size, nt), dtype=complex)
    e = np.sqrt(dev) * h
    times = (np.datetime64(start) + np.arange(nt) * np.timedelta64(600, "s"))
    ds = xr.Dataset(
        data_vars={"hx": (("frequency", "time"), h),
                   "hy": (("frequency", "time"), h),
                   "ex": (("frequency", "time"), e),
                   "ey": (("frequency", "time"), e)},
        coords={"frequency": FREQS, "time": times,
                "run_id": ("time", np.array([rid] * nt))},
        attrs={"input_channels": "hx,hy", "output_channels": "ex,ey",
               "sample_rate": 1.0},
    )
    return ds


def _payload(extra=None):
    """5 clean anchor runs + one 10x-inflated run + two mild (1.8x) runs:
    'noon' starts 2024-01-01T04:00 UTC (= local noon at UTC+8), 'midnight'
    starts 2024-01-01T16:00 UTC (= local midnight). Runs are concatenated
    along time into one merged level (the Build FC output shape)."""
    import xarray as xr

    runs = {f"clean_{k}": (1.0, "2024-01-01T00:00:00") for k in range(5)}
    runs["noisy"] = (10.0, "2024-01-01T08:00:00")
    runs["mild_noon"] = (1.8, "2024-01-01T04:00:00")
    runs["mild_midnight"] = (1.8, "2024-01-01T16:00:00")
    if extra:
        runs.update(extra)
    parts = [_ds_for_run(rid, dev, start)
             for rid, (dev, start) in runs.items()]
    merged = xr.concat(parts, dim="time")
    merged.attrs.update(parts[0].attrs)
    return {0: merged}


@pytest.fixture()
def ctx(tmp_path):
    return RunContext(workspace=tmp_path, store=DataStore())


def test_select_runs_drops_noisy_and_strict_noon(ctx):
    res = SelectRuns().execute(ctx, _payload(), deviation_max=2.0,
                               use_prior=True, prior_max=2.0,
                               local_utc_offset=8.0)
    out = res["fc"][0]
    kept = set(np.unique(out["run_id"].values).astype(str))
    assert kept == {f"clean_{k}" for k in range(5)} | {"mild_midnight"}
    assert out.attrs["runs_dropped"] == "mild_noon,noisy"
    assert out.attrs["select_runs_band_s"] == "1.0-10.0"
    import json

    rows = json.loads(Path(res["table"]["path"]).read_text())
    by_id = {r["run_id"]: r for r in rows}
    assert by_id["noisy"]["deviation"] == pytest.approx(10.0)
    assert by_id["mild_noon"]["threshold"] == pytest.approx(1.0)
    assert by_id["mild_midnight"]["threshold"] == pytest.approx(2.0)


def test_select_runs_no_prior_judges_all_runs_alike(ctx):
    res = SelectRuns().execute(ctx, _payload(), deviation_max=2.0,
                               use_prior=False)
    out = res["fc"][0]
    kept = set(np.unique(out["run_id"].values).astype(str))
    assert kept == {f"clean_{k}" for k in range(5)} | {
        "mild_noon", "mild_midnight"}


def test_select_runs_flags_dead_channels_too(ctx):
    """A dead electric channel (|Z| -> 0) deviates as much as an inflated
    one — Weckmann's low-power noise tail."""
    payload = _payload({"dead": (0.02, "2024-01-01T00:00:00")})
    res = SelectRuns().execute(ctx, payload, deviation_max=2.0,
                               use_prior=False)
    out = res["fc"][0]
    kept = set(np.unique(out["run_id"].values).astype(str))
    assert "dead" not in kept


def test_select_runs_requires_run_id(ctx):
    import xarray as xr

    ds = _ds_for_run("a", 1.0, "2024-01-01T00:00:00").drop_vars("run_id")
    with pytest.raises(ValueError, match="run_id"):
        SelectRuns().execute(ctx, {0: ds})


def test_select_runs_band_miss_raises(ctx):
    with pytest.raises(ValueError, match="reference band"):
        SelectRuns().execute(ctx, _payload(), period_min=100.0,
                             period_max=200.0, use_prior=False)
