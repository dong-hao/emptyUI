"""Phase 7 aurora-group LOW regressions (L-A3..L-A12)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from _synthetic import compute_fc_seconds
from emptyui_aurora._processing import (
    _container_geometry,
    _pair_remote_times,
    _resolve_cross_level_claims,
    coherence_sort_fc,
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
        compute_fc_seconds(runts, ["hx"], ["hx"], 32.0, 1.0, "hamming")


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


# --- L-A9: actionable errors ---------------------------------------------------


def test_coherence_sort_rejects_zero_ensemble():
    local = _fake_level(1.0)
    with pytest.raises(ValueError, match="ensemble_windows must be >= 1"):
        coherence_sort_fc(local, None, ["hx", "hy"], ["ex", "ey"],
                          ensemble_windows=0)


def test_all_zero_calibration_response_raises(tmp_path):
    """An all-zero calibration response divides every FC by zero (inf
    garbage); it raises with the remedy instead. The DC-only guard missed
    it because max(|resp[1:]|) is 0."""
    from emptyui_aurora._processing import _calibrate_stft_vars
    from mt_metadata.timeseries.filters import CoefficientFilter

    cf = CoefficientFilter(gain=0.0)  # all-zero response
    f = np.array([1.0, 2.0])
    fc_vars = {"hx": (("frequency", "time"),
                      np.ones((2, 3), dtype=complex))}
    with pytest.raises(ValueError, match="all zero"):
        _calibrate_stft_vars(fc_vars, f, {"hx": [(cf, True)]}, "test")


def test_ref_fc_missing_channel_named():
    """A ref fc lacking an input channel raised a raw KeyError — the error
    names the channel(s) and the remedy."""
    local = _fake_level(1.0)
    ref = _fake_level(1.0)
    ref = ref.drop_vars(["hx"])
    with pytest.raises(ValueError, match="'hx'"):
        coherence_sort_fc(local, ref, ["hx", "hy"], ["ex", "ey"])


def test_magnetic_only_fc_rejected():
    """A magnetic-only fc (no electric outputs) raised a raw numpy stack
    error — the remedy names the channels."""
    from emptyui_aurora.process import SelectRuns

    ds = _fake_level(1.0).drop_vars(["ex", "ey"])
    ds = ds.assign_coords(run_id=("time", np.full(ds.sizes["time"], "a")))
    with pytest.raises(ValueError, match="output"):
        SelectRuns().execute(None, {0: ds})


# --- L-A4: the documented "0 = dipole from run metadata" promise --------------


def test_build_fc_dipole_from_run_metadata(tmp_path):
    """When e_dipole_length_m is 0/empty, the E dipole length is taken from
    the run's electric channel metadata (mt_metadata Electric.dipole_length)
    — the documented-but-unimplemented promise."""
    import warnings

    warnings.filterwarnings("ignore")
    from mt_metadata.timeseries import Electric
    from mt_metadata.timeseries import Run as RunMD
    from mt_timeseries import ChannelTS, RunTS
    from mth5.mth5 import MTH5

    rng = np.random.default_rng(1)
    chs = []
    for comp in ("hx", "hy", "ex", "ey"):
        cls = "magnetic" if comp.startswith("h") else "electric"
        c = ChannelTS(cls, data=rng.standard_normal(64), start="2024-01-01",
                      sample_rate=1.0)
        c.component = comp
        if cls == "electric":
            emd = Electric()
            emd.dipole_length = 100.0
            c.channel_metadata = emd
        chs.append(c)
    runts = RunTS(array_list=chs)
    runts.run_metadata = RunMD()
    runts.run_metadata.id = "a"

    path = tmp_path / "dip.h5"
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey("4P")
    stg = sg.stations_group.add_station("SDX")
    rg = stg.add_run("a")
    rg.from_runts(runts)
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()
    archive = MTH5()
    archive.open_mth5(str(path), "r", locking=False)

    from emptyui.context import RunContext
    from emptyui.store import DataStore
    from emptyui_aurora.process import BuildFC

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    fc = BuildFC().execute(
        ctx, archive, runs="a",
        decimation_levels=0, factor=4, window_samples=16, overlap=0.5,
        window_type="boxcar", clock_zero="epoch", persist=False,
        e_dipole_length_m=0.0)["fc"]
    # the lookup found the metadata length (the archive convention is m)
    assert fc and all(
        float(lv.attrs["e_dipole_length_m"]) > 0 for lv in fc.values())
