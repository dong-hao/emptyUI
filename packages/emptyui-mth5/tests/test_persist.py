"""FC persistence round-trip through the workspace MTH5 (history §14
Phase 1). Error surfaces live in test_errors.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_aurora._processing import compute_fc
from emptyui_mth5.persist import (
    WORKSPACE_MTH5_NAME,
    load_fc_level,
    persist_fc_level,
    runts_context,
)


def test_fc_persist_roundtrip(tmp_path: Path, fake_runts):
    ctx = RunContext(workspace=tmp_path, store=DataStore())
    runts = fake_runts

    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")
    # the fc must carry an absolute (datetime64) time axis for persistence
    assert "datetime64" in str(fc.time.dtype)
    assert fc.attrs["start_datetime"]

    survey, station, run = runts_context(runts)
    assert (survey, station, run) == ("ws", "local", "run1")

    path = persist_fc_level(ctx, fc, 0, fc.attrs["sample_rate"],
                            survey=survey, station=station, run=run)
    assert path.name == WORKSPACE_MTH5_NAME
    assert path.exists()

    out = load_fc_level(ctx, 0, survey=survey, station=station, run=run)
    assert set(out.data_vars) == set(fc.data_vars)
    # mth5 stores (time, frequency); our fc is (frequency, time)
    assert out["hx"].shape == (fc.sizes["time"], fc.sizes["frequency"])
    np.testing.assert_allclose(
        np.abs(out["hx"].values.T), np.abs(fc["hx"].values), rtol=1e-6
    )


def test_persist_multiple_decimation_levels(tmp_path: Path, fake_runts):
    """ComputeFC's multi-level fc container persists each level separately."""
    from emptyui_aurora._processing import compute_fc_levels

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    runts = fake_runts
    levels = compute_fc_levels(runts, ["hx", "hy"], ["ex", "ey"],
                               num_decimations=1, factor=4, num_samples_window=128,
                               overlap=0.5, window_type="hamming")

    survey, station, run = runts_context(runts)
    for level, fc in levels.items():
        assert "datetime64" in str(fc.time.dtype)
        persist_fc_level(ctx, fc, level, fc.attrs["sample_rate"],
                         survey=survey, station=station, run=run)

    for level in (0, 1):
        out = load_fc_level(ctx, level, survey=survey, station=station, run=run)
        assert set(out.data_vars) == set(levels[level].data_vars)
        assert out["hx"].shape == (
            levels[level].sizes["time"],
            levels[level].sizes["frequency"],
        )


def test_read_back_fc_feeds_estimate_tf(tmp_path: Path, fake_runts):
    """The full FC round-trip through an mth5 archive must produce an fc that
    EstimateTF can consume: attrs enriched (sample_rate, channel roles via the
    nomenclature) and the (time, frequency) layout normalized."""
    from emptyui_aurora._processing import estimate_tf
    from emptyui_mth5.fc import FCGroupToSpectra
    from mth5.mth5 import MTH5

    ctx = RunContext(workspace=tmp_path, store=DataStore())
    runts = fake_runts
    fc = compute_fc(runts, ["hx", "hy"], ["ex", "ey"],
                    window_length_s=256, overlap=0.5, window_type="hamming")

    survey, station, run = runts_context(runts)
    path = persist_fc_level(ctx, fc, 0, fc.attrs["sample_rate"],
                            survey=survey, station=station, run=run)

    m = MTH5()
    m.open_mth5(str(path), mode="r")
    try:
        spectra = FCGroupToSpectra().execute(
            ctx, m, survey=survey, station=station, run=run,
            decimation_level=0,
        )["fc"]
    finally:
        m.close_mth5()

    # enrichment: the read-back fc carries the metadata EstimateTF needs
    assert spectra.attrs["sample_rate"] > 0
    assert spectra.attrs["channel_nomenclature"] == "default"
    assert spectra.attrs["input_channels"] == "hx,hy"
    assert spectra.attrs["output_channels"] == "ex,ey"

    periods, Z, _ = estimate_tf(spectra, ["hx", "hy"], ["ex", "ey"],
                                estimator_engine="Least Squares",
                                periods_per_octave=8)
    assert periods.size > 1
    assert np.all(np.isfinite(Z))


# --- payload-truth identity (runts_context) -------------------------------


class _Duck:
    """Payload without metadata: params are the only id source."""

    def __init__(self):
        self.survey = "S"
        self.station = "B"
        self.run = "r"


class _WithRunMd:
    """mth5-style payload: run_metadata.id is the truth."""

    class _Md:
        def __init__(self, i):
            self.id = i

    def __init__(self, run_id):
        self.run_metadata = self._Md(run_id)


def test_runts_context_param_wins_without_payload_metadata(tmp_path):
    from emptyui_mth5.persist import runts_context

    assert runts_context(_Duck()) == ("S", "B", "r")
    # the payload's plain fields are metadata too — a disagreeing param
    # loses (same payload-truth rule); params fill only the true gap
    _survey, _station, run = runts_context(
        _Duck(), run="override",
        logger=__import__("logging").getLogger("emptyui.test"))
    assert run == "r"

    class _Bare:
        pass

    assert runts_context(_Bare()) == ("workspace", "local", "run")
    assert runts_context(_Bare(), run="override") == (
        "workspace", "local", "override")


def test_runts_context_payload_truth_overrides_stale_param(caplog):
    """A stale param (e.g. a hand-typed run id from a previous test) must
    not poison the fc/TF metadata: the payload id wins, loudly."""
    from emptyui_mth5.persist import runts_context

    _survey, _station, run = runts_context(
        _WithRunMd("sr1_0001"), survey="NO46", station="NO46", run="a,b",
        logger=__import__("logging").getLogger("emptyui.test"))
    assert run == "sr1_0001"
    assert "disagrees with the payload's own id" in caplog.text
    assert "'a,b'" in caplog.text


def test_runts_context_agreeing_params_pass_through():
    from emptyui_mth5.persist import runts_context

    assert runts_context(_WithRunMd("sr1_0001"), run="sr1_0001") == (
        "workspace", "local", "sr1_0001")
    assert runts_context(_WithRunMd("sr1_0001")) == (
        "workspace", "local", "sr1_0001")


def test_runts_context_ignores_default_ids(caplog):
    """mt_metadata's DEFAULT id ('0') means 'unset' — payload truth does not
    include unset defaults: a payload built via RunTS(array_list=...)
    (default ids) must not override real user params (the fc/TF identity
    used to become survey='0', station='0', run='0')."""
    import logging
    import warnings

    import numpy as np
    from mt_timeseries import ChannelTS, RunTS

    warnings.filterwarnings("ignore")
    n = 64
    chs = []
    for comp in ("hx", "hy"):
        c = ChannelTS("magnetic", data=np.zeros(n),
                      start="2024-01-01T00:00:00", sample_rate=1.0)
        c.component = comp
        chs.append(c)
    runts = RunTS(array_list=chs)  # default metadata: ids == "0"
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        survey, station, run = runts_context(runts, survey="4P",
                                             station="SDX", run="r1")
    assert (survey, station, run) == ("4P", "SDX", "r1")
    # and no spurious disagreement warnings
    assert not any("disagrees" in r.message for r in caplog.records)
