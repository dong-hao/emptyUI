"""Auxiliary QC channels pass the core filter nodes UNTOUCHED (M26).

The latent hazard this pins: ``coerce_run`` converts EVERY channel to
float64, so ``Load Phoenix -> Detrend -> Compute FC`` used to filtfilting
the ``saturated`` mask (a 0/1 step train) — a silently corrupted gate.
The mask must survive Detrend/Bandpass bit-for-bit, on the single path
AND the collection map.
"""

from __future__ import annotations

import numpy as np
import pytest
from emptyui.context import RunContext
from emptyui.store import DataStore
from emptyui_core_nodes.types import MASK_CHANNELS, TimeSeries, coerce_run

pytestmark = pytest.mark.filterwarnings("ignore")


def _payload(n: int = 256) -> TimeSeries:
    rng = np.random.default_rng(3)
    mask = np.zeros(n, dtype=np.uint8)
    mask[100:110] = 1  # a flagged stretch a filter would smear
    return TimeSeries(
        channels={"hx": rng.standard_normal(n), "saturated": mask},
        sample_rate=1.0, start=0.0, run="a",
        aux_channels=frozenset({"saturated"}))


def _collection():
    from types import SimpleNamespace

    import xarray as xr

    t = np.arange(64) * np.timedelta64(1, "s")
    mask = np.zeros(64, dtype=np.uint8)
    mask[10:12] = 1
    ds = xr.Dataset(
        {"hx": ("time", np.random.default_rng(5).standard_normal(64)),
         "saturated": ("time", mask, {"type": "auxiliary", "units": "1"})},
        coords={"time": t})
    duck = SimpleNamespace(dataset=ds, sample_rate=1.0, start=t[0],
                           filters={})
    # extents rows let the map name the member
    return SimpleNamespace(
        runs=[duck],
        extents=[{"survey": "s", "station": "st", "run": "a",
                  "start": "", "end": "", "sample_rate": 1.0}])


def test_coerce_run_detects_aux_channels():
    from types import SimpleNamespace

    import xarray as xr

    t = np.arange(8) * np.timedelta64(1, "s")
    ds = xr.Dataset(
        {"hx": ("time", np.arange(8.0)),
         "quality_mask": ("time", np.zeros(8, dtype=np.uint8),
                          {"type": "auxiliary"})},
        coords={"time": t})
    ts = coerce_run(SimpleNamespace(
        dataset=ds, sample_rate=1.0, start=t[0], filters={}))
    assert ts.aux_channels == frozenset({"quality_mask"})


def test_coerce_run_masks_are_aux_by_name():
    """A mask channel counts as auxiliary even without the type attr
    (the MASK_CHANNELS names are the contract)."""
    ts = _payload()
    assert ts.aux_channels == frozenset({"saturated"})


def test_timeseries_copy_preserves_aux():
    ts = _payload()
    assert ts.copy().aux_channels == frozenset({"saturated"})


def test_detrend_passes_the_mask_through_bit_for_bit():
    from emptyui_core_nodes.filters import Detrend

    ts = _payload()
    out = Detrend().execute(
        RunContext(workspace=None, store=DataStore()), ts,
        method="linear")["data"]
    assert "saturated" in out.aux_channels
    np.testing.assert_array_equal(
        out.channels["saturated"], ts.channels["saturated"])
    # the DATA channel was actually detrended (not a blanket passthrough)
    assert not np.array_equal(out.channels["hx"], ts.channels["hx"])


def test_bandpass_passes_the_mask_through_bit_for_bit():
    from emptyui_core_nodes.filters import Bandpass

    ts = TimeSeries(
        channels={"hx": np.random.default_rng(7).standard_normal(512),
                  "saturated": np.zeros(512, dtype=np.uint8)},
        sample_rate=8.0, aux_channels=frozenset({"saturated"}))
    ts.channels["saturated"][100:104] = 1
    out = Bandpass().execute(
        RunContext(workspace=None, store=DataStore()), ts,
        low_freq=0.1, high_freq=2.0)["filtered"]
    np.testing.assert_array_equal(
        out.channels["saturated"], ts.channels["saturated"])
    assert not np.array_equal(out.channels["hx"], ts.channels["hx"])


def test_collection_map_passes_the_mask_through():
    from emptyui_core_nodes.filters import Bandpass

    coll = _collection()
    before = coll.runs[0].dataset["saturated"].values.copy()
    out = Bandpass().execute(
        RunContext(workspace=None, store=DataStore()), coll,
        low_freq=0.05, high_freq=0.4)["filtered"]
    np.testing.assert_array_equal(
        out.runs[0].dataset["saturated"].values, before)
    assert set(out.runs[0].dataset.data_vars) == set(
        coll.runs[0].dataset.data_vars)


def test_mask_channel_names_are_the_canonical_pair():
    assert MASK_CHANNELS == ("saturated", "quality_mask")
