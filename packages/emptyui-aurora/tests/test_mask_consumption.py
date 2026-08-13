"""Compute FC consumes every quality-mask channel (M26, §9.5).

The Phoenix loader's ``saturated`` and Mask TS's ``quality_mask`` both
feed the pooled window-fraction rule; the union is the policy (a sample
flagged by ANY mask is bad). The span concat pads missing masks with
zeros (no flagged samples there) instead of dropping the channel.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from _synthetic import synthetic_runts, synthetic_source
from emptyui_core_nodes.types import MASK_CHANNELS

pytestmark = pytest.mark.filterwarnings("ignore")


def _payload_with_mask(name: str, n: int = 2048, flagged=None):
    """A synthetic duck payload with a mask channel added to the dataset."""

    src = synthetic_source(1, n=n, sr=8.0)
    runts = synthetic_runts(src)
    ds = runts.dataset
    mask = np.zeros(ds.sizes["time"], dtype=np.uint8)
    if flagged:
        mask[flagged] = 1
    ds = ds.copy()
    ds[name] = ("time", mask, {"type": "auxiliary", "units": "1"})
    return SimpleNamespace(
        dataset=ds, sample_rate=float(runts.sample_rate),
        start=getattr(runts, "start", None),
        filters=getattr(runts, "filters", None))


from types import SimpleNamespace


def _compute(payload, **kw):
    from emptyui_aurora._processing import compute_fc_levels

    kw.setdefault("num_decimations", 0)
    kw.setdefault("num_samples_window", 64)
    kw.setdefault("window_type", "boxcar")
    kw.setdefault("clock_zero", "epoch")
    return compute_fc_levels(payload, "hx,hy", "ex,ey", **kw)


def test_quality_mask_rejects_windows(tmp_path):
    """A quality_mask channel NaNs the windows overlapping its flags —
    the same policy the saturated mask has always had."""
    payload = _payload_with_mask("quality_mask", flagged=range(100, 112))
    levels = _compute(payload)
    # skip the DC bin (NaN under first-difference prewhitening by design)
    ex = np.asarray(levels[0]["ex"].values)[1:]
    nan_windows = np.isnan(ex).any(axis=0)
    assert nan_windows.any()
    # the rejected windows are exactly those whose span includes the flags
    assert levels[0].attrs["saturated_windows_rejected"] == int(
        nan_windows.sum())
    assert levels[0].attrs["mask_channels"] == "quality_mask"


def test_masks_union_across_names(tmp_path):
    """saturated + quality_mask compose: the rejection count is the union,
    and the attrs name both contributors."""
    payload = _payload_with_mask("saturated", flagged=range(100, 112))
    ds = payload.dataset
    extra = np.zeros(ds.sizes["time"], dtype=np.uint8)
    extra[500:512] = 1
    ds["quality_mask"] = ("time", extra, {"type": "auxiliary", "units": "1"})
    levels = _compute(payload)
    n = levels[0].attrs["saturated_windows_rejected"]
    only_sat = _compute(_payload_with_mask("saturated", flagged=range(100, 112)))
    assert n > only_sat[0].attrs["saturated_windows_rejected"]
    assert levels[0].attrs["mask_channels"] == "saturated,quality_mask"


def test_no_mask_channel_is_a_no_op(tmp_path):
    payload = _payload_with_mask("hx", flagged=range(100, 112))  # not a mask
    levels = _compute(payload)
    assert levels[0].attrs["saturated_windows_rejected"] == 0


def test_span_concat_pads_missing_masks(caplog):
    """_concat_span_channels pads a missing mask channel with zeros (no
    flagged samples there) instead of dropping it or fabricating data."""

    import xarray as xr
    from emptyui_aurora.process import _concat_span_channels

    a = {"hx": xr.DataArray(np.arange(8.0), dims=("time",)),
         "quality_mask": xr.DataArray(np.zeros(8), dims=("time",))}
    b = {"hx": xr.DataArray(np.arange(8.0) + 8, dims=("time",))}
    out = _concat_span_channels(
        [("a", a), ("b", b)], "span", logging.getLogger("emptyui"))
    assert "quality_mask" in out
    np.testing.assert_array_equal(
        np.asarray(out["quality_mask"]), np.zeros(16))


def test_mask_channels_is_the_canonical_pair():
    from emptyui_core_nodes.types import MASK_CHANNELS as canonical

    assert MASK_CHANNELS == canonical == ("saturated", "quality_mask")
