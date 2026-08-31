"""M6 (b): the library entry point's defaults are the shipped recipe (boxcar +
first-difference prewhitening, DESIGN §10.1). A direct caller that passes no
recipe arguments must get the same FC as the node. Before the fix it silently
got hamming with no prewhitening (docs/issues.md M6).
"""

from __future__ import annotations

import numpy as np
from emptyui_aurora._processing import compute_fc_levels
from emptyui_core_nodes.types import TimeSeries


def _series(n=2048, sr=8.0):
    rng = np.random.default_rng(6)
    return TimeSeries(
        channels={c: rng.standard_normal(n) for c in ("hx", "hy", "ex", "ey")},
        sample_rate=sr)


def test_library_defaults_are_the_shipped_recipe():
    ts = _series()
    kw = {"num_decimations": 0, "num_samples_window": 256, "overlap": 0.5}
    default = compute_fc_levels(ts, "hx,hy", "ex,ey", **kw)
    shipped = compute_fc_levels(ts, "hx,hy", "ex,ey", window_type="boxcar",
                                prewhitening="first difference", **kw)
    for comp in ("ex", "ey", "hx", "hy"):
        np.testing.assert_array_equal(default[0][comp].values,
                                      shipped[0][comp].values)
    # the first-difference recoloring leaves the DC bin NaN (DESIGN §10.1)
    assert np.isnan(default[0]["ex"].values[0]).all()
