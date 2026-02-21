"""emptyui-mth5 package."""

from __future__ import annotations

from ._compat import apply_channel_rate_fix, apply_crs_cache

# datum validation via pyproj must not rebuild CRS objects thousands of
# times per pass (DESIGN §24.8 performance note)
apply_crs_cache()
# ChannelTS's endpoint sample-rate method is biased for short high-rate
# runs (a 2-s 2400 Hz slot derives 2401 — §24.8 performance note)
apply_channel_rate_fix()
