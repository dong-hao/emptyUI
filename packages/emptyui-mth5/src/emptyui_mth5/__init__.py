"""emptyui-mth5 package."""

from __future__ import annotations

from ._compat import (
    apply_channel_rate_fix,
    apply_crs_cache,
    apply_metadata_class_cache,
    apply_unit_object_cache,
)

# datum validation via pyproj must not rebuild CRS objects thousands of
# times per pass (history §24.8 performance note)
apply_crs_cache()
# ChannelTS's endpoint sample-rate method is biased for short high-rate
# runs (a 2-s 2400 Hz slot derives 2401 — §24.8 performance note)
apply_channel_rate_fix()
# mth5 builds a new pydantic model class per group/channel init — an
# ~80 s tax on a ~1300-run burst ingest (history §24.8 performance note)
apply_metadata_class_cache()
# unit validation rebuilds Unit objects from a DataFrame lookup per
# attribute update — ~18 s on a ~1000-run burst ingest
apply_unit_object_cache()
