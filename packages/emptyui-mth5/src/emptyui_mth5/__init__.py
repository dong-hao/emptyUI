"""emptyui-mth5 package."""

from __future__ import annotations

from ._compat import apply_crs_cache

# datum validation via pyproj must not rebuild CRS objects thousands of
# times per pass (DESIGN §24.8 performance note)
apply_crs_cache()
