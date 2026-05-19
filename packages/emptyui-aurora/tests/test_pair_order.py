"""M11: tolerance pairing returns matched windows in local time order.

Coherence Sort's sliding ensemble is defined by array index, so the matched
pairs must come back chronological (issues.md M11).
"""

from __future__ import annotations

import numpy as np
from emptyui_aurora._processing import _pair_remote_times


def test_tolerance_pairs_come_back_in_local_time_order():
    """The closest pair is found first, but the matched pairs must be returned
    in local time order (Coherence Sort indexes its ensemble by position)."""
    lt = np.array([0.0, 10.0])
    rt = np.array([0.5, 10.1])
    li, ri = _pair_remote_times(lt, rt, 1.0)
    assert li.tolist() == [0, 1]
    assert ri.tolist() == [0, 1]
