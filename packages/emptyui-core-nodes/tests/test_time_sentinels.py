"""Placeholder-epoch start handling (review #10 bundle).

MT's pre-digital era (Cagniard/Tikhonov, 1950s on) means digitized paper
records with placeholder timestamps are legitimate data: unknown starts are
warned about and degraded — never rejected — and genuinely old stamps are
preserved.
"""

from __future__ import annotations

import pandas as pd
import pytest
from emptyui_core_nodes.types import (
    UNKNOWN_EPOCH_STARTS,
    is_unknown_start,
)


def test_placeholder_epochs_detected():
    for e in UNKNOWN_EPOCH_STARTS:
        assert is_unknown_start(pd.Timestamp(e))
        assert is_unknown_start(e)  # ISO strings too
    assert not is_unknown_start(pd.Timestamp("1957-06-15T12:00:00"))
    assert not is_unknown_start(pd.Timestamp("2024-01-01"))
    assert not is_unknown_start(None)
    assert not is_unknown_start("not a time")


@pytest.mark.parametrize("epoch", UNKNOWN_EPOCH_STARTS)
def test_aurora_start_datetime_treats_placeholders_as_unknown(epoch):
    """The aurora fc-time path shares the sentinel (consistency with
    runs._run_time_info): no fake absolute stamps on the fc time axis."""
    from emptyui_aurora._processing import _start_datetime

    assert _start_datetime(epoch) is None
    assert _start_datetime(pd.Timestamp(epoch)) is None


def test_genuinely_old_records_are_preserved():
    """A 1957 (Cagniard-era) digitized start is a real time — kept as-is."""
    from emptyui_aurora._processing import _start_datetime

    ts = pd.Timestamp("1957-06-15T12:00:00")
    assert _start_datetime(ts) == ts
    assert not is_unknown_start(ts)
