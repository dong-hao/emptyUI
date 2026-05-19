"""M22: exact remote-reference pairing must be linear in the number of windows.

The fix replaces a per-row list membership test with a boolean mask. The
keep-first duplicate semantics (L-A12) must survive (issues.md M22).
"""

from __future__ import annotations

import time

import numpy as np
from emptyui_aurora._processing import _pair_remote_times


def test_exact_pairing_keeps_the_first_of_duplicate_local_times():
    """L-A12 semantics: a local window time that occurs twice pairs only with
    the first occurrence."""
    lt = np.array([0.0, 0.0, 128.0])
    rt = np.array([0.0, 128.0])
    li, ri = _pair_remote_times(lt, rt, 0.0)
    assert li.tolist() == [0, 2]
    assert ri.tolist() == [0, 1]


def test_exact_pairing_scales_to_tens_of_thousands_of_windows():
    """5*10^4 windows on identical grids pair in well under a second; the old
    per-row list scan took about 9 s at this size (37 s at 10^5)."""
    n = 50000
    t = np.arange(n, dtype=float) * 128.0
    t_start = time.perf_counter()
    li, ri = _pair_remote_times(t, t, 0.0)
    elapsed = time.perf_counter() - t_start
    assert li.size == n and ri.size == n
    assert elapsed < 2.0, f"exact pairing took {elapsed:.1f} s for {n} windows"
