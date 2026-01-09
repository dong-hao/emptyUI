"""Compatibility patches for upstream library pathologies.

pyproj ``CRS.from_user_input`` is called by mt_metadata's datum validator
(``Location.validate_datum``) on essentially every metadata attribute
update — mth5's group/metadata rebuilds trigger it thousands of times on
burst archives (25k+ CRS constructions — 75 s — for a single Build FC
pass over 121 slot runs, DESIGN §24.8 performance note). CRS objects are
immutable and datum strings repeat exhaustively, so an LRU cache makes
repeat validations free without changing behavior.
"""

from __future__ import annotations

import functools

_patched = False


def apply_crs_cache() -> None:
    """Idempotently LRU-cache ``pyproj.CRS.from_user_input``."""
    global _patched
    if _patched:
        return
    try:
        from pyproj import CRS
    except ImportError:  # pyproj optional outside mt_metadata installs
        return
    CRS.from_user_input = functools.lru_cache(maxsize=64)(
        CRS.from_user_input)
    _patched = True
