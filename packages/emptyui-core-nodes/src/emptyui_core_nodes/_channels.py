"""Channel name resolution for core display nodes.

The mth5/aurora plugins carry the full mt_metadata-based nomenclature
(``emptyui_mth5.nomenclature`` — the canonical implementation). Core nodes
cannot depend on those plugins, so this module mirrors the *minimal*
primitives here: standard ⇄ actual channel mapping via mt_metadata's
canonical ``CHANNEL_MAPS`` (lazy import — mt_metadata is a plugin-tier
dependency) with a built-in fallback copy for standalone use.
"""

from __future__ import annotations

#: Mirrored from mt_metadata.transfer_functions.CHANNEL_MAPS (v1.0.10) —
#: used only when mt_metadata is not importable (core-nodes must stay
#: plugin-free). See emptyui_mth5/nomenclature.py for the canonical copy.
_FALLBACK_MAPS: dict[str, dict[str, str]] = {
    "default": {"ex": "ex", "ey": "ey", "hx": "hx", "hy": "hy", "hz": "hz"},
    "lemi12": {"ex": "e1", "ey": "e2", "hx": "bx", "hy": "by", "hz": "bz"},
    "lemi34": {"ex": "e3", "ey": "e4", "hx": "bx", "hy": "by", "hz": "bz"},
    "musgraves": {"ex": "ex", "ey": "ey", "hx": "bx", "hy": "by", "hz": "bz"},
    "nims": {"ex": "ex", "ey": "ey", "hx": "hx", "hy": "hy", "hz": "hz"},
    "phoenix123": {"ex": "e1", "ey": "e2", "hx": "h1", "hy": "h2", "hz": "h3"},
}

#: Tie-break priority when several keywords match equally well (matches
#: emptyui_mth5.nomenclature._PRIORITY: lemi12 wins for LEMI 4-E runs).
_PRIORITY: tuple[str, ...] = (
    "lemi12", "lemi34", "default", "nims", "musgraves", "phoenix123",
)


def _maps() -> dict[str, dict[str, str]]:
    try:
        from mt_metadata.transfer_functions import CHANNEL_MAPS

        return {k: dict(v) for k, v in CHANNEL_MAPS.items()}
    except ImportError:  # pragma: no cover — mt_metadata present in CI
        return {k: dict(v) for k, v in _FALLBACK_MAPS.items()}


def infer_keyword(available) -> str:
    """Best-effort nomenclature keyword from actual channel names (mirror of
    emptyui_mth5.nomenclature.infer_keyword)."""
    names = {str(c).strip().lower() for c in available}
    if not names:
        return "default"
    best: str | None = None
    best_score = -1
    for kw in _PRIORITY:
        recognized = {a.lower() for a in _maps()[kw].values()}
        score = sum(1 for n in names if n in recognized)
        if score > best_score:
            best, best_score = kw, score
    return best if best_score > 0 else "default"


def standard_name(keyword: str, standard: str) -> str:
    """Map a standard component name to its instrument name for a keyword."""
    return _maps().get(keyword, _FALLBACK_MAPS["default"]).get(standard, standard)


def is_electric_name(name: str) -> bool:
    n = str(name).strip().lower()
    return n in ("ex", "ey") or (len(n) >= 2 and n[0] == "e" and n[1].isdigit())


def is_magnetic_name(name: str) -> bool:
    return str(name).strip().lower() in (
        "hx", "hy", "hz", "bx", "by", "bz", "h1", "h2", "h3",
    )


def _lookup(available_lower: dict[str, str], name: str) -> str | None:
    low = str(name).strip().lower()
    for actual, low_actual in available_lower.items():
        if low_actual == low:
            return actual
    return None


def resolve_channels(available, spec) -> list[str]:
    """Resolve a comma-separated channel spec to actual names in ``available``.

    - empty spec → all channels (data order)
    - names match case-insensitively; the data's exact casing is used
    - standard component names (ex/ey/hx/hy/hz) are mapped through the
      mt_metadata ``CHANNEL_MAPS`` keyword inferred from the available
      channels (e.g. ``ex`` → ``e1`` on LEMI data)
    - unknown names raise ValueError listing the available channels
    """
    available = [str(c) for c in available]
    if not str(spec or "").strip():
        return list(available)
    available_lower = {c: c.lower() for c in available}
    requested = [s.strip() for s in str(spec).split(",") if s.strip()]
    keyword = infer_keyword(available)
    resolved: list[str] = []
    for name in requested:
        actual = _lookup(available_lower, name)
        if actual is None:
            mapped = standard_name(keyword, name.lower())
            if mapped.lower() != name.lower():
                actual = _lookup(available_lower, mapped)
        if actual is None:
            raise ValueError(
                f"channel '{name}' not in data {available} "
                f"(resolved through nomenclature '{keyword}')"
            )
        resolved.append(actual)
    return resolved


def auto_channel(available) -> str:
    """Pick a default channel for single-channel display: the first electric
    channel (by nomenclature), else the first available channel."""
    available = [str(c) for c in available]
    if not available:
        raise ValueError("data has no channels")
    for c in available:
        if is_electric_name(c):
            return c
    return available[0]
