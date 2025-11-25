"""Channel nomenclature helpers built on mt_metadata's canonical CHANNEL_MAPS.

mt_metadata is the single source of truth for standard <-> instrument channel
naming (keywords: default, lemi12, lemi34, phoenix123, musgraves, nims).
Following the aurora/mt_metadata design, channels are **never renamed** in the
data (a RunTS keeps ``e1``/``bx``/... names); a nomenclature keyword is carried
as metadata (fc attrs) and used only to *interpret* channel names — unit
normalization (B vs H), TF standard coordinates (ex/ey/hz x hx/hy), and
split/merge component resolution.
"""

from __future__ import annotations

from mt_metadata.transfer_functions import (
    CHANNEL_MAPS,
    STANDARD_INPUT_CHANNELS,
    STANDARD_OUTPUT_CHANNELS,
)

#: Canonical standard channel names (mt_metadata STANDARD_*_CHANNELS).
STANDARD_INPUTS: tuple[str, ...] = tuple(STANDARD_INPUT_CHANNELS)
STANDARD_OUTPUTS: tuple[str, ...] = tuple(STANDARD_OUTPUT_CHANNELS)

#: User-facing nomenclature keywords (order kept from mt_metadata CHANNEL_MAPS).
KEYWORDS: tuple[str, ...] = tuple(CHANNEL_MAPS.keys())
DEFAULT_KEYWORD: str = "default"

#: Tie-break priority when several keywords match equally well (e.g. a LEMI
#: 4-E-channel run where lemi12 and lemi34 both cover 5 channels): aurora's
#: LEMI-424 reader defaults to the e1/e2 pair, so lemi12 wins.
_PRIORITY: dict[str, int] = {
    "lemi12": 0,
    "lemi34": 1,
    "default": 2,
    "nims": 3,
    "musgraves": 4,
    "phoenix123": 5,
}


def channel_map(keyword: str | None) -> dict[str, str]:
    """Standard -> actual channel names for a keyword (mt_metadata map)."""
    if not keyword or keyword not in CHANNEL_MAPS:
        keyword = DEFAULT_KEYWORD
    return dict(CHANNEL_MAPS[keyword])


def standard_name(keyword: str | None, standard: str) -> str:
    """Map a standard component to its actual name, e.g. ex -> e1 (lemi12)."""
    return channel_map(keyword).get(standard, standard)


def actual_to_standard(keyword: str | None, name: str) -> str | None:
    """Map an actual channel name to the standard component, e.g. e1 -> ex
    (lemi12), e3 -> ex (lemi34), bx -> hx. Returns None when the name is not
    part of the given nomenclature."""
    name = str(name).strip().lower()
    for standard, actual in channel_map(keyword).items():
        if actual.lower() == name:
            return standard
    return None


def is_electric_name(name: str) -> bool:
    """True for electric component names (ex/ey/e1..e4)."""
    n = str(name).strip().lower()
    return n in ("ex", "ey") or (len(n) >= 2 and n[0] == "e" and n[1].isdigit())


def is_magnetic_name(name: str) -> bool:
    """True for magnetic component names (hx/hy/hz/bx/by/bz/h1/h2/h3)."""
    n = str(name).strip().lower()
    return n in ("hx", "hy", "hz", "bx", "by", "bz", "h1", "h2", "h3")


def magnetic_unit_for(keyword: str | None) -> str:
    """'b' when the nomenclature's magnetic channels are B (nT, fluxgate),
    'h' when they are H (A/m, coil sensor) — derived from CHANNEL_MAPS."""
    hx = channel_map(keyword).get("hx", "hx")
    return "b" if hx.lower().startswith("b") else "h"


def magnetic_unit_from_channels(channels) -> str:
    """Infer the magnetic unit from actual channel names (bx/by/bz -> 'b')."""
    names = {str(c).strip().lower() for c in channels}
    if any(n in ("bx", "by", "bz") for n in names):
        return "b"
    return "h"


def infer_keyword(channels) -> str:
    """Best-effort nomenclature keyword from actual channel names.

    Scores every known nomenclature by how many of the given channel names it
    recognizes; the best-scoring keyword wins (with a fixed priority tie-break,
    e.g. a LEMI 4-E-channel run resolves to lemi12, matching aurora's default
    e1/e2 pair). Returns 'default' when nothing matches.
    """
    names = {str(c).strip().lower() for c in channels}
    if not names:
        return DEFAULT_KEYWORD
    best = None
    best_score = -1
    for kw in KEYWORDS:
        recognized = {a.lower() for a in channel_map(kw).values()}
        score = sum(1 for n in names if n in recognized)
        if score > best_score or (
            score == best_score
            and best is not None
            and _PRIORITY.get(kw, 99) < _PRIORITY.get(best, 99)
        ):
            best, best_score = kw, score
    if best_score <= 0:
        return DEFAULT_KEYWORD
    return best


def default_channels(keyword: str | None) -> tuple[list[str], list[str]]:
    """(input, output) actual channel names for a keyword, mt_metadata-style:
    inputs are the horizontal magnetic pair, outputs are the electric pair plus
    the vertical magnetic channel."""
    m = channel_map(keyword)
    return [m["hx"], m["hy"]], [m["ex"], m["ey"], m["hz"]]


def classify_channels(channels, keyword: str | None = None):
    """Split actual channel names into (inputs, outputs, unused) via the
    nomenclature: horizontal magnetic -> input, electric + vertical magnetic ->
    output, anything unmapped -> unused."""
    kw = keyword or infer_keyword(channels)
    inputs: list[str] = []
    outputs: list[str] = []
    unused: list[str] = []
    for c in channels:
        std = actual_to_standard(kw, c)
        if std in ("hx", "hy"):
            inputs.append(str(c))
        elif std in ("ex", "ey", "hz"):
            outputs.append(str(c))
        else:
            unused.append(str(c))
    return inputs, outputs, unused
