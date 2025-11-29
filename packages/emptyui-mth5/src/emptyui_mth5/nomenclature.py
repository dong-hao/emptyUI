"""Channel nomenclature helpers built on mt_metadata's canonical CHANNEL_MAPS.

mt_metadata is the single source of truth for standard <-> instrument channel
naming (keywords: default, lemi12, lemi34, phoenix123, musgraves, nims).
Following the aurora/mt_metadata design, channels are **never renamed** in the
data (a RunTS keeps ``e1``/``bx``/... names); a nomenclature keyword is carried
as metadata (fc attrs) and used only to *interpret* channel names — unit
normalization (B vs H), TF standard coordinates (ex/ey/hz x hx/hy), and
split/merge component resolution.

The standard <-> actual mapping primitives (``infer_keyword``,
``standard_name``, ``is_electric_name``, ``is_magnetic_name``) live in
``emptyui_core_nodes._channels`` (plugin-free mirror, same mt_metadata source
with a built-in fallback) and are re-exported here — core-nodes must not
depend on this plugin, so the single implementation is the core one.
"""

from __future__ import annotations

from emptyui_core_nodes._channels import (
    infer_keyword,
    is_electric_name,
    is_magnetic_name,
    standard_name,
)
from mt_metadata.transfer_functions import (
    CHANNEL_MAPS,
    STANDARD_INPUT_CHANNELS,
    STANDARD_OUTPUT_CHANNELS,
)

#: Public API — the primitives are re-exported from emptyui-core-nodes (the
#: single implementation) so plugin-tier callers import them here.
__all__ = [
    "DEFAULT_KEYWORD",
    "KEYWORDS",
    "STANDARD_INPUTS",
    "STANDARD_OUTPUTS",
    "actual_to_standard",
    "channel_map",
    "classify_channels",
    "default_channels",
    "infer_keyword",
    "is_electric_name",
    "is_magnetic_name",
    "magnetic_unit_for",
    "magnetic_unit_from_channels",
    "standard_name",
]

#: Canonical standard channel names (mt_metadata STANDARD_*_CHANNELS).
STANDARD_INPUTS: tuple[str, ...] = tuple(STANDARD_INPUT_CHANNELS)
STANDARD_OUTPUTS: tuple[str, ...] = tuple(STANDARD_OUTPUT_CHANNELS)

#: User-facing nomenclature keywords (order kept from mt_metadata CHANNEL_MAPS).
KEYWORDS: tuple[str, ...] = tuple(CHANNEL_MAPS.keys())
DEFAULT_KEYWORD: str = "default"


def channel_map(keyword: str | None) -> dict[str, str]:
    """Standard -> actual channel names for a keyword (mt_metadata map)."""
    if not keyword or keyword not in CHANNEL_MAPS:
        keyword = DEFAULT_KEYWORD
    return dict(CHANNEL_MAPS[keyword])


def actual_to_standard(keyword: str | None, name: str) -> str | None:
    """Map an actual channel name to the standard component, e.g. e1 -> ex
    (lemi12), e3 -> ex (lemi34), bx -> hx. Returns None when the name is not
    part of the given nomenclature."""
    name = str(name).strip().lower()
    for standard, actual in channel_map(keyword).items():
        if actual.lower() == name:
            return standard
    return None


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
