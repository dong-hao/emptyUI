"""L10 — DESIGN figures pinned to the manifests they describe.

The §9.4 session-length guidance stated a deepest-window span (~68 min)
that contradicted its own formula (window 256, 5 levels x factor 4 ->
256·4^4 s = 18.2 h at 1 Hz; 68 min is the L=2 span). The test parses the
figure straight out of docs/DESIGN.md and compares it against the manifest
defaults, so the doc cannot silently drift again.
"""

from __future__ import annotations

import re
from pathlib import Path

DESIGN = Path(__file__).resolve().parents[3] / "docs/DESIGN.md"


def test_design_deepest_window_figure_matches_manifest_defaults():
    from emptyui.registry import Registry, discover_plugins

    reg = Registry()
    discover_plugins(reg)
    node = reg.get("aurora.process.compute_fc")
    defaults = {p.name: p.default for p in node.params}
    span_s = (defaults["window_samples"]
              * defaults["factor"] ** defaults["decimation_levels"])

    text = DESIGN.read_text()
    m = re.search(
        r"deepest-level window spans ~(\d+(?:\.\d+)?) (min|h|hours?) of "
        r"raw 1 Hz data", text)
    assert m, "the §9.4 deepest-window figure vanished from docs/DESIGN.md"
    value = float(m.group(1))
    documented_s = value * 60 if m.group(2) == "min" else value * 3600
    assert abs(documented_s - span_s) / span_s < 0.10, (
        f"DESIGN says {value} {m.group(2)} ({documented_s:.0f} s) but the "
        f"manifest defaults give {span_s:.0f} s "
        f"(window {defaults['window_samples']}, "
        f"{defaults['decimation_levels']} levels x factor "
        f"{defaults['factor']})")
