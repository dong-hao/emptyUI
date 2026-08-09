"""Minimal shared plot style for core Display nodes (DESIGN.md §10.6).

Core must not depend on the mth5 plugin, so this mirrors the tiny subset of
the TF style module (``emptyui-mth5/plotting.py``) that applies to
time-series/spectrogram figures: DPI, grid/font conventions, and unit hints
for channel labels. The mth5-side module carries the full MT conventions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

DPI = 150


def apply_style(fig, axes=()):
    """Apply the shared publication style to a figure and its axes.

    Grids follow the EMTF reference figure: solid major gridlines with
    dotted minor gridlines (the USArray MT figures show the fine minor
    grid on the log axes). M5: the ONE copy — mth5's plotting used to
    carry its own, which had already drifted from this one."""
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.tick_params(labelsize=9, which="both")
        ax.grid(True, which="major", alpha=0.30, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.20, linewidth=0.5,
                linestyle=":")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)


def channel_unit_hint(name: str) -> str:
    """Unit hint for raw-instrument channels (LEMI/BGS convention).

    Electric channels arrive in mV/km and fluxgate magnetic channels in nT;
    only nomenclature-recognized names get a hint so unknown channels stay
    unlabeled rather than mislabeled.
    """
    from ._channels import is_electric_name, is_magnetic_name

    if is_electric_name(name):
        return "mV/km"
    if is_magnetic_name(name):
        return "nT"
    return ""
