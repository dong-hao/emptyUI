"""Bundled minimal nodes for emptyui.

M1: CSV import, bandpass/detrend filters, time-series and spectrogram
plots. M2 will add the mth5/aurora plugins; the ts.run payload here is a
lightweight numpy container that will migrate to mth5 RunTS.
"""

from __future__ import annotations

from .filters import Bandpass, Detrend
from .io import LoadCsv
from .plots import PlotSpectrogramNode, PlotTimeSeriesNode
from .plugin_core import Echo
from .preview import PreviewPlotNode
from .save_plot import SavePlotNode
from .table_view import ViewTableNode


def register(registry) -> None:
    registry.add(Echo)
    registry.add(LoadCsv)
    registry.add(Bandpass)
    registry.add(Detrend)
    registry.add(PlotTimeSeriesNode)
    registry.add(PlotSpectrogramNode)
    registry.add(PreviewPlotNode)
    registry.add(SavePlotNode)
    registry.add(ViewTableNode)


# bundled plugin: always on, not toggleable in the Settings dialog (§5.1)
register.bundled = True
