"""emptyui-mth5 plugin.

Data I/O and selection nodes backed by mth5 (DESIGN.md §3, M2 pins:
mth5==0.6.8, mt_metadata==1.0.10, mt_timeseries==0.0.2). The mth5 archive is
the server-side container; RunTS/ChannelTS (re-exported by mth5.timeseries)
are the ts.run/ts.channel payloads; KernelDataset (mth5.processing) is the
mt.selection payload.
"""

from __future__ import annotations

from .fc import FCGroupToSpectra, SpectraToFCGroup
from .io import LoadMth5, SaveMth5
from .merge import MergeRuns
from .plots import PlotAppResPhase
from .runs import ExtractRun, MergeChannels, SplitChannels
from .tf import ExportTF, ImportTF


def register(registry) -> None:
    registry.add(LoadMth5)
    registry.add(SaveMth5)
    registry.add(ExtractRun)
    registry.add(SplitChannels)
    registry.add(MergeChannels)
    registry.add(MergeRuns)
    registry.add(ImportTF)
    registry.add(ExportTF)
    registry.add(FCGroupToSpectra)
    registry.add(SpectraToFCGroup)
    registry.add(PlotAppResPhase)
