"""emptyui-aurora plugin.

aurora processing nodes (DESIGN.md §3, pin aurora==0.6.2): ComputeFC
(TS→FC, the uniform route — collections cascade per run/_session and
merge along time), EstimateTF (FC→TF), per-level TF merge. The TS→FC→TF
pipeline is split at the FC boundary; estimators (OLS / robust RME /
remote-reference RME_RR) are all aurora-backed.
"""

from __future__ import annotations

from .plots import CoherenceMap, PlotCoherency, PlotSpectrogramFC
from .process import (
    CoherenceSort,
    ComputeFC,
    EstimateTF,
    MergeTFLevels,
    ResidualSort,
    SelectRuns,
)


def register(registry) -> None:
    registry.add(ComputeFC)
    registry.add(CoherenceSort)
    registry.add(SelectRuns)
    registry.add(EstimateTF)
    registry.add(ResidualSort)
    registry.add(MergeTFLevels)
    registry.add(PlotCoherency)
    registry.add(PlotSpectrogramFC)
    registry.add(CoherenceMap)


# bundled plugin: always on, not toggleable in the Settings dialog (§5.1)
register.bundled = True
