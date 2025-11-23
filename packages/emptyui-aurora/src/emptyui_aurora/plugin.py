"""emptyui-aurora plugin.

aurora processing nodes (DESIGN.md §3, pin aurora==0.6.2): config assembly,
run selection, ComputeFC (TS→FC), EstimateTF (FC→TF), per-level TF merge.
KernelDataset/RunSummary come from mth5.processing.
"""

from __future__ import annotations

from .config import BuildConfig, FrequencyBands
from .process import ComputeFC, EstimateTF, MergeTFLevels
from .select import SelectRuns


def register(registry) -> None:
    registry.add(BuildConfig)
    registry.add(FrequencyBands)
    registry.add(SelectRuns)
    registry.add(ComputeFC)
    registry.add(EstimateTF)
    registry.add(MergeTFLevels)
