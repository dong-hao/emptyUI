"""aurora pipeline nodes: ComputeFC (TS→FC), EstimateTF (FC→TF), MergeTFLevels.

M2 scaffold: the node manifests and orchestration are defined; the actual
aurora STFT/regression calls are wired to aurora's public API. These are the
two halves of the (former) monolithic RunAurora — estimator params live only
on EstimateTF so changing the estimator reuses cached/saved FCs.
"""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param


class ComputeFC(Node):
    type = "aurora.process.compute_fc"
    title = "Compute FC"
    description = "Decimation + STFT from time series to Fourier coefficients."
    category = "Processing"
    inputs = (
        Input("data", "ts.run", label="time series"),
        Input("bands", "bands", required=False),
    )
    outputs = (Output("fc", "fc"),)
    params = (
        Param("num_levels", "integer", default=4, min=1),
        Param("factor", "integer", default=4, min=2),
        Param("window_length", "number", label="Window length (s)", default=64.0),
        Param("overlap", "number", default=0.5),
        Param("window_type", "select", options=("hamming", "hanning"), default="hamming"),
    )

    def execute(self, ctx, data, bands=None, num_levels: int = 4, factor: int = 4,
                window_length: float = 64.0, overlap: float = 0.5,
                window_type: str = "hamming") -> dict[str, Any]:
        from emptyui_mth5.runs import ExtractRun  # noqa: F401  (type import)

        # M2: build decimation levels from the shared params, run aurora STFT,
        # and return an xarray fc (time x frequency). FC persistence into
        # the archive's FCGroup is wired here when an archive is available.
        raise NotImplementedError(
            "ComputeFC: aurora STFT wiring lands with the LEMI424 end-to-end test"
        )


class EstimateTF(Node):
    type = "aurora.process.estimate_tf"
    title = "Estimate TF"
    description = "Robust regression from Fourier coefficients to a transfer function."
    category = "Processing"
    inputs = (
        Input("fc", "fc"),
        Input("selection", "selection", required=False),
        Input("bands", "bands", required=False),
    )
    outputs = (Output("tf", "tf"),)
    params = (
        Param("estimator_engine", "select", options=("RME_RR", "RME", "OLS"),
              default="RME_RR"),
        Param("remote_reference", "boolean", default=False),
    )

    def execute(self, ctx, fc, selection=None, bands=None,
                estimator_engine: str = "RME_RR",
                remote_reference: bool = False) -> dict[str, Any]:
        # M2: run aurora regression on FCs -> mt_metadata TF.
        raise NotImplementedError(
            "EstimateTF: aurora regression wiring lands with the LEMI424 test"
        )


class MergeTFLevels(Node):
    type = "aurora.process.merge_tf_levels"
    title = "Merge TF Levels"
    description = "Merge per-decimation-level TFs into a single transfer function."
    category = "Processing"
    inputs = (Input("tf_levels", "tf.level", label="tf level"),)
    outputs = (Output("tf", "tf"),)
    params = ()

    def execute(self, ctx, tf_levels) -> dict[str, Any]:
        from aurora.transfer_function.transfer_function_collection import (
            TransferFunctionCollection,
        )

        collection = TransferFunctionCollection(tf_dict=tf_levels)
        merged = collection.get_merged_dict()
        return {"tf": merged}


def register(registry) -> None:
    registry.add(ComputeFC)
    registry.add(EstimateTF)
    registry.add(MergeTFLevels)
