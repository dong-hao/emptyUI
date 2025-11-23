"""Aurora processing-config assembly nodes.

Decimation levels are independent by default in mt_metadata (fresh
Window/Estimator per level) — BuildConfig explicitly copies the shared window,
estimator and regression settings to every level (DESIGN.md M2 pins).
"""

from __future__ import annotations

from typing import Any

from emptyui import Input, Node, Output, Param


class FrequencyBands(Node):
    type = "aurora.config.bands"
    title = "Frequency Bands"
    description = "Define frequency/period band edges for a processing config."
    category = "Processing"
    inputs = ()
    outputs = (Output("bands", "bands"),)
    params = (
        Param("period_min", "number", label="Longest period (s)", default=10000.0),
        Param("period_max", "number", label="Shortest period (s)", default=0.01),
        Param("bands_per_decade", "integer", label="Bands per decade", default=8),
    )

    def execute(self, ctx, period_min: float = 10000.0, period_max: float = 0.01,
                bands_per_decade: int = 8) -> dict[str, Any]:
        from aurora.time_series.frequency_band import FrequencyBands

        bands = FrequencyBands(
            num_bands_per_decade=bands_per_decade,
            period_min=period_min,
            period_max=period_max,
        )
        return {"bands": bands}


class BuildConfig(Node):
    type = "aurora.config.build"
    title = "Build Config"
    description = (
        "Build an aurora Processing config: N decimation levels sharing the "
        "same window/estimator, copied to each level."
    )
    category = "Processing"
    inputs = (Input("bands", "bands", required=False),)
    outputs = (Output("config", "config"),)
    params = (
        Param("num_levels", "integer", label="Decimation levels", default=4, min=1),
        Param("factor", "integer", label="Decimation factor", default=4, min=2),
        Param("input_channels", "string", label="Input channels", default="hx,hy"),
        Param("output_channels", "string", label="Output channels", default="ex,ey,hz"),
        Param("estimator_engine", "select", label="Estimator",
              options=("RME_RR", "RME", "OLS"), default="RME_RR"),
        Param("window_type", "select", label="Window", options=("hamming", "hanning"),
              default="hamming"),
        Param("window_length", "number", label="Window length (s)", default=64.0),
        Param("overlap", "number", label="Overlap", default=0.5),
    )

    def execute(self, ctx, bands=None, num_levels: int = 4, factor: int = 4,
                input_channels: str = "hx,hy", output_channels: str = "ex,ey,hz",
                estimator_engine: str = "RME_RR", window_type: str = "hamming",
                window_length: float = 64.0, overlap: float = 0.5) -> dict[str, Any]:
        from mt_metadata.processing.aurora import (
            STFT,
            Decimation,
            DecimationLevel,
            Estimator,
            Processing,
            Regression,
            Window,
        )

        input_ch = [c.strip() for c in input_channels.split(",") if c.strip()]
        output_ch = [c.strip() for c in output_channels.split(",") if c.strip()]

        config = Processing()
        window = Window(
            type=window_type,
            num_samples=int(window_length),  # num_samples in samples? see note
            overlap=overlap,
        )
        stft = STFT(window=window)
        estimator = Estimator(engine=estimator_engine)
        regression = Regression()

        for level in range(num_levels):
            dec = Decimation(factor=factor, level=level)
            dec_level = DecimationLevel(
                decimation=dec,
                stft=stft,
                estimator=estimator,
                regression=regression,
                input_channels=input_ch,
                output_channels=output_ch,
            )
            config.add_decimation_level(dec_level)

        if bands is not None:
            # attach band edges (aurora FrequencyBands -> per-level Band list)
            for dec_level, band in zip(config.decimations, bands.get_band_list()):
                dec_level.bands.append(band)

        return {"config": config}


def register(registry) -> None:
    registry.add(FrequencyBands)
    registry.add(BuildConfig)
