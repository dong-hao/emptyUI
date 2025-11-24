"""Fourier-coefficient <-> xarray spectra nodes.

`spectra` payload is an mth5 FCGroup/FCDecimationGroup (xarray, time x freq).
These nodes convert between a spectra handle and the FC storage layer.
"""

from __future__ import annotations

from typing import Any, ClassVar

from emptyui import Input, Node, Output, Param


class FCGroupToSpectra(Node):
    type = "mth5.fc.read"
    title = "Read FC"
    description = "Read Fourier coefficients from an mth5 FC group into an fc handle."
    category = "Data I/O"
    inputs = (Input("mth5", "mth5"),)
    outputs = (Output("fc", "fc"),)
    params = (
        Param("survey", "string", default=""),
        Param("station", "string", default=""),
        Param("run", "string", default=""),
        Param("decimation_level", "integer", default=0, min=0),
        Param("channels", "string", label="Channels (comma-sep, empty=all)",
              default="", optional=True),
    )
    inherit: ClassVar[dict[str, dict[str, str]]] = {
        "mth5": {"survey": "survey_id", "station": "station_id"},
    }

    def execute(self, ctx, mth5, survey: str = "", station: str = "",
                run: str = "", decimation_level: int = 0,
                channels: str = "") -> dict[str, Any]:
        survey_obj = mth5.get_survey(survey) if survey else mth5.survey_group
        station_obj = survey_obj.stations_group.get_station(station)
        fc_group = station_obj.fourier_coefficients_group.get_fc_group(run)
        dec = fc_group.get_decimation_level(f"{decimation_level}")
        chan_list = [c.strip() for c in channels.split(",") if c.strip()] or None
        spectra = dec.to_xarray(channels=chan_list)
        return {"fc": spectra}


class SpectraToFCGroup(Node):
    type = "mth5.fc.write"
    title = "Write FC"
    description = "Persist an fc handle into an mth5 FC group."
    category = "Data I/O"
    inputs = (
        Input("fc", "fc"),
        Input("mth5", "mth5", required=False),
    )
    outputs = ()
    params = (
        Param("survey", "string", default=""),
        Param("station", "string", default=""),
        Param("run", "string", default=""),
        Param("sample_rate", "number", label="Decimation-level sample rate", default=1.0),
    )
    inherit: ClassVar[dict[str, dict[str, str]]] = {
        "fc": {"survey": "survey", "station": "station", "run": "run"},
    }

    def execute(self, ctx, fc, mth5=None, survey: str = "",
                station: str = "", run: str = "", sample_rate: float = 1.0) -> dict[str, Any]:
        if mth5 is None:
            raise ValueError("Spectra → FC requires an archive to write into")
        survey_obj = mth5.get_survey(survey) if survey else mth5.survey_group
        station_obj = survey_obj.stations_group.get_station(station)
        fc_group = station_obj.fourier_coefficients_group.add_fc_group(run)
        dec = fc_group.add_decimation_level("0")
        dec.from_xarray(fc, sample_rate_decimation_level=sample_rate)
        return {}


def register(registry) -> None:
    registry.add(FCGroupToSpectra)
    registry.add(SpectraToFCGroup)
