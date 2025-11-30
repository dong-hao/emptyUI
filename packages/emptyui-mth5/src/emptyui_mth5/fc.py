"""Fourier-coefficient <-> xarray spectra nodes.

`spectra` payload is an mth5 FCGroup/FCDecimationGroup (xarray, time x freq).
These nodes convert between a spectra handle and the FC storage layer.
"""

from __future__ import annotations

from typing import Any, ClassVar

from emptyui import Input, Node, Output, Param

from .nomenclature import classify_channels, infer_keyword
from .runs import _resolve_run, _resolve_station, _resolve_survey


class FCGroupToSpectra(Node):
    type = "mth5.fc.read"
    title = "Read FC"
    tooltip = "Read Fourier coefficients from an mth5 FC group into an fc handle."
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
        survey_obj = _resolve_survey(mth5, survey)
        station_obj = _resolve_station(survey_obj, station)
        fc_group = station_obj.fourier_coefficients_group.get_fc_group(run)
        dec = fc_group.get_decimation_level(f"{decimation_level}")
        chan_list = [c.strip() for c in channels.split(",") if c.strip()] or None
        spectra = dec.to_xarray(channels=chan_list)
        # mth5's to_xarray drops the FC metadata; enrich the attrs so the
        # downstream EstimateTF can derive periods/channels (sample_rate is
        # required by _period_bands, channel roles via the nomenclature).
        _enrich_fc_attrs(spectra, dec, decimation_level)
        return {"fc": spectra}


def _enrich_fc_attrs(spectra, dec, decimation_level: int) -> None:
    """Attach sample rate + channel roles to a read-back fc (mth5 to_xarray
    returns a bare Dataset without attrs)."""
    summary = dec.channel_summary
    if len(summary) and "sample_rate" not in spectra.attrs:
        spectra.attrs["sample_rate"] = float(
            summary["sample_rate_decimation_level"].iloc[0]
        )
    spectra.attrs.setdefault("decimation_level", int(decimation_level))
    names = [str(c) for c in spectra.data_vars]
    kw = infer_keyword(names)
    ins, outs, _ = classify_channels(names, kw)
    if "input_channels" not in spectra.attrs:
        spectra.attrs["input_channels"] = ",".join(ins)
    if "output_channels" not in spectra.attrs:
        spectra.attrs["output_channels"] = ",".join(outs)
    if "channel_nomenclature" not in spectra.attrs:
        spectra.attrs["channel_nomenclature"] = kw


class SpectraToFCGroup(Node):
    type = "mth5.fc.write"
    title = "Write FC"
    tooltip = "Persist an fc handle into an mth5 FC group."
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
        # dual-layout resolution (modern surveys_group + legacy single-survey)
        survey_obj = _resolve_survey(mth5, survey)
        station_obj = _resolve_station(survey_obj, station)
        run = _resolve_run(station_obj, run).metadata.id if not run else run
        fc_group = station_obj.fourier_coefficients_group.add_fc_group(run)
        dec = fc_group.add_decimation_level("0")
        dec.from_xarray(fc, sample_rate_decimation_level=sample_rate)
        return {}


def register(registry) -> None:
    registry.add(FCGroupToSpectra)
    registry.add(SpectraToFCGroup)
