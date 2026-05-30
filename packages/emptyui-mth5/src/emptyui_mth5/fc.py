"""Fourier-coefficient <-> xarray spectra nodes.

`spectra` payload is an mth5 FCGroup/FCDecimationGroup (xarray, time x freq).
These nodes convert between a spectra handle and the FC storage layer.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from emptyui import Input, Node, Output, Param

from .nomenclature import classify_channels, infer_keyword
from .runs import _resolve_run, _resolve_station, _resolve_survey

_log = logging.getLogger("emptyui")

#: Window-family mapping into mth5's WindowTypeEnum vocabulary.
_MTH5_WINDOW_TYPES = {"hanning": "hann"}

#: emptyUI's human-readable per-window-detrend values -> mth5's
#: ShortTimeFourierTransform.per_window_detrend_type enum (['linear',
#: 'constant', '']): "off" is emptyUI's spelling of "no per-window
#: detrend". (The former model round trip VALIDATED this value and — on a
#: mismatch — silently dropped the whole geometry record; direct writes
#: must map it instead.)
_MTH5_DETREND_TYPES = {"off": "", "none": ""}

#: same mapping for prewhitening (mth5's enum: ['first difference', 'other']
#: — an empty string is NOT valid, so "off" (emptyUI's "no prewhitening")
#: must leave the attr untouched, like the former model round trip did)
_MTH5_PREWHITENING_TYPES = {"off": None, "none": None}

#: The decimation-group attribute keys that carry the fc geometry (mth5's
#: own metadata model). Written directly: mth5's ``write_metadata``
#: serializes the FULL model and chokes on nested FCChannel objects once
#: ``from_xarray`` populated them.
_GEOMETRY_KEYS = (
    "time_series_decimation.level",
    "time_series_decimation.factor",
    "time_series_decimation.sample_rate",
    "short_time_fourier_transform.window.num_samples",
    "short_time_fourier_transform.window.overlap",
    "short_time_fourier_transform.window.type",
    "short_time_fourier_transform.prewhitening_type",
    "short_time_fourier_transform.per_window_detrend_type",
)


def write_fc_geometry(dec, attrs: dict[str, Any]) -> None:
    """Record an fc's window/decimation geometry in mth5's own FC metadata
    model — it round-trips the archive and benefits any mth5/aurora
    consumer (aurora reads the same STFT metadata).

    Unknown HDF5 attributes are deliberately NOT used: mt_metadata's
    ``read_metadata`` parses EVERY group attribute into its pydantic model
    and unknown keys raise there — a foreign consumer reading the archive
    would crash. What the FC model cannot express (electric scaling,
    calibration provenance, saturated-window counts) stays in-memory
    provenance on the fc payload only.

    The geometry attrs are computed directly from the payload (same
    serialization mth5's model produces: overlap in samples, the hann
    type mapping, ``none`` for missing) — going through the model
    (``dec.metadata``) costs a pydantic rebuild per level (§24.8).
    """
    n = int(attrs.get("num_samples_window") or 0)
    ov = attrs.get("overlap")
    wt = str(attrs.get("window_type") or "")
    values: dict[str, Any] = {
        "time_series_decimation.level": (
            int(attrs["decimation_level"])
            if attrs.get("decimation_level") is not None else None),
        "time_series_decimation.factor": (
            float(attrs["factor"]) if attrs.get("factor") else None),
        "time_series_decimation.sample_rate": (
            float(attrs["sample_rate"]) if attrs.get("sample_rate") else None),
        "short_time_fourier_transform.window.num_samples": n or None,
        "short_time_fourier_transform.window.overlap": (
            round(float(ov) * n) if (ov is not None and n) else None),
        "short_time_fourier_transform.window.type": (
            _MTH5_WINDOW_TYPES.get(wt, wt) if wt else None),
        "short_time_fourier_transform.prewhitening_type": (
            _MTH5_PREWHITENING_TYPES.get(str(attrs["prewhitening"]).lower(),
                                         str(attrs["prewhitening"]))
            if attrs.get("prewhitening") else None),
        "short_time_fourier_transform.per_window_detrend_type": (
            _MTH5_DETREND_TYPES.get(str(attrs["per_window_detrend"]).lower(),
                                    str(attrs["per_window_detrend"]))
            if attrs.get("per_window_detrend") is not None else None),
    }
    group = dec.hdf5_group
    for key, value in values.items():
        if value is None:
            continue
        try:
            if key in group.attrs:
                group.attrs.modify(key, value)
            else:
                group.attrs.create(key, value)
        except Exception as exc:  # noqa: BLE001
            _log.debug("fc geometry attr %s not written: %s", key, exc)


def _fc_identity_attrs(dec, station_obj) -> dict[str, Any]:
    """Station identity attrs resolved from the archive hierarchy (the
    mth5-idiomatic source — the FC group sits under the station, whose
    metadata carries id/location).

    Read from the HDF5 group paths/attrs (free) — the station
    ``.metadata`` property rebuilds the FULL hierarchy through pydantic
    (~30-40 s per access on a burst archive, §24.8).
    """
    attrs: dict[str, Any] = {}
    try:
        from .runs import _group_id

        attrs["station_id"] = _group_id(station_obj)
        # survey id = the survey group's own attrs (cheap; works for both
        # the modern nested layout and legacy single-survey files)
        try:
            survey_id = str(
                station_obj.hdf5_group.parent.parent.attrs.get("id") or "")
        except Exception:  # noqa: BLE001 — identity is best effort
            survey_id = ""
        if survey_id:
            attrs["survey_id"] = survey_id
        a = station_obj.hdf5_group.attrs

        def _loc(key):
            try:
                v = a[key]
                return float(v) if v is not None else None
            except (KeyError, TypeError, ValueError):
                return None

        lat, lon, elev = (_loc("location.latitude"),
                          _loc("location.longitude"),
                          _loc("location.elevation"))
        if lat not in (None, 0.0):
            attrs["station_latitude"] = lat
        if lon not in (None, 0.0):
            attrs["station_longitude"] = lon
        if elev not in (None, 0.0):
            attrs["station_elevation"] = elev
    except Exception as exc:  # noqa: BLE001 — identity is best effort
        _log.debug("fc identity partially resolved: %s", exc)
    try:
        attrs["run_id"] = dec.hdf5_group.parent.name.split("/")[-1]
    except Exception as exc:  # noqa: BLE001
        _log.debug("fc run id not resolved: %s", exc)
    return attrs


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
        # empty ids fall back to the first available (same contract as the
        # write node; get_fc_group("") would raise)
        run = _resolve_run(station_obj, run).metadata.id if not run else run
        fc_group = station_obj.fourier_coefficients_group.get_fc_group(run)
        dec = fc_group.get_decimation_level(f"{decimation_level}")
        chan_list = [c.strip() for c in channels.split(",") if c.strip()] or None
        spectra = dec.to_xarray(channels=chan_list)
        # mth5's to_xarray drops the FC metadata; enrich the attrs so the
        # downstream EstimateTF can derive periods/channels (sample_rate is
        # required by _period_bands, channel roles via the nomenclature).
        _enrich_fc_attrs(spectra, dec, decimation_level, station_obj)
        return {"fc": spectra}


def _enrich_fc_attrs(spectra, dec, decimation_level: int,
                     station_obj=None) -> None:
    """Attach geometry, identity and channel roles to a read-back fc (mth5
    to_xarray returns a bare Dataset without attrs).

    Sources, in priority order: (1) the fc's own attrs (a payload that
    already carries them, e.g. merged containers), (2) mth5's native FC
    metadata written by :func:`write_fc_geometry`, (3) the archive
    hierarchy (station identity), (4) derivations from the channel names
    (nomenclature, magnetic unit). Older archives keep working via (3)+(4).
    """
    try:
        md = dec.metadata
        tsd = md.time_series_decimation
        stft = md.short_time_fourier_transform
        w = stft.window

        def _clean(value, enum_cls=None) -> str:
            # mt_metadata enum round trip: "<TypeEnum>.hamming" -> the
            # enum's VALUE ("hamming", "first difference", ...)
            text = str(value)
            if "TypeEnum." not in text:
                return text
            member = text.split(".", 1)[1]
            if enum_cls is not None:
                try:
                    return str(enum_cls[member].value)
                except KeyError:
                    return member
            return member

        if tsd.sample_rate:
            spectra.attrs.setdefault("sample_rate", float(tsd.sample_rate))
        if tsd.factor and float(tsd.factor) != 1.0:
            spectra.attrs.setdefault("factor", float(tsd.factor))
        if w.num_samples:
            spectra.attrs.setdefault("num_samples_window", int(w.num_samples))
        if w.type is not None:
            spectra.attrs.setdefault(
                "window_type", _clean(w.type, type(w.type)))
        if w.overlap and w.num_samples:
            spectra.attrs.setdefault(
                "overlap", float(w.overlap) / float(w.num_samples))
        if stft.prewhitening_type:
            spectra.attrs.setdefault(
                "prewhitening",
                _clean(stft.prewhitening_type, type(stft.prewhitening_type)))
        if stft.per_window_detrend_type is not None:
            spectra.attrs.setdefault(
                "per_window_detrend",
                _clean(stft.per_window_detrend_type,
                       type(stft.per_window_detrend_type)))
    except Exception as exc:  # noqa: BLE001 — enrichment is best effort
        _log.debug("fc geometry enrichment skipped: %s", exc)
    if station_obj is not None:
        for k, v in _fc_identity_attrs(dec, station_obj).items():
            spectra.attrs.setdefault(k, v)
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
    if "magnetic_unit" not in spectra.attrs:
        from .nomenclature import magnetic_unit_from_channels

        spectra.attrs["magnetic_unit"] = magnetic_unit_from_channels(names)


class SpectraToFCGroup(Node):
    type = "mth5.fc.write"
    # raw-data/FC-scale work: a generous own budget (issues.md M2)
    timeout = 3600.0
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
        Param("decimation_level", "integer", label="Decimation level", default=0,
              min=0,
              description="Level id written into the archive (default 0). "
                          "Set it when persisting a decimated fc so the "
                          "archive's decimation metadata is correct."),
        Param("sample_rate", "number", label="Decimation-level sample rate",
              default=None, optional=True,
              description="Sample rate of THIS decimation level (Hz). Empty "
                          "= derive from the fc's metadata (set by Compute "
                          "FC); falls back to 1.0 with a warning when the "
                          "fc carries no rate."),
    )
    inherit: ClassVar[dict[str, dict[str, str]]] = {
        "fc": {"survey": "survey", "station": "station", "run": "run"},
    }

    def execute(self, ctx, fc, mth5=None, survey: str = "",
                station: str = "", run: str = "",
                decimation_level: int = 0, sample_rate: float | None = None
                ) -> dict[str, Any]:
        if mth5 is None:
            raise ValueError("Spectra → FC requires an archive to write into")
        if sample_rate is None:
            # derive from the fc's metadata (Compute FC records it; the read
            # node enriches read-back fcs with it) — never silently label a
            # wrong rate
            sample_rate = fc.attrs.get("sample_rate")
            if sample_rate is None:
                _log.warning(
                    "fc carries no sample_rate metadata and none was set — "
                    "writing decimation-level rate 1.0 (set 'Decimation-"
                    "level sample rate' explicitly for correct archive "
                    "metadata)")
                sample_rate = 1.0
        # dual-layout resolution (modern surveys_group + legacy single-survey)
        survey_obj = _resolve_survey(mth5, survey)
        station_obj = _resolve_station(survey_obj, station)
        run = _resolve_run(station_obj, run).metadata.id if not run else run
        fc_group = station_obj.fourier_coefficients_group.add_fc_group(run)
        dec = fc_group.add_decimation_level(f"{int(decimation_level)}")
        dec.from_xarray(fc, sample_rate_decimation_level=float(sample_rate))
        write_fc_geometry(dec, dict(fc.attrs))
        return {}


