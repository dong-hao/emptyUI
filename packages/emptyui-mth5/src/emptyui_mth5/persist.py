"""Workspace MTH5 + Fourier-coefficient persistence (DESIGN.md §3, §14).

The server workspace is itself an MTH5 archive: every intermediate result a
node chooses to persist is written into ``<workspace>/workspace.mth5``. FCs are
stored per (survey, station, run) and per decimation level, mirroring aurora's
``save_fourier_coefficients`` layout, so any mth5/aurora script can read them
back. Using a specially named file (``workspace.mth5``) is consistent and maps
directly to the long-term "saved mth5 as workspace" design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Name of the workspace archive inside ``ctx.workspace``.
WORKSPACE_MTH5_NAME = "workspace.mth5"


def workspace_mth5_path(ctx) -> Path:
    """Absolute path of the workspace archive for a run context."""
    return Path(ctx.workspace) / WORKSPACE_MTH5_NAME


def _ensure_survey(m, survey: str):
    if survey in m.surveys_group.groups_list:
        return m.get_survey(survey)
    return m.add_survey(survey)


def _ensure_station(survey_obj, station: str):
    if station in survey_obj.stations_group.groups_list:
        return survey_obj.stations_group.get_station(station)
    return survey_obj.stations_group.add_station(station)


def _ensure_fc_group(station_obj, run: str):
    fc_parent = station_obj.fourier_coefficients_group
    if run in fc_parent.groups_list:
        return fc_parent.get_fc_group(run)
    return fc_parent.add_fc_group(run)


def _ensure_dec_level(fc_group, level):
    if str(level) in fc_group.groups_list:
        return fc_group.get_decimation_level(str(level))
    return fc_group.add_decimation_level(str(level))


def persist_fc_level(
    ctx,
    fc,
    decimation_level,
    sample_rate,
    survey: str = "workspace",
    station: str = "local",
    run: str = "run",
) -> Path:
    """Persist one decimation level's Fourier coefficients to the workspace mth5.

    ``fc`` must carry a ``datetime64`` ``time`` coordinate (i.e. an absolute
    start time), which mth5's ``from_xarray`` requires.
    """
    if "datetime64" not in str(fc.time.dtype):
        raise ValueError(
            "cannot persist FC without an absolute (datetime64) time axis; the "
            "ts.run source must carry a start time (e.g. an mth5 RunTS)"
        )
    from mth5.mth5 import MTH5

    path = workspace_mth5_path(ctx)
    m = MTH5()
    m.open_mth5(str(path), mode="a", locking=False)
    try:
        survey_obj = _ensure_survey(m, survey)
        station_obj = _ensure_station(survey_obj, station)
        fc_group = _ensure_fc_group(station_obj, run)
        dec_level = _ensure_dec_level(fc_group, decimation_level)
        dec_level.from_xarray(fc, sample_rate_decimation_level=float(sample_rate))
        dec_level.update_metadata()
        fc_group.update_metadata()
        # window/decimation geometry rides mth5's own FC metadata model
        # (unknown HDF5 attrs would break mt_metadata's read_metadata for
        # any foreign consumer — see write_fc_geometry)
        from .fc import write_fc_geometry

        write_fc_geometry(dec_level, dict(fc.attrs))
    finally:
        m.close_mth5()
    return path


def load_fc_level(
    ctx,
    decimation_level,
    survey: str = "workspace",
    station: str = "local",
    run: str = "run",
    channels: list[str] | None = None,
) -> Any:
    """Read a persisted decimation level's FC back from the workspace mth5."""
    from mth5.mth5 import MTH5

    path = workspace_mth5_path(ctx)
    if not path.exists():
        raise FileNotFoundError(f"workspace archive not found: {path}")
    m = MTH5()
    m.open_mth5(str(path), mode="r", locking=False)
    try:
        dec = (
            m.get_survey(survey)
            .stations_group.get_station(station)
            .fourier_coefficients_group.get_fc_group(run)
            .get_decimation_level(str(decimation_level))
        )
        spectra = dec.to_xarray(channels=channels)
        from .fc import _enrich_fc_attrs

        _enrich_fc_attrs(
            spectra, dec, decimation_level,
            m.get_survey(survey).stations_group.get_station(station))
        return spectra
    finally:
        m.close_mth5()


def _payload_field(data, name: str) -> str:
    """Best-effort plain-field probe that never triggers mt_timeseries'
    logging ``__getattr__`` (missing attrs raise NameError — which
    ``getattr``'s default does not catch — and log an ERROR line).

    Instance ``__dict__`` first (core TimeSeries carries survey/station/run
    as plain string fields); explicit attribute access only when the
    attribute is visible via ``dir()`` (class properties — no
    ``__getattr__`` fallback), so duck payloads are probed safely.
    """
    d = getattr(data, "__dict__", None)
    if isinstance(d, dict) and name in d:
        v = d.get(name)
        return "" if v is None else str(v)
    if name in dir(data):
        try:
            v = getattr(data, name)
        except Exception:  # noqa: BLE001
            return ""
        return "" if v is None else str(v)
    return ""


def runts_context(data, survey: str = "", station: str = "", run: str = "",
                  logger=None) -> tuple[str, str, str]:
    """Derive (survey, station, run) ids for an fc's identity attrs.

    The payload's own metadata is the **truth**: when the payload carries an
    id and an explicitly provided param DISAGREES with it, the payload id
    wins with a warning naming both — a stale param (e.g. a hand-typed run
    id from a previous test) must not silently poison the fc/TF metadata
    only to explode three nodes downstream at mt_metadata's id-pattern
    validation. The params are authoritative only for payloads without
    metadata (synthetic/duck payloads); workspace defaults come last.
    """
    def _meta_id(obj):
        try:
            return obj.id
        except Exception:  # noqa: BLE001
            return ""

    fields = []
    for param_value, meta_obj, plain_name, fallback in (
            (survey, "survey_metadata", "survey", "workspace"),
            (station, "station_metadata", "station", "local"),
            (run, "run_metadata", "run", "run")):
        payload_id = _meta_id(getattr(data, meta_obj, None)) or _payload_field(
            data, plain_name)
        if param_value and payload_id and str(param_value) != str(payload_id):
            if logger is not None:
                logger.warning(
                    "'%s' param %r disagrees with the payload's own id %r — "
                    "using the payload id (params are conveniences; the "
                    "archive metadata is the truth)", plain_name,
                    param_value, payload_id)
            param_value = payload_id
        fields.append(str(param_value or payload_id or fallback))
    return fields[0], fields[1], fields[2]
