"""Workspace MTH5 + Fourier-coefficient persistence (DESIGN.md §3, history §14).

The server workspace is itself an MTH5 archive: every intermediate result a
node chooses to persist is written into ``<workspace>/workspace.mth5``. FCs are
stored per (survey, station, run) and per decimation level, mirroring aurora's
``save_fourier_coefficients`` layout, so any mth5/aurora script can read them
back. Using a specially named file (``workspace.mth5``) is consistent and maps
directly to the long-term "saved mth5 as workspace" design.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger("emptyui")

#: Name of the workspace archive inside ``ctx.workspace``.
WORKSPACE_MTH5_NAME = "workspace.mth5"

#: Cached open workspace handles keyed by archive path (history §24.8): opening and
#: closing the workspace PER PERSISTED LEVEL is quadratic — every
#: ``MTH5.close_mth5()`` re-summarizes ALL summary tables (the fc summary
#: recursively walks every FC level in the archive — ~96 s when persisting
#: 294 levels). One handle per node execution, closed once at the end.
_workspace_handles: dict[str, Any] = {}


def workspace_mth5_path(ctx) -> Path:
    """Absolute path of the workspace archive for a run context."""
    return Path(ctx.workspace) / WORKSPACE_MTH5_NAME


def get_workspace_mth5(ctx):
    """A cached, open workspace archive handle (append mode) for ``ctx``.

    The caller must eventually call :func:`close_workspace_mth5` (which
    flushes and closes; mth5's close re-summarizes its summary tables —
    once per execution instead of once per persisted level).
    """
    key = str(workspace_mth5_path(ctx))
    m = _workspace_handles.get(key)
    if m is not None:
        try:
            if m.h5_is_write():
                return m
        except Exception as exc:  # noqa: BLE001 — stale handle, reopen
            _log.debug(
                "workspace mth5: cached handle write-state check failed "
                "(%s) — reopening", exc)
        _workspace_handles.pop(key, None)
    from mth5.mth5 import MTH5

    from ._compat import ensure_h5_openable

    ensure_h5_openable(key)
    m = MTH5()
    m.open_mth5(key, mode="a", locking=False)
    from ._compat import register_write_open

    register_write_open(key)
    _workspace_handles[key] = m
    return m


def close_workspace_mth5(ctx) -> None:
    """Flush and close the cached workspace handle for ``ctx`` (if open)."""
    key = str(workspace_mth5_path(ctx))
    m = _workspace_handles.pop(key, None)
    if m is not None:
        try:
            m.close_mth5()
        except Exception as exc:  # noqa: BLE001 — close is best effort
            _log.debug("workspace mth5: close failed (%s)", exc)
        finally:
            from ._compat import release_write_open

            release_write_open(key)


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
    m=None,
) -> Path:
    """Persist one decimation level's Fourier coefficients to the workspace mth5.

    ``fc`` must carry a ``datetime64`` ``time`` coordinate (i.e. an absolute
    start time), which mth5's ``from_xarray`` requires. Pass a cached
    handle (``get_workspace_mth5``) to persist many levels through one
    open/close cycle — the per-call open/close is quadratic (history §24.8).
    """
    if "datetime64" not in str(fc.time.dtype):
        raise ValueError(
            "cannot persist FC without an absolute (datetime64) time axis; the "
            "ts.run source must carry a start time (e.g. an mth5 RunTS)"
        )
    if int(fc.sizes.get("time", 0)) < 2:
        # mth5's from_xarray derives the decimation's dt from t[1]-t[0] — a
        # single-window FC is degenerate (skipped with a warning, not a crash)
        import logging

        logging.getLogger("emptyui").warning(
            "FC level %s carries a single window — nothing to persist "
            "(mth5 needs >= 2 windows for the dt derivation)", decimation_level)
        return workspace_mth5_path(ctx)
    from mth5.mth5 import MTH5

    from ._compat import ensure_h5_openable

    path = workspace_mth5_path(ctx)
    ensure_h5_openable(str(path))
    own = m is None
    if own:
        from ._compat import register_write_open, release_write_open

        m = MTH5()
        m.open_mth5(str(path), mode="a", locking=False)
        register_write_open(path)
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
        if own:
            m.close_mth5()
            release_write_open(path)
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

    from ._compat import ensure_h5_openable

    path = workspace_mth5_path(ctx)
    if not path.exists():
        raise FileNotFoundError(f"workspace archive not found: {path}")
    ensure_h5_openable(str(path))
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
        if str(payload_id) == "0":
            # mt_metadata's DEFAULT id means "unset" (the plot layer
            # already excludes rid == "0") — payload truth does not
            # include unset defaults; the param (or fallback) wins
            payload_id = ""
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
