"""Compatibility patches for upstream library pathologies.

pyproj ``CRS.from_user_input`` is called by mt_metadata's datum validator
(``Location.validate_datum``) on essentially every metadata attribute
update — mth5's group/metadata rebuilds trigger it thousands of times on
burst archives (25k+ CRS constructions — 75 s — for a single Build FC
pass over 121 slot runs, DESIGN §24.8 performance note). CRS objects are
immutable and datum strings repeat exhaustively, so an LRU cache makes
repeat validations free without changing behavior.
"""

from __future__ import annotations

import functools
import logging
import shutil
from pathlib import Path

_log = logging.getLogger("emptyui")

_patched = False

#: In-process registry of archives emptyUI holds open FOR WRITE (loader
#: outputs — server-lifetime handles — and the workspace persistence layer's
#: per-node handles). ``ensure_h5_openable`` consults it: a "already open"
#: failure on a tracked path is a live writer, not a stale flag. The
#: registry is process-local state: it dies with the process, so after a
#: crash the flag is genuinely stale and the recovery path clears it.
_open_write_paths: set[str] = set()
_open_paths_lock = None


def _paths_lock():
    global _open_paths_lock
    if _open_paths_lock is None:
        import threading

        _open_paths_lock = threading.Lock()
    return _open_paths_lock


def _path_key(path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def register_write_open(path) -> None:
    """Record that this process holds ``path`` open for write."""
    with _paths_lock():
        _open_write_paths.add(_path_key(path))


def release_write_open(path) -> None:
    """Record that this process no longer holds ``path`` open for write."""
    with _paths_lock():
        _open_write_paths.discard(_path_key(path))


def is_write_open_in_process(path) -> bool:
    with _paths_lock():
        return _path_key(path) in _open_write_paths


def close_archive_payload(value, log=None) -> bool:
    """Close an open MTH5 payload and release its write-open registry entry.

    Used by Save MTH5's close-in-place branch and the executor's
    failed/cancelled-run cleanup. Returns False when the value is not an
    open-archive payload (duck: has ``close_mth5``)."""
    close = getattr(value, "close_mth5", None)
    if close is None or not callable(close):
        return False
    fn = getattr(value, "filename", None)
    try:
        close()
    finally:
        if fn:
            release_write_open(fn)
    (log or _log).debug("closed MTH5 archive %s", fn)
    return True


def apply_crs_cache() -> None:
    """Idempotently LRU-cache ``pyproj.CRS.from_user_input``."""
    global _patched
    if _patched:
        return
    try:
        from pyproj import CRS
    except ImportError:  # pyproj optional outside mt_metadata installs
        return
    CRS.from_user_input = functools.lru_cache(maxsize=64)(
        CRS.from_user_input)
    _patched = True


_rate_patched = False


def apply_channel_rate_fix() -> None:
    """Idempotently replace ``ChannelTS.compute_sample_rate``'s endpoint
    method with the median of the time diffs (§24.8).

    The endpoint method (``size / (t[-1] - t[0])``) is biased by
    ``n / (n - 1)``: a 2-second 2400 Hz slot (4800 samples) derives
    2400.5 → rounds to 2401, which then fails ``RunTS.add_channel``'s
    sample-rate consistency check during Calibrate. The median of the
    diffs is unbiased for consistently-quantized axes (pandas date_range
    rounds ns inconsistently between samples — upstream's own HF path
    uses the mode for exactly this reason; the median is the robust
    generalization).
    """
    global _rate_patched
    if _rate_patched:
        return
    try:
        from mt_timeseries import ChannelTS
    except ImportError:
        return

    def _median_rate(self):
        import numpy as np

        try:
            # force ns BEFORE diffing — np.diff on a pandas DatetimeIndex
            # can produce a coarser unit (timedelta64[us]) whose int64
            # values are 1000x off
            t = np.asarray(self.data_array.coords.indexes["time"],
                           dtype="datetime64[ns]")
            dt = np.diff(t).astype("int64")
        except (TypeError, ValueError, KeyError):
            return 0.0
        if dt.size == 0:
            return 0.0
        best = int(np.median(dt))
        if best <= 0:
            return 0.0
        sr = 1e9 / best
        return np.round(sr, 0) if sr >= 1 else np.round(sr, 6)

    ChannelTS.compute_sample_rate = _median_rate
    _rate_patched = True


def ensure_h5_openable(path, log=None) -> None:
    """Detect and clear a stale HDF5 write flag before opening (§24.8).

    A process killed while a loader/Save handle had the archive open for
    write leaves the superblock's consistency flags set — every later open
    fails with ``file is already open for write (may use <h5clear file> to
    clear file consistency flags)``. When ``h5clear`` is on PATH, clear
    the flag (with a loud warning — the data itself is intact as long as
    the last write transaction completed) and return; without it, raise
    the original error plus the remedy.

    A path emptyUI itself holds open for write (loader outputs live for the
    server lifetime, the workspace persistence layer) is tracked in-process
    (:func:`register_write_open`): the failure is a live writer, not a
    stale flag — clearing under it can corrupt the archive, so this raises
    with the remedy instead. The registry dies with the process: after a
    crash the flag is genuinely stale and the next instance's recovery
    clears it (the documented interrupted-server case, unchanged).
    """
    import os
    import subprocess

    import h5py
    if not os.path.exists(path):
        return
    try:
        with h5py.File(path, "r", locking=False):
            return
    except OSError as exc:
        if "already open" not in str(exc):
            raise  # not the consistency flag — report the real error
        if is_write_open_in_process(path):
            raise OSError(
                f"{exc} — the archive is currently open for write by THIS "
                "server (a loader output or the workspace persistence "
                "layer; handles live for the server lifetime). Save MTH5 "
                "closes it, or restart the server; h5clear is refused "
                "while a live writer holds the file"
            ) from exc
        h5clear = shutil.which("h5clear")
        if h5clear is None:
            raise OSError(
                f"{exc} — the archive carries a stale HDF5 write flag "
                "(its last writer was interrupted). Install the HDF5 tools "
                f"and run `h5clear -s {path}`") from exc
        (log or _log).warning(
            "archive %s carries a stale HDF5 write flag (its last writer "
            "was interrupted) — clearing the flag via `h5clear -s`", path)
        proc = subprocess.run([h5clear, "-s", str(path)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise OSError(
                f"{exc} — `h5clear -s` failed: {proc.stderr.strip() or proc.stdout.strip()}"
            ) from exc
