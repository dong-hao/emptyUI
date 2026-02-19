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

_log = logging.getLogger("emptyui")

_patched = False


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


def ensure_h5_openable(path, log=None) -> None:
    """Detect and clear a stale HDF5 write flag before opening (§24.8).

    A process killed while a loader/Save handle had the archive open for
    write leaves the superblock's consistency flags set — every later open
    fails with ``file is already open for write (may use <h5clear file> to
    clear file consistency flags)``. When ``h5clear`` is on PATH, clear
    the flag (with a loud warning — the data itself is intact as long as
    the last write transaction completed) and return; without it, raise
    the original error plus the remedy.

    NB: this cannot distinguish a stale flag from a LIVE in-process writer
    — but emptyUI's loaders close their write phase before returning
    handles, so a flagged archive in practice means the previous server
    was interrupted.
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
