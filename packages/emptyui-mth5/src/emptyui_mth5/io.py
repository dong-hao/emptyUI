"""mth5 archive I/O nodes."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from emptyui import Input, Node, Output, Param

from ._compat import ensure_h5_openable


class LoadMth5(Node):
    type = "mth5.io.load"
    title = "Load MTH5"
    tooltip = "Open an MTH5 HDF5 archive as the data source."
    description = "Open an MTH5 HDF5 archive as the data source."
    category = "Data I/O"
    inputs = ()
    outputs = (Output("mth5", "mth5"),)
    params = (
        Param("path", "file", label="MTH5 file path", path_kind="in"),
        Param("mode", "select", label="Mode",
              # L-M12: 'w' is gone — open_mth5(mode="w") TRUNCATES an
              # existing archive silently (data loss with one misclick);
              # 'r+'/'a' cover every legitimate use
              options=("r", "r+", "a"), default="r"),
    )

    def execute(self, ctx, path: str, mode: str = "r") -> dict[str, Any]:
        from mth5 import mth5

        if str(mode).strip() == "w":
            # L-M12: dropped from the select — a legacy document carrying it
            # must fail loudly (open_mth5("w") TRUNCATES the archive:
            # silent data loss), not silently truncate
            raise ValueError(
                "Load MTH5 mode 'w' is not supported — it would TRUNCATE "
                "an existing archive; use 'a' (read/write) or 'r+'")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"MTH5 not found: {p}")
        ensure_h5_openable(str(p), log=ctx.logger)
        archive = mth5.MTH5()
        # UNIFORM locking policy (DESIGN §9.3): every emptyUI open disables
        # HDF5 file locking (mth5 forwards kwargs to h5py.File). Handles live
        # for the server lifetime (cached payloads), so a default-locking
        # reader would pin the file against every other open (errno 35) — and
        # HDF5 rejects MIXED locking configurations across concurrent handles
        # in one process ("file locking flag values don't match"), which is
        # exactly the loader-creates + Load-MTH5-reads flow in one graph.
        # HDF5's consistency flag still guards write-active files.
        archive.open_mth5(str(p), mode=mode, locking=False)
        if mode != "r":
            # live-writer registry (§9.3 h5clear recovery): read handles
            # don't set the superblock write flag, write modes do
            from ._compat import register_write_open

            register_write_open(p)
        ctx.logger.info("opened %s (mode=%s)", p, mode)
        return {"mth5": archive}


class SaveMth5(Node):
    type = "mth5.io.save"
    title = "Save MTH5"
    tooltip = "Close (flush) the open MTH5 archive, or copy it to a new path (Save As)."
    description = (
        "Persist the archive: with no path (or the archive's own path) it "
        "is closed/flushed in place so any mth5 script can read it back. "
        "With a DIFFERENT path it is copied byte-level to the target (Save "
        "As) — the copy is self-contained and relocatable, while the "
        "handle stays open and bound to its original path (moving an open "
        "HDF5 is unsupported; downstream nodes may still consume it)."
    )
    category = "Data I/O"
    inputs = (Input("mth5", "mth5"),)
    outputs = ()
    #: side effect IS writing/closing — never served from the structural
    #: cache (a hit would report "done (cached)" with no file written, §7.1)
    cacheable = False
    #: the close-in-place branch closes the shared input archive — the
    #: upstream producers' cache entries must be evicted (§7.1 immutability)
    invalidates_inputs = True
    params = (
        Param("path", "file", label="Output MTH5 path", default="", optional=True, path_kind="out"),
    )

    def execute(self, ctx, mth5, path: str = "") -> dict[str, Any]:
        dest = Path(path) if path else None
        if dest is not None and not _same_file(dest, mth5.filename):
            # Save As: byte-level copy of the on-disk archive (preserves
            # attrs/compression). Loaders close their write phase before
            # returning the handle, so the on-disk state is complete; flush
            # the handle's buffers defensively anyway (a read-only handle
            # flushes as a no-op).
            try:
                mth5.flush()
            except Exception:  # noqa: BLE001 — flush best-effort
                pass
            if dest.exists():
                ctx.logger.warning("overwriting existing MTH5 at %s", dest)
                # replace via temp file: a failed copy must not destroy the
                # user's pre-existing archive (§7.1 degrade, never destroy)
                tmp = dest.with_name(dest.name + ".emptyui-tmp")
                try:
                    shutil.copy2(mth5.filename, tmp)
                    tmp.replace(dest)
                finally:
                    tmp.unlink(missing_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mth5.filename, dest)
            ctx.logger.info(
                "saved MTH5 copy -> %s (archive remains open at %s)",
                dest, mth5.filename)
            return {}
        from ._compat import close_archive_payload

        close_archive_payload(mth5)
        ctx.logger.info("saved MTH5 (closed)")
        return {}


def _same_file(dest: Path, other) -> bool:
    """True when ``dest`` and ``other`` name the same file — path-string
    comparison is not enough (``./arch.h5`` vs its canonical form used to
    take the Save-As branch and delete the live archive)."""
    dest = Path(dest)
    src = Path(other)
    try:
        if dest.exists() and src.exists():
            return dest.samefile(src)
    except OSError:
        pass
    return dest.expanduser().resolve() == src.expanduser().resolve()
