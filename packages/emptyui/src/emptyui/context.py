"""Execution context handed to node ``execute`` implementations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emptyui.store import DataStore


class RunContext:
    """Per-node execution context: workspace, logging, progress, artifacts."""

    def __init__(
        self,
        workspace: Path,
        store: DataStore,
        progress_cb=None,
        logger: logging.Logger | None = None,
        node_id: str = "",
    ) -> None:
        self.workspace = Path(workspace) if workspace is not None else None
        self.store = store
        self.node_id = node_id or "node"
        self._progress_cb = progress_cb
        self.logger = logger or logging.getLogger("emptyui")
        self.published: list[dict[str, Any]] = []
        #: Watchdog flag (issues.md M2): True once the executor abandoned
        #: this node for exceeding its deadline. The abandoned thread keeps
        #: running — late progress/artifact calls must not emit events for
        #: a run that already ended. Checked, never reset.
        self.abandoned = False

    def artifact_path(self, stem: str, suffix: str) -> Path:
        """Workspace artifact filename unique per node instance.

        Display/diagnostic nodes must not share fixed filenames: two
        instances of the same node type in one run would otherwise overwrite
        each other's files (artifacts reference paths, so earlier viewers
        would silently show the later figure).
        """
        return (self.workspace / f"{stem}_{self.node_id}{suffix}"
                if self.workspace is not None
                else Path(f"{stem}_{self.node_id}{suffix}"))

    def progress(self, frac: float, message: str = "") -> None:
        if self.abandoned:
            return  # the run ended; the abandoned thread's calls are void
        frac = max(0.0, min(1.0, frac))
        if self._progress_cb:
            self._progress_cb(frac, message)

    def publish_artifact(self, path: str | Path, kind: str) -> dict[str, Any]:
        """Register a file produced by a node as an artifact."""
        if self.abandoned:
            # keep the return contract without registering (nothing reads a
            # doomed run's published list) — the file itself still exists
            return {"path": str(path), "kind": kind}
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        artifact = {"path": str(p), "kind": kind}
        self.published.append(artifact)
        return artifact
