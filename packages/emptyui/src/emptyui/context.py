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
    ) -> None:
        self.workspace = workspace
        self.store = store
        self._progress_cb = progress_cb
        self.logger = logger or logging.getLogger("emptyui")
        self.published: list[dict[str, Any]] = []

    def progress(self, frac: float, message: str = "") -> None:
        frac = max(0.0, min(1.0, frac))
        if self._progress_cb:
            self._progress_cb(frac, message)

    def publish_artifact(self, path: str | Path, kind: str) -> dict[str, Any]:
        """Register a file produced by a node as an artifact."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        artifact = {"path": str(p), "kind": kind}
        self.published.append(artifact)
        return artifact
