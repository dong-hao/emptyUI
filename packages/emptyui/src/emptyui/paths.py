"""Path confinement for remote/embedded mode (DESIGN.md §11.2).

Two storage realms: a read-only data root (platform-provisioned original
data — inputs resolve only inside it) and the writable per-run workspace
(artifacts/exports — output params resolve there). Confinement activates
only when a data root is configured; local dev is untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PathPolicy:
    """Validates/resolves ``path_kind`` params for one node before execute."""

    def __init__(self, data_root: Path | None, workspace: Path | None) -> None:
        self.data_root = Path(data_root) if data_root is not None else None
        self.workspace = Path(workspace) if workspace is not None else None

    def resolve_params(self, manifest, params: dict[str, Any]) -> dict[str, Any]:
        if self.data_root is None:
            return params
        resolved = dict(params)
        for param in manifest.params:
            kind = getattr(param, "path_kind", "")
            name = param.name
            if not kind or name not in resolved:
                continue
            value = resolved[name]
            if kind == "in":
                if value in (None, ""):
                    continue  # optional inputs left to the node's own handling
                resolved[name] = str(self._resolve_input(name, value))
            elif kind == "out":
                resolved[name] = str(self._resolve_output(name, value))
        return resolved

    def _resolve_input(self, name: str, value: str) -> Path:
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = self.data_root / p
        if not self._under(p, self.data_root):
            raise ValueError(
                f"param '{name}': path is outside the data root "
                f"{self.data_root}: {p}"
            )
        return p

    def _resolve_output(self, name: str, value: str) -> Path:
        if value in (None, ""):
            # no explicit output dir → the run workspace itself (history §17.2:
            # writable outputs live there; local dev keeps its defaults)
            return self.workspace if self.workspace is not None else Path("")
        p = Path(value).expanduser()
        if p.is_absolute():
            # remote-mode confinement: writable outputs stay in the run
            # workspace; absolute paths are rejected.
            raise ValueError(
                f"param '{name}': absolute output paths are not allowed in "
                "remote mode; use a relative path under the run workspace"
            )
        if self.workspace is None:
            return p
        joined = self.workspace / p
        if not self._under(joined, self.workspace):
            raise ValueError(
                f"param '{name}': output path escapes the run workspace "
                f"{self.workspace}: {p}"
            )
        return joined

    @staticmethod
    def _under(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False
