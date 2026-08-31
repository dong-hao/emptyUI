"""Path confinement for remote/embedded mode (docs/DESIGN.md §11.2).

Storage realms: a read-only data root (platform-provisioned original
data — inputs resolve only inside it), the writable per-run workspace
(artifacts/exports — output params resolve there) and the project
directory (docs/issues.md M23) — the anchor for portable relative input
paths in local mode. Data-root confinement activates only when a data
root is configured; project-dir anchoring activates when a project dir
is configured and no data root takes precedence; local dev with neither
is untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PathPolicy:
    """Validates/resolves ``path_kind`` params for one node before execute.

    Three realms: the read-only data root (remote-mode confinement), the
    writable per-run workspace (outputs) and the project directory
    (docs/issues.md M23) — the anchor that makes relative input paths portable:
    a workflow saved with project-relative paths resolves against
    ``project_dir`` on any machine, instead of silently depending on the
    server's CWD.
    """

    def __init__(
        self,
        data_root: Path | None,
        workspace: Path | None,
        project_dir: Path | None = None,
    ) -> None:
        self.data_root = Path(data_root) if data_root is not None else None
        self.workspace = Path(workspace) if workspace is not None else None
        self.project_dir = Path(project_dir) if project_dir is not None else None

    def resolve_params(self, manifest, params: dict[str, Any]) -> dict[str, Any]:
        if self.data_root is None and self.project_dir is None:
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
            if self.data_root is not None:
                # remote-mode confinement: relative inputs live in the
                # data root (and must stay inside it)
                p = self.data_root / p
                if not self._under(p, self.data_root):
                    raise ValueError(
                        f"param '{name}': path is outside the data root "
                        f"{self.data_root}: {p}"
                    )
                return p
            if self.project_dir is not None:
                # local-dev portability (M23): anchor relative inputs to
                # the project directory. No confinement — the project dir
                # is an anchor, not a realm boundary; absolute paths are
                # untouched (local dev is unrestricted, §11.2).
                return self.project_dir / p
        elif self.data_root is not None and not self._under(p, self.data_root):
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
