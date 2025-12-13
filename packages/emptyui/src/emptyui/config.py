"""Runtime configuration (DESIGN.md §17 embedded mode).

Env-first knobs mirrored as ``create_app(...)`` keyword arguments so the
platform can embed emptyUI as a module without env coupling (kwargs win).
Only remote-mode compatibility knobs live here — run/cache internals stay
code-configurable.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    #: Read-only input confinement (§17.2): /api/files and input path params
    #: resolve only inside this directory. None = unrestricted (local dev).
    data_root: Path | None = None
    #: Directory of the built web frontend (dist/) to serve same-origin;
    #: None = don't serve (dev runs Vite separately). See §17.3.
    serve_dist: str | None = None
    #: CORS origins ("" disables the middleware; default "*" keeps the
    #: dev/local behavior — same-origin serving needs no CORS at all).
    cors_origins: tuple[str, ...] = ("*",)
    #: ASGI root path for sub-path mounts (e.g. "/tools/emptyui") — sets
    #: FastAPI root_path; usually provided by uvicorn/proxy instead.
    root_path: str | None = None
    #: Remove completed run workspaces after this many seconds. None or <= 0
    #: disables retention cleanup (local-development default).
    workspace_ttl_seconds: float | None = None
    #: Maximum queued/running runs per process. None or <= 0 means unlimited.
    max_concurrent_runs: int | None = None

    @classmethod
    def from_env(cls) -> Settings:
        data_root = os.environ.get("EMPTYUI_DATA_ROOT", "").strip()
        serve_dist = os.environ.get("EMPTYUI_SERVE_DIST", "").strip()
        raw_cors = os.environ.get("EMPTYUI_CORS_ORIGINS")
        if raw_cors is None:
            cors_origins = ("*",)
        else:
            cors = raw_cors.strip()
            cors_origins = tuple(o.strip() for o in cors.split(",") if o.strip())

        raw_ttl = os.environ.get("EMPTYUI_WORKSPACE_TTL", "").strip()
        raw_max_runs = os.environ.get("EMPTYUI_MAX_CONCURRENT_RUNS", "").strip()
        try:
            workspace_ttl_seconds = float(raw_ttl) if raw_ttl else None
        except ValueError as exc:
            raise ValueError("EMPTYUI_WORKSPACE_TTL must be a number") from exc
        if (
            workspace_ttl_seconds is not None
            and not math.isfinite(workspace_ttl_seconds)
        ):
            raise ValueError("EMPTYUI_WORKSPACE_TTL must be finite")
        try:
            max_concurrent_runs = int(raw_max_runs) if raw_max_runs else None
        except ValueError as exc:
            raise ValueError("EMPTYUI_MAX_CONCURRENT_RUNS must be an integer") from exc
        return cls(
            data_root=Path(data_root).expanduser() if data_root else None,
            serve_dist=serve_dist or None,
            cors_origins=cors_origins,
            root_path=None,
            workspace_ttl_seconds=workspace_ttl_seconds,
            max_concurrent_runs=max_concurrent_runs,
        )
