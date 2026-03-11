"""Runtime configuration (DESIGN.md §17 embedded mode).

Env-first knobs mirrored as ``create_app(...)`` keyword arguments so the
platform can embed emptyUI as a module without env coupling (kwargs win).
Only remote-mode compatibility knobs live here — run/cache internals stay
code-configurable.
"""

from __future__ import annotations

import math
import os
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
    #: Plugin entry-point names disabled at startup (DESIGN.md §5.1) — the
    #: platform-level baseline; runtime UI toggles are in-memory only (no
    #: settings file is created by default) and the environment wins for
    #: the baseline. Unknown names are ignored.
    disabled_plugins: tuple[str, ...] = ()
    #: JSON settings file path (UI plugin toggles persist here, §5.1).
    #: None = memory-only for now — no file is created unless configured
    #: (config-file persistence is deferred to the configure/env file work).
    settings_path: Path | None = None

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
        if max_concurrent_runs is not None and max_concurrent_runs < 0:
            # L-C10: a negative bound would clamp to... nothing — the
            # executor treats only None as "unlimited", so a negative int
            # silently behaves as an unlimited permit pool
            raise ValueError(
                "EMPTYUI_MAX_CONCURRENT_RUNS must be >= 0 "
                "(0 or empty = unlimited)")

        raw_disabled = os.environ.get("EMPTYUI_DISABLED_PLUGINS", "").strip()
        disabled_plugins = (
            tuple(p.strip() for p in raw_disabled.split(",") if p.strip())
            if raw_disabled
            else ()
        )
        raw_settings = os.environ.get("EMPTYUI_SETTINGS_PATH", "").strip()
        settings_path = Path(raw_settings).expanduser() if raw_settings else None
        raw_base = os.environ.get("EMPTYUI_BASE_PATH", "").strip()
        if raw_base and (
            not raw_base.startswith("/")
            or raw_base.endswith("/")
            or "//" in raw_base
        ):
            # L-C10: a malformed mount breaks asset/API URL derivation
            # subtly (the frontend derives its API base from the document
            # URL) — reject it at startup instead
            raise ValueError(
                "EMPTYUI_BASE_PATH must look like '/tools/emptyui' "
                "(leading '/', no trailing slash, no double slashes)")
        return cls(
            data_root=Path(data_root).expanduser() if data_root else None,
            serve_dist=serve_dist or None,
            cors_origins=cors_origins,
            root_path=raw_base or None,
            workspace_ttl_seconds=workspace_ttl_seconds,
            max_concurrent_runs=max_concurrent_runs,
            disabled_plugins=disabled_plugins,
            settings_path=settings_path,
        )
