"""Runtime configuration (DESIGN.md §17 embedded mode).

Env-first knobs mirrored as ``create_app(...)`` keyword arguments so the
platform can embed emptyUI as a module without env coupling (kwargs win).
Only remote-mode compatibility knobs live here — run/cache internals stay
code-configurable.
"""

from __future__ import annotations

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

    @classmethod
    def from_env(cls) -> Settings:
        data_root = os.environ.get("EMPTYUI_DATA_ROOT", "").strip()
        serve_dist = os.environ.get("EMPTYUI_SERVE_DIST", "").strip()
        cors = os.environ.get("EMPTYUI_CORS_ORIGINS", "").strip()
        return cls(
            data_root=Path(data_root).expanduser() if data_root else None,
            serve_dist=serve_dist or None,
            cors_origins=tuple(o.strip() for o in cors.split(",") if o.strip())
            or ("*",),
            root_path=None,
        )
