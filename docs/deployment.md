# emptyUI deployment — embedded/remote mode

## Quick start (a lab server)

The two-command recipe; see the README's Deploying section for the
user-level view. One-time: `./scripts/install.sh --backend-only` (skip
npm on servers), build the frontend into `packages/web/dist` with Node
(`cd packages/web && npm install && npm run build`), then:

```bash
./scripts/serve.sh --host 0.0.0.0 --data-root /srv/mt/data --port 8080
```

That serves the frontend + API on one port with file access confined to
the data root (input paths and `/api/files` resolve only inside it).
Put your real TLS/reverse-proxy layer in front — see the contract below.

---

How to deploy emptyUI as an embedded module behind an existing
authentication gateway. The contract is designed in **docs/DESIGN.md §11** —
read it first; this guide is the operational counterpart.

## Deployment shape

- **One backend instance per user or team** (not a shared multi-user
  process). Process-global state — runs, artifacts, the cross-run cache —
  is then per-user by construction, and emptyUI stays identity-blind.
- **The platform gateway owns authentication** (certs, SSO, session
  cookies), TLS termination, authorization, routing and rate limiting.
  emptyUI contains no auth code and never enforces identity.
- **Same-origin serving**: emptyUI serves the built web frontend itself,
  so REST and WebSocket traffic share one origin and session cookies flow
  naturally. No CORS is needed; the dev default (`*`) is harmless but
  should be locked down or disabled in production.

## Configuration reference

| Env var | Purpose | Default |
|---|---|---|
| `EMPTYUI_DATA_ROOT` | Read-only root for platform-provisioned data. Input path params (CSV/MTH5/calibration/LEMI data) resolve only inside it; `/api/files` is clamped to it; output params resolve into the per-run workspace (absolute output paths are rejected in remote mode). **Unset = unrestricted (local dev).** | unset |
| `EMPTYUI_SERVE_DIST` | Directory of the built frontend (e.g. `/opt/emptyui/web/dist`). When set, FastAPI serves it at `/` (single origin). | unset (Vite dev serves itself) |
| `EMPTYUI_CORS_ORIGINS` | Comma-separated allowed origins. Empty value **disables** the CORS middleware (same-origin serving needs none). | `*` |
| `EMPTYUI_BASE_PATH` | ASGI sub-path for module embedding (e.g. `/tools/emptyui`) — sets FastAPI `root_path`. Usually provided by the proxy/uvicorn `--root-path` instead. | unset |
| `EMPTYUI_WORKSPACE_TTL` | Retain completed run workspaces and artifacts for this many seconds before cleanup. `0` or unset disables cleanup. | disabled |
| `EMPTYUI_MAX_CONCURRENT_RUNS` | Maximum queued/running runs per backend process. `0` or unset is unlimited. Excess submissions receive HTTP 429. | unlimited |
| `EMPTYUI_NODE_TIMEOUT` | Per-node watchdog budget in seconds; a node exceeding it is abandoned and the run ends with a timeout. `0` disables. | `600` |
| `EMPTYUI_PROJECT_DIR` | Stable project directory anchoring portable relative input paths (and the future disk-cache home); unset = server CWD. | unset (CWD) |
| `EMPTYUI_SETTINGS_PATH` | JSON settings file for runtime settings (plugin enable/disable §5.1 + live knobs §7.4: `workspace_ttl_seconds`, `max_concurrent_runs`, `cache_max_entries`, `cache_max_gb`, `node_timeout_seconds`). **Unset = in-memory only** — no settings file is created. | unset (memory-only) |
| `EMPTYUI_DISABLED_PLUGINS` | Comma-separated plugin entry-point names disabled at startup (the platform-level baseline; unknown names are silently ignored). | none |

All knobs are mirrored as `create_app(...)` keyword arguments
(`data_root=`, `serve_dist=`, `cors_origins=`, `root_path=`,
`workspace_ttl_seconds=`, `max_concurrent_runs=`, `node_timeout_seconds=`,
`disabled_plugins=`, `settings_path=`, `project_dir=`) so a platform can
embed the app as a module without env coupling; explicit non-`None` kwargs
win over env.

> Planned, not implemented: `EMPTYUI_MODELS_PATH` (model weights, DESIGN
> §11.4) and `EMPTYUI_CACHE_MAX_GB`/`EMPTYUI_CACHE_TTL_DAYS` (disk cache,
> DESIGN §12.3). The live knobs `cache_max_entries`/`cache_max_gb` are
> settings-file/UI values, not env vars. There is no `EMPTYUI_ROOT_PATH`;
> the mount env var is `EMPTYUI_BASE_PATH`.

## Building the frontend

```bash
cd packages/web && npm install && npm run build   # -> packages/web/dist
```

Point `EMPTYUI_SERVE_DIST` at that directory (or copy it to the deployment
location).

## Sub-path mounting

The app is designed to be mounted under a sub-path, e.g.
`https://platform.example/tools/emptyui/`. The frontend derives its API
base at runtime from the document location (one segment below the app
root), so **no rebuild or extra config is needed** — it works at `/`,
under a mount, and in dev. The backend's `root_path` is set by the proxy
or uvicorn (`--root-path`); `create_app(root_path=...)` exists for module
embedding. Configure the reverse proxy to forward the prefix and the
WebSocket upgrade:

```nginx
location /tools/emptyui/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
    proxy_set_header X-Forwarded-Prefix /tools/emptyui;
    # X-Forwarded-* are forwarded for future URL generation; the backend
    # does not consume them yet (DESIGN §11.3, planned)
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;      # WebSocket
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;                     # runs are long-lived
}
```

> The `proxy_pass .../` (trailing slash) is what strips the prefix before
> forwarding. If the platform instead forwards the full path, set the
> backend `root_path` accordingly.

## Data provisioning

The platform provisions original data (time-series CSVs, MTH5 archives,
calibration files) into `EMPTYUI_DATA_ROOT` — read-only. Users never modify
it: input path params are validated to stay inside the root (a clear error
otherwise) and writable outputs (exports, plots, saved archives) land in
the per-run workspace (relative paths only; absolute output paths are
rejected). The file picker can never browse beyond the root.

## Running

```bash
EMPTYUI_DATA_ROOT=/srv/emptyui/data \
EMPTYUI_SERVE_DIST=/opt/emptyui/web/dist \
EMPTYUI_CORS_ORIGINS= \
uvicorn emptyui.server.app:create_app --factory --host 127.0.0.1 --port 8000
```

or, as a module (the platform's own process):

```python
from emptyui.server.app import create_app
app = create_app(data_root="/srv/emptyui/data", serve_dist="/opt/emptyui/web/dist",
                 cors_origins=(), root_path="/tools/emptyui",
                 node_timeout_seconds=600, project_dir="/srv/emptyui/project")
```

## Operational notes

- **WebSocket**: browsers cannot set `Authorization` headers on WS — the
  gateway authenticates the upgrade (cookie/cert) at the proxy; same-origin
  serving makes this seamless. The UI's polling fallback keeps working when
  a proxy drops the socket.
- **Long runs**: keep proxy read/idle timeouts generous; the UI survives
  disconnects (run status is re-fetched).
- **Plugin availability**: the platform controls it via
  `EMPTYUI_DISABLED_PLUGINS`; runtime toggling (§5.1) is available in the
  web UI's Settings dialog (in-memory by default; optional JSON persistence
  via `EMPTYUI_SETTINGS_PATH`).
- **Retention & limits**: set `EMPTYUI_WORKSPACE_TTL` to garbage-collect
  terminal run records, artifact indexes, and per-run workspaces; active runs
  are never removed. Set `EMPTYUI_MAX_CONCURRENT_RUNS` to apply admission
  control and receive HTTP 429 when the process is at capacity. Both controls
  are disabled by default for local development.

## Gateway contract (what the platform must guarantee)

1. Authenticate every request **and** every WebSocket upgrade.
2. Forward the sub-path prefix (or set `root_path`).
3. Provision read-only data under `EMPTYUI_DATA_ROOT` (system of record
   stays with the platform/database).
4. Keep one process per user/team; scale out = more instances.
5. Forwarded headers (`X-Forwarded-Proto/Host/Prefix`) are accepted for
   future URL generation; the handling is planned (DESIGN §11.3) — current
   deployments rely on `root_path`/prefix stripping.
