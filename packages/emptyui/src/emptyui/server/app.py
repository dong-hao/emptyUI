"""FastAPI application (docs/DESIGN.md §7.3).

M1: runs execute on a worker thread; events stream to WebSocket clients.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

import emptyui
from emptyui.cache import NodeCache

logger = logging.getLogger("emptyui.server")
from emptyui.config import Settings
from emptyui.executor import Executor
from emptyui.registry import Registry, build_default_registry
from emptyui.schema import ValidationError_, Workflow, validate_workflow
from emptyui.settings import LIVE_KNOBS, PLANNED_KNOBS, RuntimeSettings, SettingsStore


class RunRequest(BaseModel):
    workflow: dict
    # partial execution ("execute to here"): run only these nodes plus their
    # ancestors; omitted/empty = run everything
    targets: list[str] | None = None


class SettingsUpdate(BaseModel):
    """PUT /api/settings payload (history §22): partial update of live knobs."""
    values: dict[str, float | int | None]


class EventHub:
    """Broadcasts run events to connected WebSocket clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    def broadcast_threadsafe(self, message: dict) -> None:
        """Called from worker threads; schedules async sends on the loop."""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def _broadcast(self, message: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 — drop broken clients
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


_IMAGE_MEDIA = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
}


def _artifact_media(kind: str, path: str) -> tuple[str, str]:
    """(media_type, file extension) for an artifact, derived from kind and path."""
    suffix = Path(path).suffix.lower()
    if kind == "image" or suffix in _IMAGE_MEDIA:
        return _IMAGE_MEDIA.get(suffix, "image/png"), suffix or ".png"
    if kind == "table" or suffix == ".json":
        # L-C9: a table artifact's media type follows its file — a .csv
        # table is not JSON
        if suffix == ".csv":
            return "text/csv", suffix
        if suffix == ".tsv":
            return "text/tab-separated-values", suffix
        return "application/json", ".json"
    return "application/octet-stream", suffix or ".bin"


#: L-C7: per-run event history cap (memory hygiene when retention is off;
#: late-joining pollers replay the surviving tail)
_MAX_EVENTS_PER_RUN = 5000

_TERMINAL = frozenset({"completed", "failed", "cancelled", "timeout"})


def _merge_summary(current: dict, incoming: dict) -> dict:
    """Merge a worker/executor summary into a run summary.

    Terminal status is **monotonic**: once a summary reports completed /
    failed / cancelled, a later status (e.g. a cancel that raced a
    fast-completing run) must not flap it to a different terminal value.
    Non-status keys (elapsed_s, error, ...) always merge.
    """
    merged = dict(current)
    for k, v in incoming.items():
        if k == "status" and merged.get("status") in _TERMINAL:
            continue
        merged[k] = v
    return merged


def _history_label(workflow: dict, registry: Registry) -> str:
    """L32: session-history label — the input data identity, never node titles.

    The source nodes are the common prefix of nearly every workflow, so they
    do not disambiguate runs. The useful key is the data the run consumed:
    the basename of each ``path_kind="in"`` data path. Calibration inputs
    (``calibration_path`` / ``calibration_folder`` / Load Calibration) are
    excluded — they are not the dataset. Multiple sources are joined, capped
    at two plus ``+N``; the document name is the fallback, then "Untitled".
    """
    labels: list[str] = []
    for raw in workflow.get("nodes", []) or []:
        if not isinstance(raw, dict):
            continue
        node_type = str(raw.get("type", ""))
        node = registry.get(node_type)
        if node is None:
            continue
        # Load Calibration's `path` is a calibration input, not the dataset
        if "calibration" in node_type.lower():
            continue
        params = raw.get("params") or {}
        for spec in node.params:
            if getattr(spec, "path_kind", "") != "in":
                continue
            if "calibration" in spec.name.lower():
                continue
            value = params.get(spec.name)
            if not isinstance(value, str) or not value.strip():
                continue
            base = Path(value.rstrip("/\\")).name
            if base and base not in labels:
                labels.append(base)
    if labels:
        label = " + ".join(labels[:2])
        if len(labels) > 2:
            label += f" +{len(labels) - 2}"
        return label[:120]
    name = (workflow.get("meta") or {}).get("name")
    if isinstance(name, str) and name.strip() and name.strip() != "untitled":
        return name.strip()[:120]
    return "Untitled"


def create_app(
    registry: Registry | None = None,
    *,
    settings: Settings | None = None,
    data_root=None,
    serve_dist: str | None = None,
    cors_origins: tuple[str, ...] | None = None,
    root_path: str | None = None,
    workspace_ttl_seconds: float | None = None,
    max_concurrent_runs: int | None = None,
    node_timeout_seconds: float | None = None,
    disabled_plugins: tuple[str, ...] | None = None,
    settings_path: Path | str | None = None,
    project_dir=None,
) -> FastAPI:
    """Build the application (docs/DESIGN.md §7.3, remote-mode knobs §11/§11.2).

    Explicit keyword arguments win over ``settings``/env: embedding platforms
    pass what they need without env coupling. ``data_root`` confines input
    paths (read-only platform data), ``serve_dist`` serves the built web
    frontend same-origin, ``cors_origins`` replaces the dev default (empty
    tuple disables the middleware — same-origin serving needs no CORS).
    ``disabled_plugins`` sets the plugin baseline (§5.1: env/platform wins)
    and ``settings_path`` overrides where UI plugin toggles persist.
    ``project_dir`` (M23) is the anchor for portable relative input paths
    (and the future §12.3 disk cache home); None = the server CWD.
    """
    if settings is None:
        settings = Settings.from_env()
    data_root = data_root if data_root is not None else settings.data_root
    serve_dist = serve_dist if serve_dist is not None else settings.serve_dist
    cors_origins = cors_origins if cors_origins is not None else settings.cors_origins
    root_path = root_path if root_path is not None else settings.root_path
    workspace_ttl_seconds = (
        workspace_ttl_seconds
        if workspace_ttl_seconds is not None
        else settings.workspace_ttl_seconds
    )
    max_concurrent_runs = (
        max_concurrent_runs
        if max_concurrent_runs is not None
        else settings.max_concurrent_runs
    )
    node_timeout_seconds = (
        node_timeout_seconds
        if node_timeout_seconds is not None
        else settings.node_timeout_seconds
    )
    disabled_plugins = (
        disabled_plugins if disabled_plugins is not None else settings.disabled_plugins
    )
    settings_path = (
        settings_path if settings_path is not None else settings.settings_path
    )
    if isinstance(settings_path, str):
        settings_path = Path(settings_path)
    project_dir = (
        project_dir if project_dir is not None else settings.project_dir
    )
    if isinstance(project_dir, str):
        project_dir = Path(project_dir).expanduser() if project_dir else None

    app = FastAPI(title="emptyui", version=emptyui.__version__, root_path=root_path or "")
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # plugin baseline (§5.1): the env/platform-disabled set wins over the
    # settings file at startup; UI toggles go through the store below
    plugin_store = SettingsStore(settings_path)
    env_plugin_baseline = set(disabled_plugins)

    # runtime settings (history §22): kwarg → env → settings file → default.
    # Knobs arriving from kwarg/env are platform-baselined (not UI-editable,
    # §5.1 precedent); file-section values are the recorded UI tier.
    file_settings = plugin_store.ui_settings()
    file_live = {}
    try:
        file_live = plugin_store.validate_live_values(
            {k: v for k, v in file_settings.items() if k in LIVE_KNOBS})
    except ValueError:
        pass  # corrupt file values fall back to defaults
    _ttl_kwarg = workspace_ttl_seconds
    _runs_kwarg = max_concurrent_runs
    runtime = RuntimeSettings(
        workspace_ttl_seconds=(
            _ttl_kwarg if _ttl_kwarg is not None
            else (settings.workspace_ttl_seconds
                  if settings.workspace_ttl_seconds is not None
                  else file_live.get("workspace_ttl_seconds"))
        ),
        max_concurrent_runs=(
            _runs_kwarg if _runs_kwarg is not None
            else (settings.max_concurrent_runs
                  if settings.max_concurrent_runs is not None
                  else file_live.get("max_concurrent_runs"))
        ),
        # 0 is a valid bound (the cache clamps to 1 entry live) — do not
        # conflate it with unset via a falsy `or` (M14: a restart-time 0
        # silently became the 256 default)
        cache_max_entries=(
            256 if file_live.get("cache_max_entries") is None
            else int(file_live["cache_max_entries"])),
        # the memory-tier byte budget (L5, decision 2026-10-13): default
        # off — unbounded local mode stays the default (§7.4 policy)
        cache_max_gb=file_live.get("cache_max_gb"),
        # per-node watchdog budget (M2): kwarg → env → file → default;
        # 0 = off is a real value (same falsy-0 discipline as above) — an
        # explicit None at any stage falls through to the next stage, the
        # LIVE_KNOBS default only applies when every stage is unset
        node_timeout_seconds=(
            node_timeout_seconds
            if node_timeout_seconds is not None
            else (settings.node_timeout_seconds
                  if settings.node_timeout_seconds is not None
                  else (file_live["node_timeout_seconds"]
                        if file_live.get("node_timeout_seconds") is not None
                        else LIVE_KNOBS["node_timeout_seconds"][1]))
        ),
        env_baselined=frozenset(
            k for k, v in (("workspace_ttl_seconds", _ttl_kwarg),
                           ("max_concurrent_runs", _runs_kwarg),
                           ("node_timeout_seconds", node_timeout_seconds))
            if v is not None or getattr(settings, k) is not None
        ),
    )
    registry = registry or build_default_registry(
        disabled_plugins=plugin_store.effective_disabled(env_plugin_baseline)
    )
    executor = Executor(registry)
    hub = EventHub()
    runs: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}  # artifact id -> {path, kind, run_id}
    # GiB -> bytes; 0/None = unbounded (the estimator is approximate —
    # the budget bounds the memory, it does not audit it, L5)
    _cache_budget = runtime.cache_max_gb
    node_cache = NodeCache(  # §7, history §22
        max_entries=runtime.cache_max_entries,
        max_bytes=(int(_cache_budget * 2**30)
                   if _cache_budget is not None and _cache_budget > 0
                   else None),
    )
    state = {"run_counter": 0, "active_runs": 0}
    lock = threading.Lock()

    def _reclaim_workspace(workspace: str) -> None:
        """Invalidate cached payloads backed by ``workspace``, then remove it.

        M-C8: cached payloads holding open files inside the doomed workspace
        are invalidated FIRST (the rmtree would otherwise leave a cache hit
        restoring a payload whose backing file is gone — reads/writes on
        unlinked inodes). Handles stay open (a concurrent run may share them);
        the dropped entries force fresh re-execution on the next run.
        """
        try:
            Executor.invalidate_workspace_cache(node_cache, workspace, logger)
        except Exception:  # GC hygiene must never mask the reclaim
            logger.exception(
                "workspace cache invalidation failed for %s", workspace)
        try:
            shutil.rmtree(workspace)
        except OSError:
            # M17: ignore_errors=True leaked failed deletions silently — the
            # directory is lost either way, so surface it (best effort: later
            # workspaces are still reclaimed)
            logger.exception("workspace reclaim failed for %s", workspace)

    def cleanup_expired_runs(now: float | None = None) -> int:
        """Remove completed run state and workspaces older than the TTL."""
        ttl = runtime.workspace_ttl_seconds
        if ttl is None or ttl <= 0:
            return 0
        cutoff = (time.time() if now is None else now) - ttl
        expired: list[tuple[str, str]] = []
        with lock:
            for run_id, info in list(runs.items()):
                summary = info["summary"]
                if (
                    info.get("worker_done")
                    and summary.get("status")
                    in {"completed", "failed", "cancelled", "timeout"}
                    and info.get("finished_at") is not None
                    and info["finished_at"] <= cutoff
                ):
                    expired.append((run_id, info["workspace"]))
                    runs.pop(run_id, None)
                    for artifact_id, artifact in list(artifacts.items()):
                        if artifact.get("run_id") == run_id:
                            artifacts.pop(artifact_id, None)
        for _run_id, workspace in expired:
            _reclaim_workspace(workspace)
        return len(expired)

    async def cleanup_loop() -> None:
        # retention (history §22) is live-editable: re-read the TTL each pass; the
        # loop idles at 60 s when disabled and picks up an enable/change on
        # the next pass
        while True:
            ttl = runtime.workspace_ttl_seconds
            interval = 60.0 if ttl is None or ttl <= 0 else max(
                0.1, min(ttl / 4.0, 60.0))
            await asyncio.sleep(interval)
            # M17: the reclaim does path resolution, cache scans and
            # rmtree — a multi-GB MTH5 workspace would stall the loop that
            # drives every WS broadcast; it runs on a worker thread. The
            # pass guard keeps the reaper alive whatever one pass raises.
            try:
                await asyncio.to_thread(cleanup_expired_runs)
            except Exception:  # the reaper must survive anything
                logger.exception("workspace reclaim pass failed")

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        cleanup_task = asyncio.create_task(cleanup_loop())
        try:
            yield
        finally:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
            hub.loop = None

    app.router.lifespan_context = lifespan
    app.state.cleanup_expired_runs = cleanup_expired_runs

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/nodes")
    def nodes() -> dict:
        """Node catalog: manifests + port-type UI metadata."""
        return registry.manifests()

    @app.get("/api/plugins")
    def list_plugins() -> list[dict]:
        """Installed plugins with enable state (docs/DESIGN.md §5.1).

        Bundled plugins are always enabled; third-party plugins carry the
        user-toggleable state. The palette is catalog-driven, so hiding/
        showing a plugin's nodes is automatic on the frontend side.
        """
        with lock:
            return registry.plugins()

    def _toggle_plugin(name: str, disable: bool) -> dict:
        """Shared enable/disable handler (§5.1): hot-reload the registry in
        place (no process respawn), persist the UI toggle, clear the
        cross-run cache on disable. Serialized by the app lock."""
        with lock:
            info = registry.plugin(name)
            if info is None:
                raise HTTPException(
                    status_code=404, detail=f"unknown plugin '{name}'"
                )
            if info.bundled:
                # bundled plugins are always on — the toggle is a no-op
                return {"name": name, "enabled": True, "plugins": registry.plugins()}
            # M-C3: rebuild BEFORE persisting — the settings file must not
            # claim an enable state the live registry doesn't have;
            # rebuild swaps atomically, so a raise leaves the live table
            # untouched. The rebuild set keeps the
            # env baseline (§5.1: a UI toggle cannot enable a
            # platform-disabled plugin). Under the M20 degradation a
            # duplicate-type plugin no longer raises — the guard below
            # skips the persistence instead.
            disabled_after = plugin_store.effective_disabled(env_plugin_baseline)
            if disable or name in env_plugin_baseline:
                disabled_after.add(name)
            else:
                disabled_after.discard(name)
            try:
                registry.rebuild(disabled_after)
            except ValueError as exc:
                # a manual-node contract violation ("does not define
                # 'type'"): a clean 409 beats a raw 500 — the registry
                # stays on its old tables and the settings file is
                # untouched (M-C3)
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            effective_info = registry.plugin(name)
            if not disable and (
                effective_info is None or not effective_info.enabled
            ):
                # M20: the rebuild degraded the plugin (e.g. a duplicate
                # node type) — the conflict is in its error row. Do NOT
                # persist an enable state the live registry doesn't have
                # (M-C3): the settings file keeps the plugin disabled.
                return {
                    "name": name,
                    "enabled": False,
                    "plugins": registry.plugins(),
                }
            try:
                plugin_store.set_disabled(name, disable)
            except OSError as exc:
                # L6: the memory state already rolled back (L-C8) — the
                # client must know the file disagrees, with the reason
                raise HTTPException(
                    status_code=500,
                    detail=f"settings write failed: {exc}",
                ) from exc
            if disable:
                # disabled nodes' cached results are unreachable — drop them
                node_cache.clear()
            effective = effective_info.enabled if effective_info else False
            return {
                "name": name,
                "enabled": effective,
                "plugins": registry.plugins(),
            }

    @app.post("/api/plugins/{name}/enable")
    def enable_plugin(name: str) -> dict:
        """Hot-reload the registry with the plugin active (§5.1)."""
        return _toggle_plugin(name, disable=False)

    @app.post("/api/plugins/{name}/disable")
    def disable_plugin(name: str) -> dict:
        """Hot-reload without the plugin; clear the cross-run cache (§5.1)."""
        return _toggle_plugin(name, disable=True)

    # --- runtime settings (history §22) -------------------------------------------

    def _settings_view() -> dict:
        """Effective settings with sources: platform (env/kwarg — read-only),
        ui (UI edit, file-recorded when configured), default."""
        with lock:
            live = {}
            for key, (vtype, default, desc) in LIVE_KNOBS.items():
                value = getattr(runtime, key)
                if key in runtime.env_baselined:
                    source = "platform"
                elif key in plugin_store.ui_settings():
                    source = "ui"
                else:
                    source = "default"
                live[key] = {
                    "value": value,
                    "type": vtype.__name__,
                    "default": default,
                    "description": desc,
                    "editable": key not in runtime.env_baselined,
                    "source": source,
                }
            deployment = {
                "data_root": str(data_root) if data_root else None,
                # M23: the effective relative-path anchor — the configured
                # project dir, or the server CWD when unset (what relative
                # paths resolve against today). The frontend relativizes
                # saved input paths against it.
                "project_dir": str(project_dir or Path.cwd()),
                "serve_dist": serve_dist,
                "cors_origins": list(cors_origins),
                "root_path": root_path or None,
            }
            planned = {
                key: {"value": plugin_store.ui_settings().get(key, default),
                      "type": vtype.__name__, "default": default,
                      "description": desc}
                for key, (vtype, default, desc) in PLANNED_KNOBS.items()
            }
            return {"live": live, "deployment": deployment, "planned": planned}

    @app.get("/api/settings")
    def get_settings() -> dict:
        """Effective runtime settings (history §22): live-editable knobs, read-only
        deployment rows and the reserved (planned) §12.3 cache knobs."""
        return _settings_view()

    @app.put("/api/settings")
    def put_settings(req: SettingsUpdate) -> dict:
        """Live-edit whitelisted runtime knobs (history §22).

        Validated + coerced against LIVE_KNOBS; knobs baselined by the
        environment/platform are rejected with 409 (platform wins, §5.1
        precedent — restart-level change). Applied immediately (TTL loop,
        run admission, cache resize) and persisted to the settings file
        when one is configured (memory-only otherwise).
        """
        try:
            coerced = plugin_store.validate_live_values(req.values)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with lock:
            conflicts = sorted(set(coerced) & set(runtime.env_baselined))
            if conflicts:
                raise HTTPException(
                    status_code=409,
                    detail="set via environment/platform — restart-level "
                           f"change required: {conflicts}",
                )
            try:
                plugin_store.set_settings(coerced)
            except OSError as exc:
                # L6: the merge already rolled back (L-C8) — surface the
                # write failure with its reason, not a bare traceback
                raise HTTPException(
                    status_code=500,
                    detail=f"settings write failed: {exc}",
                ) from exc
            if coerced.get("cache_max_entries") is not None:
                # L6 (§7.4 policy): 0 = off — honored (the worker skips
                # the cache) and REPORTED as 0, not clamped to 1; 1..n
                # are real bounds. A restart restores the same value.
                runtime.cache_max_entries = round(
                    coerced["cache_max_entries"])
                node_cache.resize(runtime.cache_max_entries)
            if "workspace_ttl_seconds" in coerced:
                runtime.workspace_ttl_seconds = coerced["workspace_ttl_seconds"]
            if "max_concurrent_runs" in coerced:
                runtime.max_concurrent_runs = (
                    round(coerced["max_concurrent_runs"])
                    if coerced["max_concurrent_runs"] is not None else None)
            if "node_timeout_seconds" in coerced:
                # per-node watchdog budget (M2): applied live — the next
                # worker() call reads the holder; 0 = off
                runtime.node_timeout_seconds = coerced["node_timeout_seconds"]
            if "cache_max_gb" in coerced:
                # the memory-tier byte budget (L5): applied live — the
                # next put() evicts under the new bound; 0/None = off
                runtime.cache_max_gb = coerced["cache_max_gb"]
                node_cache.set_budget(
                    int(coerced["cache_max_gb"] * 2**30)
                    if coerced["cache_max_gb"] is not None
                    and coerced["cache_max_gb"] > 0 else None)
        return _settings_view()

    @app.post("/api/runs")
    def start_run(req: RunRequest) -> dict:
        try:
            workflow = Workflow.model_validate(req.workflow)
            validate_workflow(workflow, registry)
        except ValidationError_ as exc:
            raise HTTPException(status_code=422, detail=exc.errors) from exc
        except PydanticValidationError as exc:
            # malformed document (§6 envelope: missing/wrong-typed fields) —
            # surface pydantic's field errors as 422, not an unhandled 500
            raise HTTPException(
                status_code=422,
                detail=[str(e) for e in exc.errors()],
            ) from exc

        if req.targets:
            node_ids = {n["id"] if isinstance(n, dict) else n.id
                        for n in req.workflow.get("nodes", [])}
            unknown = sorted(set(req.targets) - node_ids)
            if unknown:
                raise HTTPException(
                    status_code=422,
                    detail=f"unknown target node(s): {unknown}",
                )

        with lock:
            limit = runtime.max_concurrent_runs or 0  # live-editable (history §22)
            if limit > 0 and state["active_runs"] >= limit:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"maximum concurrent runs reached ({limit}); "
                        "wait for an active run to finish"
                    ),
                )
            state["run_counter"] += 1
            state["active_runs"] += 1
            run_id = f"r{state['run_counter']}"
        try:
            workspace = Path(tempfile.mkdtemp(prefix=f"emptyui-{run_id}-"))
        except Exception:
            with lock:
                state["active_runs"] -= 1
            raise
        cancel_event = threading.Event()
        with lock:
            runs[run_id] = {
                "events": [],
                "summary": {"status": "queued"},
                "workspace": str(workspace),
                "cancel_event": cancel_event,
                "worker_done": False,
                "finished_at": None,
                "active_slot": True,
                # L32: session-only run history — the exact submitted document,
                # the derived data label and a compact result snapshot. The
                # table is in-memory, so a restart clears the history with it.
                "workflow": copy.deepcopy(req.workflow),
                "targets": list(req.targets or []),
                "label": _history_label(req.workflow, registry),
                "created_at": time.time(),
                "results": {
                    "nodes": {},
                    "artifacts": [],
                    "payloads": {},
                    "status": "queued",
                    "error": None,
                },
            }

        def emit(event: dict) -> None:
            event = {"run_id": run_id, **event}
            # L3: the artifact's server-side path is routing data for the
            # preview endpoint, not client payload — it never reaches the
            # event list, the replay or the WS broadcast (§7.2: `handle`
            # and `url` are the client-side routers)
            artifact_path = None
            if event["event"] == "artifact" and "path" in event:
                artifact_path = event.pop("path")
            with lock:
                info = runs.get(run_id)
                if info is None:
                    return
                info["events"].append(event)
                # L-C7: cap events per run — a long-running node emitting
                # progress with retention off would otherwise accumulate
                # unbounded (per-run memory growth). The newest tail wins;
                # late-joining pollers replay whatever remains (the
                # frontend dedupes by content, so replays stay safe).
                # L4: trim from the MIDDLE — the first event (run_started)
                # and the newest tail survive, so a late poller's replay
                # stays self-describing (index 0 is always the run's
                # run_started; the terminal run_finished rides the tail).
                if len(info["events"]) > _MAX_EVENTS_PER_RUN:
                    excess = len(info["events"]) - _MAX_EVENTS_PER_RUN
                    del info["events"][1:1 + excess]
                # L32: compact result snapshot for the history detail. The
                # event list is capped (L-C7), so node states/artifacts/
                # payloads are folded here as they arrive instead of being
                # re-derived from a possibly-trimmed replay.
                results = info.get("results")
                if results is not None:
                    kind = event.get("event")
                    if kind == "node_status":
                        node = event.get("node")
                        if node:
                            entry = results["nodes"].setdefault(node, {})
                            entry["status"] = event.get("status")
                            if event.get("error"):
                                entry["error"] = event["error"]
                    elif kind == "node_warning":
                        node = event.get("node")
                        msg = event.get("msg")
                        if node and msg:
                            entry = results["nodes"].setdefault(node, {})
                            warnings = entry.setdefault("warnings", [])
                            if msg not in warnings:
                                warnings.append(msg)
                    elif kind == "artifact":
                        results["artifacts"].append({
                            "id": event.get("id"),
                            "node": event.get("node"),
                            "kind": event.get("kind"),
                            "handle": event.get("handle"),
                            "url": event.get("url"),
                        })
                    elif kind == "payload":
                        node = event.get("node")
                        port = event.get("port")
                        if node and port:
                            results["payloads"].setdefault(node, {})[port] = (
                                event.get("summary"))
                    elif kind == "run_started":
                        results["status"] = "running"
                    elif kind == "run_finished":
                        results["status"] = event.get("status")
                        results["error"] = event.get("error")
                if event["event"] == "run_started":
                    if info["summary"].get("status") == "queued":
                        info["summary"]["status"] = "running"
                        info["started_at"] = time.time()
                elif event["event"] == "run_finished":
                    status = event.get("status")
                    if status in _TERMINAL and info["summary"].get("status") not in _TERMINAL:
                        info["summary"]["status"] = status
                elif event["event"] == "artifact":
                    artifacts[event["id"]] = {
                        "path": artifact_path,
                        "kind": event["kind"],
                        "run_id": run_id,
                    }
            hub.broadcast_threadsafe(event)

        def worker() -> None:
            try:
                summary = executor.run(
                    workflow,
                    emit=emit,
                    workspace=workspace,
                    cancel_event=cancel_event,
                    # L6: 0 = off (§7.4) — the worker skips the cache
                    # entirely; the NodeCache itself clamps to >= 1 and
                    # never learns about the off state
                    cache=node_cache if runtime.cache_max_entries else None,
                    targets=req.targets,
                    data_root=data_root,
                    project_dir=project_dir,
                    node_timeout_seconds=runtime.node_timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 — reported to clients
                # The reason is BOTH logged here and carried on run_finished
                # (the executor attaches it there, H3): the UI reads only the
                # terminal event, so a bare "failed" leaves the toolbar with
                # nothing to show while the run looks broken.
                logger.warning("run %s failed: %s", run_id, exc)
                summary = {"status": "failed", "error": str(exc)}
                with lock:
                    info = runs.get(run_id)
                    already_finished = bool(
                        info
                        and info["summary"].get("status")
                        in {"completed", "failed", "cancelled", "timeout"}
                    )
                if not already_finished:
                    emit({
                        "event": "run_finished",
                        "status": "failed",
                        "error": str(exc),
                    })
            finally:
                with lock:
                    info = runs.get(run_id)
                    if info is not None:
                        info["summary"] = _merge_summary(info["summary"], summary)
                        info["finished_at"] = time.time()
                        info["worker_done"] = True
                        if info["active_slot"]:
                            info["active_slot"] = False
                            state["active_runs"] -= 1

        threading.Thread(target=worker, daemon=True, name=f"run-{run_id}").start()
        return {"run_id": run_id, "status": "queued"}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        with lock:
            info = runs.get(run_id)
            if info is None:
                raise HTTPException(status_code=404, detail="unknown run")
            return {
                "run_id": run_id,
                **info["summary"],
                "events": list(info["events"]),
            }

    def _history_entry(run_id: str, info: dict) -> dict:
        summary = info.get("summary") or {}
        workflow = info.get("workflow") or {}
        created = info.get("created_at")
        finished = info.get("finished_at")
        artifact_count = sum(
            1 for a in artifacts.values() if a.get("run_id") == run_id)
        return {
            "run_id": run_id,
            "label": info.get("label") or "Untitled",
            "created_at": created,
            "finished_at": finished,
            "status": summary.get("status"),
            "duration_s": (
                finished - created
                if isinstance(created, (int, float))
                and isinstance(finished, (int, float)) else None),
            "node_count": len(workflow.get("nodes", []) or []),
            "artifact_count": artifact_count,
        }

    @app.get("/api/history")
    def history() -> list[dict]:
        """Session-only run history (L32): finished runs, newest first.

        The table is in-memory; a server restart clears it (ComfyUI parity).
        Active runs are excluded until their worker finishes.
        """
        with lock:
            entries = [
                _history_entry(run_id, info)
                for run_id, info in runs.items()
                if info.get("finished_at") is not None
                and (info.get("summary") or {}).get("status") in _TERMINAL
            ]
        entries.sort(key=lambda e: e.get("created_at") or 0, reverse=True)
        return entries

    @app.get("/api/history/{run_id}")
    def history_entry(run_id: str) -> dict:
        """One history entry: the submitted snapshot + compact results."""
        with lock:
            info = runs.get(run_id)
            if info is None:
                raise HTTPException(status_code=404, detail="unknown run")
            if info.get("finished_at") is None:
                raise HTTPException(
                    status_code=409, detail="run is still active")
            entry = _history_entry(run_id, info)
            entry.update({
                "workflow": copy.deepcopy(info.get("workflow") or {}),
                "targets": list(info.get("targets") or []),
                "summary": dict(info.get("summary") or {}),
                "results": copy.deepcopy(info.get("results") or {}),
            })
            return entry

    @app.delete("/api/history/{run_id}")
    def delete_history_entry(run_id: str) -> dict:
        """Delete one finished history entry, its artifacts and workspace."""
        with lock:
            info = runs.get(run_id)
            if info is None:
                raise HTTPException(status_code=404, detail="unknown run")
            if info.get("finished_at") is None:
                raise HTTPException(
                    status_code=409, detail="run is still active")
            runs.pop(run_id, None)
            for artifact_id, artifact in list(artifacts.items()):
                if artifact.get("run_id") == run_id:
                    artifacts.pop(artifact_id, None)
            workspace = info.get("workspace")
        if workspace:
            _reclaim_workspace(workspace)
        return {"deleted": True, "run_id": run_id}

    @app.delete("/api/history")
    def clear_history() -> dict:
        """Delete every finished history entry; active runs are left alone."""
        with lock:
            expired = [
                (run_id, info.get("workspace"))
                for run_id, info in runs.items()
                if info.get("finished_at") is not None
                and (info.get("summary") or {}).get("status") in _TERMINAL
            ]
            for run_id, _workspace in expired:
                runs.pop(run_id, None)
                for artifact_id, artifact in list(artifacts.items()):
                    if artifact.get("run_id") == run_id:
                        artifacts.pop(artifact_id, None)
        for _run_id, workspace in expired:
            if workspace:
                _reclaim_workspace(workspace)
        return {"deleted": len(expired)}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict:
        """Request cancellation; the executor checks the token between nodes."""
        with lock:
            info = runs.get(run_id)
            if info is None:
                raise HTTPException(status_code=404, detail="unknown run")
            info["cancel_event"].set()
            if info["summary"]["status"] in ("queued", "running"):
                info["summary"]["status"] = "cancelled"
            status = info["summary"]["status"]
        return {"run_id": run_id, "status": status}

    @app.delete("/api/cache")
    def clear_cache() -> dict:
        """Drop all cached node results (also cleared by a server restart)."""
        return {"cleared": node_cache.clear()}

    @app.get("/api/files")
    def list_files(path: str = "") -> dict:
        """Server-side directory listing for the file-picker widget.

        With a data root configured (docs/DESIGN.md §11.2, remote/embedded mode)
        the listing is confined to that root: requests outside it resolve to
        the root itself, so users can never browse beyond the
        platform-provisioned data. Without one (local dev) the whole
        filesystem is browsable.
        """
        root = Path(data_root).resolve() if data_root is not None else None
        raw = Path(path).expanduser() if path.strip() else None
        if root is not None:
            if raw is None:
                p = root
            else:
                try:
                    p = raw.resolve().relative_to(root)
                    p = root / p
                except ValueError:
                    p = root  # requested path is outside the data root → clamp
        else:
            p = raw.resolve() if raw is not None else Path.cwd()
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"path not found: {p}")
        if p.is_file():
            return {"path": str(p), "parent": str(p.parent), "entries": []}
        dirs, files = [], []
        try:
            for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
                if child.name.startswith("."):
                    continue
                entry = {"name": child.name, "path": str(child), "is_dir": child.is_dir()}
                (dirs if child.is_dir() else files).append(entry)
        except PermissionError:
            pass
        return {"path": str(p), "parent": str(p.parent), "entries": dirs + files}

    def _artifact_response(artifact_id: str, download: bool) -> FileResponse:
        info = artifacts.get(artifact_id)
        if info is None:
            raise HTTPException(status_code=404, detail="unknown artifact")
        path = Path(info["path"])
        if not path.exists():
            # L-C5: a reclaimed workspace (TTL retention) removes artifact
            # files — a FileResponse open() would 500; 404 with the remedy
            raise HTTPException(
                status_code=404,
                detail="artifact file no longer exists (its run's workspace "
                       "was reclaimed by retention) — rerun the producing "
                       "node to regenerate it")
        media, ext = _artifact_media(info["kind"], info["path"])
        headers = {
            # L-C6: served artifacts must never execute in the app origin —
            # sandbox the content and defeat MIME sniffing (an SVG artifact
            # is active content)
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox",
        }
        if download:
            # L-C9: keep the producing file's name (stem + extension) — the
            # bare artifact id erased any hint of what the file is
            return FileResponse(
                info["path"],
                media_type=media,
                filename=path.name or f"{artifact_id}{ext}",
                headers=headers,
            )
        return FileResponse(info["path"], media_type=media, headers=headers)

    @app.get("/api/artifacts/{artifact_id}/preview")
    def artifact_preview(artifact_id: str) -> FileResponse:
        return _artifact_response(artifact_id, download=False)

    @app.get("/api/artifacts/{artifact_id}")
    def artifact_download(artifact_id: str) -> FileResponse:
        """Full artifact download (Content-Disposition: attachment)."""
        return _artifact_response(artifact_id, download=True)

    @app.websocket("/api/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive / future cancel messages
        except WebSocketDisconnect:
            pass
        finally:
            # any exit path (disconnect, abrupt close raising RuntimeError,
            # server shutdown) must drop the client from the hub
            hub.disconnect(ws)

    # single-origin serving (docs/DESIGN.md §11.3): mount the built frontend
    # last, so /api/* routes win; html=True serves index.html at the root.
    # API routes stay on top — a same-origin deployment needs no CORS at all.
    if serve_dist:
        dist = Path(serve_dist).expanduser()
        if not dist.is_dir():
            raise ValueError(f"EMPTYUI_SERVE_DIST: not a directory: {dist}")
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")

    return app


def main() -> None:
    import uvicorn

    uvicorn.run("emptyui.server.app:create_app", factory=True, host="127.0.0.1", port=8000)
