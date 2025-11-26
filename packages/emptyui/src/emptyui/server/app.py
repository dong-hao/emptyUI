"""FastAPI application (DESIGN.md §8).

M1: runs execute on a worker thread; events stream to WebSocket clients.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from emptyui.executor import Executor
from emptyui.registry import Registry, build_default_registry
from emptyui.schema import ValidationError_, Workflow, validate_workflow


class RunRequest(BaseModel):
    workflow: dict


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
        return "application/json", ".json"
    return "application/octet-stream", suffix or ".bin"


def create_app(registry: Registry | None = None) -> FastAPI:
    app = FastAPI(title="emptyui", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    registry = registry or build_default_registry()
    executor = Executor(registry)
    hub = EventHub()
    runs: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}  # artifact id -> {path, kind, run_id}
    state = {"run_counter": 0}
    lock = threading.Lock()

    @app.on_event("startup")
    async def _startup() -> None:
        hub.loop = asyncio.get_running_loop()

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/nodes")
    def nodes() -> dict:
        """Node catalog: manifests + port-type UI metadata."""
        return registry.manifests()

    @app.post("/api/runs")
    def start_run(req: RunRequest) -> dict:
        try:
            workflow = Workflow.model_validate(req.workflow)
            validate_workflow(workflow, registry)
        except ValidationError_ as exc:
            raise HTTPException(status_code=422, detail=exc.errors) from exc

        with lock:
            state["run_counter"] += 1
            run_id = f"r{state['run_counter']}"
        workspace = Path(tempfile.mkdtemp(prefix=f"emptyui-{run_id}-"))
        cancel_event = threading.Event()
        runs[run_id] = {
            "events": [],
            "summary": {"status": "queued"},
            "workspace": str(workspace),
            "cancel_event": cancel_event,
        }

        def emit(event: dict) -> None:
            event = {"run_id": run_id, **event}
            runs[run_id]["events"].append(event)
            if event["event"] == "artifact":
                artifacts[event["id"]] = {
                    "path": event["path"],
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
                )
                with lock:
                    runs[run_id]["summary"] = summary
            except Exception as exc:  # noqa: BLE001 — reported to clients
                with lock:
                    runs[run_id]["summary"] = {"status": "failed", "error": str(exc)}

        threading.Thread(target=worker, daemon=True, name=f"run-{run_id}").start()
        return {"run_id": run_id, "status": "queued"}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        if run_id not in runs:
            raise HTTPException(status_code=404, detail="unknown run")
        return {"run_id": run_id, **runs[run_id]["summary"],
                "events": runs[run_id]["events"]}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict:
        """Request cancellation; the executor checks the token between nodes."""
        if run_id not in runs:
            raise HTTPException(status_code=404, detail="unknown run")
        info = runs[run_id]
        info["cancel_event"].set()
        # A queued run that has not started yet would otherwise wait for the
        # worker to wake; mark it cancelled immediately (the worker's own
        # summary, also "cancelled", is written last and wins). The lock
        # linearizes against the worker's final summary write.
        with lock:
            if info["summary"]["status"] in ("queued", "running"):
                info["summary"]["status"] = "cancelled"
            status = info["summary"]["status"]
        return {"run_id": run_id, "status": status}

    @app.get("/api/files")
    def list_files(path: str = "") -> dict:
        """Server-side directory listing for the file-picker widget.

        MVP/local-mode convenience: no sandboxing yet — a must-have before
        any remote deployment (see DESIGN.md §12 auth open question).
        """
        p = Path(path).expanduser().resolve() if path.strip() else Path.cwd()
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

    @app.get("/api/artifacts/{artifact_id}/preview")
    def artifact_preview(artifact_id: str) -> FileResponse:
        info = artifacts.get(artifact_id)
        if info is None:
            raise HTTPException(status_code=404, detail="unknown artifact")
        media, _ext = _artifact_media(info["kind"], info["path"])
        return FileResponse(info["path"], media_type=media)

    @app.get("/api/artifacts/{artifact_id}")
    def artifact_download(artifact_id: str) -> FileResponse:
        """Full artifact download (Content-Disposition: attachment)."""
        info = artifacts.get(artifact_id)
        if info is None:
            raise HTTPException(status_code=404, detail="unknown artifact")
        media, ext = _artifact_media(info["kind"], info["path"])
        return FileResponse(
            info["path"],
            media_type=media,
            filename=f"{artifact_id}{ext}",
        )

    @app.websocket("/api/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive / future cancel messages
        except WebSocketDisconnect:
            hub.disconnect(ws)

    return app


def main() -> None:
    import uvicorn

    uvicorn.run("emptyui.server.app:create_app", factory=True, host="127.0.0.1", port=8000)
