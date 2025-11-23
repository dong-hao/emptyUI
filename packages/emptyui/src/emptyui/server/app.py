"""FastAPI application (DESIGN.md §8)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from emptyui.executor import Executor
from emptyui.registry import Registry, build_default_registry
from emptyui.schema import ValidationError_, Workflow, validate_workflow


class RunRequest(BaseModel):
    workflow: dict


def create_app(registry: Registry | None = None) -> FastAPI:
    app = FastAPI(title="emptyui", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    registry = registry or build_default_registry()
    executor = Executor(registry)
    runs: dict[str, dict] = {}
    run_counter = {"n": 0}

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
        run_counter["n"] += 1
        run_id = f"r{run_counter['n']}"
        events: list[dict] = []
        workspace = Path(tempfile.mkdtemp(prefix=f"emptyui-{run_id}-"))
        summary = executor.run(workflow, emit=events.append, workspace=workspace)
        runs[run_id] = {"events": events, "summary": summary}
        return {"run_id": run_id, **summary}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        if run_id not in runs:
            raise HTTPException(status_code=404, detail="unknown run")
        return runs[run_id]

    return app


def main() -> None:
    import uvicorn

    uvicorn.run("emptyui.server.app:create_app", factory=True, host="127.0.0.1", port=8000)
