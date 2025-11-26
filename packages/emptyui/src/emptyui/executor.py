"""DAG executor (DESIGN.md §7).

M0: sequential topological execution over validated workflows. Cancellation
is supported via a threading.Event polled between nodes; concurrency and
disk caching remain M3. The handle-keyed structure below is already
cache-ready.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from emptyui.context import RunContext
from emptyui.schema import Workflow
from emptyui.store import DataStore


class ExecutionError(Exception):
    def __init__(self, node_id: str, original: BaseException) -> None:
        self.node_id = node_id
        self.original = original
        super().__init__(f"node '{node_id}' failed: {original}")


class Executor:
    """Executes a workflow against a registry. Emits events via callback."""

    def __init__(self, registry) -> None:
        self.registry = registry
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _topo_order(self, workflow: Workflow) -> list[str]:
        indegree = {n.id: 0 for n in workflow.nodes}
        adj: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        for e in workflow.edges:
            adj[e.source].append(e.target)
            indegree[e.target] += 1
        queue = sorted(nid for nid, d in indegree.items() if d == 0)
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for nxt in sorted(adj[nid]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return order

    def run(
        self,
        workflow: Workflow,
        emit=print,
        workspace=None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Execute the workflow; returns a run summary.

        `cancel_event` is polled between nodes: when set, the run stops
        early and returns ``{"status": "cancelled", ...}``. When omitted,
        the legacy ``cancel()`` token is used (and cleared on entry).
        """
        if cancel_event is None:
            cancel_event = self._cancel_event
            cancel_event.clear()
        store = DataStore()
        started = time.monotonic()
        if workspace is not None:
            Path(workspace).mkdir(parents=True, exist_ok=True)

        emit({"event": "run_started"})
        outputs_by_node: dict[str, dict[str, str]] = {}
        order = self._topo_order(workflow)
        by_id = {n.id: n for n in workflow.nodes}

        for node_id in order:
            if cancel_event.is_set():
                emit({"event": "run_finished", "status": "cancelled"})
                return {"status": "cancelled", "outputs": outputs_by_node}

            wnode = by_id[node_id]
            manifest = self.registry.get(wnode.type)
            emit({"event": "node_status", "node": node_id, "status": "running"})

            kwargs: dict[str, Any] = {}
            for edge in workflow.edges:
                if edge.target != node_id:
                    continue
                upstream = store.get(outputs_by_node[edge.source][edge.source_port])
                kwargs[edge.target_port] = upstream.value

            # optional inputs left unconnected arrive as None
            for port in manifest.inputs:
                if not port.required:
                    kwargs.setdefault(port.name, None)

            for param in manifest.params:
                if param.name in wnode.params:
                    kwargs[param.name] = wnode.params[param.name]
                elif param.default is not None:
                    kwargs.setdefault(param.name, param.default)

            ctx = RunContext(
                workspace=workspace,
                store=store,
                progress_cb=lambda frac, msg, _nid=node_id: emit(
                    {"event": "node_progress", "node": _nid, "frac": frac, "msg": msg}
                ),
            )
            try:
                result = manifest.execute(ctx, **kwargs)
            except Exception as exc:
                emit(
                    {
                        "event": "node_status",
                        "node": node_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                emit({"event": "run_finished", "status": "failed"})
                raise ExecutionError(node_id, exc) from exc

            handles: dict[str, str] = {}
            value_to_handle: dict[int, str] = {}
            for port in manifest.outputs:
                value = (result or {}).get(port.name)
                handle = store.put(port.type, value)
                handles[port.name] = handle.id
                value_to_handle[id(value)] = handle.id
            outputs_by_node[node_id] = handles

            for artifact in ctx.published:
                artifact_id = f"a{uuid4().hex[:8]}"
                emit(
                    {
                        "event": "artifact",
                        "node": node_id,
                        "id": artifact_id,
                        "kind": artifact["kind"],
                        "path": artifact["path"],
                        "handle": value_to_handle.get(id(artifact)),
                        "url": f"/api/artifacts/{artifact_id}/preview",
                    }
                )
            emit({"event": "node_status", "node": node_id, "status": "done"})

            # Cancellation is polled between nodes — including after the last
            # one, so cancelling a run that is mid-way through its final node
            # still reports "cancelled" instead of "completed".
            if cancel_event.is_set():
                emit({"event": "run_finished", "status": "cancelled"})
                return {"status": "cancelled", "outputs": outputs_by_node}

        emit(
            {
                "event": "run_finished",
                "status": "completed",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        )
        return {"status": "completed", "outputs": outputs_by_node}
