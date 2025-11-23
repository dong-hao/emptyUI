"""DAG executor (DESIGN.md §7).

M0: sequential topological execution over validated workflows.
Concurrency, disk caching and cancellation land in M1–M3; the handle-keyed
structure below is already cache-ready.
"""

from __future__ import annotations

import time
from typing import Any

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
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

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

    def run(self, workflow: Workflow, emit=print, workspace=None) -> dict[str, Any]:
        """Execute the workflow; returns a run summary."""
        self._cancelled = False
        store = DataStore()
        started = time.monotonic()

        emit({"event": "run_started"})
        outputs_by_node: dict[str, dict[str, str]] = {}
        order = self._topo_order(workflow)
        by_id = {n.id: n for n in workflow.nodes}

        for node_id in order:
            if self._cancelled:
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

            from emptyui.api import Param

            for param in manifest.params:
                if param.name in wnode.params:
                    kwargs[param.name] = wnode.params[param.name]
                elif param.default is not None or not isinstance(param, Param):
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
            for port in manifest.outputs:
                value = (result or {}).get(port.name)
                handle = store.put(port.type, value)
                handles[port.name] = handle.id
            outputs_by_node[node_id] = handles
            emit({"event": "node_status", "node": node_id, "status": "done"})

        emit(
            {
                "event": "run_finished",
                "status": "completed",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        )
        return {"status": "completed", "outputs": outputs_by_node}
