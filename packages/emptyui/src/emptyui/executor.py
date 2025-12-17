"""DAG executor (DESIGN.md §7).

M0: sequential topological execution over validated workflows. Cancellation
is supported via a threading.Event polled between nodes. M3: results are
content-addressed in a cross-run :class:`NodeCache` (structural signatures
propagate upstream changes), and runs can be restricted to a target node's
ancestral closure ("execute to here"). Concurrency (thread-pool branches)
remains future work.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from emptyui.cache import CacheEntry, NodeCache
from emptyui.context import RunContext
from emptyui.paths import PathPolicy
from emptyui.schema import Workflow, WorkflowNode
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

    @staticmethod
    def _ancestor_closure(workflow: Workflow, targets: list[str]) -> set[str]:
        """Target nodes plus everything they (transitively) depend on."""
        parents: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
        for e in workflow.edges:
            if e.source in parents and e.target in parents:
                parents[e.target].append(e.source)
        seen: set[str] = set()
        stack = list(targets)
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(parents[nid])
        return seen

    @staticmethod
    def _value_signature(value: Any) -> Any:
        """Cache-signature form of one effective param value.

        String values that point at existing files contribute their stat
        signature ``(path, size, mtime_ns)`` instead of the path text alone
        (Nextflow-style: cheap, detects externally modified inputs; content
        hashing is deliberately avoided — too expensive for data files).
        Everything else is used as-is (workflow params are JSON-borne).
        """
        if isinstance(value, str):
            try:
                st = Path(value).stat()
            except (OSError, ValueError):
                return value
            return [value, st.st_size, st.st_mtime_ns]
        return value

    def _node_key(
        self,
        wnode: WorkflowNode,
        manifest,
        incoming: list[dict[str, str]],
    ) -> str:
        """Structural cache signature for one node (DESIGN.md §7).

        ``sha256(type, manifest version, effective params, per-input upstream
        signatures)`` — node ids are deliberately excluded, so structurally
        identical nodes (e.g. a copy/pasted duplicate) share cache entries.
        Signatures are computed in topological order, embedding upstream
        keys recursively: any upstream change changes every downstream key.
        """
        params: dict[str, Any] = {}
        for param in manifest.params:
            if param.name in wnode.params:
                value: Any = wnode.params[param.name]
            elif param.default is not None:
                value = param.default
            else:
                continue
            params[param.name] = self._value_signature(value)
        payload = {
            "type": wnode.type,
            "version": manifest.version,
            "params": params,
            "inputs": sorted(
                incoming, key=lambda e: (e["port"], e["source"], e["source_port"])
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _record_artifacts(
        ctx: RunContext, result: dict[str, Any] | None, node_id: str
    ) -> list[dict[str, Any]]:
        """Summarize published artifacts for the cache entry.

        ``port`` records which output port's value *is* the artifact dict
        (display nodes return it), so a restored run can re-wire the event's
        ``handle`` to the fresh store handles.
        """
        recorded = []
        for artifact in ctx.published:
            p = Path(artifact["path"])
            marker = f"_{node_id}{p.suffix}"
            name = p.name
            stem = name[: -len(marker)] if name.endswith(marker) else p.stem
            port = next((pn for pn, v in (result or {}).items() if v is artifact), "")
            recorded.append(
                {
                    "src": str(p),
                    "kind": artifact["kind"],
                    "stem": stem,
                    "suffix": p.suffix,
                    "port": port,
                }
            )
        return recorded

    @staticmethod
    def _artifact_dest(rec: dict[str, Any], node_id: str, workspace) -> Path:
        """Workspace path for a restored artifact, renamed for this run's
        node instance (mirrors ``RunContext.artifact_path`` so same-type
        instances still cannot overwrite each other's files)."""
        name = f"{rec['stem']}_{node_id}{rec['suffix']}"
        return workspace / name if workspace is not None else Path(name)

    def run(
        self,
        workflow: Workflow,
        emit=print,
        workspace=None,
        cancel_event: threading.Event | None = None,
        cache: NodeCache | None = None,
        targets: list[str] | None = None,
        data_root=None,
    ) -> dict[str, Any]:
        """Execute the workflow; returns a run summary.

        `cancel_event` is polled between nodes: when set, the run stops
        early and returns ``{"status": "cancelled", ...}``. When omitted a
        fresh (never-set) event is used.

        `cache` enables cross-run result reuse: nodes whose structural
        signature (§7) matches a cached entry skip execution — cached
        outputs are re-registered in this run's store and published
        artifacts are copied into this run's workspace.

        `targets` restricts execution to the given nodes plus their
        ancestors ("execute to here"); other nodes emit no events.

        `data_root` (DESIGN.md §17.2) confines input path params to a
        read-only root and resolves output path params into the run
        workspace; None keeps local-dev behavior.
        """
        if cancel_event is None:
            cancel_event = threading.Event()
        if targets:
            known = {n.id for n in workflow.nodes}
            unknown = sorted(set(targets) - known)
            if unknown:
                raise ValueError(f"unknown target node(s): {unknown}")
        store = DataStore()
        started = time.monotonic()
        if workspace is not None:
            Path(workspace).mkdir(parents=True, exist_ok=True)

        emit({"event": "run_started"})
        outputs_by_node: dict[str, dict[str, str]] = {}
        order = self._topo_order(workflow)
        by_id = {n.id: n for n in workflow.nodes}
        if len(order) != len(by_id):
            # direct executor.run callers may skip validate_workflow; a cycle
            # (incl. a self-loop) must fail loudly, not silently skip nodes
            missing = sorted(set(by_id) - set(order))
            emit({"event": "run_finished", "status": "failed"})
            raise ExecutionError(
                missing[0],
                ValueError(
                    f"workflow contains a cycle involving nodes: {missing}"
                ),
            )

        # Snapshot the manifests this run needs at start (DESIGN.md §5.1):
        # a plugin enable/disable toggle mid-run must not yank a node out of
        # an executing workflow — the run completes under the manifests it
        # started with (toggling is serialized by the app lock).
        manifests = {nid: self.registry.get(by_id[nid].type) for nid in order}

        # Structural signatures for every node, in topo order so each embeds
        # the already-computed signatures of its inputs (§7). Deliberately
        # independent of execution: unchanged structure ⇒ unchanged key.
        keys: dict[str, str] = {}
        for node_id in order:
            wnode = by_id[node_id]
            manifest = manifests[node_id]
            incoming = [
                {
                    "port": e.target_port,
                    "source_port": e.source_port,
                    "source": keys[e.source],
                }
                for e in workflow.edges
                if e.target == node_id
            ]
            keys[node_id] = self._node_key(wnode, manifest, incoming)

        execute = (
            self._ancestor_closure(workflow, targets)
            if targets
            else set(order)
        )
        for node_id in order:
            if node_id not in execute:
                continue

            if cancel_event.is_set():
                emit({"event": "run_finished", "status": "cancelled"})
                return {"status": "cancelled", "outputs": outputs_by_node}

            wnode = by_id[node_id]
            manifest = manifests[node_id]

            # cache hit: reuse the previous run's outputs and artifacts —
            # unless a recorded artifact file has disappeared, in which case
            # the node is re-executed (never serve a half-restored result)
            entry = cache.get(keys[node_id]) if cache is not None else None
            if entry is not None and any(
                not Path(rec["src"]).exists() for rec in entry.artifacts
            ):
                entry = None
            if entry is not None:
                handles: dict[str, str] = {}
                for port in manifest.outputs:
                    handle = store.put(port.type, entry.outputs.get(port.name))
                    handles[port.name] = handle.id
                outputs_by_node[node_id] = handles
                for rec in entry.artifacts:
                    dest = self._artifact_dest(rec, node_id, workspace)
                    # a run can restore into the original run's workspace
                    # (same-workspace partial re-runs) — skip then
                    if not (dest.exists() and os.path.samefile(rec["src"], dest)):
                        shutil.copyfile(rec["src"], dest)
                    artifact_id = f"a{uuid4().hex[:8]}"
                    emit(
                        {
                            "event": "artifact",
                            "node": node_id,
                            "id": artifact_id,
                            "kind": rec["kind"],
                            "path": str(dest),
                            "handle": handles.get(rec["port"], ""),
                            "url": f"/api/artifacts/{artifact_id}/preview",
                        }
                    )
                emit(
                    {
                        "event": "node_status",
                        "node": node_id,
                        "status": "done",
                        "cached": True,
                    }
                )
                continue

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

            # remote-mode path confinement (§17.2): inputs stay inside the
            # data root, outputs resolve into this run's workspace
            if data_root is not None:
                kwargs = PathPolicy(data_root, workspace).resolve_params(
                    manifest, kwargs
                )

            ctx = RunContext(
                workspace=workspace,
                store=store,
                node_id=node_id,
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

            # success ⇒ cacheable: record outputs + artifact summaries so a
            # later run with the same structural signature can skip the node
            if cache is not None:
                cache.put(
                    keys[node_id],
                    CacheEntry(
                        outputs={
                            port.name: (result or {}).get(port.name)
                            for port in manifest.outputs
                        },
                        artifacts=self._record_artifacts(ctx, result, node_id),
                    ),
                )

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
