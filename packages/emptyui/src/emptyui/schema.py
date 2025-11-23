"""Workflow document validation (DESIGN.md §6, format ``emptyui/workflow`` v1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Position(BaseModel):
    x: float = 0
    y: float = 0


class WorkflowNode(BaseModel):
    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    position: Position = Field(default_factory=Position)

    @field_validator("id")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("node id must be non-empty")
        return v


class Edge(BaseModel):
    id: str
    source: str
    source_port: str
    target: str
    target_port: str


class Workflow(BaseModel):
    """A portable workflow document. UI-only fields (positions) live here but
    are ignored by the executor."""

    format: str = "emptyui/workflow"
    version: int = 1
    meta: dict[str, Any] = Field(default_factory=dict)
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class ValidationError_(Exception):
    """Aggregated workflow validation errors with actionable messages."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_workflow(workflow: Workflow, registry) -> None:
    """Validate a workflow against the current node catalog.

    Raises :class:`ValidationError_` listing all problems found.
    """
    errors: list[str] = []
    from emptyui.api import port_type_compatible

    by_id = {}
    for wnode in workflow.nodes:
        if wnode.id in by_id:
            errors.append(f"duplicate node id '{wnode.id}'")
        by_id[wnode.id] = wnode

        manifest = registry.get(wnode.type)
        if manifest is None:
            errors.append(
                f"node '{wnode.id}': unknown type '{wnode.type}' "
                "(plugin not installed?)"
            )
            continue

        known_params = {p.name for p in manifest.params}
        for name in wnode.params:
            if name not in known_params:
                errors.append(
                    f"node '{wnode.id}' ({wnode.type}): unknown param '{name}'"
                )

        required_inputs = {i.name for i in manifest.inputs if i.required}
        for name in required_inputs:
            if not any(e.target == wnode.id and e.target_port == name for e in workflow.edges):
                errors.append(
                    f"node '{wnode.id}' ({wnode.type}): required input "
                    f"'{name}' is not connected"
                )

    for edge in workflow.edges:
        src = by_id.get(edge.source)
        tgt = by_id.get(edge.target)
        if src is None or tgt is None:
            errors.append(f"edge '{edge.id}' references missing node")
            continue

        src_manifest, tgt_manifest = registry.get(src.type), registry.get(tgt.type)
        if src_manifest is None or tgt_manifest is None:
            continue  # already reported as unknown type

        out_port = next((o for o in src_manifest.outputs if o.name == edge.source_port), None)
        in_port = next((i for i in tgt_manifest.inputs if i.name == edge.target_port), None)
        if out_port is None:
            errors.append(
                f"edge '{edge.id}': source node '{src.id}' has no output port "
                f"'{edge.source_port}'"
            )
        if in_port is None:
            errors.append(
                f"edge '{edge.id}': target node '{tgt.id}' has no input port "
                f"'{edge.target_port}'"
            )
        if out_port and in_port and not port_type_compatible(out_port.type, in_port):
            errors.append(
                f"edge '{edge.id}': type mismatch — output "
                f"'{src.id}.{edge.source_port}:{out_port.type}' cannot connect to "
                f"input '{tgt.id}.{edge.target_port}:{in_port.type}'"
            )

    # cycle detection via Kahn's algorithm
    indegree = {n.id: 0 for n in workflow.nodes}
    adj: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
    for e in workflow.edges:
        if e.source in indegree and e.target in indegree:
            adj[e.source].append(e.target)
            indegree[e.target] += 1
    queue = [nid for nid, d in indegree.items() if d == 0]
    seen = 0
    while queue:
        nid = queue.pop()
        seen += 1
        for nxt in adj[nid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if seen != len(indegree):
        cyclic = sorted(nid for nid, d in indegree.items() if d > 0)
        errors.append(f"workflow contains a cycle involving nodes: {cyclic}")

    if errors:
        raise ValidationError_(errors)
