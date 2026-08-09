"""Workflow document validation (DESIGN.md §6, format ``emptyui/workflow`` v1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from emptyui.params import coerce_param


class Position(BaseModel):
    x: float = 0
    y: float = 0


class WorkflowNode(BaseModel):
    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    position: Position = Field(default_factory=Position)
    #: The plugin distribution version recorded when the document was
    #: saved (issues.md M23, DESIGN §5). Informational only — the run
    #: always uses the installed codebase; a mismatch surfaces as a
    #: node_warning (executor._warn_version_drift), never an error.
    #: Tolerated absent (documents saved before the field existed).
    plugin_version: str | None = None

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


def _missing_param_error(wnode: WorkflowNode, name: str) -> str:
    return (
        f"node '{wnode.id}' ({wnode.type}): required parameter '{name}' "
        "is not provided"
    )


def _param_value_problems(wnode: WorkflowNode, manifest) -> list[str]:
    """Validate param *values* against the manifest's own declarations (M1).

    The API is the trust boundary in remote mode (DESIGN §11): widgets
    guard the UI, but a POSTed document bypasses them. Each param present
    in the document is coerced with the executor's own
    :func:`emptyui.params.coerce_param` — validation and execution
    therefore agree on what a value means — and then checked against the
    declared ``min``/``max`` (number/integer), the declared ``options``
    (select/radio) and the ``optional`` flag. Manifest defaults are not
    re-checked (plugin-declared, not client-supplied), and stale params a
    node no longer declares stay tolerated per the history §24.9 policy (the
    executor warns about them).

    Option checks are skipped for params whose option set is extended at
    runtime: ``suggest_from`` (payload-summary suggestions) and inherit
    targets (upstream-fed editable comboboxes).
    """
    problems: list[str] = []
    inherited: set[str] = set()
    for mapping in (getattr(manifest, "inherit", {}) or {}).values():
        inherited.update(mapping.keys())
    for param in getattr(manifest, "params", ()) or ():
        name = param.name
        if name in wnode.params:
            raw: Any = wnode.params[name]
        else:
            if param.default is None and not param.optional:
                problems.append(_missing_param_error(wnode, name))
            continue
        # explicit null, or an empty path widget, is an unprovided value
        if raw is None or (
            getattr(param, "widget", "") in ("file", "directory")
            and isinstance(raw, str) and not raw.strip()
        ):
            if not param.optional and param.default is None:
                problems.append(_missing_param_error(wnode, name))
            continue
        try:
            value = coerce_param(param, raw)
        except ValueError as exc:
            problems.append(f"node '{wnode.id}' ({wnode.type}): {exc}")
            continue
        widget = getattr(param, "widget", "")
        if widget in ("number", "integer"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(
                    f"node '{wnode.id}' ({wnode.type}): param '{name}': "
                    f"{raw!r} is not a number"
                )
                continue
            if widget == "integer" and float(value) != int(value):
                problems.append(
                    f"node '{wnode.id}' ({wnode.type}): param '{name}': "
                    f"{raw!r} requires an integer value"
                )
                continue
            lo, hi = getattr(param, "min", None), getattr(param, "max", None)
            if (lo is not None and value < lo) or (
                    hi is not None and value > hi):
                bounds = lo if hi is None else (
                    hi if lo is None else f"[{lo}, {hi}]")
                problems.append(
                    f"node '{wnode.id}' ({wnode.type}): param '{name}': "
                    f"{raw!r} is out of range ({bounds})"
                )
        elif widget in ("select", "radio"):
            opts = getattr(param, "options", ()) or ()
            if (opts and not getattr(param, "suggest_from", "")
                    and name not in inherited and value not in opts):
                problems.append(
                    f"node '{wnode.id}' ({wnode.type}): param '{name}': "
                    f"{raw!r} is not one of the declared options "
                    f"({', '.join(str(o) for o in opts)})"
                )
    return problems


def validate_workflow(workflow: Workflow, registry) -> None:
    """Validate a workflow against the current node catalog.

    Raises :class:`ValidationError_` listing all problems found.
    """
    errors: list[str] = []
    # document envelope (M-C2/L-C2, DESIGN.md §6): the frontend validates
    # format/version at load time, but the executor is a public API — a
    # foreign or future document must be rejected here too (a silent
    # best-effort parse would run half-understood workflows)
    if workflow.format != "emptyui/workflow":
        errors.append(
            f"not an emptyUI workflow document (format: {workflow.format!r})")
    if workflow.version != 1:
        errors.append(f"unsupported workflow version: {workflow.version}")
    from emptyui.api import port_type_compatible

    by_id = {}
    for wnode in workflow.nodes:
        if wnode.id in by_id:
            errors.append(f"duplicate node id '{wnode.id}'")
        by_id[wnode.id] = wnode

        manifest = registry.get(wnode.type)
        if manifest is None:
            # actionable message (§5.1): a disabled plugin is different from
            # a plugin that is simply not installed
            if registry.status_of(wnode.type) == "disabled":
                plugin = registry.plugin_name_of(wnode.type) or "unknown"
                errors.append(
                    f"node '{wnode.id}': type '{wnode.type}' belongs to plugin "
                    f"'{plugin}', which is disabled — enable it in Settings"
                )
            else:
                errors.append(
                    f"node '{wnode.id}': unknown type '{wnode.type}' "
                    "(plugin not installed?)"
                )
            continue

        # Stale params (history §24.9) are deliberately tolerated here: a saved
        # workflow may carry params a node no longer declares. The executor
        # resolves only manifest params and emits a node-level progress
        # note naming them — validation must not hard-reject what the
        # documented policy is to warn about.

        errors.extend(_param_value_problems(wnode, manifest))

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
        if edge.source == edge.target:
            errors.append(
                f"edge '{edge.id}': node '{edge.source}' cannot connect to "
                "itself (self-loop)"
            )
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

    # fan-in duplicates: an input port receives exactly one payload — multiple
    # edges into the same port would silently overwrite (executor last-wins)
    seen_targets: set[tuple[str, str]] = set()
    for edge in workflow.edges:
        key = (edge.target, edge.target_port)
        if key in seen_targets:
            errors.append(
                f"edge '{edge.id}': input port '{edge.target_port}' of node "
                f"'{edge.target}' already has an incoming connection — an "
                "input port accepts exactly one edge"
            )
        seen_targets.add(key)

    # cycle detection via Kahn's algorithm (self-loops reported above —
    # excluded here so one problem produces one error)
    indegree = {n.id: 0 for n in workflow.nodes}
    adj: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
    for e in workflow.edges:
        if e.source in indegree and e.target in indegree and e.source != e.target:
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
