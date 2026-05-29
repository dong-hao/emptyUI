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
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from emptyui.cache import CacheEntry, NodeCache
from emptyui.context import RunContext
from emptyui.params import coerce_param
from emptyui.paths import PathPolicy
from emptyui.schema import Workflow, WorkflowNode
from emptyui.store import DataStore


class ExecutionError(Exception):
    def __init__(self, node_id: str, original: BaseException) -> None:
        self.node_id = node_id
        self.original: BaseException = original
        super().__init__(f"node '{node_id}' failed: {original}")


def _as_float(value: Any) -> float:
    """Degrading float coercion — 0.0 for non-numeric/None (M-C4: the
    summarizer must never crash the run; the bar merely goes unusable)."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ts_summary(value: Any) -> dict[str, Any] | None:
    """Lightweight extent summary for a time-series payload (DESIGN.md §8.3).

    Drives the frontend time-selector bar. Duck-typed over the payload
    shapes so the core package stays independent of mth5:
    - ``ts`` kind — mth5 RunTS (``.dataset``), the core ``TimeSeries``
      (``.channels``) or a ChannelTS (``.component``): absolute ISO bounds
      (UTC) when the record carries a real start time, relative seconds
      otherwise;
    - ``mth5`` kind — an open archive (``.run_summary``): per-run extents
      (survey/station/run/start/end/sample_rate) for the Extract Run bar.
    Returns None for anything that is not (interpretable as) either.
    """
    if isinstance(value, dict):
        return None
    # NB: mt_timeseries objects convert missing-attribute AttributeError into
    # NameError AND log an ERROR line — probe via dir() (no __getattr__
    # invocation) so summarizing an mth5 RunTS stays quiet
    visible = dir(value)
    rs = value.run_summary if "run_summary" in visible else None
    if rs is not None and hasattr(rs, "iterrows") and len(rs):
        runs = []
        for _, row in rs.iterrows():
            runs.append({
                "survey": str(row.get("survey", "")),
                "station": str(row.get("station", "")),
                "run": str(row.get("run", "")),
                "start": str(row.get("start", ""))[:19],
                "end": str(row.get("end", ""))[:19],
                # non-numeric sample_rate degrades to 0.0 (an unusable bar
                # is better than crashing the run — M-C4)
                "sample_rate": _as_float(row.get("sample_rate", 0.0)),
            })
        return {"kind": "mth5", "runs": runs}
    # run collection (Extract Run 'time range' mode): union bounds on top,
    # per-member extents for provenance — the frontend renders the union
    # track directly (non-mth5 kinds use the top-level start/end)
    extents = value.extents if "extents" in visible else None
    if (extents is not None and "runs" in visible
            and isinstance(extents, list) and extents):
        starts = [str(e.get("start", ""))[:19] for e in extents]
        ends = [str(e.get("end", ""))[:19] for e in extents]
        runs = [{
            "survey": str(e.get("survey", "")),
            "station": str(e.get("station", "")),
            "run": str(e.get("run", "")),
            "start": str(e.get("start", ""))[:19],
            "end": str(e.get("end", ""))[:19],
            "sample_rate": _as_float(e.get("sample_rate", 0.0)),
        } for e in extents]
        bounds = [s for s in starts if s]
        if not bounds:
            # all extents carry empty starts — no interpretable union
            # (M-C4: min() on an empty generator crashed the run outside
            # the try); emit the per-member runs only — the frontend
            # renders no top-level bar and degrades gracefully
            return {"kind": "ts.runs", "runs": runs}
        # guard the symmetric edge: starts parse but every end is empty
        ends_ok = [e for e in ends if e]
        if not ends_ok:
            return {"kind": "ts.runs", "runs": runs}
        return {
            "kind": "ts.runs",
            "start": min(bounds),
            "end": max(ends_ok),
            "absolute": True,
            "runs": runs,
        }
    if "sample_rate" not in visible:
        return None
    try:
        sample_rate = float(value.sample_rate)
    except Exception:  # noqa: BLE001
        return None
    if "n_samples" in visible:
        try:
            n = int(value.n_samples)
        except Exception:  # noqa: BLE001 - mt_timeseries raises NameError
            n = None
    else:
        try:
            n = len(value.dataset.time)  # mth5 RunTS
        except Exception:  # noqa: BLE001
            return None
    if n is None or sample_rate <= 0 or n <= 0:
        return None

    channels: list[str] | None = None
    start_iso = ""
    start_rel = 0.0
    if hasattr(value, "dataset"):  # mth5 RunTS
        try:
            channels = [str(k) for k in value.dataset.data_vars]
        except AttributeError:
            channels = None
        raw = getattr(value, "start", None) if "start" in visible else None
        if raw is not None:
            try:
                start_iso = datetime.fromisoformat(str(raw)[:19]).strftime(
                    "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                start_iso = ""
    elif hasattr(value, "channels"):  # core TimeSeries
        channels = [str(k) for k in value.channels]
        start_iso = str(getattr(value, "start_datetime", "") or "")
        try:
            start_rel = float(getattr(value, "start", 0.0) or 0.0)
        except (TypeError, ValueError):
            start_rel = 0.0
        if start_iso:
            try:
                anchor = datetime.fromisoformat(start_iso[:19])
            except ValueError:
                start_iso = ""
            else:
                # `start` is an offset from the record start; the absolute
                # bound of sample 0 is anchor + start
                start_iso = (anchor + timedelta(seconds=start_rel)).strftime(
                    "%Y-%m-%dT%H:%M:%S")
                start_rel = 0.0
    elif hasattr(value, "component"):  # single channel (ts.channel port)
        channels = [str(getattr(value, "component", "channel") or "channel")]
        raw = getattr(value, "start", None) if "start" in visible else None
        if raw is not None:
            try:
                start_iso = datetime.fromisoformat(str(raw)[:19]).strftime(
                    "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                start_iso = ""
    # Placeholder epochs (mt_timeseries' 1980 default, digitizers' 1970 —
    # core-nodes UNKNOWN_EPOCH_STARTS, mirrored here because core cannot
    # import its plugins) mean "start unknown": degrade to relative bounds.
    if start_iso[:19] in ("1980-01-01T00:00:00", "1970-01-01T00:00:00"):
        start_iso = ""
    if channels is None:
        return None
    end = (start_rel + n / sample_rate) if not start_iso else (
        datetime.fromisoformat(start_iso) + timedelta(seconds=n / sample_rate)
    ).strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "kind": "ts",
        "start": start_iso if start_iso else start_rel,
        "end": end,
        "absolute": bool(start_iso),
        "sample_rate": sample_rate,
        "channels": channels,
    }


def _fc_summary(value) -> dict[str, Any] | None:
    """Lightweight rate-tier summary for an fc payload (DESIGN.md §8.3).

    A merged burst FC database is rate-keyed (`{tier: Dataset}`); the plot
    nodes select a tier by its actual sample rate — the summary streams the
    available rates so the frontend can offer them as select suggestions.
    Duck-typed: a dict of Datasets (or one Dataset) whose data carry a
    `frequency` dimension and a `sample_rate` attr.
    """
    try:
        items = (value.items() if isinstance(value, dict)
                 else [(0, value)])
        rates = []
        for _, ds in items:
            r = float(getattr(ds, "attrs", {}).get("sample_rate", 0.0) or 0.0)
            if r > 0 and "frequency" in getattr(ds, "dims", {}):
                rates.append(r)
        if not rates:
            return None
        return {
            "kind": "fc",
            "rates": sorted({round(r, 9) for r in rates}, reverse=True),
        }
    except Exception:  # noqa: BLE001 - not an fc payload
        return None


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
    def invalidate_workspace_cache(cache, workspace, logger) -> int:
        """TTL-GC support (M-C8): drop cache entries whose cached payloads
        hold open files inside ``workspace`` — the GC is about to rmtree
        it, and a cache hit on such an entry would restore a payload whose
        backing file is gone (reads/writes land on unlinked inodes; Save
        As from it silently produces nothing on disk).

        Open handles are deliberately NOT closed here: a concurrent run may
        hold the same shared payload instance (immutability contract, §7.1)
        and reads on the unlinked inode keep succeeding until the last
        reference drops. Dropping the entry forces the next run to
        re-execute the producing node into its own fresh workspace.
        Returns how many entries were dropped."""
        if cache is None or not workspace:
            return 0
        try:
            ws = str(Path(workspace).resolve())
        except OSError:
            ws = str(workspace)
        if not ws:
            return 0

        def _refers(value: Any) -> bool:
            fn = getattr(value, "filename", None)
            if not fn:
                return False
            try:
                p = str(Path(fn).resolve())
            except OSError:
                p = str(fn)
            return p == ws or p.startswith(ws + os.sep)

        return cache.drop_referencing(_refers)

    @staticmethod
    def _sweep_run_payloads(store, outputs_by_node, cache, logger) -> None:
        """Failed/cancelled-run hygiene: close open MTH5 payloads the run
        left in its store (§9.3 stale-flag hygiene — an interrupted run must
        not leave a live writer holding a user archive).

        Cached payloads are shared with later runs (§7.1) — only identities
        NOT in any cache entry may be closed, so the sweep never poisons the
        cache. Best effort: a failure here must never mask the run's error.
        """
        cached = {id(v) for v in (cache.cached_values() if cache else [])}
        for handles in outputs_by_node.values():
            for hid in handles.values():
                try:
                    value = store.get(hid).value
                except KeyError:
                    continue
                try:
                    close = getattr(value, "close_mth5", None)
                except Exception:  # noqa: BLE001, S112 — best effort: a
                    # mt_timeseries converts a missing-attribute AttributeError
                    # into NameError — which defeats getattr's default; a
                    # non-archive payload (RunTS) must simply be skipped (the
                    # NameError otherwise escapes and MASKS the run's error)
                    continue
                if close is None or id(value) in cached:
                    continue
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 — best effort
                    logger.warning(
                        "run cleanup: closing an open archive failed: %s", exc)

    @staticmethod
    def _value_signature(
        value: Any,
        data_root=None,
        path_kind: str = "",
        widget: str = "",
        project_dir=None,
    ) -> Any:
        """Cache-signature form of one effective param value.

        Only path params (``widget`` ``file``/``directory``) pointing at an
        existing path contribute a stat signature ``(path, size, mtime_ns)``
        (Nextflow-style: cheap, detects externally modified inputs; content
        hashing is deliberately avoided — too expensive for data files).
        Every other value is used as-is (workflow params are JSON-borne), so
        a plain string that coincidentally names an existing path never
        becomes a stat signature.

        The path part is taken data-root-relative (POSIX form) when the
        file lives under the data root or project dir, so cache keys
        survive relocation (DESIGN.md §7.1, §12.3); the absolute path is
        used otherwise. The stat half still moves with the file: a plain
        copy refreshes ``mtime_ns`` and misses the cache (recompute — the
        safe direction); only mtime-preserving moves stay cache hits.

        ``data_root`` + ``path_kind="in"``: a RELATIVE input path is
        resolved under the data root before statting (execution resolves
        the same way, §17.2) — without this the stat silently missed for
        remote-mode workflows and external modifications never
        invalidated the cache.
        """
        if not isinstance(value, str) or widget not in ("file", "directory"):
            return value
        if path_kind == "out":
            # L-C3: an output path is a DESTINATION, not an input — its
            # stat at key time says nothing about what the node will
            # write, and an empty out-path stat()s the server CWD
            # (Path("") == Path(".")): a spurious cross-run
            # invalidation trigger on every CWD-affecting event (§17.2
            # resolves out-paths per-run into the workspace anyway)
            return value
        p = Path(value).expanduser()
        if path_kind == "in" and data_root is not None and not p.is_absolute():
            p = Path(data_root) / p
        try:
            st = p.stat()
        except (OSError, ValueError):
            return value
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        for root in (data_root, project_dir):
            if not root:
                continue
            try:
                base = Path(root).resolve()
                if resolved.is_relative_to(base):
                    return [resolved.relative_to(base).as_posix(),
                            st.st_size, st.st_mtime_ns]
            except (OSError, ValueError):
                continue
        return [str(p), st.st_size, st.st_mtime_ns]

    def _node_key(
        self,
        wnode: WorkflowNode,
        manifest,
        incoming: list[dict[str, str]],
        data_root=None,
        project_dir=None,
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
                raw: Any = wnode.params[param.name]
            elif param.default is not None:
                raw = param.default
            else:
                continue
            # §7 hashes the EFFECTIVE params — coerce first, so the string
            # combobox value "4" and the number 4 share one cache entry
            # (execution coerces identically, §18 review fix). The numeric
            # canonicalization here normalizes the KEY form only (int 4 vs
            # float 4.0); execution keeps its own semantics.
            value = self._coerce_param(param, raw)
            widget = getattr(param, "widget", "")
            if widget in ("number", "integer") and isinstance(
                    value, (int, float)) and not isinstance(value, bool):
                value = (float(value) if widget == "number"
                         else round(float(value)))
            params[param.name] = self._value_signature(
                value, data_root=data_root,
                path_kind=getattr(param, "path_kind", ""),
                widget=getattr(param, "widget", ""),
                project_dir=project_dir,
            )
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
    def _artifact_dest(rec: dict[str, Any], node_id: str, workspace) -> Path | None:
        """Workspace path for a restored artifact, renamed for this run's
        node instance (mirrors ``RunContext.artifact_path`` so same-type
        instances still cannot overwrite each other's files).

        None without a workspace (headless/CLI embedding): restoring into
        the process CWD was a side effect outside the workspace model —
        the restore skips file copying instead (the cache entry still
        re-registers the output payloads).
        """
        name = f"{rec['stem']}_{node_id}{rec['suffix']}"
        return workspace / name if workspace is not None else None

    @staticmethod
    def _coerce_param(param, value: Any) -> Any:
        """Coerce a workflow param value to the manifest's declared type.

        Delegates to the single shared :func:`emptyui.params.coerce_param`
        (M1) — schema validation applies the identical coercion before its
        min/max/options checks, so the trust boundary and execution can
        never disagree about what a value means.
        """
        return coerce_param(param, value)

    def run(
        self,
        workflow: Workflow,
        emit=print,
        workspace=None,
        cancel_event: threading.Event | None = None,
        cache: NodeCache | None = None,
        targets: list[str] | None = None,
        data_root=None,
        project_dir=None,
        include_store: bool = False,
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

        `data_root` (DESIGN.md §11.2) confines input path params to a
        read-only root and resolves output path params into the run
        workspace; None keeps local-dev behavior. `project_dir` (§12.3,
        planned) is the stable user-designated directory whose relative
        layout cache keys may share; inert until the disk cache lands.
        """
        if cancel_event is None:
            cancel_event = threading.Event()
        if targets:
            known = {n.id for n in workflow.nodes}
            unknown = sorted(set(targets) - known)
            if unknown:
                raise ValueError(f"unknown target node(s): {unknown}")
        by_id = {n.id: n for n in workflow.nodes}
        # malformed edges (M-C10): an edge referencing a missing node id
        # would surface as a raw KeyError from _topo_order/_node_key (a
        # crash without the node-failure contract) — name it instead
        for e in workflow.edges:
            for endpoint in (e.source, e.target):
                if endpoint not in by_id:
                    raise ExecutionError(
                        endpoint,
                        ValueError(
                            f"edge '{getattr(e, 'id', '?')}' references "
                            f"unknown node '{endpoint}'"),
                    )
        store = DataStore()
        started = time.monotonic()
        if workspace is not None:
            Path(workspace).mkdir(parents=True, exist_ok=True)

        emit({"event": "run_started"})
        outputs_by_node: dict[str, dict[str, str]] = {}
        order = self._topo_order(workflow)
        if len(order) != len(by_id):
            # direct executor.run callers may skip validate_workflow; a cycle
            # (incl. a self-loop) must fail loudly, not silently skip nodes
            missing = sorted(set(by_id) - set(order))
            cycle_error = f"workflow contains a cycle involving nodes: {missing}"
            emit(
                {
                    "event": "run_finished",
                    "status": "failed",
                    "node": missing[0],
                    "error": cycle_error,
                }
            )
            raise ExecutionError(missing[0], ValueError(cycle_error))

        # Snapshot the manifests this run needs at start (DESIGN.md §5.1):
        # a plugin enable/disable toggle mid-run must not yank a node out of
        # an executing workflow — the run completes under the manifests it
        # started with. NB the whole TABLE is fetched once (M-C5): per-node
        # registry.get() calls are not a snapshot — each read may land on a
        # different table across a concurrent rebuild's swap, mixing
        # old/new manifests (a just-disabled plugin could execute).
        table = self.registry.table()
        manifests = {nid: table.get(by_id[nid].type) for nid in order}
        unknown = sorted(nid for nid, m in manifests.items() if m is None)
        if unknown:
            # direct executor.run callers may skip validate_workflow; also
            # covers a type vanishing between validation and this snapshot
            missing_types = sorted({by_id[nid].type for nid in unknown})
            type_error = f"unknown node type(s): {missing_types}"
            emit(
                {
                    "event": "run_finished",
                    "status": "failed",
                    "node": unknown[0],
                    "error": type_error,
                }
            )
            raise ExecutionError(unknown[0], ValueError(type_error))

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
            try:
                keys[node_id] = self._node_key(wnode, manifest, incoming,
                                               data_root=data_root,
                                               project_dir=project_dir)
            except ValueError as exc:
                # param coercion now happens at key time (§7 hashes the
                # EFFECTIVE params) — surface the failure as the same
                # ExecutionError the execution phase would produce
                emit(
                    {
                        "event": "run_finished",
                        "status": "failed",
                        "node": node_id,
                        "error": str(exc),
                    }
                )
                raise ExecutionError(node_id, exc) from exc

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
                summary = {"status": "cancelled",
                           "outputs": outputs_by_node}
                if include_store:
                    summary["store"] = store
                return summary

            wnode = by_id[node_id]
            manifest = manifests[node_id]

            # cache hit: reuse the previous run's outputs and artifacts —
            # unless a recorded artifact file has disappeared, in which case
            # the node is re-executed (never serve a half-restored result).
            # Non-cacheable nodes (save/export — the side effect IS writing
            # the file) never hit: the file must be (re-)written this run.
            entry = cache.get(keys[node_id]) if (
                cache is not None and manifest.cacheable) else None
            if entry is not None and any(
                not Path(rec["src"]).exists() for rec in entry.artifacts
            ):
                entry = None
            if entry is not None:
                try:
                    handles: dict[str, str] = {}
                    for port in manifest.outputs:
                        handle = store.put(port.type, entry.outputs.get(port.name))
                        handles[port.name] = handle.id
                        summary = (_ts_summary(entry.outputs.get(port.name))
                                   or _fc_summary(entry.outputs.get(port.name)))
                        if summary is not None:
                            emit({"event": "payload", "node": node_id,
                                  "port": port.name, "type": port.type,
                                  "summary": summary})
                    outputs_by_node[node_id] = handles
                    for rec in entry.artifacts:
                        dest = self._artifact_dest(rec, node_id, workspace)
                        if dest is not None:
                            # a run can restore into the original run's
                            # workspace (same-workspace partial re-runs) —
                            # skip then
                            if not (dest.exists()
                                    and os.path.samefile(rec["src"], dest)):
                                shutil.copyfile(rec["src"], dest)
                            event_path = str(dest)
                        else:
                            # headless/CLI run without a workspace: skip the
                            # file copy (no CWD side effect) — the artifact
                            # event resolves to the cache entry's own file
                            event_path = rec["src"]
                        artifact_id = f"a{uuid4().hex[:8]}"
                        emit(
                            {
                                "event": "artifact",
                                "node": node_id,
                                "id": artifact_id,
                                "kind": rec["kind"],
                                "path": event_path,
                                "handle": handles.get(rec["port"], ""),
                                "url": f"/api/artifacts/{artifact_id}/preview",
                            }
                        )
                except Exception as exc:
                    # restore/summary failures must surface as the node's
                    # error, never as a raw crash escaping the run (a closed
                    # cached payload would otherwise leave the node
                    # "running" forever — §7.1 failure contract)
                    emit(
                        {
                            "event": "node_status",
                            "node": node_id,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
                    emit(
                        {
                            "event": "run_finished",
                            "status": "failed",
                            "node": node_id,
                            "error": str(exc),
                        }
                    )
                    self._sweep_run_payloads(store, outputs_by_node, cache,
                                             logging.getLogger("emptyui"))
                    raise ExecutionError(node_id, exc) from exc
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

            ctx = RunContext(
                workspace=workspace,
                store=store,
                node_id=node_id,
                progress_cb=lambda frac, msg, _nid=node_id: emit(
                    {"event": "node_progress", "node": _nid, "frac": frac, "msg": msg}
                ),
            )
            try:
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
                        kwargs[param.name] = self._coerce_param(
                            param, wnode.params[param.name])
                    elif param.default is not None:
                        kwargs.setdefault(param.name, param.default)

                # stale params (§24.9): a saved workflow may carry params a
                # node no longer declares (removed/refactored options) —
                # they are ignored, but silently, so warn once per node
                stale = sorted(set(wnode.params) - {p.name for p in manifest.params})
                if stale:
                    emit({
                        "event": "node_progress",
                        "node": node_id,
                        "frac": 0.0,
                        "msg": f"ignoring params not declared by node type "
                               f"'{wnode.type}': {', '.join(stale)}",
                    })

                # remote-mode path confinement (§17.2): inputs stay inside
                # the data root, outputs resolve into this run's workspace
                if data_root is not None:
                    kwargs = PathPolicy(data_root, workspace).resolve_params(
                        manifest, kwargs
                    )
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
                emit(
                    {
                        "event": "run_finished",
                        "status": "failed",
                        "node": node_id,
                        "error": str(exc),
                    }
                )
                self._sweep_run_payloads(store, outputs_by_node, cache,
                                         logging.getLogger("emptyui"))
                raise ExecutionError(node_id, exc) from exc

            handles: dict[str, str] = {}
            value_to_handle: dict[int, str] = {}
            try:
                for port in manifest.outputs:
                    value = (result or {}).get(port.name)
                    handle = store.put(port.type, value)
                    handles[port.name] = handle.id
                    value_to_handle[id(value)] = handle.id
                    # time-series extent summary for the frontend time selector
                    # (§23.4) — lightweight, duck-typed, emitted per ts output
                    summary = _ts_summary(value) or _fc_summary(value)
                    if summary is not None:
                        emit({"event": "payload", "node": node_id,
                              "port": port.name, "type": port.type,
                              "summary": summary})
            except Exception as exc:
                # output materialization/summary failures surface as the
                # node's error, never as a raw crash escaping the run (the
                # node_status "running" would otherwise stick forever)
                emit(
                    {
                        "event": "node_status",
                        "node": node_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                emit(
                    {
                        "event": "run_finished",
                        "status": "failed",
                        "node": node_id,
                        "error": str(exc),
                    }
                )
                self._sweep_run_payloads(store, outputs_by_node, cache,
                                         logging.getLogger("emptyui"))
                raise ExecutionError(node_id, exc) from exc
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
            # later run with the same structural signature can skip the node.
            # Non-cacheable nodes (save/export) re-execute every run.
            if cache is not None and manifest.cacheable:
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

            # input-invalidating nodes (e.g. Save MTH5 closing the open
            # archive in place, §9.3): the direct-input producers' cache
            # entries now hold a poisoned payload (a closed archive handle)
            # — evict them so later structurally-identical runs re-execute
            # the producers instead of serving the closed handle (§7.1
            # immutability contract). Derived payloads are unaffected (their
            # data was materialized eagerly), so only the direct producers
            # need eviction.
            if manifest.invalidates_inputs and cache is not None:
                for edge in workflow.edges:
                    if edge.target == node_id:
                        cache.drop(keys[edge.source])

            # Cancellation is polled between nodes — including after the last
            # one, so cancelling a run that is mid-way through its final node
            # still reports "cancelled" instead of "completed".
            if cancel_event.is_set():
                emit({"event": "run_finished", "status": "cancelled"})
                self._sweep_run_payloads(store, outputs_by_node, cache,
                                         logging.getLogger("emptyui"))
                summary = {"status": "cancelled",
                           "outputs": outputs_by_node}
                if include_store:
                    summary["store"] = store
                return summary

        emit(
            {
                "event": "run_finished",
                "status": "completed",
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        )
        summary = {"status": "completed", "outputs": outputs_by_node}
        if include_store:
            # in-process embedding/scripts: resolve handle ids to payloads
            # (the server layer resolves handles its own way and never
            # passes this flag — the store never leaks through the API)
            summary["store"] = store
        return summary
