"""Node registry and plugin discovery via entry points."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from emptyui.api import Node

ENTRY_POINT_GROUP = "emptyui.nodes"

#: UI metadata for the core port types (DESIGN.md §3) — the backend is the
#: single source of truth for socket colors; the frontend's built-in table
#: is only a fallback. Colors mirror the frontend table verbatim so the
#: rendered result is unchanged. Node-declared ``port_types`` (plugin guide
#: §"Port types") override these per tag.
CORE_PORT_TYPES: dict[str, dict[str, str]] = {
    "any": {"color": "#95a5a6", "label": "Any"},
    "table": {"color": "#f39c12", "label": "Table"},
    "plot": {"color": "#d81b8c", "label": "Plot"},
    "mth5": {"color": "#9b59b6", "label": "MTH5 Archive"},
    "calibration": {"color": "#af7ac5", "label": "Calibration / Instrument Response"},
    # NB: "selection" is intentionally absent (issues.md L2, decision
    # 2026-10-14): no bundled node produces it (the Kernel Dataset surface
    # is M4-deferred) — the type returns together with its producer.
    # ViewSelection (mth5.table.selection) stays registered but hidden.
    "decimation": {"color": "#a569bd", "label": "Decimation Level"},
    "config": {"color": "#6c3483", "label": "Processing Config"},
    "bands": {"color": "#d2b4de", "label": "Frequency Bands"},
    "ts.run": {"color": "#2980b9", "label": "Run Time Series"},
    "ts.channel": {"color": "#7fb3d5", "label": "Channel Time Series"},
    "ts.runs": {"color": "#5dade2", "label": "Run Collection"},
    "fc": {"color": "#186a3b", "label": "Fourier Coefficients"},
    "tf.level": {"color": "#e74c3c", "label": "TF (decimation level)"},
    "tf": {"color": "#c0392b", "label": "Transfer Function"},
}

logger = logging.getLogger("emptyui")


@dataclass
class PluginInfo:
    """Metadata about one discovered plugin (DESIGN.md §5.1).

    ``enabled`` is derived at discovery time from the disabled set; bundled
    plugins (``register.bundled = True``) are always enabled.
    """

    name: str
    version: str
    bundled: bool
    node_types: list[str] = field(default_factory=list)
    enabled: bool = True
    #: when discovery or registration failed, the reason (the plugin is
    #: skipped — recorded instead of crashing discovery/startup)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "node_count": len(self.node_types),
            "bundled": self.bundled,
            "enabled": self.enabled,
            "error": self.error,
        }


class Registry:
    """Holds all known node types; serializes manifests for the frontend.

    Plugins are discovered through the ``emptyui.nodes`` entry-point group.
    A plugin can be disabled (§5.1): its nodes are removed from the node
    table (so the catalog/palette hides them and workflow validation
    rejects them with an actionable message), but its :class:`PluginInfo`
    is retained so the settings dialog can re-enable it. Nodes added
    directly via :meth:`add` (framework tests, embedding platforms) are
    re-registered on :meth:`rebuild`.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._plugins: dict[str, PluginInfo] = {}
        self._plugin_of: dict[str, str] = {}
        self._manual: dict[str, type[Node]] = {}

    def add(self, node_cls: type[Node]) -> None:
        node = node_cls()
        if not getattr(node, "type", None):
            raise ValueError(f"{node_cls.__name__} does not define 'type'")
        if node.type in self._nodes:
            raise ValueError(f"duplicate node type: {node.type}")
        self._nodes[node.type] = node
        self._manual[node.type] = node_cls

    def get(self, node_type: str) -> Node | None:
        return self._nodes.get(node_type)

    def table(self) -> dict[str, Node]:
        """A consistent snapshot of the node table (M-C5): fetch ONCE and
        read from the reference — :meth:`rebuild` builds into a scratch
        registry and REPLACES this dict, never mutates it in place, so a
        single fetch is a consistent snapshot even if a toggle lands
        mid-snapshot. Per-key ``get()`` calls are not: each read may land
        on a different table across the swap (a torn snapshot mixing
        old/new manifests, letting a just-disabled plugin execute).
        Returns a shallow copy: ``add`` mutates the live dict in place."""
        return dict(self._nodes)

    def types(self) -> list[str]:
        return sorted(self._nodes)

    def manifests(self) -> dict[str, Any]:
        # port_types: UI metadata for every port type (DESIGN.md §3) — the
        # backend decides the colors (the frontend table is only a fallback).
        # The core tags ship here; node-declared ``port_types`` (plugin guide
        # §"Port types") override per tag, first declaration wins among nodes.
        # L13: "first" is by SORTED node type — raw insertion order follows
        # entry-point iteration and flipped the winner across installs.
        port_types: dict[str, dict[str, str]] = {
            tag: dict(meta) for tag, meta in CORE_PORT_TYPES.items()
        }
        for node_type in self.types():
            node = self._nodes[node_type]
            for ptype, meta in getattr(node, "port_types", {}).items():
                port_types.setdefault(ptype, dict(meta))
        # M23: each manifest carries its plugin's distribution version —
        # the frontend records it per node when saving, and the executor
        # compares it at run time (the drift warning). Manual/test nodes
        # outside any plugin carry no key (nothing to compare).
        manifests = []
        for node_type, node in self._nodes.items():
            m = node.manifest()
            pv = self.plugin_version_of(node_type)
            if pv:
                m["plugin_version"] = pv
            manifests.append(m)
        return {"nodes": manifests, "port_types": port_types}

    # --- plugin management (§5.1) -----------------------------------------

    def plugin(self, name: str) -> PluginInfo | None:
        return self._plugins.get(name)

    def plugins(self) -> list[dict[str, Any]]:
        """Installed plugins — enabled or not, bundled included — for the UI."""
        return [info.to_dict() for _, info in sorted(self._plugins.items())]

    def status_of(self, node_type: str) -> str:
        """``active`` | ``disabled`` | ``unknown`` for a node type (§5.1).

        Lets workflow validation distinguish *"plugin 'x' is disabled —
        enable it in Settings"* from *"plugin not installed?"*: a node type
        is ``disabled`` when a discovered plugin declares it but the plugin
        is currently off, and ``unknown`` when no plugin declares it.
        """
        if node_type in self._nodes:
            return "active"
        for info in self._plugins.values():
            if node_type in info.node_types:
                return "disabled" if not info.enabled else "active"
        return "unknown"

    def plugin_name_of(self, node_type: str) -> str | None:
        """Name of the plugin providing ``node_type``, if known."""
        if node_type in self._plugin_of:
            return self._plugin_of[node_type]
        for name, info in self._plugins.items():
            if node_type in info.node_types:
                return name
        return None

    def plugin_version_of(self, node_type: str) -> str | None:
        """Distribution version of the plugin providing ``node_type``
        (issues.md M23) — the drift-warning comparison key. None for
        manual/test nodes registered outside any plugin."""
        name = self._plugin_of.get(node_type)
        if name is None:
            return None
        info = self._plugins.get(name)
        return info.version if info else None

    def rebuild(self, disabled: Iterable[str] = ()) -> list[PluginInfo]:
        """In-place hot reload (§5.1): re-discover plugins and rebuild the
        node table with the given set of disabled plugin names.

        Module imports are cached by Python, so re-running discovery is
        cheap; the node table is rebuilt into a scratch registry and
        swapped in atomically, so readers see either the old or the new
        ``_nodes`` table — never an empty or partial one. The plugin
        tables are updated in three separate assignments immediately
        after, so a concurrent ``/api/nodes`` read can briefly pair the
        new node table with the old plugin tables (harmless: manifests()
        only touches ``_nodes``; status_of during a race may misattribute
        a just-disabled type). Nodes added directly via :meth:`add`
        survive a rebuild (they are re-registered unless a plugin claims
        the same type — plugin nodes win on conflict).
        """
        scratch = Registry()
        scratch._manual = self._manual
        infos = discover_plugins(scratch, set(disabled))
        claimed = {t for info in infos for t in info.node_types}
        for node_type, node_cls in self._manual.items():
            if node_type not in claimed and node_type not in scratch._nodes:
                scratch.add(node_cls)
        self._nodes = scratch._nodes
        self._plugins = scratch._plugins
        self._plugin_of = scratch._plugin_of
        return infos


def _plugin_version(ep) -> str:
    """Best-effort plugin version from distribution metadata."""
    dist = getattr(ep, "dist", None)
    if dist is not None:
        try:
            return str(dist.version)
        except Exception:  # noqa: BLE001, S110 — version is best-effort metadata
            pass
    return ""


def _record_plugin_failure(
    registry: Registry, infos: list[PluginInfo], ep, bundled: bool, exc: Exception
) -> None:
    """Degrade one failed plugin (issues.md M20): log it, and record the
    PluginInfo BOTH in the discovery list and in the registry's plugin
    table — ``GET /api/plugins`` must show the error row (previously the
    info existed only in the local list and the API never saw it).
    """
    logger.warning(
        "plugin '%s' failed and was skipped: %s: %s",
        ep.name, type(exc).__name__, exc)
    info = PluginInfo(
        name=ep.name, version=_plugin_version(ep), bundled=bundled,
        node_types=[], enabled=False,
        error=f"{type(exc).__name__}: {exc}")
    infos.append(info)
    registry._plugins[ep.name] = info


def _rollback_partial(registry: Registry, before: set[str]) -> None:
    """Remove the nodes a failed register() already added (issues.md M20):
    a plugin is skipped whole — a half-registered plugin would expose
    nodes whose siblings vanished, and pollute the shared manual table
    across rebuilds."""
    added = set(registry._nodes) - before
    for node_type in added:
        registry._nodes.pop(node_type, None)
        registry._manual.pop(node_type, None)


def discover_plugins(
    registry: Registry, disabled: set[str] | None = None
) -> list[PluginInfo]:
    """Discover plugins advertising the ``emptyui.nodes`` entry-point group.

    Registers the enabled plugins' nodes into ``registry`` and records
    metadata for **every** discovered plugin — including disabled ones,
    whose node types are still learned from the register call (module
    imports are cached by Python, so this is cheap) so
    :meth:`Registry.status_of` can tell a disabled plugin from a missing
    one. Names in ``disabled`` that match no plugin are silently ignored
    (§5.1 requirement 4 — config files may travel between machines);
    bundled plugins (``register.bundled = True``) can never be disabled.
    """
    registry._plugins = {}
    registry._plugin_of = {}
    infos: list[PluginInfo] = []
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: e.name):
        # a broken third-party plugin (missing heavy dep, raising register)
        # must not take down the whole instance — skip it and record the
        # error; the plugin shows up disabled with a note in /api/plugins
        try:
            register = ep.load()
        except Exception as exc:  # noqa: BLE001 — any import failure degrades
            _record_plugin_failure(registry, infos, ep, False, exc)
            continue
        bundled = bool(getattr(register, "bundled", False))
        enabled = bundled or (disabled is None or ep.name not in disabled)
        if enabled:
            before = set(registry._nodes)
            try:
                register(registry)
            except ValueError as exc:
                # M20 (decision 2026-10-13): a duplicate node type degrades
                # like any other registration failure — one mis-packaged
                # wheel must not brick startup (nor make every toggle
                # 409). The partial adds are rolled back; the first
                # registration wins the type; the conflict is named in
                # the plugin's error row.
                _rollback_partial(registry, before)
                _record_plugin_failure(registry, infos, ep, bundled, exc)
                continue
            except Exception as exc:  # noqa: BLE001 — degrade, never crash startup
                _rollback_partial(registry, before)
                _record_plugin_failure(registry, infos, ep, bundled, exc)
                continue
            node_types = sorted(set(registry._nodes) - before)
        else:
            # learn the disabled plugin's node types in a scratch registry:
            # this must not mutate (or crash on type collisions with) the
            # live table — the nodes are popped below either way
            scratch = Registry()
            # same degradation as the enabled path: a disabled plugin that
            # crashes register() must not kill discovery (it would take
            # down startup AND make every toggle 500 — the scratch table
            # is a different registry, but its import side effects are not)
            try:
                register(scratch)
            except Exception as exc:  # noqa: BLE001 — degrade, never crash startup
                _record_plugin_failure(registry, infos, ep, False, exc)
                continue
            node_types = sorted(scratch._nodes)
        info = PluginInfo(
            name=ep.name,
            version=_plugin_version(ep),
            bundled=bundled,
            node_types=node_types,
            enabled=enabled,
        )
        registry._plugins[ep.name] = info
        if enabled:
            for t in node_types:
                registry._plugin_of[t] = ep.name
        infos.append(info)
    return infos


def build_default_registry(disabled_plugins: Iterable[str] = ()) -> Registry:
    registry = Registry()
    discover_plugins(registry, set(disabled_plugins))
    return registry
