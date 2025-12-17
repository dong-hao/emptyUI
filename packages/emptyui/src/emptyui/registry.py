"""Node registry and plugin discovery via entry points."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from emptyui.api import Node

ENTRY_POINT_GROUP = "emptyui.nodes"


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "node_count": len(self.node_types),
            "bundled": self.bundled,
            "enabled": self.enabled,
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

    def types(self) -> list[str]:
        return sorted(self._nodes)

    def manifests(self) -> dict[str, Any]:
        # port_types: UI metadata for every port type declared by any node
        # (plugin guide §"Port types") — the frontend colors edges with it,
        # falling back to gray for undeclared types
        port_types: dict[str, dict[str, str]] = {}
        for node in self._nodes.values():
            for ptype, meta in getattr(node, "port_types", {}).items():
                port_types.setdefault(ptype, dict(meta))
        return {"nodes": [node.manifest() for node in self._nodes.values()],
                "port_types": port_types}

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

    def rebuild(self, disabled: Iterable[str] = ()) -> list[PluginInfo]:
        """In-place hot reload (§5.1): re-discover plugins and rebuild the
        node table with the given set of disabled plugin names.

        Module imports are cached by Python, so re-running discovery is
        cheap; the node table is rebuilt in place so the catalog served to
        the frontend reflects the toggle immediately. Nodes added directly
        via :meth:`add` survive a rebuild (they are re-registered unless a
        plugin claims the same type — plugin nodes win on conflict).
        """
        self._nodes = {}
        self._plugins = {}
        self._plugin_of = {}
        infos = discover_plugins(self, set(disabled))
        claimed = {t for info in infos for t in info.node_types}
        for node_type, node_cls in self._manual.items():
            if node_type not in claimed and node_type not in self._nodes:
                self.add(node_cls)
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
        register = ep.load()
        bundled = bool(getattr(register, "bundled", False))
        enabled = bundled or (disabled is None or ep.name not in disabled)
        before = set(registry._nodes)
        register(registry)
        node_types = sorted(set(registry._nodes) - before)
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
        else:
            for t in node_types:
                registry._nodes.pop(t, None)
        infos.append(info)
    return infos


def find_plugins(registry: Registry) -> list[str]:
    """Discover plugins advertising the ``emptyui.nodes`` entry point group.

    Backward-compatible wrapper returning the loaded entry-point names
    (all plugins enabled). See :func:`discover_plugins` for plugin
    metadata and enable/disable support.
    """
    return [info.name for info in discover_plugins(registry)]


def build_default_registry(disabled_plugins: Iterable[str] = ()) -> Registry:
    registry = Registry()
    discover_plugins(registry, set(disabled_plugins))
    return registry
