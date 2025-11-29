"""Node registry and plugin discovery via entry points."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from emptyui.api import Node

ENTRY_POINT_GROUP = "emptyui.nodes"


class Registry:
    """Holds all known node types; serializes manifests for the frontend."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}

    def add(self, node_cls: type[Node]) -> None:
        node = node_cls()
        if not getattr(node, "type", None):
            raise ValueError(f"{node_cls.__name__} does not define 'type'")
        if node.type in self._nodes:
            raise ValueError(f"duplicate node type: {node.type}")
        self._nodes[node.type] = node

    def get(self, node_type: str) -> Node | None:
        return self._nodes.get(node_type)

    def types(self) -> list[str]:
        return sorted(self._nodes)

    def manifests(self) -> dict[str, Any]:
        return {"nodes": [node.manifest() for node in self._nodes.values()]}


def find_plugins(registry: Registry) -> list[str]:
    """Discover plugins advertising the ``emptyui.nodes`` entry point group."""
    loaded: list[str] = []
    eps = entry_points(group=ENTRY_POINT_GROUP)
    for ep in eps:
        register = ep.load()
        register(registry)
        loaded.append(ep.name)
    return loaded


def build_default_registry() -> Registry:
    registry = Registry()
    find_plugins(registry)
    return registry
