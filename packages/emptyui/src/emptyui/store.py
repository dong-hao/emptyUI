"""Handle-based data store (docs/DESIGN.md §3).

Nodes exchange opaque handles; payloads live in this in-process store,
serialized to the MTH5-backed workspace (Compute FC's ``persist``) — the
M2 "arrives later" note is history: persistence shipped with the FC
database and per-run workspaces (docs/DESIGN.md §9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Handle:
    id: str
    type: str  # port type tag, e.g. "ts.run"
    value: Any = None


class DataStore:
    """Maps handle IDs to in-memory values. Executor-scoped."""

    def __init__(self) -> None:
        self._handles: dict[str, Handle] = {}
        self._counter = 0

    def put(self, type: str, value: Any) -> Handle:
        self._counter += 1
        handle = Handle(id=f"h{self._counter}", type=type, value=value)
        self._handles[handle.id] = handle
        return handle

    def get(self, handle_id: str) -> Handle:
        if handle_id not in self._handles:
            raise KeyError(f"unknown handle: {handle_id}")
        return self._handles[handle_id]
