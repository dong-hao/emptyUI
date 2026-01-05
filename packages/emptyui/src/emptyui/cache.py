"""Node-result cache shared across runs (DESIGN.md §7).

Server-lifetime, in-memory LRU. Entries are keyed by the executor's
structural node signature — ``(type, version, effective params, upstream
signatures)`` — so an unchanged upstream automatically yields unchanged
downstream keys (ComfyUI-style structural invalidation; data payloads are
never hashed).

**Immutability contract**: cached output values are handed to downstream
nodes and later runs as the *same object instances* (no defensive copies —
they would defeat the purpose for large arrays). Nodes must therefore be
functional: never mutate inputs or previously returned outputs.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """One node's cached result.

    ``outputs`` maps output-port name → value (the same instances the
    original run put into its store). ``artifacts`` records published files
    so a cache hit can restore them into a new run's workspace:
    ``src``/``kind``/``suffix`` identify the original file, ``stem`` is the
    artifact filename without the ``_<node_id>`` marker (renamed per
    restoring node instance), and ``port`` is the output port whose value
    *is* the artifact dict (display nodes return it as their output so
    viewers can resolve the receiving input edge).
    """

    outputs: dict[str, Any]
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    created: float = field(default_factory=time.time)


class NodeCache:
    """Thread-safe LRU of :class:`CacheEntry` keyed by structural signature."""

    def __init__(self, max_entries: int = 256) -> None:
        self._max = max(1, max_entries)
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def resize(self, max_entries: int) -> int:
        """Change the LRU bound live (§22 settings); returns evicted count."""
        self._max = max(1, int(max_entries))
        with self._lock:
            evicted = 0
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
                evicted += 1
        return evicted

    @property
    def max_entries(self) -> int:
        return self._max

    def clear(self) -> int:
        """Drop all entries; returns how many were evicted."""
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
