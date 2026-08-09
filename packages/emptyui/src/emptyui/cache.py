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

import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

#: Recursion guard for :func:`_approx_size` — the payloads the executor
#: caches nest at most a few levels (RunTS -> dataset -> variables).
_APPROX_DEPTH = 6

#: A size floor per leaf so container overhead is not undercounted.
_APPROX_FLOOR = 64


def _probe(value: Any, name: str) -> Any:
    """getattr that survives the mt_timeseries quirk: its ``__getattr__``
    converts a missing-attribute AttributeError into NameError, which
    defeats getattr's default (the landmine documented at the executor's
    cleanup — a RunTS payload probed for ``nbytes`` would otherwise fail
    the whole node, caught by the pipeline e2e on 2026-10-14)."""
    try:
        return getattr(value, name, None)
    except (AttributeError, NameError):
        return None


def _approx_size(value: Any, depth: int = 0) -> int:
    """Approximate deep size of a cached payload in bytes (issues.md L5).

    The budget is a bound, not an accounting system: numpy/xarray payloads
    dominate, and their exact ``nbytes`` is what matters — scalars and
    small strings round up to a floor. Duck-typed so the estimator never
    needs a plugin's classes: ``nbytes`` (numpy/xarray), ``dataset``
    (RunTS-ish wrappers), ``variables`` (xarray datasets), pandas
    ``memory_usage``, and plain containers recurse. Attribute probes go
    through :func:`_probe` — payloads with hostile ``__getattr__``s (the
    mt_timeseries NameError quirk) size to their floor instead of raising.
    """
    if depth > _APPROX_DEPTH:
        return 0
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return max(_APPROX_FLOOR, sys.getsizeof(value))
    nbytes = _probe(value, "nbytes")
    if isinstance(nbytes, int):
        return nbytes
    dataset = _probe(value, "dataset")  # RunTS-ish wrapper
    if dataset is not None:
        return _approx_size(dataset, depth + 1)
    variables = _probe(value, "variables")  # xarray dataset
    if variables is not None:
        return sum(_approx_size(v, depth + 1) for v in variables.values())
    if hasattr(value, "memory_usage"):  # pandas
        return int(value.memory_usage(index=False).sum())
    if isinstance(value, dict):
        return sum(_approx_size(v, depth + 1) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_approx_size(v, depth + 1) for v in value)
    return max(_APPROX_FLOOR, sys.getsizeof(value))


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
    """Thread-safe LRU of :class:`CacheEntry` keyed by structural signature.

    Bounded two ways (issues.md L5): by entry *count* (``max_entries``)
    and — when ``max_bytes`` is set — by the approximate deep size of the
    cached payloads. The byte accounting is computed once per ``put`` via
    :func:`_approx_size`; it bounds the memory, it does not audit it.
    """

    def __init__(self, max_entries: int = 256,
                 max_bytes: int | None = None) -> None:
        self._max = max(1, max_entries)
        self._max_bytes = max_bytes
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._bytes = 0
        self._lock = threading.Lock()

    def _pop_oldest_locked(self) -> None:
        key, _ = self._entries.popitem(last=False)
        self._bytes -= self._sizes.pop(key, 0)

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        size = sum(_approx_size(v) for v in (entry.outputs or {}).values())
        size += sum(_approx_size(a) for a in entry.artifacts)
        with self._lock:
            old = self._sizes.pop(key, 0)
            self._entries[key] = entry
            self._sizes[key] = size
            self._bytes += size - old
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._pop_oldest_locked()
            # the byte budget: the newest entry always survives (evicting
            # it would empty the cache for a payload we just computed)
            while (
                self._max_bytes is not None
                and self._bytes > self._max_bytes
                and len(self._entries) > 1
            ):
                self._pop_oldest_locked()

    def set_budget(self, max_bytes: int | None) -> int:
        """Change the byte bound live (history §22 settings); returns evicted count.

        Unlike ``put`` (which never evicts the just-stored entry), a live
        shrink enforces the new bound against EVERYTHING — the user asked
        for the budget; a 32 MiB payload under a 1 MiB budget goes.
        """
        self._max_bytes = max_bytes
        with self._lock:
            evicted = 0
            while (
                self._max_bytes is not None
                and self._bytes > self._max_bytes
                and self._entries
            ):
                self._pop_oldest_locked()
                evicted += 1
        return evicted

    @property
    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes

    def resize(self, max_entries: int) -> int:
        """Change the LRU bound live (history §22 settings); returns evicted count."""
        self._max = max(1, int(max_entries))
        with self._lock:
            evicted = 0
            while len(self._entries) > self._max:
                self._pop_oldest_locked()
                evicted += 1
        return evicted

    def drop(self, *keys: str) -> int:
        """Evict specific entries by key (§7.1 input invalidation: a node
        that closed/mutated its inputs poisons the producers' entries).
        Returns how many entries were removed."""
        n = 0
        with self._lock:
            for key in keys:
                if self._entries.pop(key, None) is not None:
                    self._bytes -= self._sizes.pop(key, 0)
                    n += 1
        return n

    def drop_referencing(self, predicate) -> int:
        """Evict every entry where at least one output value satisfies
        ``predicate(value)`` (M-C8: the TTL GC drops entries whose cached
        payloads hold open files inside a workspace it is about to
        reclaim). Returns how many entries were removed."""
        with self._lock:
            doomed = [
                key for key, entry in self._entries.items()
                if any(predicate(v) for v in (entry.outputs or {}).values())
            ]
            for key in doomed:
                self._entries.pop(key, None)
                self._bytes -= self._sizes.pop(key, 0)
            return len(doomed)

    def cached_values(self) -> list:
        """All values currently held by cache entries (identity sweep for
        the failed-run handle cleanup — never close a cached payload)."""
        with self._lock:
            return [
                v for entry in self._entries.values()
                for v in (entry.outputs or {}).values()
            ]

    def clear(self) -> int:
        """Drop all entries; returns how many entries were removed."""
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            self._sizes.clear()
            self._bytes = 0
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
