"""Plugin enable/disable settings persistence (DESIGN.md §5.1).

The ``EMPTYUI_DISABLED_PLUGINS`` environment variable is the platform-level
baseline (env wins at startup; in embedded mode §17 the platform decides
plugin availability). Runtime UI toggles are **in-memory only** for now —
no settings file is created unless an explicit ``EMPTYUI_SETTINGS_PATH``
(or ``create_app(settings_path=...)``) is configured; the config-file
persistence layer is deferred until the dedicated configure/env file lands.
Unknown plugin names are silently ignored (requirement 4: config files may
travel between machines).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

KEY_DISABLED = "disabled_plugins"


@dataclass
class SettingsStore:
    """Plugin toggle state (DESIGN.md §5.1).

    With ``path=None`` (the default) the store is **memory-only**: toggles
    live for the server process lifetime and nothing is written to disk.
    When a path is configured (env ``EMPTYUI_SETTINGS_PATH`` or the
    ``create_app`` kwarg), the state is additionally persisted as a JSON
    file (atomic write) so toggles survive restarts. A missing or corrupt
    file is silently ignored, as are unknown names.
    """

    path: Path | None = None
    _disabled: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self.load()

    def load(self) -> None:
        """Read the settings file; a missing/corrupt file is ignored."""
        self._disabled = set()
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        names = data.get(KEY_DISABLED, [])
        if isinstance(names, list):
            self._disabled = {str(n) for n in names if isinstance(n, str)}

    def save(self) -> None:
        """Atomically persist the current disabled set."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({KEY_DISABLED: sorted(self._disabled)}, indent=2)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".settings-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def disabled(self) -> set[str]:
        """Plugin names currently disabled (memory state)."""
        return set(self._disabled)

    def set_disabled(self, name: str, disabled: bool) -> None:
        """Toggle one plugin (UI-driven changes only).

        Persists to the settings file when a path is configured; otherwise
        the change stays in memory for the process lifetime.
        """
        if disabled:
            self._disabled.add(name)
        else:
            self._disabled.discard(name)
        self.save()

    def effective_disabled(self, env_baseline: set[str]) -> set[str]:
        """Environment baseline wins at startup (§5.1): a plugin disabled by
        the platform stays disabled even when the settings file enables it.
        Unknown names are carried along harmlessly — discovery ignores them.
        """
        return set(env_baseline) | self._disabled
