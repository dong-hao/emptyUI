"""Runtime settings persistence (DESIGN.md §5.1, §22).

The ``EMPTYUI_DISABLED_PLUGINS`` environment variable is the platform-level
plugin baseline (env wins at startup; in embedded mode §17 the platform
decides plugin availability). All runtime settings follow the same
precedence: **env/platform baseline wins**, UI edits are recorded on top.

UI edits are **in-memory only** unless ``EMPTYUI_SETTINGS_PATH`` (or
``create_app(settings_path=...)``) configures a JSON settings file (atomic
write, sectioned: ``disabled_plugins`` + ``settings``); no file is created
unless configured. Unknown names/keys are ignored when loading (config
files may travel between machines). Deployment-level knobs (data root,
serve dist, CORS, root path) are deliberately NOT UI-editable — they shape
confinement/serving semantics and change only via env + restart (§22).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

KEY_DISABLED = "disabled_plugins"
KEY_SETTINGS = "settings"

#: Live-editable runtime knobs (whitelist for PUT /api/settings, §22):
#: name -> (type, default, description). Values may be None (= "off").
LIVE_KNOBS: dict[str, tuple[type, object, str]] = {
    "workspace_ttl_seconds": (
        float, None,
        ("Remove completed run workspaces after this many seconds "
         "(0/None = keep forever)"),
    ),
    "max_concurrent_runs": (
        int, None,
        "Maximum queued/running runs per process (0/None = unlimited)",
    ),
    "cache_max_entries": (
        int, 256,
        "Cross-run in-memory node-result cache size (entries)",
    ),
}

#: Knobs reserved for the postponed §21 disk cache — defined in the schema
#: so the UI/API scaffolding is ready, without any effect yet.
PLANNED_KNOBS: dict[str, tuple[type, object, str]] = {
    "project_dir": (
        str, None,
        ("Project directory: stable, relocatable state home "
         "(.emptyui-cache) — reserved for the §21 disk cache"),
    ),
    "cache_max_gb": (
        float, 10.0, "Disk-cache size budget in GB — reserved (§21)",
    ),
    "cache_ttl_days": (
        float, 30.0, "Disk-cache entry expiry in days — reserved (§21)",
    ),
}


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
    _settings: dict[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.load()

    def load(self) -> None:
        """Read the settings file; a missing/corrupt file is ignored."""
        self._disabled = set()
        self._settings: dict[str, object] = {}
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        names = data.get(KEY_DISABLED, [])
        if isinstance(names, list):
            self._disabled = {str(n) for n in names if isinstance(n, str)}
        section = data.get(KEY_SETTINGS, {})
        if isinstance(section, dict):
            # whitelist: unknown keys are ignored (files may travel)
            known = set(LIVE_KNOBS) | set(PLANNED_KNOBS)
            self._settings = {
                k: v for k, v in section.items() if k in known
            }

    def save(self) -> None:
        """Atomically persist the current disabled set + settings."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                KEY_DISABLED: sorted(self._disabled),
                KEY_SETTINGS: self._settings,
            },
            indent=2,
        )
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

    # --- runtime settings section (§22) -----------------------------------

    def ui_settings(self) -> dict[str, object]:
        """UI/file-recorded settings (may be overridden by the env baseline)."""
        return dict(getattr(self, "_settings", {}))

    def set_settings(self, values: dict[str, object]) -> None:
        """Merge UI-edited settings and persist (when a path is configured)."""
        self._settings.update(values)
        self.save()

    def validate_live_values(self, values: dict[str, object]) -> dict[str, object]:
        """Validate a PUT payload against the LIVE_KNOBS whitelist.

        Returns the coerced values; raises ValueError naming the offending
        key on unknown keys, wrong types or out-of-range values.
        """
        out: dict[str, object] = {}
        for key, value in values.items():
            spec = LIVE_KNOBS.get(key)
            if spec is None:
                raise ValueError(
                    f"unknown or not live-editable setting: '{key}'")
            vtype, default, _desc = spec
            if value is None:
                # §22: null = "off" only for knobs whose default is None
                # (off by design); a knob with a non-null default (e.g.
                # cache_max_entries = 256) does not accept null
                if default is not None:
                    raise ValueError(
                        f"setting '{key}' does not accept null "
                        f"(default {default!r}; 0 or a number to bound it)")
                out[key] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                # surface as HTTP 422 (bad request), not a TypeError
                raise ValueError(  # noqa: TRY004
                    f"setting '{key}' expects a number (or null)")
            value = float(value)
            if value < 0:
                raise ValueError(f"setting '{key}' must be >= 0")
            out[key] = round(value) if vtype is int else value
        return out


@dataclass
class RuntimeSettings:
    """Live-editable runtime knobs (DESIGN.md §22) — a mutable holder so UI
    changes take effect without a restart (the app closures read it).

    Initialized from the effective start values (kwarg → env → settings
    file → default); `env_baselined` lists knobs whose start value arrived
    from the environment/platform — those are not UI-editable (platform
    wins, §5.1 precedent).
    """

    workspace_ttl_seconds: float | None = None
    max_concurrent_runs: int | None = None
    cache_max_entries: int = 256
    env_baselined: frozenset[str] = frozenset()
