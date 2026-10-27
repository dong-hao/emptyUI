"""Output-path defaults shared by the archive loaders (docs/issues.md M29).

The loaders accept an optional ``save_path``. An explicit value wins; an
empty one resolves to a dedicated subfolder next to the input data — never
the data directory itself, where a re-run could overwrite a source archive.
Remote mode resolves output params through ``PathPolicy`` before the node
sees them, so this is the local/headless default.
"""

from __future__ import annotations

from pathlib import Path

#: Subfolder an empty ``save_path`` resolves to (M29).
DEFAULT_OUTPUT_DIRNAME = "emptyui_output"


def resolve_output_dir(save_path: str | Path | None,
                       data_path: str | Path) -> Path:
    """Return (and create) the directory the archive should be written to.

    ``save_path`` wins when set. When empty, the result is
    ``<data_path>/emptyui_output`` — a persistent, discoverable folder that
    keeps the input tree's own files untouched. Parents are created so the
    caller can write immediately; a path that exists as a file raises.
    """
    out = (Path(save_path).expanduser().resolve() if save_path
           else Path(data_path).expanduser().resolve() / DEFAULT_OUTPUT_DIRNAME)
    out.mkdir(parents=True, exist_ok=True)
    return out
