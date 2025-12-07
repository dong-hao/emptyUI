"""Plugin registration — the only integration point with emptyUI core.

Discovery is via the ``emptyui.nodes`` entry-point group (pyproject.toml);
no core edits, no config files. Keep ``register`` trivial: instantiate and
add node classes.
"""

from emptyui.registry import Registry

from emptyui_example.nodes import Gain, RmsMeter


def register(registry: Registry) -> None:
    registry.add(Gain)
    registry.add(RmsMeter)
