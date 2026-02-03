"""Every node class in this plugin must end up in the default registry.

Regression: SelectRuns and ResidualSort were fully implemented and tested
but never added in plugin.py, so the nodes were unreachable from the UI
(DESIGN.md §24.13 claims both delivered). This guards the plugin wiring
itself, not node behavior — mirroring emptyui-mth5's test_registration.py.
"""

from emptyui import Registry
from emptyui.registry import build_default_registry


def _plugin_nodes() -> set[str]:
    registry = build_default_registry()
    return {
        t for t in registry.types() if t.startswith(("aurora.",))
    }


def test_select_runs_and_residual_sort_are_registered():
    assert "aurora.process.select_runs" in _plugin_nodes()
    assert "aurora.process.residual_sort" in _plugin_nodes()


def test_plugin_register_covers_all_declared_nodes():
    from emptyui_aurora import plugin as aurora_plugin

    scratch = Registry()
    aurora_plugin.register(scratch)
    declared = set(scratch.types())
    assert declared == _plugin_nodes(), (
        "plugin.py register() and the built registry disagree: "
        f"{declared ^ _plugin_nodes()}"
    )
