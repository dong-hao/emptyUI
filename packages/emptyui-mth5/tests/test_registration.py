"""Every node class in this plugin must end up in the default registry.

Regression: RotateTF was fully implemented and tested but never added in
plugin.py, so the node was unreachable from the UI (docs/DESIGN.md §11 claims it
delivered). This guards the plugin wiring itself, not node behavior.
"""

from emptyui import Registry
from emptyui.registry import build_default_registry


def _plugin_nodes() -> set[str]:
    registry = build_default_registry()
    return {
        t for t in registry.types() if t.startswith(("mth5.",))
    }


def test_rotate_tf_is_registered():
    assert "mth5.tf.rotate" in _plugin_nodes()


def test_plugin_register_covers_all_declared_nodes():
    from emptyui_mth5 import plugin as mth5_plugin

    scratch = Registry()
    mth5_plugin.register(scratch)
    declared = set(scratch.types())
    assert declared == _plugin_nodes(), (
        "plugin.py register() and the built registry disagree: "
        f"{declared ^ _plugin_nodes()}"
    )
