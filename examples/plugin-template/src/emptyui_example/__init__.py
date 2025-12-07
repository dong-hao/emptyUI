"""emptyUI example plugin.

Copy this package as a starting point for your own plugin — it demonstrates
every convention from ``docs/plugin-guide.md``: entry-point registration,
namespaced node and port types, the manifest/tooltip protocol, functional
(immutable-input) execute implementations, and the required self-tests.
"""

from emptyui_example.plugin import register

__all__ = ["register"]
