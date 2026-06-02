"""H7 — node warnings must reach the UI, not die in the server log.

Loaders that degrade (a missing Phoenix calibration, a weak remote
coherence, an interpolated gap) warn via ``ctx.logger`` — invisible to a
browser session. ``ctx.warn`` emits a ``node_warning`` run event the same
way ``ctx.progress`` emits ``node_progress``; the frontend collects them
per node and renders them on the node card and in the properties panel.
"""

from __future__ import annotations

import logging

from emptyui.executor import Executor
from emptyui.registry import Registry


def _registry():
    from emptyui import Node, Output

    reg = Registry()

    class Warner(Node):
        type = "test.warner"
        outputs = (Output("data", "any"),)

        def execute(self, ctx):
            ctx.warn("no box calibration (*_CLB.asc) — shape correction "
                     "missing")
            ctx.warn("no box calibration (*_CLB.asc) — shape correction "
                     "missing")  # deduped by the frontend, not here
            ctx.warn("no coil calibration for hx")
            return {"data": "done"}

    reg.add(Warner)
    return reg


def _wf():
    from emptyui.schema import Workflow, WorkflowNode

    return Workflow(nodes=[WorkflowNode(id="n1", type="test.warner")])


def test_ctx_warn_emits_node_warning_events():
    events: list[dict] = []
    summary = Executor(_registry()).run(_wf(), emit=events.append)
    assert summary["status"] == "completed"
    warnings = [e for e in events if e["event"] == "node_warning"]
    assert len(warnings) == 3  # the backend emits raw; the UI dedupes
    assert all(w["node"] == "n1" for w in warnings)
    assert warnings[0]["msg"] == warnings[1]["msg"]  # the duplicate rides
    assert warnings[2]["msg"] == "no coil calibration for hx"


def test_ctx_warn_also_logs(caplog):
    """The warning keeps its server-log presence (belt and braces — the
    log outlives the run's event replay)."""
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        Executor(_registry()).run(_wf(), emit=lambda e: None)
    assert any("no box calibration" in r.getMessage()
               for r in caplog.records)


def test_ctx_warn_without_callback_stays_log_only(caplog):
    """Headless embedding constructs RunContext without a warn_cb — warn
    degrades to the logger (never a crash)."""
    from emptyui.context import RunContext

    ctx = RunContext(workspace=None, store=None, node_id="n1")
    with caplog.at_level(logging.WARNING, logger="emptyui"):
        ctx.warn("headless warning")
    assert any("headless warning" in r.getMessage()
               for r in caplog.records)
