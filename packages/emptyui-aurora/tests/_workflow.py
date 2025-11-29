"""Workflow-building helpers for executor-based tests."""

from __future__ import annotations

from emptyui.schema import Edge, WorkflowNode


def node(nid, typ, params):
    return WorkflowNode(id=nid, type=typ, params=params,
                        position={"x": 0, "y": 0})


def edge(eid, s, sp, t, tp):
    return Edge(id=eid, source=s, source_port=sp, target=t, target_port=tp)
