"""Node declaration API.

Nodes subclass :class:`Node` and declare typed ports and parameters as class
attributes. The registry serializes these declarations into JSON manifests
that drive the frontend (palette, node rendering, properties panel).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class Input:
    """A typed input port."""

    name: str
    type: str = "any"
    required: bool = True
    accepts: tuple[str, ...] = ()
    description: str = ""
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "accepts": list(self.accepts),
            "description": self.description,
            "label": self.label or self.name,
        }


@dataclass(frozen=True)
class Output:
    """A typed output port."""

    name: str
    type: str = "any"
    description: str = ""
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "label": self.label or self.name,
        }


@dataclass(frozen=True)
class Param:
    """A parameter rendered by the frontend properties panel."""

    name: str
    widget: str  # number | integer | string | text | boolean | select |
    #               file | directory | code | list
    label: str = ""
    default: Any = None
    optional: bool = False
    options: tuple[Any, ...] = ()
    min: float | None = None
    max: float | None = None
    step: float | None = None
    description: str = ""
    #: Auto-change rules driven from the UI: when this param changes to a given
    #: value (key), set the listed (other_param, value) pairs. Keys are
    #: serialized as strings (bool -> "true"/"false") for JSON transport.
    cascade: dict[Any, tuple[tuple[str, Any], ...]] | None = None

    def _cascade_dict(self) -> dict[str, list[list[Any]]] | None:
        if not self.cascade:
            return None
        out: dict[str, list[list[Any]]] = {}
        for key, actions in self.cascade.items():
            if isinstance(key, bool):
                out["true" if key else "false"] = [list(pair) for pair in actions]
            else:
                out[str(key)] = [list(pair) for pair in actions]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "widget": self.widget,
            "label": self.label or self.name.replace("_", " ").title(),
            "default": self.default,
            "optional": self.optional,
            "options": list(self.options),
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "description": self.description,
            "cascade": self._cascade_dict(),
        }


class Node:
    """Base class for all node types.

    Subclasses declare ``type``, ports, params and implement ``execute``.
    """

    type: ClassVar[str]
    title: ClassVar[str] = ""
    description: ClassVar[str] = ""
    category: ClassVar[str] = "Processing"
    version: ClassVar[str] = "0.1.0"
    inputs: ClassVar[tuple[Input, ...]] = ()
    outputs: ClassVar[tuple[Output, ...]] = ()
    params: ClassVar[tuple[Param, ...]] = ()
    deps: ClassVar[tuple[str, ...]] = ()

    #: Params to auto-fill (as dropdowns) from an upstream node connected to a
    #: given input port. Maps input port name -> {target_param: upstream_param}
    #: (identity mapping = same name), e.g.
    #: ``{"data": {"survey": "survey", "station": "station", "run": "run"}}``.
    #: Resolved by the frontend via a recursive chain-walking resolver
    #: (DESIGN.md §4 "Chain-walking inheritance").
    inherit: ClassVar[dict[str, dict[str, str]]] = {}

    #: subclasses override; signature: execute(ctx, **inputs_and_params)
    execute: Callable[..., dict[str, Any]]

    def manifest(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "version": self.version,
            "title": self.title or self.type.rsplit(".", 1)[-1],
            "description": self.description,
            "category": self.category,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "params": [p.to_dict() for p in self.params],
            "deps": list(self.deps),
            "inherit": {k: dict(v) for k, v in self.inherit.items()},
        }


def port_type_compatible(source_type: str, target: Input) -> bool:
    """Connection rule from DESIGN.md §3: equal tags, declared accepts, or any."""
    if source_type == target.type:
        return True
    if target.type == "any" or source_type == "any":
        return True
    return source_type in target.accepts
