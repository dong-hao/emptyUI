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
    #: Rendering-only (DESIGN.md §4): the port is populated only when the
    #: named sibling param holds one of the listed values (same shape as
    #: Param.enabled_when) — Extract Run's dual outputs. The UI dims ports
    #: that will not be populated; the executor stores unpopulated outputs
    #: as None.
    populated_when: tuple[str, tuple[Any, ...]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "label": self.label or self.name,
            "populated_when": (
                {"param": self.populated_when[0],
                 "values": [str(v) for v in self.populated_when[1]]}
                if self.populated_when else None
            ),
        }


@dataclass(frozen=True)
class Param:
    """A parameter rendered by the frontend properties panel."""

    name: str
    widget: str  # number | integer | string | text | boolean | select |
    #               radio | file | directory | code | list
    label: str = ""
    default: Any = None
    optional: bool = False
    options: tuple[Any, ...] = ()
    min: float | None = None
    max: float | None = None
    step: float | None = None
    description: str = ""
    #: Placeholder text for text inputs (e.g. "auto (detect from data)").
    placeholder: str = ""
    #: Remote-mode path confinement (DESIGN.md §17.2): "" = not a path,
    #: "in" = read-only input resolved under the data root (validated),
    #: "out" = writable output resolved into the run workspace.
    path_kind: str = ""
    #: Auto-change rules driven from the UI: when this param changes to a given
    #: value (key), set the listed (other_param, value) pairs. Keys are
    #: serialized as strings (bool -> "true"/"false") for JSON transport.
    cascade: dict[Any, tuple[tuple[str, Any], ...]] | None = None
    #: Conditional-UX rule (DESIGN.md §4): the widget is editable only when
    #: the named sibling param currently holds one of the listed values,
    #: e.g. ``enabled_when=("estimator_engine", ("Robust",))``. Purely a
    #: rendering/UX rule — the executor never enforces it (nodes validate
    #: their own params).
    enabled_when: tuple[str, tuple[Any, ...]] | None = None
    #: Split inherited comma-separated candidates into individual combobox
    #: options (§4): an upstream ``output_channels="e1,e2,bz"`` offers e1 /
    #: e2 / bz as separate choices. Rendering-only.
    split_inherited: bool = False
    #: Time-selector bar binding (§23.4): the *input port name* whose
    #: payload summary (extent streamed via `payload` run events) drives the
    #: frontend time bar for this param pair. Rendering-only.
    time_bar: str = ""

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
            "placeholder": self.placeholder,
            "cascade": self._cascade_dict(),
            "path_kind": self.path_kind,
            "enabled_when": (
                {"param": self.enabled_when[0],
                 "values": [str(v) for v in self.enabled_when[1]]}
                if self.enabled_when else None
            ),
            "split_inherited": self.split_inherited,
            "time_bar": self.time_bar,
        }


def first_sentence(text: str, limit: int = 140) -> str:
    """First sentence of a description (tooltip fallback, DESIGN.md §4).

    Whitespace-normalized, cut at the first sentence terminator; a sentence
    with no terminator is hard-capped at ``limit`` chars.
    """
    text = " ".join((text or "").split())
    if not text:
        return ""
    for sep in (". ", "! ", "? "):
        i = text.find(sep)
        if i > 0:
            return text[: i + 1]
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Node:
    """Base class for all node types.

    Subclasses declare ``type``, ports, params and implement ``execute``.
    """

    type: ClassVar[str]
    title: ClassVar[str] = ""
    #: One-sentence hover tip (hover protocol, DESIGN.md §4): what the node
    #: does, in a single sentence. Empty = derived from ``description``.
    tooltip: ClassVar[str] = ""
    #: Full details for the right-hand properties panel (shown on click).
    description: ClassVar[str] = ""
    #: Utility/test nodes are excluded from the palette by default (still
    #: executable when present in a saved workflow).
    hidden: ClassVar[bool] = False
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

    #: New port types introduced by this node (plugin guide §"Port types").
    #: Core types (``ts.run``, ``fc.container``, ``tf``, …) need no entry;
    #: genuinely new types must be namespaced (``<org>.<type>``) and declare
    #: UI metadata so edges render readably:
    #: ``{"ml.model": {"color": "#8e44ad", "label": "Model"}}``.
    #: The registry collects these into the catalog for the frontend.
    port_types: ClassVar[dict[str, dict[str, str]]] = {}

    #: External model weights this node needs (plugin guide §"Models").
    #: Declare — do not bundle: binaries live in a configured model
    #: directory, never inside the plugin wheel. Each entry:
    #: ``{"id": "denoiser-cnn", "filename": "denoiser_v2.onnx", "sha256":
    #: "<…>", "url": "https://…", "license": "CC-BY-4.0"}``.
    models: ClassVar[list[dict[str, str]]] = []

    #: subclasses override; signature: execute(ctx, **inputs_and_params)
    execute: Callable[..., dict[str, Any]]

    def manifest(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "version": self.version,
            "title": self.title or self.type.rsplit(".", 1)[-1],
            # hover protocol: one sentence — explicit tooltip wins, else the
            # first sentence of the description (old plugins stay consistent)
            "tooltip": self.tooltip or first_sentence(self.description),
            "description": self.description,
            "category": self.category,
            "hidden": self.hidden,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "params": [p.to_dict() for p in self.params],
            "deps": list(self.deps),
            "inherit": {k: dict(v) for k, v in self.inherit.items()},
            # models declared but not bundled (plugin guide §"Models")
            "models": [dict(m) for m in self.models],
        }


def port_type_compatible(source_type: str, target: Input) -> bool:
    """Connection rule from DESIGN.md §3: equal tags, declared accepts, or any."""
    if source_type == target.type:
        return True
    if target.type == "any" or source_type == "any":
        return True
    return source_type in target.accepts
