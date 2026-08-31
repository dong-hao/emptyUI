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
    #: One-port convention (§4): this input accepts a run collection
    #: (ts.runs) arriving on the static port — the node dispatches on the
    #: payload shape. The frontend's wiring check consults the source
    #: port's runtime payload summary: a collection may only be wired into
    #: collection_ok inputs. Rendering-only.
    collection_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "accepts": list(self.accepts),
            "description": self.description,
            "label": self.label or self.name,
            "collection_ok": self.collection_ok,
        }


@dataclass(frozen=True)
class Output:
    """A typed output port."""

    name: str
    type: str = "any"
    description: str = ""
    label: str = ""
    #: Rendering-only (docs/DESIGN.md §4): the port is populated only when the
    #: named sibling param holds one of the listed values (same shape as
    #: Param.enabled_when). The UI dims ports that will not be populated;
    #: the executor stores unpopulated outputs as None. Currently dormant:
    #: Extract Run's former dual outputs were collapsed into the single
    #: collection-carrying `data` output; kept for mutually exclusive
    #: port variants (output or input — ComfyUI-style "one detector input
    #: at a time"), where the sibling param selects which port is active.
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
    #: Remote-mode path confinement (docs/DESIGN.md §11.2): "" = not a path,
    #: "in" = read-only input resolved under the data root (validated),
    #: "out" = writable output resolved into the run workspace.
    path_kind: str = ""
    #: Auto-change rules driven from the UI: when this param changes to a given
    #: value (key), set the listed (other_param, value) pairs. Keys are
    #: serialized as strings (bool -> "true"/"false") for JSON transport.
    cascade: dict[Any, tuple[tuple[str, Any], ...]] | None = None
    #: Conditional-UX rule (docs/DESIGN.md §4): the widget is editable only when
    #: the named sibling param currently holds one of the listed values,
    #: e.g. ``enabled_when=("estimator_engine", ("Robust",))``. Purely a
    #: rendering/UX rule — the executor never enforces it (nodes validate
    #: their own params).
    enabled_when: tuple[str, tuple[Any, ...]] | None = None
    #: Split inherited comma-separated candidates into individual combobox
    #: options (§4): an upstream ``output_channels="e1,e2,bz"`` offers e1 /
    #: e2 / bz as separate choices. Rendering-only.
    split_inherited: bool = False
    #: Time-selector bar binding (history §23.4): the *input port name* whose
    #: payload summary (extent streamed via `payload` run events) drives the
    #: frontend time bar for this param pair. Rendering-only.
    time_bar: str = ""
    #: Select-option suggestions (history §20.1): the *input port name* whose
    #: payload summary offers the option list (e.g. an fc database's
    #: sample-rate tiers streamed via `payload` run events). The frontend
    #: extends the static options with the summary's values; rendering-only.
    suggest_from: str = ""

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
            "suggest_from": self.suggest_from or None,
        }


#: Tooltip hard cap (docs/DESIGN.md §4) — a sentence with no terminator stops here.
_TOOLTIP_CAP = 140


def first_sentence(text: str) -> str:
    """First sentence of a description (tooltip fallback, docs/DESIGN.md §4).

    Whitespace-normalized, cut at the first sentence terminator; a sentence
    with no terminator is hard-capped at :data:`_TOOLTIP_CAP` chars. (The
    ``limit`` parameter was never used by a caller — dead parameter.)
    """
    text = " ".join((text or "").split())
    if not text:
        return ""
    for sep in (". ", "! ", "? "):
        i = text.find(sep)
        if i > 0:
            return text[: i + 1]
    return (
        text if len(text) <= _TOOLTIP_CAP
        else text[: _TOOLTIP_CAP - 1] + "…"
    )


class Node:
    """Base class for all node types.

    Subclasses declare ``type``, ports, params and implement ``execute``.
    """

    type: ClassVar[str]
    title: ClassVar[str] = ""
    #: One-sentence hover tip (hover protocol, docs/DESIGN.md §4): what the node
    #: does, in a single sentence. Empty = derived from ``description``.
    tooltip: ClassVar[str] = ""
    #: Full details for the right-hand properties panel (shown on click).
    description: ClassVar[str] = ""
    #: Utility/test nodes are excluded from the palette by default (still
    #: executable when present in a saved workflow).
    hidden: ClassVar[bool] = False
    #: Inline canvas-body renderer the frontend mounts inside the node card
    #: ("plot" | "table"); None = the standard port rows. The frontend
    #: switches on this manifest flag — never on node-type literals
    #: (L-F12, plugin guide §"Manifest fields").
    viewer: ClassVar[str | None] = None
    #: M13: "this run produces something the user can see" — a manifest-
    #: declared flag the frontend tests instead of the category string
    #: (never a category used as semantics). Display/viewer nodes declare
    #: ``terminal = True``; a node with no outputs counts too.
    terminal: ClassVar[bool] = False
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
    #: (docs/DESIGN.md §4 "Chain-walking inheritance").
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

    #: Whether structural cache hits may skip this node's execution
    #: (docs/DESIGN.md §7.1). Save/export nodes are ``False``: their side effect IS writing
    #: the file — a cache hit that skips it reports "done (cached)" while the
    #: file is never (re-)written (and a deleted destination stays gone).
    #: Cheap to re-run: the inputs come from the cache either way.
    cacheable: ClassVar[bool] = True

    #: The node closes/mutates its inputs (e.g. Save MTH5 closing the open
    #: archive in place). After it executes, the executor evicts the cache
    #: entries of the direct-input producers — the poisoned payload must not
    #: be served to later structurally-identical runs (§7.1 immutability).
    invalidates_inputs: ClassVar[bool] = False

    #: Per-node watchdog budget in seconds (docs/issues.md M2, docs/DESIGN.md §7.1).
    #: The server's ``node_timeout_seconds`` setting is the default budget
    #: every node gets; a node may declare its own: ``None`` (default)
    #: inherits the knob, ``0`` opts out entirely (this node knows its
    #: workload is legitimately long), ``> 0`` is this node's own budget.
    #: Heavy raw-data/FC nodes declare a generous 3600 so genuine long
    #: computations survive the 600 s default.
    timeout: ClassVar[float | None] = None

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
            "viewer": self.viewer,
            "terminal": self.terminal,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "params": [p.to_dict() for p in self.params],
            "deps": list(self.deps),
            "inherit": {k: dict(v) for k, v in self.inherit.items()},
            "cacheable": self.cacheable,
            "invalidates_inputs": self.invalidates_inputs,
            "timeout": self.timeout,
            # models declared but not bundled (plugin guide §"Models")
            "models": [dict(m) for m in self.models],
        }


def port_type_compatible(source_type: str, target: Input) -> bool:
    """Connection rule from docs/DESIGN.md §3: equal tags, declared accepts, or any."""
    if source_type == target.type:
        return True
    if target.type == "any" or source_type == "any":
        return True
    return source_type in target.accepts
