"""Shared param coercion — the single copy used by the executor AND schema
validation (issues.md M1/M5).

Workflows are JSON-borne and older saved documents (or free-text combobox
entries) can carry numbers as strings. ``coerce_param`` turns a raw workflow
param value into what node code should receive, per the manifest's declared
widget. Validation (``schema.validate_workflow``) applies the *same*
function before its min/max/options checks, so the trust boundary and the
executor can never disagree about what a value means.
"""

from __future__ import annotations

from typing import Any


def coerce_param(param, value: Any) -> Any:
    """Coerce a workflow param value to the manifest's declared type.

    number/integer params coerce numeric strings (empty string -> None);
    boolean params coerce "true"/"false"-style strings (L-C1: the raw
    string would reach node code truthy); select params whose declared
    options are numeric are matched back by their string form. Unknown
    formats raise a clean ValueError naming the param.
    """
    if value is None:
        return None
    widget = getattr(param, "widget", "")
    if widget in ("number", "integer") and isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        try:
            f = float(s)
        except ValueError:
            raise ValueError(
                f"param '{param.name}': '{value}' is not a number"
            ) from None
        return round(f) if widget == "integer" else f
    if widget == "boolean" and isinstance(value, str):
        # L-C1: a JSON workflow can carry "false" as a STRING (older
        # documents, free-text editing) — the raw string reaches node
        # code truthy ('false' is truthy!). Empty string = unchecked.
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
        raise ValueError(
            f"param '{param.name}': '{value}' is not a boolean")
    if widget == "select" and isinstance(value, str):
        opts = getattr(param, "options", ()) or ()
        if opts and all(
            isinstance(o, (int, float)) and not isinstance(o, bool)
            for o in opts
        ):
            for o in opts:
                if str(o) == value.strip():
                    return o
    return value
