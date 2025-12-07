"""Example nodes — the conventions of docs/plugin-guide.md in practice.

Conventions demonstrated here:
- node types are namespaced ``<org>.<domain>.<name>`` ("example" here)
- port types: reuse the core types (``ts.run`` …) where sensible; genuinely
  new types are namespaced and declare UI metadata via ``port_types``
- the §4 manifest protocol: one-sentence ``tooltip`` + full ``description``
- execute is functional: never mutate inputs (cached payloads are shared
  across runs, DESIGN.md §7 immutability contract) — always copy
"""

from __future__ import annotations

from typing import Any, ClassVar

from emptyui import Input, Node, Output, Param


class Gain(Node):
    """Scale every channel of a time series by a constant factor.

    Reuses the core ``ts.run`` port type; the payload is duck-typed (the
    documented interchange attributes ``channels``/``sample_rate``) so this
    plugin does not depend on any bundled plugin's internals.
    """

    type = "example.ts.gain"
    title = "Gain"
    tooltip = "Scales every channel of a time series by a constant factor."
    description = (
        "Multiplies every channel of the input time series by the gain "
        "factor. The input is copied, never mutated (cached payloads are "
        "shared across runs). Demonstrates reuse of the core ts.run port "
        "type with a duck-typed payload."
    )
    category = "Example"
    version = "0.1.0"
    inputs = (Input("data", "ts.run"),)
    outputs = (Output("scaled", "ts.run"),)
    params = (Param("gain", "number", default=2.0),)

    def execute(self, ctx, data, gain: float = 2.0) -> dict[str, Any]:
        out = data.copy()  # functional: copy, then modify the copy
        out.channels = {name: values * gain for name, values in out.channels.items()}
        return {"scaled": out}


class RmsMeter(Node):
    """Per-channel RMS summary of a time series.

    Introduces a NEW port type, ``example.signal`` — namespaced by this
    plugin's org and declared with UI metadata (color/label) so the
    frontend renders its edges readably (plugin guide §"Port types").
    """

    type = "example.demo.rms"
    title = "RMS Meter"
    tooltip = "Computes the per-channel RMS of a time series."
    description = (
        "Computes the root-mean-square of every channel. Outputs the "
        "example.signal type introduced by this plugin — new port types "
        "must be namespaced and declared via port_types with UI metadata."
    )
    category = "Example"
    version = "0.1.0"
    inputs = (Input("data", "ts.run"),)
    outputs = (Output("signal", "example.signal"),)
    params = ()
    port_types: ClassVar[dict[str, dict[str, str]]] = {
        "example.signal": {"color": "#16a085", "label": "Example signal"},
    }

    def execute(self, ctx, data) -> dict[str, Any]:
        import numpy as np

        rms = {
            name: float(np.sqrt(np.mean(np.asarray(values) ** 2)))
            for name, values in data.channels.items()
        }
        return {"signal": {"rms": rms, "sample_rate": data.sample_rate}}

    # Models are DECLARED, never bundled (plugin guide §"Models"): binaries
    # live in a configured model directory (platform-provisioned in embedded
    # mode); the manifest records id/filename/checksum/license. A node that
    # needs weights would look like:
    #
    #   models = [{
    #       "id": "denoiser-cnn",
    #       "filename": "denoiser_v2.onnx",
    #       "sha256": "<checksum of the exact file>",
    #       "url": "https://example.org/models/denoiser_v2.onnx",
    #       "license": "CC-BY-4.0",
    #   }]
