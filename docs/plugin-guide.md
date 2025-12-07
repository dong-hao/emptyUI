# emptyUI plugin guide

How to build, test, and publish emptyUI plugins. This guide is the
foundation of the ecosystem contract: core ships only the framework and
one bundled core-nodes plugin — everything domain-specific (mth5, aurora,
ML denoisers, …) arrives as an ordinary Python package.

> Copy **`examples/plugin-template/`** as your starting point; it
> demonstrates every convention below and its self-tests pass out of the
> box.

---

## 1. Philosophy

- **Entry points, no core edits.** Discovery uses Python entry points
  (`importlib.metadata`): your package advertises an `emptyui.nodes` entry
  point; emptyUI imports it at startup. No config files, no core
  registration lists.
- **Heavy deps are yours.** Each plugin declares its own dependencies as
  pip extras (`pip install emptyui-mth5[full]`), so users install only what
  they need. Core stays light (numpy/matplotlib-level).
- **Namespaces prevent collisions.** Node types and new port types are
  namespaced by org (`<org>.<domain>.<name>`). The ComfyUI ecosystem's
  biggest quality problem is unnamespaced node names colliding across
  thousands of packs — don't repeat it.
- **Manifests are the contract.** The frontend is fully manifest-driven
  (palette, inline params, properties panel); there is no plugin-side UI
  code. A well-written manifest *is* the documentation.
- **Functional nodes.** `execute` must never mutate its inputs — cached
  payloads (DESIGN.md §7) are handed to downstream nodes and later runs as
  the *same object instances*.

## 2. Anatomy of a plugin

```
emptyui-example-plugin/
├── pyproject.toml            # metadata + entry point + extras
├── README.md
├── src/emptyui_example/
│   ├── __init__.py
│   ├── plugin.py             # register(registry) — the integration point
│   └── nodes.py              # your Node subclasses
└── tests/
    └── test_nodes.py         # required self-tests (§8)
```

`pyproject.toml` — the essential parts:

```toml
[project]
name = "emptyui-example-plugin"
dependencies = ["numpy", "emptyui"]          # emptyui = framework only

[project.optional-dependencies]
dev = ["pytest", "ruff"]                      # heavy domain deps go here:
                                              # [full] = mth5==0.6.8, etc.

[project.entry-points."emptyui.nodes"]
example = "emptyui_example.plugin:register"
```

```python
# plugin.py — keep it trivial
def register(registry):
    registry.add(Gain)
    registry.add(RmsMeter)
```

`register` runs at app startup. Keep it side-effect-free: instantiate and
add classes, nothing else (no file I/O, no downloads).

## 3. Naming and namespacing

- Node types: `<org>.<domain>.<name>` — e.g. `example.ts.gain`,
  `simpeg.aurora.compute_fc`. The bundled plugin uses the reserved `core.`
  prefix; third-party plugins must not.
- Port types: **reuse the core types** where sensible — the ecosystem stays
  connectable only if producers and consumers share type tags. Introduce a
  new type only when the payload is genuinely different (see §5).

## 4. Node manifest protocol (DESIGN.md §4)

Every `Node` subclass declares class variables that become the manifest the
frontend renders:

| Field | Purpose |
|---|---|
| `type` | namespaced node type (§3) |
| `title` | palette display name |
| `tooltip` | **one sentence** — the hover protocol; empty = first sentence of `description` |
| `description` | full details for the properties panel (shown on click) |
| `category` | palette grouping; unknown categories are appended alphabetically |
| `hidden` | utility/test nodes excluded from the palette (still executable) |
| `version` | bump on behavior changes; workflows record versions for reproducibility |
| `deps` | human-readable dependency notes surfaced in the properties panel |
| `inherit` | params auto-filled (as dropdowns) from the upstream chain — see DESIGN §4 |

Ports: `Input(name, type, required=True, accepts=(), label="", description="")`,
`Output(name, type, ...)`. Params: `Param(name, widget, ...)` with the widget
vocabulary `number | integer | string | text | boolean | select | radio |
file | directory | code | list`, plus `options`, `min/max/step`,
`placeholder`, and `cascade` (value-driven auto-change rules, e.g. remote
reference forcing the robust estimator).

## 5. Port types

The type system is **open**: a port type is a string tag; connections are
validated by equality, declared `accepts`, or the `any` escape hatch
(`port_type_compatible`, DESIGN.md §3).

**Core types** (fixed names — reuse these whenever your payload matches):

| Tag | Payload |
|---|---|
| `ts.run` | multi-channel time series (duck-typed: `channels: dict[str, ndarray]`, `sample_rate`, `copy()`, metadata attrs) |
| `ts.channel` | single-channel series |
| `fc` | Fourier-coefficient container |
| `tf` | transfer-function estimate |
| `tf.level` | per-decimation-level TF |
| `mth5` | MTH5 file/archive handle |
| `calibration` | instrument calibration filters |
| `selection` / `decimation` / `config` / `bands` | metadata/config payloads |
| `plot` / `table` | display artifacts |
| `any` | escape hatch (avoid in new nodes) |

**Declaring a new type** (when your payload is genuinely different, e.g. an
ML model handle):

```python
class ModelLoader(Node):
    type = "example.ml.model_loader"
    ...
    outputs = (Output("model", "example.model"),)
    port_types = {"example.model": {"color": "#8e44ad", "label": "Model"}}
```

Rules:
- new types are **namespaced** (`<org>.<type>`) — the catalog collects every
  installed node's `port_types` and the frontend colors edges with them;
  undeclared types fall back to gray (readable but anonymous).
- document the payload's duck-typed attributes in the port `description` —
  consumers must not import your container classes (that would couple
  plugins); they duck-type the documented attributes, as the bundled
  `ts.run` payload does.

## 6. The functional execute contract

```python
def execute(self, ctx, **inputs_and_params) -> dict[str, Any]:
```

- **Never mutate inputs** (§7 immutability contract — cached payloads are
  shared): copy first, modify the copy, return new objects.
- Return `dict[port_name, value]` for every declared output.
- Errors: raise with an actionable message — a failed node halts the run and
  the message is surfaced verbatim on the node and in the run status. Fail
  loudly instead of silently degrading (this project's correctness policy).
- Progress: `ctx.progress(frac, "message")` for long operations.
- Artifacts: `ctx.publish_artifact(ctx.artifact_path("stem", ".png"), "image")`
  — filenames are node-instance-unique by construction; publish files, never
  return open handles.
- **Determinism**: given identical inputs and params, execution must produce
  equivalent outputs — results are cached across runs keyed on structure
  (§7). Expose randomness (e.g. seeds) as params instead of hiding it.

## 7. Self-tests (required)

Every plugin ships a pytest suite — this is the ecosystem's trust contract
(the ComfyUI ecosystem's recurring quality problems trace back to packs
without tests). The template demonstrates the pattern:

- registry-level: `register(reg)` into a fresh `Registry`, assert manifests
  (namespaced types, non-empty `tooltip`, declared `port_types`);
- node-level: call `execute` directly with duck-typed payloads; assert
  **inputs are not mutated**;
- interop-level: wire your node to a fake upstream producer and a capturing
  sink, run through `Executor`, assert the data flow.

Run from the package root: `pip install -e ".[dev]" && pytest`. CI (yours)
should run this on every push; a plugin whose tests do not pass should not
be published.

## 8. Models: declare, don't bundle

Model weights (e.g. CNN noise-suppression networks) are **never bundled**
inside the plugin wheel — the same rule ComfyUI's ecosystem converged on.
Declare the requirement in the manifest; binaries live in a configured model
directory (`EMPTYUI_MODELS_PATH`), platform-provisioned in embedded remote
mode (DESIGN.md §17's read-only data root):

```python
class Denoiser(Node):
    type = "example.ml.denoiser"
    ...
    models = [{
        "id": "denoiser-cnn",
        "filename": "denoiser_v2.onnx",
        "sha256": "<checksum of the exact file>",
        "url": "https://example.org/models/denoiser_v2.onnx",
        "license": "CC-BY-4.0",
    }]
```

Rules:
- `sha256` is mandatory — results must be reproducible and tamper-evident.
- `execute` fails with an actionable error when a declared model is missing
  at its expected path (name the setting and the expected filename); never
  download silently at run time.
- Small lookup tables/config files that are part of your source code may
  ship in the wheel; anything with trained weights does not.

## 9. Versioning & reproducibility

- Bump `version` whenever behavior changes; manifests surface it in the
  properties panel and the workflow document records it.
- Changing a node's semantics without a version bump silently invalidates
  reproducibility for saved workflows — treat `version` as part of the
  cache signature (it is: DESIGN.md §7 keys include it).

## 10. Security & quality guidelines

- `execute` runs on a worker thread inside the user's process — treat input
  payloads as untrusted structure (duck-type, don't assume).
- No network access, subprocesses, or filesystem writes outside the
  workspace without declaring it in the node `description`; remote/embedded
  deployments (§17) confine file access to the data root — path parameters
  must resolve inside it.
- No `eval`/`exec` of workflow-borne strings; no secrets in params (they are
  stored in plain workflow JSON).
- Lint (`ruff`) and keep the manifest honest: `hidden` for test nodes,
  tooltips that actually fit one sentence.

## 11. Publishing & distribution

- Publish to PyPI as an ordinary package — entry points make installation
  `pip install emptyui-<yourplugin>` with no registry or index required.
- Bundled plugins shipped in this repository (`emptyui-core-nodes`,
  `emptyui-mth5`, `emptyui-aurora`) mark their register function with
  `register.bundled = True`; third-party plugins never set this — it keeps
  them toggleable in the Settings dialog (DESIGN.md §5.1) while bundled
  plugins stay always-on.
- Pin heavy domain deps exactly (e.g. `mth5==0.6.8`): emptyUI nodes wrap
  third-party APIs that move.
- A plugin index/publish flow (marketplace-style) is an M4 item; PyPI plus
  the guide's conventions are the MVP distribution path.
- Embedded/remote deployments (DESIGN.md §17) decide plugin availability at
  the platform level — `EMPTYUI_DISABLED_PLUGINS` sets the baseline there;
  runtime toggling persists to the settings file (§5.1).

## 12. Publishing checklist

- [ ] package renamed, entry point points to your `register`
- [ ] node types namespaced `<org>.<domain>.<name>`; no `core.` prefix
- [ ] new port types namespaced + declared in `port_types` with metadata
- [ ] manifests follow §4: one-sentence `tooltip`, full `description`,
      sensible `category`, `hidden` on test nodes
- [ ] `execute` copies inputs; raises actionable errors; declares progress
- [ ] models declared (id/filename/sha256/license), not bundled
- [ ] `pip install -e ".[dev]" && pytest` green
- [ ] `ruff check` clean
- [ ] version bumped if replacing an existing published release
