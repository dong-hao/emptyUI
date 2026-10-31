# emptyUI example plugin

Copy this package as the starting point for your own emptyUI plugin — it
demonstrates the core conventions from [`docs/plugin-guide.md`](../../docs/plugin-guide.md):

- **entry-point registration** (`emptyui.nodes` group — no core edits)
- **namespaced node types** (`example.ts.gain`, `example.demo.rms`)
- **new port types** with UI metadata (`example.signal`)
- the **§4 manifest protocol** (one-sentence tooltip + full description)
- **functional execute** (inputs copied, never mutated — cached payloads are
  shared across runs)
- the **required self-tests** (`tests/`)

## Try it

```bash
pip install -e ".[dev]"
pytest                      # self-tests must pass
# start emptyUI — the plugin is installed and appears in ⚙ Settings, but
# both demo nodes are palette-hidden (`hidden = True`, L30); use them in a
# saved workflow or the self-tests
```

License: Apache-2.0 (matching the template `pyproject.toml`).

## Make it yours

1. rename the package (`src/emptyui_example/`) and the PyPI name
2. change the entry point in `pyproject.toml` to your module
3. replace the example nodes; keep node types namespaced `<org>.<domain>.<name>`
4. keep `pytest` green — self-tests are the ecosystem's trust contract
