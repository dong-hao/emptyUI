"""API contract between the backend and the handwritten frontend types.

DESIGN.md §2: the frontend API types are handwritten
(``packages/web/src/api/client.ts``) — the node manifest is plugin-open
(unknown fields are ignored downstream), so catalog types stay deliberately
loose. This test pins the server-side *shapes* the frontend depends on:
removals/renames of these keys would break the UI, so they fail loudly here.
Additions are tolerated (the frontend ignores unknown fields — the same
"files may travel" philosophy as the settings file).
"""

from fastapi.testclient import TestClient
import pytest

from emptyui import Input, Node, Output, Param
from emptyui.registry import PluginInfo, Registry
from emptyui.server.app import create_app


MANIFEST_KEYS = {
    "type", "version", "title", "tooltip", "description", "category",
    "hidden", "inputs", "outputs", "params", "deps", "inherit", "models",
}
PORT_KEYS = {"name", "type", "description", "label"}
INPUT_KEYS = PORT_KEYS | {"required", "accepts"}
PARAM_KEYS = {
    "name", "widget", "label", "default", "optional", "options", "min",
    "max", "step", "description", "placeholder", "cascade", "path_kind",
    "enabled_when", "split_inherited", "time_bar",
}
PLUGIN_KEYS = {
    "name", "version", "node_count", "bundled", "enabled", "error",
}
FILES_KEYS = {"path", "parent", "entries"}
ENTRY_KEYS = {"name", "path", "is_dir"}


def _registry() -> Registry:
    """A node exercising every manifest surface the frontend reads."""

    class Full(Node):
        type = "test.full"
        version = "1.2.3"
        title = "Full Surface"
        description = "Exercises every manifest field. Second sentence here."
        category = "Processing"
        hidden = True
        inputs = (
            Input("data", "ts.run", accepts=("ts.run", "any"),
                  description="in port"),
            Input("aux", "any", required=False),
        )
        outputs = (Output("out", "plot", description="figure"),)
        params = (
            Param("path", "file", path_kind="in"),
            Param("mode", "select", options=("a", "b"),
                  cascade={"a": (("path", ""),)}),
            Param("gate", "boolean", default=False,
                  enabled_when=("mode", ("a",))),
            Param("n", "integer", default=1, min=0, max=10, step=1),
            Param("ids", "string", split_inherited=True),
            Param("span", "string", time_bar="data"),
            Param("note", "text", placeholder="…", optional=True),
        )
        inherit = {"data": {"survey": "survey"}}
        deps = ("numpy>=1.24",)
        port_types = {"test.kind": {"color": "#123456", "label": "Test"}}

        def execute(self, ctx, **kw):
            return {"out": {"kind": "image", "path": ""}}

    reg = Registry()
    reg.add(Full)
    return reg


@pytest.fixture()
def client():
    return TestClient(create_app(_registry()))


def _manifest_of(client: TestClient, node_type: str) -> dict:
    catalog = client.get("/api/nodes").json()
    return next(m for m in catalog["nodes"] if m["type"] == node_type)


def test_nodes_catalog_shape(client: TestClient):
    catalog = client.get("/api/nodes").json()
    assert set(catalog) == {"nodes", "port_types"}
    manifest = _manifest_of(client, "test.full")
    assert MANIFEST_KEYS <= set(manifest)
    assert manifest["version"] == "1.2.3"
    # tooltip fallback: first sentence of the description
    assert manifest["tooltip"] == "Exercises every manifest field."
    for port in manifest["inputs"]:
        assert INPUT_KEYS <= set(port)
    for port in manifest["outputs"]:
        assert PORT_KEYS <= set(port)
    for param in manifest["params"]:
        assert PARAM_KEYS <= set(param), param["name"]
    by_name = {p["name"]: p for p in manifest["params"]}
    assert by_name["mode"]["cascade"] == {"a": [["path", ""]]}
    assert by_name["gate"]["enabled_when"] == {"param": "mode", "values": ["a"]}
    assert by_name["ids"]["split_inherited"] is True
    assert by_name["span"]["time_bar"] == "data"
    assert by_name["path"]["path_kind"] == "in"
    assert manifest["inherit"] == {"data": {"survey": "survey"}}


def test_port_types_metadata_shape(client: TestClient):
    catalog = client.get("/api/nodes").json()
    assert catalog["port_types"]["test.kind"] == {
        "color": "#123456", "label": "Test"}


def test_plugins_shape():
    info = PluginInfo(name="demo", version="1.0", bundled=False,
                      node_types=["demo.a"])
    assert PLUGIN_KEYS <= set(info.to_dict())


def test_files_shape(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "data.txt").write_text("x")
    app = create_app(data_root=tmp_path)
    body = TestClient(app).get("/api/files", params={"path": str(tmp_path)}).json()
    assert FILES_KEYS <= set(body)
    assert body["path"] == str(tmp_path)
    entry_keys = {ENTRY_KEYS <= set(e) for e in body["entries"]}
    assert entry_keys == {True}


def test_settings_shape(client: TestClient):
    view = client.get("/api/settings").json()
    assert set(view) == {"live", "deployment", "planned"}
    for knob in view["live"].values():
        assert {"value", "type", "default", "description", "editable",
                "source"} <= set(knob)
    assert {"data_root", "serve_dist", "cors_origins", "root_path"} <= set(
        view["deployment"])
    for knob in view["planned"].values():
        assert {"value", "type", "default", "description"} <= set(knob)


def test_health_shape(client: TestClient):
    assert client.get("/api/health").json() == {"status": "ok"}
