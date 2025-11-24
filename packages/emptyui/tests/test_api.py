from typing import ClassVar

from emptyui import Input, Node, Output, Param
from emptyui.api import port_type_compatible


def test_manifest_roundtrip():
    class Dummy(Node):
        type = "test.dummy"
        title = "Dummy"
        inputs = (Input("data", "ts.run"),)
        outputs = (Output("out", "ts.run"),)
        params = (Param("order", "integer", default=4, min=1, max=10),)

    m = Dummy().manifest()
    assert m["type"] == "test.dummy"
    assert m["inputs"][0]["type"] == "ts.run"
    assert m["params"][0]["default"] == 4
    assert m["title"] == "Dummy"


def test_manifest_cascade_and_inherit():
    class Dummy(Node):
        type = "test.linked"
        params = (
            Param("unit", "radio", default="h",
                  cascade={"h": (("input_channels", "hx,hy"),
                                 ("output_channels", "ex,ey")),
                           "b": (("input_channels", "bx,by"),
                                 ("output_channels", "e1,e2"))}),
            Param("remote", "boolean", cascade={True: (("engine", "Robust"),)}),
        )
        inherit: ClassVar[dict[str, dict[str, str]]] = {
            "data": {"survey": "survey", "station": "station", "run": "run"},
        }

    m = Dummy().manifest()
    # bool cascade keys normalize to "true"/"false" for JSON transport
    assert m["params"][1]["cascade"] == {"true": [["engine", "Robust"]]}
    assert m["params"][0]["cascade"]["b"] == [["input_channels", "bx,by"],
                                              ["output_channels", "e1,e2"]]
    # inherit serializes as {port: {target: upstream}} (name mapping)
    assert m["inherit"] == {"data": {"survey": "survey", "station": "station", "run": "run"}}


def test_port_compatibility_rules():
    from emptyui.api import Input

    assert port_type_compatible("ts.run", Input("a", "ts.run"))
    assert port_type_compatible("any", Input("a", "ts.run"))
    assert port_type_compatible("ts.run", Input("a", "any"))
    assert not port_type_compatible("spectra", Input("a", "ts.run"))
    assert port_type_compatible(
        "ts.channel", Input("a", "ts.run", accepts=("ts.channel",))
    )
