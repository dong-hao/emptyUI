"""Visual review: render the TF plots on the real BGS RR workflow."""
import sys
from pathlib import Path

sys.path.insert(0, "packages/emptyui-mth5/src")
sys.path.insert(0, "packages/emptyui-aurora/src")
sys.path.insert(0, "packages/emptyui/src")

import tempfile

from emptyui.executor import Executor
from emptyui.registry import build_default_registry
from emptyui.schema import Edge, Workflow, WorkflowNode

DATA_DIR = Path("examples/data/bgs-lemi-424")
NO46_DIR = DATA_DIR / "lemi_txt" / "NO46"
NX37_DIR = DATA_DIR / "lemi_txt" / "NX37"


def node(nid, typ, params):
    return WorkflowNode(id=nid, type=typ, params=params, position={"x": 0, "y": 0})


def edge(eid, s, sp, t, tp):
    return Edge(id=eid, source=s, source_port=sp, target=t, target_port=tp)


def main():
    out_dir = Path(tempfile.mkdtemp(prefix="plotreview-"))
    workflow = Workflow(
        nodes=[
            node("l1", "mth5.io.lemi", {"data_path": str(NO46_DIR), "survey_id": "bgs",
                                        "station_id": "NO46", "mth5_filename": "no46.mth5",
                                        "save_path": str(out_dir), "channel_nomenclature": "lemi12"}),
            node("x1", "mth5.io.extract_run", {"survey": "bgs", "station": "NO46"}),
            node("l2", "mth5.io.lemi", {"data_path": str(NX37_DIR), "survey_id": "bgs",
                                        "station_id": "NX37", "mth5_filename": "nx37.mth5",
                                        "save_path": str(out_dir), "channel_nomenclature": "lemi12"}),
            node("x2", "mth5.io.extract_run", {"survey": "bgs", "station": "NX37"}),
            node("al", "mth5.runs.align", {}),
            node("c1", "aurora.process.compute_fc", {"input_channels": "bx,by",
                                                     "output_channels": "e1,e2,bz",
                                                     "channel_nomenclature": "lemi12",
                                                     "window_samples": 256, "decimation_levels": 2}),
            node("c2", "aurora.process.compute_fc", {"input_channels": "bx,by",
                                                     "output_channels": "e1,e2,bz",
                                                     "channel_nomenclature": "lemi12",
                                                     "window_samples": 256, "decimation_levels": 2}),
            node("cs", "aurora.process.coherence_sort", {"coherence_min": 0.25,
                                                         "coherence_max": 0.98,
                                                         "mode": "weight"}),
            node("t", "aurora.process.estimate_tf", {"estimator_engine": "Robust",
                                                     "remote_reference": True}),
            node("p1", "mth5.plot.appres_phase", {"mode": "off-diagonal"}),
            node("p2", "mth5.plot.tipper", {}),
            node("s1", "core.display.preview_plot", {}),
            node("s2", "core.display.preview_plot", {}),
        ],
        edges=[
            edge("e1", "l1", "mth5", "x1", "mth5"),
            edge("e2", "l2", "mth5", "x2", "mth5"),
            edge("e3", "x1", "data", "al", "data1"),
            edge("e4", "x2", "data", "al", "data2"),
            edge("e5", "al", "data1", "c1", "data"),
            edge("e6", "al", "data2", "c2", "data"),
            edge("e7", "c1", "fc", "cs", "fc"),
            edge("e8", "c2", "fc", "cs", "ref_fc"),
            edge("e9", "cs", "fc", "t", "fc"),
            edge("e10", "cs", "ref_fc", "t", "ref_fc"),
            edge("e11", "t", "tf", "p1", "tf"),
            edge("e12", "t", "tf", "p2", "tf"),
            edge("e13", "p1", "plot", "s1", "plot"),
            edge("e14", "p2", "plot", "s2", "plot"),
        ],
    )
    summary = Executor(build_default_registry()).run(
        workflow, emit=lambda e: print(e.get("event"), e.get("status", "")),
        workspace=str(out_dir))
    print("STATUS:", summary["status"])
    for p in sorted(Path(out_dir).glob("*.png")):
        print("PNG:", p.name, p.stat().st_size)
    return out_dir


if __name__ == "__main__":
    print(main())
