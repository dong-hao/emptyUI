"""Head-to-head: the §24.3 refined SDG31 recipe through the NODE GRAPH
(the stock workflow a user builds in the UI, executed by the Executor)
vs the parity harness's low-level realization (recorded in §24.3).

Node chain (all stock nodes, no harness code in the loop):

  Load MTH5 (SDG31) ─▶ Extract Run (b) ─▶ Calibrate ───────────┐
  Load MTH5 (SDH28) ─▶ Merge Runs (b,c) ─▶ Calibrate ─▶ Align ─┴─▶ Compute FC ×2
      (window 128, boxcar, deep taper hanning from level 3,
       prewhitening first difference)
  ─▶ Coherence Sort (weight 0.7, apply below 20 s) ─▶ Estimate TF
      (Robust + remote reference, band_style=manual: official band
       centers + "6,6,6,6,4,2") ─▶ Rotate TF (−5.7°)

Recorded harness numbers (§24.3, low-level with exact-band sort):
head 0.034/0.080 (83/100), short 0.007/0.022, tail 0.010/0.036 (97/100),
reliable full band 0.009 median / 0.029 p90 / 98% within gate.

Expected (and measured here) node-graph deviations: the node Coherence
Sort bands with the l2u scheme (no band-table input) while the harness
sorted on the exact official table — slightly different per-window
weights at the head. Everything else is parameter-identical.

Run:  .venv/bin/python examples/data/nims-usarray-test/node_vs_harness_sdg31.py
"""

from __future__ import annotations

import json

import numpy as np
from emptyui.executor import Executor
from emptyui.registry import build_default_registry
from emptyui.schema import Edge, Workflow, WorkflowNode
from parity_sdg31 import summarize_split
from verify_sdg31 import (
    DATA_DIR,
    read_official_edi,
    tensor_compare,
)

RECIPE = {
    "decimation_levels": 5, "factor": 4, "window_samples": 128,
    "overlap": 0.5, "window_type": "boxcar", "window_type_deep": "hanning",
    "deep_from_level": 3, "prewhitening": "first difference", "persist": False,
}
SORT = {
    "coherence_min": 0.7, "coherence_max": 1.0, "mode": "weight",
    "apply_below_period": 20.0, "band_style": "l2u", "periods_per_octave": 3,
}
ROTATION_DEG = -5.7

# recorded harness numbers (§24.3 refined recipe, long_period_sdg31.py)
HARNESS = {
    "head": (0.034, 0.080, 83, 100),
    "short_period": (0.007, 0.022, 100, 100),
    "reliable_tail": (0.010, 0.036, 97, 100),
    "reliable_full": (0.009, 0.029, 98, 100),
}


def build_workflow(band_centers: str) -> Workflow:
    fc_params = dict(RECIPE)
    return Workflow(
        nodes=[
            WorkflowNode(id="load_local", type="mth5.io.load",
                         params={"path": str(DATA_DIR / "4P_SDG31.h5"),
                                 "mode": "r"}),
            WorkflowNode(id="load_remote", type="mth5.io.load",
                         params={"path": str(DATA_DIR / "4P_SDH28.h5"),
                                 "mode": "r"}),
            WorkflowNode(id="extract_local", type="mth5.io.extract_run",
                         params={"survey": "4P", "station": "SDG31",
                                 "selection": "run id", "run": "b"}),
            WorkflowNode(id="merge_remote", type="mth5.runs.merge",
                         params={"survey": "4P", "station": "SDH28",
                                 "runs": "b,c", "new_run_id": "bc"}),
            WorkflowNode(id="cal_local", type="mth5.calibrate"),
            WorkflowNode(id="cal_remote", type="mth5.calibrate"),
            WorkflowNode(id="align", type="mth5.runs.align"),
            WorkflowNode(id="fc_local", type="aurora.process.compute_fc",
                         params=fc_params),
            WorkflowNode(id="fc_remote", type="aurora.process.compute_fc",
                         params=fc_params),
            WorkflowNode(id="sort", type="aurora.process.coherence_sort",
                         params={**SORT}),
            WorkflowNode(id="est", type="aurora.process.estimate_tf",
                         params={"estimator_engine": "Robust",
                                 "remote_reference": True,
                                 "band_style": "manual",
                                 "band_centers": band_centers,
                                 "bands_per_level": "6,6,6,6,4,2"}),
            WorkflowNode(id="rot", type="mth5.tf.rotate",
                         params={"angle": ROTATION_DEG}),
        ],
        edges=[
            Edge(id="e1", source="load_local", source_port="mth5",
                 target="extract_local", target_port="mth5"),
            Edge(id="e1b", source="load_remote", source_port="mth5",
                 target="merge_remote", target_port="mth5"),
            Edge(id="e1c", source="extract_local", source_port="data",
                 target="cal_local", target_port="data"),
            Edge(id="e1d", source="merge_remote", source_port="merged_run",
                 target="cal_remote", target_port="data"),
            Edge(id="e3", source="cal_local", source_port="data",
                 target="align", target_port="data1"),
            Edge(id="e4", source="cal_remote", source_port="data",
                 target="align", target_port="data2"),
            Edge(id="e5", source="align", source_port="data1",
                 target="fc_local", target_port="data"),
            Edge(id="e6", source="align", source_port="data2",
                 target="fc_remote", target_port="data"),
            Edge(id="e7", source="fc_local", source_port="fc",
                 target="sort", target_port="fc"),
            Edge(id="e8", source="fc_remote", source_port="fc",
                 target="sort", target_port="ref_fc"),
            Edge(id="e9", source="sort", source_port="fc",
                 target="est", target_port="fc"),
            Edge(id="e10", source="sort", source_port="ref_fc",
                 target="est", target_port="ref_fc"),
            Edge(id="e11", source="est", source_port="tf",
                 target="rot", target_port="tf"),
        ],
    )


def main():
    periods_off, zo, _var_o = read_official_edi()
    freqs_off = 1.0 / np.asarray(periods_off)  # descending
    band_centers = ", ".join(f"{f:.8g}" for f in freqs_off)

    reg = build_default_registry()
    wf = build_workflow(band_centers)
    print("running the node graph through the Executor ...")
    summary = Executor(reg).run(wf, emit=lambda *a, **k: None,
                                include_store=True)
    assert summary["status"] == "completed", summary["status"]
    store = summary["store"]
    tf = store.get(summary["outputs"]["rot"]["tf"]).value
    arr = tf.dataset["transfer_function"]
    Z = arr.sel(output=["ex", "ey"], input=["hx", "hy"]).values
    periods = arr.coords["period"].values
    print(f"node TF: {len(periods)} periods "
          f"({periods.min():.1f} - {periods.max():.1f} s)")

    rows = tensor_compare(periods, Z, periods_off, zo)
    stats = summarize_split(rows, "node_graph")

    print(f"\n{'region':>14} | {'NODE med/p90':>13} {'pass r/p':>9} "
          f"| {'HARNESS med/p90':>15} {'pass r/p':>9}")
    print("-" * 72)
    for name in ("head", "short_period", "reliable_tail", "reliable_full"):
        s = stats[name]
        h = HARNESS[name]
        print(f"{name:>14} | {s['median_abs_log10_rho']:.3f}/"
              f"{s['p90_abs_log10_rho']:.3f}       "
              f"{100 * s['pass_rho']:>3.0f}/{100 * s['pass_phase']:<3.0f} | "
              f"{h[0]:.3f}/{h[1]:.3f}         {h[2]:>3.0f}/{h[3]:<3.0f}")

    worst = sorted(rows, key=lambda r: max(
        abs(r["delta_log10_rho_xy"]), abs(r["delta_log10_rho_yx"]),
        abs(r["delta_phi_deg_xy"]), abs(r["delta_phi_deg_yx"])),
        reverse=True)[:5]
    print("\nworst node-graph rows:")
    for r in worst:
        print(f"  T={r['period_official']:>9.1f}  d_rho "
              f"{r['delta_log10_rho_xy']:+.3f}/{r['delta_log10_rho_yx']:+.3f}  "
              f"d_phi {r['delta_phi_deg_xy']:+.1f}/{r['delta_phi_deg_yx']:+.1f}")

    out = {"node_stats": stats,
           "harness_reference": {k: dict(zip(
               ("median", "p90", "pass_rho_pct", "pass_phase_pct"), v))
               for k, v in HARNESS.items()},
           "rows": rows}
    path = DATA_DIR / "sdg31_node_vs_harness_metrics.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nmetrics -> {path}")


if __name__ == "__main__":
    main()
