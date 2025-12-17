"""aurora pipeline nodes: ComputeFC (TS→FC), EstimateTF (FC→TF), MergeTFLevels.

ComputeFC/EstimateTF split the aurora pipeline at the FC boundary (DESIGN.md
§3, §14). FC is passed in-memory between nodes and, when ``persist`` is on,
also written into the workspace MTH5's FC layer so it can be reused / merged.
Estimators are OLS / robust RME / remote-reference RME_RR (aurora-backed).
"""

from __future__ import annotations

from typing import Any, ClassVar

from emptyui import Input, Node, Output, Param
from emptyui_mth5.nomenclature import (
    KEYWORDS,
    classify_channels,
)

from ._processing import (
    build_tf,
    coherence_sort_fcs,
    compute_fc_levels,
    estimate_tf_levels,
    merge_tf_levels,
)

_NOMENCLATURE_OPTIONS = ("auto",) + KEYWORDS


def _station_location(data):
    """(latitude, longitude, elevation) of the run's station, or (None,)*3."""
    try:
        loc = data.station_metadata.location
        return float(loc.latitude), float(loc.longitude), float(loc.elevation)
    except Exception:  # noqa: BLE001 — core TimeSeries carries no location
        return None, None, None


def _station_identity_attrs(data, survey: str, station: str, run: str) -> dict[str, Any]:
    """Primitive station-identity attrs to carry on an fc (DESIGN.md §15).

    The ts.run payload (mth5 RunTS) exposes survey/station/run metadata and the
    station location; only primitives are stored (xarray attrs must survive
    persistence), and missing values are omitted so downstream nodes can tell
    "unknown" apart from a default.
    """
    attrs: dict[str, Any] = {
        "survey_id": survey,
        "station_id": station,
        "run_id": run,
    }
    lat, lon, elev = _station_location(data)
    if lat is not None:
        attrs["station_latitude"] = lat
        attrs["station_longitude"] = lon
        attrs["station_elevation"] = elev
    return attrs

#: Nomenclature presets (actual channel names, mt_metadata CHANNEL_MAPS).
#: Each preset includes the vertical magnetic channel (bz/h3) so the tipper is
#: computed by default; ComputeFC drops it when the data has no vertical
#: channel (best-effort extension, see _processing._resolve_io_channels).
_NOMENCLATURE_PRESETS: dict[str, tuple[tuple[str, str], ...]] = {
    "lemi12": (("input_channels", "bx,by"), ("output_channels", "e1,e2,bz")),
    "lemi34": (("input_channels", "bx,by"), ("output_channels", "e3,e4,bz")),
    "musgraves": (("input_channels", "bx,by"), ("output_channels", "ex,ey,bz")),
    "phoenix123": (("input_channels", "h1,h2"), ("output_channels", "e1,e2,h3")),
    "default": (("input_channels", "hx,hy"), ("output_channels", "ex,ey,hz")),
}


class ComputeFC(Node):
    type = "aurora.process.compute_fc"
    title = "Compute FC"
    tooltip = "Windowed STFT: time series to Fourier coefficients over cascading decimation levels."
    description = (
        "Windowed STFT to Fourier coefficients (fc) over N cascading "
        "decimation levels (constant window in samples; sample rate halves by "
        "``factor`` per level). Specify input (horizontal magnetic) and output "
        "channel component names — outputs may be the electric pair (ex,ey) "
        "and/or a vertical magnetic channel (bz) for a geomagnetic transfer "
        "function (tipper), e.g. inputs bx,by / outputs e1,e2,bz. Empty input/"
        "output channels auto-detect from the data and its nomenclature — the "
        "vertical magnetic channel is included by default when present (tipper "
        "by default; impedance-only or tipper-only FCs are derived from what "
        "the run actually contains). The channel "
        "nomenclature (mt_metadata CHANNEL_MAPS: lemi12/lemi34/...) decides "
        "which names mean what — magnetic B vs H is derived from it, no manual "
        "switch needed. Persists each level's FC into the workspace MTH5 when "
        "an absolute start time is available."
    )
    category = "Processing"
    inputs = (Input("data", "ts.run", label="time series"),)
    outputs = (Output("fc", "fc"),)
    params = (
        Param("channel_nomenclature", "select", label="Channel nomenclature",
              options=_NOMENCLATURE_OPTIONS, default="auto",
              description="mt_metadata CHANNEL_MAPS keyword: maps standard "
                          "components (ex/ey/hx/hy/hz) to the instrument's "
                          "actual channel names and derives magnetic units",
              cascade={k: list(v) for k, v in _NOMENCLATURE_PRESETS.items()}),
        Param("input_channels", "string", label="Input channels", default="",
              placeholder="auto (detect from data)",
              description="Two horizontal magnetic channel names (e.g. 'bx,by' "
                          "or 'h1,h2'). Empty = resolve from the nomenclature "
                          "and the data (mt_metadata CHANNEL_MAPS)."),
        Param("output_channels", "string", label="Output channels", default="",
              placeholder="auto (detect from data)",
              description="Electric pair and/or vertical magnetic channel "
                          "(e.g. 'e1,e2' or 'e1,e2,bz' for a tipper). Empty = "
                          "resolve from the nomenclature and the data: the "
                          "electric pair plus the vertical magnetic channel "
                          "when present; a vertical-only run yields a "
                          "tipper-only FC."),
        Param("magnetic_unit", "select", label="Magnetic input",
              default="auto", options=("auto", "h", "b"),
              description="auto (default) = derived from the channel "
                          "nomenclature / names (bx/by → fluxgate B, hx/hy → "
                          "coil H); h/b override for legacy workflows"),
        Param("e_dipole_length_m", "number", label="E dipole length (m)", default=None,
              optional=True,
              description="LEMI stores E as mV potential; divide by dipole length to get V/m"),
        Param("decimation_levels", "integer", label="Decimation levels", default=0,
              min=0,
              description="Number of cascading decimations (0 = none, native rate only)"),
        Param("factor", "integer", label="Decimation factor", default=4, min=2),
        Param("window_samples", "integer", label="Window (samples)", default=256, min=8,
              description="Constant window length in samples across all levels"),
        Param("overlap", "number", default=0.5),
        Param("window_type", "select", options=("hamming", "hanning"), default="hamming"),
        Param("persist", "boolean", label="Persist to workspace", default=True,
              description="Write each level's FC into the workspace MTH5 FC layer"),
        Param("survey", "string", label="Survey id", default="", optional=True),
        Param("station", "string", label="Station id", default="", optional=True),
        Param("run", "string", label="Run id", default="", optional=True),
    )
    inherit: ClassVar[dict[str, dict[str, str]]] = {
        "data": {"survey": "survey", "station": "station", "run": "run",
                 "channel_nomenclature": "channel_nomenclature"},
    }

    def execute(self, ctx, data, input_channels: str = "",
                output_channels: str = "", channel_nomenclature: str = "auto",
                magnetic_unit: str = "auto",
                e_dipole_length_m=None,
                decimation_levels: int = 0, factor: int = 4,
                window_samples: int = 256,
                overlap: float = 0.5, window_type: str = "hamming",
                persist: bool = True, survey: str = "", station: str = "",
                run: str = "") -> dict[str, Any]:
        # Channel names resolve inside compute_fc_levels (empty fields = auto
        # from nomenclature + data; tipper included when the vertical channel
        # is present, missing outputs dropped with a warning).
        levels = compute_fc_levels(
            data, input_channels, output_channels,
            num_decimations=decimation_levels, factor=factor,
            num_samples_window=window_samples, overlap=overlap, window_type=window_type,
            magnetic_unit=magnetic_unit,
            e_dipole_length_m=e_dipole_length_m,
            channel_nomenclature=channel_nomenclature,
        )
        from emptyui_mth5.persist import persist_fc_level, runts_context

        # Station identity travels with the fc (attrs) so EstimateTF can attach
        # it to the TF — the single point of truth for plot titles and export
        # headers (DESIGN.md §15).
        surv, stat, run_id = runts_context(data, survey, station, run)
        identity = _station_identity_attrs(data, surv, stat, run_id)
        for level, fc in levels.items():
            fc.attrs.update(identity)
        if persist:
            for level, fc in levels.items():
                if "datetime64" in str(fc.time.dtype):
                    persist_fc_level(ctx, fc, level, fc.attrs["sample_rate"],
                                     survey=surv, station=stat, run=run_id)
                else:
                    ctx.logger.warning(
                        f"FC level {level} has no absolute (datetime64) time axis — "
                        "skipping persistence."
                    )
                    break
        summary = ", ".join(
            f"L{lv}:{fc.sizes['frequency']}f×{fc.sizes['time']}w" for lv, fc in levels.items()
        )
        ctx.progress(1.0, f"computed FC levels [{summary}]")
        return {"fc": levels}


def _fc_io_channels(fc):
    """(input_channels, output_channels, keyword) from an fc's attrs.

    Falls back to classifying the channel names through the nomenclature when
    the attrs are absent (e.g. an fc read back from an mth5 archive).
    """
    kw = fc.attrs.get("channel_nomenclature") or None
    ins_raw = fc.attrs.get("input_channels")
    outs_raw = fc.attrs.get("output_channels")
    if ins_raw and outs_raw:
        ins = [c.strip() for c in ins_raw.split(",") if c.strip()]
        outs = [c.strip() for c in outs_raw.split(",") if c.strip()]
    else:
        ins, outs, _ = classify_channels([str(c) for c in fc.data_vars], kw)
    return ins, outs, kw


class CoherenceSort(Node):
    type = "aurora.process.coherence_sort"
    title = "Coherence Sort"
    tooltip = "Down-weight or drop low-coherence FC windows before TF estimation."
    description = (
        "EMTF-style coherence sort: down-weight (or drop) Fourier-coefficient "
        "windows whose per-window multiple coherence falls outside "
        "[coherence_min, coherence_max] — low coherence = noise-dominated "
        "windows; coherence ≈ 1 can be an overfit/coherent-noise signature. "
        "With a remote 'ref fc', the criterion is the conservative minimum of "
        "the E–remote-H signal coherence and the local↔remote magnetic "
        "transfer coherence (remote reference replaces the local magnetic "
        "channels, so local E/H coherence is the wrong measure), and the "
        "single shared weight is applied to both FCs (exact weighted-LS "
        "semantics for RME_RR). Semantics note: the criterion is a "
        "per-window multiple coherence over the band's frequency-bin "
        "ensemble (a conservative broadband QC heuristic), not EMTF's "
        "window-ensemble coherence at each frequency — a true EMTF-style "
        "variant is a planned M4 item. Optional band-edge trimming removes "
        "bands whose retained windows are still below the lower bound."
    )
    category = "Processing"
    inputs = (
        Input("fc", "fc"),
        Input("ref_fc", "fc", label="ref fc (remote)", required=False),
    )
    outputs = (
        Output("fc", "fc"),
        Output("ref_fc", "fc", label="ref fc (remote)"),
        Output("table", "table", label="coherence diagnostics"),
    )
    params = (
        Param("coherence_min", "number", label="Coherence min",
              default=0.25, min=0.0, max=1.0,
              description="Lower coherence bound (per-window coherence²). "
                          "Windows below are down-weighted/dropped."),
        Param("coherence_max", "number", label="Coherence max",
              default=0.98, min=0.0, max=1.0,
              description="Upper coherence bound; coherence ≈ 1 can be an "
                          "overfit/coherent-noise signature."),
        Param("mode", "select", label="Mode",
              options=("weight", "drop", "off"), default="weight",
              description="weight = smooth down-weighting outside the bounds "
                          "(rows scaled by √w); drop = exclude windows "
                          "outside the bounds (NaN); off = no sorting."),
        Param("coherence_trim", "boolean", label="Band-edge trimming",
              default=False,
              description="Remove bands whose retained-window median "
                          "coherence is below the lower bound (NaN)."),
        Param("periods_per_octave", "integer", label="Periods per octave",
              default=4, min=1,
              description="Band geometry — must match the 'Estimate TF' "
                          "setting for the same banding."),
        Param("band_style", "select", label="Band style",
              options=("log", "index", "emtf"), default="log",
              description="Band construction mode — must match 'Estimate TF' "
                          "for consistent banding. 'log' (default) is the "
                          "legacy continuous log-period edge partition; "
                          "'index' is the discrete, index-snapped partition "
                          "on a shared global grid (DESIGN.md §14.6); "
                          "'emtf' is the disjoint EMTF-style table — every "
                          "band assigned to exactly one decimation level "
                          "(DESIGN.md §14.7)."),
        Param("band_owner", "select", label="Band owner (emtf)",
              options=("deepest", "shallowest"), default="deepest",
              description="Only used with band style 'emtf': which decimation "
                          "level evaluates a band in the level-overlap "
                          "region. 'deepest' averages more raw harmonics but "
                          "has fewer windows; 'shallowest' is the classic "
                          "EMTF assignment. Must match 'Estimate TF'."),
    )

    def execute(self, ctx, fc, ref_fc=None,
                coherence_min: float = 0.25, coherence_max: float = 0.98,
                mode: str = "weight", coherence_trim: bool = False,
                periods_per_octave: int = 4,
                band_style: str = "log",
                band_owner: str = "deepest") -> dict[str, Any]:
        import json

        levels = fc if isinstance(fc, dict) else {0: fc}
        sample = next(iter(levels.values()))
        ins, outs, _ = _fc_io_channels(sample)
        ref_levels = ref_fc if isinstance(ref_fc, dict) else (
            {0: ref_fc} if ref_fc is not None else None)
        out_levels, out_ref, rows = coherence_sort_fcs(
            levels, ref_levels, ins, outs,
            coherence_min=coherence_min, coherence_max=coherence_max,
            mode=mode, trim=coherence_trim,
            periods_per_octave=periods_per_octave,
            band_style=band_style, band_owner=band_owner,
        )
        path = ctx.artifact_path("coherence_sort", ".json")
        path.write_text(json.dumps(rows))
        artifact = ctx.publish_artifact(path, kind="table")
        retained = sum(r["n_retained"] for r in rows)
        total = sum(r["n_windows"] for r in rows)
        trimmed = sum(1 for r in rows if r["trimmed"])
        ctx.progress(
            1.0,
            f"coherence sort: {retained}/{total} windows kept, "
            f"{trimmed} band(s) trimmed",
        )
        result: dict[str, Any] = {"fc": out_levels, "table": artifact}
        if ref_fc is not None:
            result["ref_fc"] = out_ref
        return result


class EstimateTF(Node):
    type = "aurora.process.estimate_tf"
    title = "Estimate TF"
    tooltip = "Estimate the impedance/tipper transfer function from Fourier coefficients (OLS or robust, optional remote reference)."
    description = (
        "Estimate a transfer function from Fourier coefficients, over every "
        "decimation level of the fc, and merge them into one broadband tf. "
        "Estimator: 'Least Squares' (aurora OLS) or 'Robust' (M-estimator). "
        "With 'Robust', tick 'Remote Reference' to use a second station's fc "
        "(on 'ref fc') as the remote reference; if none is connected the local "
        "channels are reused as a self-reference. No least-squares + "
        "remote-reference path by design. Also outputs per-level tfs ('tf levels')."
    )
    category = "Processing"
    inputs = (
        Input("fc", "fc"),
        Input("ref_fc", "fc", label="ref fc (remote)", required=False),
    )
    outputs = (
        Output("tf", "tf"),
        Output("tf_levels", "tf.level", label="tf levels", description="per decimation level"),
    )
    params = (
        Param("estimator_engine", "select", label="Estimator",
              options=("Robust", "Least Squares"), default="Robust"),
        Param("remote_reference", "boolean", label="Remote Reference",
              default=False,
              description="Robust remote reference (RME_RR). Only applies to "
                          "the 'Robust' estimator.",
              cascade={True: (("estimator_engine", "Robust"),)}),
        Param("periods_per_octave", "integer", label="Periods per octave",
              default=4, min=1,
              description="Log2 period bands per power-of-2 octave."),
        Param("band_style", "select", label="Band style",
              options=("log", "index", "emtf"), default="log",
              description="Band construction mode (DESIGN.md §14.6/§14.7). "
                          "'log' (default) is the legacy continuous "
                          "log-period edge partition, kept unchanged as a "
                          "comparison baseline. 'index' is the discrete, "
                          "index-snapped partition on a shared global grid: "
                          "every raw FFT bin in a level's reliable range is "
                          "assigned to exactly one band (no gaps/overlaps/"
                          "silently-empty bands); overlapping bands are "
                          "estimated redundantly and the merge keeps the "
                          "deepest level. 'emtf' is the disjoint EMTF-style "
                          "table: every band assigned to exactly one level, "
                          "each frequency evaluated exactly once."),
        Param("band_owner", "select", label="Band owner (emtf)",
              options=("deepest", "shallowest"), default="deepest",
              description="Only used with band style 'emtf': which "
                          "decimation level evaluates a band in the "
                          "level-overlap region. 'deepest' (default) "
                          "averages more raw harmonics per band but has "
                          "factor-times fewer windows; 'shallowest' is the "
                          "classic EMTF assignment. Compare both against "
                          "'log'/'index' before settling on one."),
    )

    def execute(self, ctx, fc, ref_fc=None,
                estimator_engine: str = "Robust",
                remote_reference: bool = False,
                periods_per_octave: int = 4,
                band_style: str = "log",
                band_owner: str = "deepest") -> dict[str, Any]:
        # Accept either a single fc (treated as level 0) or a multi-level
        # container {level: fc}.
        levels = fc if isinstance(fc, dict) else {0: fc}
        sample_fc = next(iter(levels.values()))
        kw = sample_fc.attrs.get("channel_nomenclature") or None
        ins_raw = sample_fc.attrs.get("input_channels")
        outs_raw = sample_fc.attrs.get("output_channels")
        if ins_raw and outs_raw:
            ins = [c.strip() for c in ins_raw.split(",") if c.strip()]
            outs = [c.strip() for c in outs_raw.split(",") if c.strip()]
        else:
            # fc without attrs (e.g. read back from an mth5 archive): classify
            # the channels through the nomenclature instead of assuming names.
            ins, outs, _ = classify_channels(
                [str(c) for c in sample_fc.data_vars], kw
            )
        if not ins or not outs:
            raise ValueError(
                "could not determine input/output channels from the fc — "
                f"data_vars: {list(sample_fc.data_vars)}"
            )
        per_level = estimate_tf_levels(
            levels, ins, outs,
            estimator_engine=estimator_engine,
            remote_reference=remote_reference,
            periods_per_octave=periods_per_octave,
            ref_levels=ref_fc,
            channel_nomenclature=kw,
            band_style=band_style,
            band_owner=band_owner,
        )
        periods, Z, Z_err = merge_tf_levels(per_level)
        notes = (
            f"estimator={estimator_engine}; remote_reference={remote_reference}; "
            f"levels={len(levels)}; "
            f"periods_per_octave={periods_per_octave}; band_style={band_style}"
            + (f"; band_owner={band_owner}" if band_style == "emtf" else "")
            + f"; nomenclature={kw or 'auto'}"
        )
        tf = build_tf(
            periods, Z, ins, outs, channel_nomenclature=kw, Z_err=Z_err,
            survey_id=sample_fc.attrs.get("survey_id"),
            station_id=sample_fc.attrs.get("station_id"),
            run_id=sample_fc.attrs.get("run_id"),
            latitude=sample_fc.attrs.get("station_latitude"),
            longitude=sample_fc.attrs.get("station_longitude"),
            elevation=sample_fc.attrs.get("station_elevation"),
            start_datetime=sample_fc.attrs.get("start_datetime"),
            processing_notes=notes,
        )
        ctx.progress(
            1.0,
            f"estimated TF ({estimator_engine}) over {len(levels)} level(s), "
            f"{len(periods)} merged periods",
        )
        return {"tf": tf, "tf_levels": per_level}


class MergeTFLevels(Node):
    type = "aurora.process.merge_tf_levels"
    title = "Merge TF Levels"
    tooltip = "Merge per-decimation-level TFs into one broadband transfer function."
    description = (
        "Merge per-decimation-level TFs (the 'tf levels' output of Estimate TF) "
        "into a single broadband transfer function, resolving duplicate periods."
    )
    category = "Processing"
    inputs = (Input("tf_levels", "tf.level", label="tf levels"),)
    outputs = (Output("tf", "tf"),)
    params = ()

    def execute(self, ctx, tf_levels) -> dict[str, Any]:
        first = next(iter(tf_levels.values()))
        ins = list(first["input_channels"])
        outs = list(first["output_channels"])
        kw = first.get("channel_nomenclature") or None
        periods, Z, Z_err = merge_tf_levels(tf_levels)
        tf = build_tf(
            periods, Z, ins, outs, channel_nomenclature=kw, Z_err=Z_err,
            survey_id=first.get("survey_id"),
            station_id=first.get("station_id"),
            run_id=first.get("run_id"),
            latitude=first.get("station_latitude"),
            longitude=first.get("station_longitude"),
            elevation=first.get("station_elevation"),
            start_datetime=first.get("start_datetime"),
        )
        ctx.progress(1.0, f"merged {len(tf_levels)} TF levels -> {len(periods)} periods")
        return {"tf": tf}


def register(registry) -> None:
    registry.add(ComputeFC)
    registry.add(CoherenceSort)
    registry.add(EstimateTF)
    registry.add(MergeTFLevels)
