"""Shared processing helpers for the aurora plugin.

ComputeFC/EstimateTF split the aurora pipeline at the FC boundary. This module
holds the signal-processing core: RunTS/TimeSeries normalization, windowed
STFT to Fourier coefficients (fc), and MT impedance estimation (OLS / robust
RME / remote-reference RME_RR, all aurora-backed via :func:`_solve_band`).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from emptyui_core_nodes.types import (
    channel_calibration,
    ensure_finite,
    is_unknown_start,
    warn_placeholder_start,
)
from emptyui_mth5.nomenclature import (
    STANDARD_INPUTS,
    STANDARD_OUTPUTS,
    channel_map,
    default_channels,
    infer_keyword,
    is_electric_name,
    is_magnetic_name,
    magnetic_unit_for,
    magnetic_unit_from_channels,
)

MU0 = 4.0 * np.pi * 1e-7

_log = logging.getLogger("emptyui")

#: M21: transient budget for one coherence-sort _min_coh call. The
#: sliding-ensemble cumsum arrays scale with F·W·channels² — unchunked, a
#: 12 h/150 Hz level-0 FC measured 5.6 GiB peak (6.0 GiB two-FC).
#: Windows are processed in haloed chunks sized from this budget; measured
#: 1.1 GiB / 1.5 GiB at the same shapes (the remaining floor is the input
#: stacks themselves). Module-level so tests can shrink it to force the
#: chunked path on small fixtures.
_COH_BUDGET_BYTES = 256 * 2**20


def _split_names(value) -> list[str]:
    """Split a comma-separated channel list (or an iterable of names); None -> []."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(c).strip() for c in value if str(c).strip()]
    return [c.strip() for c in str(value).split(",") if c.strip()]


def _resolve_magnetic_unit(magnetic_unit, keyword, input_channels) -> str:
    """Resolve the magnetic-unit mode: explicit 'h'/'b' wins; otherwise derive
    it from the nomenclature keyword (mt_metadata CHANNEL_MAPS) or, failing
    that, from the actual channel names (bx/by -> fluxgate B, hx/hy -> coil H).

    Note the mode only drives the *fallback* of :func:`_magnetic_to_h`:
    explicit unit attrs win first (A/m is never rescaled; an explicit
    magnetic-induction unit converts at its own prefix), and only payloads
    without usable unit attrs fall back to the mode — 'b' (fluxgate B is
    archived in nT) converts, 'h' keeps the legacy unitless convention
    (history §16.3 USArray verification).
    """
    mu = str(magnetic_unit or "auto").strip().lower()
    if mu in ("h", "b"):
        return mu
    if keyword:
        return magnetic_unit_for(keyword)
    return magnetic_unit_from_channels(input_channels)


def _resolve_keyword(channel_nomenclature, input_channels, output_channels,
                     available=None):
    """Resolve the nomenclature keyword: explicit value, or best-effort
    inference. With explicit channel fields, inference uses those names (so a
    user requesting the e3/e4 pair on a 4-E-channel run resolves to lemi34);
    with auto (empty) fields it falls back to the channels actually present in
    the data."""
    kw = str(channel_nomenclature or "").strip().lower()
    if kw and kw != "auto":
        return kw
    requested = _split_names(input_channels) + _split_names(output_channels)
    if requested:
        return infer_keyword(requested)
    if available:
        return infer_keyword(available)
    return "default"


def _resolve_io_channels(available, channel_nomenclature, input_channels,
                         output_channels):
    """Resolve the actual input/output channel names for a compute_fc_levels call.

    Empty (or "auto") fields are resolved automatically from the nomenclature
    and the data: inputs = the horizontal magnetic pair, outputs = the
    electric pair **plus** the vertical magnetic channel (tipper) — each
    included only when the channel is actually present. This makes the tipper
    available by default while keeping an impedance-only FC when the data has
    no vertical channel, and a tipper-only FC when the data carries only
    magnetic channels.

    Explicit names are used as-is; an explicit output channel missing from the
    data is dropped with a warning (the vertical magnetic channel is a
    best-effort extension), while a missing input channel is an error.
    """
    available = [str(c) for c in available]
    keyword = _resolve_keyword(
        channel_nomenclature, input_channels, output_channels, available
    )
    ins = _split_names(input_channels)
    outs = _split_names(output_channels)
    auto_ins = not ins or ins == ["auto"]
    auto_outs = not outs or outs == ["auto"]
    if auto_ins or auto_outs:
        d_ins, d_outs = default_channels(keyword)
    if auto_ins:
        ins = d_ins
    if auto_outs:
        outs = d_outs

    # Normalize to the exact names present in the data (case-insensitive —
    # e.g. a CSV with 'Ex,Ey,Hx,Hy' satisfies the standard 'ex,ey,hx,hy').
    def present(name) -> str | None:
        low = str(name).strip().lower()
        for c in available:
            if str(c).lower() == low:
                return c
        return None

    resolved_ins = [present(c) for c in ins]
    missing_ins = [c for c in resolved_ins if c is None]
    ins = [c for c in resolved_ins if c is not None]
    resolved_outs = [present(c) for c in outs]
    missing = [c for c in resolved_outs if c is None]
    outs = [c for c in resolved_outs if c is not None]
    if missing:
        _log.warning(
            "dropping output channel(s) %s not present in the data %s "
            "(tipper is a best-effort extension — the vertical magnetic "
            "channel is included only when available)",
            missing, available,
        )
    if len(ins) != 2:
        raise ValueError(
            f"need two input channels (horizontal magnetic Hx/Hy or Bx/By) — "
            f"resolved {ins} (missing {missing_ins}) from data {available}; "
            f"set input_channels explicitly (e.g. 'bx,by')"
        )
    if not outs:
        if auto_outs and len(ins) == 2:
            # A magnetic-only run (no electric/vertical channels) is a remote-
            # reference station: the FC carries the magnetic channels, which is
            # exactly what EstimateTF's ref_fc consumes.
            _log.warning(
                "no electric/vertical output channels in the data %s — using "
                "the magnetic channels %s as outputs (remote-reference fc)",
                available, ins,
            )
            outs = list(ins)
        else:
            raise ValueError(
                f"no output channels available — requested {output_channels!r} "
                f"against data {available}"
            )
    return ins, outs, keyword


def to_channels(value, check_finite: bool = True):
    """Normalize a ts.run payload to (channels: dict[str, np.ndarray],
    sample_rate: float, start_s: float, start_dt: pd.Timestamp | None).

    Accepts:
      - mth5 RunTS (has .dataset xarray, .sample_rate, .start datetime)
      - core TimeSeries (has .channels dict, .sample_rate, .start float s)

    ``check_finite`` guards every channel against NaN/inf; pass False to
    defer the guard until the used channel subset is known (callers that
    resolve in/out channels check only those, so a NaN in an unused
    auxiliary channel cannot hard-fail the node).
    """
    # mth5 RunTS / mt_timeseries
    if hasattr(value, "dataset") and not isinstance(value, dict):
        ds = value.dataset
        sample_rate = float(value.sample_rate)
        start_s = 0.0
        start_dt = _start_datetime(getattr(value, "start", None))
        channels = {str(k): np.asarray(v.values, dtype=float) for k, v in ds.data_vars.items()}
        if check_finite:
            ensure_finite(channels, "Compute FC")
        return channels, sample_rate, start_s, start_dt
    # core-nodes TimeSeries
    if hasattr(value, "channels") and hasattr(value, "sample_rate"):
        sample_rate = float(value.sample_rate)
        start_s = float(getattr(value, "start", 0.0))
        start_dt = _start_datetime(
            getattr(value, "start_datetime", None) or getattr(value, "start", None)
        )
        channels = {str(k): np.asarray(v, dtype=float) for k, v in value.channels.items()}
        if check_finite:
            ensure_finite(channels, "Compute FC")
        return channels, sample_rate, start_s, start_dt
    raise TypeError(f"cannot normalize ts.run value of type {type(value).__name__}")


def _start_datetime(value):
    """Return a pandas Timestamp for an absolute start time, or None.

    Handles datetime / pandas Timestamp, an ISO string (core TimeSeries), and
    mt_metadata MTime. Returns None for relative float-second starts AND for
    placeholder epochs (1980 mt_timeseries default / 1970 Unix default —
    digitized paper records): the fc time axis must not carry fake absolute
    stamps (consistency with runs._run_time_info; old records are warned
    about and degraded, never rejected).
    """
    ts = _start_datetime_raw(value)
    if ts is None:
        return None
    if is_unknown_start(ts):
        warn_placeholder_start("record", ts.isoformat())
        return None
    return ts


def _start_datetime_raw(value):
    """Parse a start value to a pandas Timestamp (or None) — no sentinel
    handling; see _start_datetime."""
    if value is None or isinstance(value, (int, float, np.number)):
        return None
    if isinstance(value, str):
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        pass
    # mt_metadata MTime and similar expose iso_str / isoformat
    for attr in ("iso_str", "isoformat"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return pd.Timestamp(str(fn()))
            except (TypeError, ValueError):
                pass
    return None


def compute_fc_levels(
    value,
    input_channels,
    output_channels,
    num_decimations=0,
    factor=4,
    num_samples_window=256,
    overlap=0.5,
    window_type="boxcar",
    magnetic_unit="auto",
    e_dipole_length_m=None,
    start_datetime=None,
    channel_nomenclature=None,
    prewhitening="first difference",
    per_window_detrend="",
    window_type_deep="",
    deep_from_level=None,
    clock_zero="",
    reject_saturated=True,
    saturated_max: float = 0.01,
):
    """Cascading-decimation multi-level STFT -> ``{level: xr.Dataset}``.

    ``saturated_max`` (§24.12 QC): an FC window is zero-weighted when the
    flagged fraction of its sample span exceeds this threshold (the original
    any-overlap rule deleted the long-period tail whenever saturation is
    frequent — see :func:`_saturated_window_mask`).
    ``num_decimations`` is the number of cascading decimations performed, so it
    also equals the deepest decimation level: 0 = no decimation (just level 0 at
    the native sample rate); 1 = decimate once (levels 0 and 1); N = levels 0..N.
    Each deeper level decimates the previous level's (already-normalized) series
    by ``factor`` and re-STFTs it with the **same constant ``num_samples_window``**
    (aurora-style), so the window *duration* grows at deeper levels and
    low-frequency resolution improves. Level *i*'s sample rate = native / factor^i.

    ``window_type_deep`` + ``deep_from_level``: per-level taper override
    (DESIGN.md §10.1) — levels >= ``deep_from_level`` are STFT'd with
    ``window_type_deep`` instead of ``window_type``. Physically motivated
    (§24.3): the dominant contamination differs by scale — white-ish
    fluxgate noise (post-prewhitening) at the short-period levels, where
    boxcar maximizes effective dof, vs line-spectra (diurnal Sq) leakage
    through boxcar sidelobes at the day-scale deep windows, where a
    zero-endpoint taper wins.

    ``clock_zero`` (§24.5 P1 v2b): "" (default) anchors windows at the run
    start; ``"epoch"`` or an explicit ISO datetime anchors windows on
    the grid ``t0 + k·step`` per level (head partial window dropped; the time
    labels are window CENTRES — issues.md L11 — so each window's first sample
    is N/(2·fs) earlier), so
    per-run FCs share one absolute window grid across runs/stations —
    required for remote-reference pairing over merged per-run FCs. Levels
    with fewer usable samples than one window (e.g. burst slots at deep
    levels) are skipped with a warning.

    Each level's fc is an xr.Dataset with ``(frequency, time)`` dims and a
    ``datetime64`` ``time`` coordinate when an absolute start is known (needed
    for mth5 FC persistence). Per-level metadata (``sample_rate``,
    ``decimation_level``, window params) is recorded in each fc's attrs.
    """
    import xarray as xr

    # the select options carry honest labels ("none", "same as shallow") —
    # alias them to the canonical off/inherit internals here so saved
    # documents from before the relabeling ("") and the labeled sentinels
    # behave identically (user report 2026-10-14: blank dropdown options)
    if window_type_deep == "same as shallow":
        window_type_deep = ""
    if prewhitening == "none":
        prewhitening = ""
    if per_window_detrend == "none":
        per_window_detrend = ""

    # review item B13: overlap >= 1 reaches scipy.signal.stft as
    # noverlap >= nperseg and crashes mid-node — reject with an actionable
    # message up front (the manifest max is frontend-only)
    if not (0.0 <= float(overlap) < 1.0):
        raise ValueError(
            f"overlap must be in [0, 1) — got {overlap!r} "
            "(0 = adjacent windows, 0.99 = near-total overlap)")
    # the retired single-level compute_fc validated this up front; keep it here
    # (the STFT check is reached only when a level has a full window)
    if prewhitening and prewhitening != "first difference":
        raise ValueError(
            f"unknown prewhitening {prewhitening!r} (supported: "
            "'first difference', 'none', or '' for off)"
        )
    channels, sample_rate, start_s, start_dt = to_channels(value, check_finite=False)
    if start_datetime is None:
        start_datetime = start_dt
    units = _channel_units(value)
    calibration = channel_calibration(value)
    input_channels, output_channels, keyword = _resolve_io_channels(
        list(channels), channel_nomenclature, input_channels, output_channels
    )
    # NaN guard only on the channels that are actually normalized/STFT'd —
    # a NaN in an unused auxiliary channel must not hard-fail the node
    ensure_finite(
        {c: channels[c] for c in dict.fromkeys(input_channels + output_channels)},
        "Compute FC",
    )
    b_to_h = _resolve_magnetic_unit(magnetic_unit, keyword, input_channels) == "b"
    e_scale = (1e-3 / e_dipole_length_m) if e_dipole_length_m else None

    num_decimations = max(0, int(num_decimations))
    deep_from = (int(deep_from_level) if deep_from_level is not None
                 and str(deep_from_level) != "" else None)
    win_deep = window_type_deep or window_type
    cz = str(clock_zero or "").strip()
    if cz and cz.lower() != "epoch":
        # validate an explicit anchor format UP FRONT: a typo must not be
        # swallowed by the no-absolute-start skip below (the skip is a data
        # condition, the anchor format is a param error)
        try:
            probe = pd.Timestamp(cz)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"clock_zero {clock_zero!r} is not a valid anchor — use "
                "'epoch', an ISO datetime, or empty for run-start anchoring"
            ) from exc
        if pd.isna(probe):
            raise ValueError(
                f"clock_zero {clock_zero!r} did not parse to a datetime — "
                "use 'epoch', an ISO datetime, or empty for run-start "
                "anchoring")
    if cz and start_datetime is None:
        _log.warning(
            "clock_zero=%r requires an absolute start time — the payload "
            "carries none; anchoring skipped (windows anchor at the run "
            "start)", clock_zero)
        cz = ""
    start_ts = (pd.Timestamp(start_datetime)
                if start_datetime is not None else None)
    levels: dict[int, xr.Dataset] = {}
    cur = channels
    cur_rate = sample_rate
    for level in range(num_decimations + 1):
        # normalize the raw data only once (level 0); deeper levels re-use the
        # already-normalized series passed down from the previous level.
        norm, e_scaling = (
            _normalize_channels(cur, input_channels, output_channels, b_to_h, e_scale,
                                units)
            if level == 0
            else (cur, "level 0 (passed down)")
        )
        win = win_deep if deep_from is not None and level >= deep_from else window_type
        nperseg = max(8, int(num_samples_window))
        noverlap = int(nperseg * overlap)
        # clock-zero anchoring (§24.5 P1 v2b): drop the head partial window so
        # window starts sit on the common t0 + k·step grid (per level — the
        # step grows with the decimation level); next level decimates the
        # UNSLICED series (aurora truncates per level too)
        n_skip, off = _clock_zero_head_skip(
            start_ts, cur_rate, nperseg, noverlap, cz)
        if start_ts is None:
            off = start_s  # no absolute time: keep the relative-offset base
        series = {k: v[n_skip:] for k, v in norm.items()}
        n_usable = max((len(v) for v in series.values()), default=0)
        if n_usable < nperseg:
            _log.warning(
                "Compute FC: decimation level %d has %d usable sample(s), "
                "fewer than one window (%d) — level and all deeper levels "
                "skipped (they are strictly shorter)", level,
                n_usable, nperseg)
            break
        f, t, fc_vars = _stft_channels(
            series, cur_rate, num_samples_window, overlap, win,
            input_channels, output_channels,
            prewhitening=prewhitening,
            per_window_detrend=per_window_detrend,
        )
        n_sat_rejected = 0
        if reject_saturated and "saturated" in series:
            flags = _saturated_window_mask(
                series["saturated"], max(8, int(num_samples_window)),
                int(max(8, int(num_samples_window)) * overlap),
                next(iter(fc_vars.values()))[1].shape[1],
                saturated_max=saturated_max)
            n_sat_rejected = int(flags.sum())
            if n_sat_rejected:
                for comp, (dims, zi) in fc_vars.items():
                    zi = zi.copy()
                    zi[:, flags] = np.nan + 1j * np.nan
                    fc_vars[comp] = (dims, zi)
                _log.info(
                    "Compute FC: level %d rejects %d/%d saturated window(s)",
                    level, n_sat_rejected, flags.size)
        cal_note = _calibrate_stft_vars(fc_vars, f, calibration, "Compute FC")
        start_iso = ""
        if start_ts is not None:
            dt_idx = start_ts + pd.to_timedelta(t + off, unit="s")
            # naive UTC (repo-wide convention) — see compute_fc_levels' note
            time_coord = (dt_idx.tz_localize(None)
                          if dt_idx.tz is not None else dt_idx)
            start_iso = (start_ts + pd.to_timedelta(off, unit="s")).isoformat()
        else:
            time_coord = t + off  # off = the payload's relative start

        fc = xr.Dataset(
            data_vars=fc_vars,
            coords={"frequency": f, "time": time_coord},
            attrs={
                "sample_rate": float(cur_rate),
                "decimation_level": int(level),
                "factor": int(factor),
                "num_samples_window": int(num_samples_window),
                "overlap": float(overlap),
                "window_type": win,
                "prewhitening": prewhitening or "off",
                "per_window_detrend": per_window_detrend or "off",
                "input_channels": ",".join(input_channels),
                "output_channels": ",".join(output_channels),
                "magnetic_unit": b_to_h and "b" or "h",
                "electric_scaling": e_scaling,
                "channel_nomenclature": keyword,
                "e_dipole_length_m": e_dipole_length_m or 0.0,
                "calibration_applied": cal_note,
                "saturated_windows_rejected": n_sat_rejected,
                "saturated_max": float(saturated_max),
                "start_datetime": start_iso,
            },
        )
        levels[int(level)] = fc

        if level < num_decimations:
            cur = _decimate_channels(norm, factor)
            cur_rate = cur_rate / float(factor)

    return levels


def _channel_units(value):
    """Per-channel unit strings from a ts.run payload (best-effort).

    Returns ``{channel: units}`` for payloads that carry units (mth5 RunTS
    xarray attrs — real archives say ``nanoTesla`` / ``milliVolt per
    kilometer``; core TimeSeries payloads carry a ``units`` dict coerced
    from those attrs at ``coerce_run``; synthetic/core payloads usually
    carry none, so ``{}``).
    """
    if hasattr(value, "dataset"):
        ds = value.dataset
        out: dict[str, str] = {}
        for name in ds.data_vars:
            u = ds[name].attrs.get("units")
            if u:
                out[str(name)] = str(u)
        return out
    units = getattr(value, "units", None)
    if isinstance(units, dict):
        return {str(k): str(v) for k, v in units.items() if v}
    return {}


#: Units that identify raw instrument counts — the un-calibrated output of
#: a digitizer. Counts carry no physical scale (the scale factor lives in
#: the instrument response), so any FC/TF built from them is garbage with a
#: plausible-looking shape. mth5 archives built from FDSN store raw counts
#: with the response attached — a Calibrate step is mandatory before
#: Compute FC (DESIGN.md §10.5).
_COUNT_UNITS = ("digital counts", "count", "counts")


def _reject_counts(x, comp, units, where):
    u = str((units or {}).get(comp, "")).strip().lower()
    if u in _COUNT_UNITS:
        raise ValueError(
            f"{where}: channel '{comp}' carries raw instrument counts "
            f"(unit attrs {u!r}) — the FC/TF would be meaningless. Connect "
            "a Calibrate node first (FDSN-built mth5 archives store raw "
            "counts with the instrument response attached)."
        )


#: Explicit H (magnetic field intensity) unit attrs — canonical mt_metadata
#: name (or raw spelling when the registry is unavailable) -> factor to A/m.
#: A payload carrying one of these already holds H and must never be
#: rescaled, whatever the nomenclature-derived mode says (DESIGN.md §10.5,
#: "explicit A/m -> leave").
_H_UNIT_FACTORS = {
    "ampere per meter": 1.0,
    "a/m": 1.0,
    "milliampere per meter": 1e-3,
    "microampere per meter": 1e-6,
}

#: Explicit B (magnetic induction) unit attrs -> factor to tesla; H follows
#: as ``x * factor / MU0`` (A/m). Exact canonical names only — a gradient
#: unit ("nanoTesla per meter") must not be mistaken for a field. The raw
#: "nT"/"nanotesla" spellings below cover a missing unit registry.
_B_UNIT_FACTORS = {
    "tesla": 1.0,
    "millitesla": 1e-3,
    "microtesla": 1e-6,
    "nanotesla": 1e-9,
    "picotesla": 1e-12,
    "nt": 1e-9,
}


def _magnetic_to_h(x, comp, b_to_h, units):
    """Convert a magnetic channel to A/m (B -> H, /mu0) — DESIGN.md §10.5.

    The fc must carry H in A/m for the impedance to be in ohm (V/m per A/m).
    Precedence, identical for modes 'b' and 'h':

    1. **explicit H units** (A/m and friends) win over every mode — the
       payload already carries H, so it is returned untouched;
    2. an **explicit magnetic-induction unit** (pT/nT/uT/mT/T, or any raw
       spelling containing "nano") is scaled by its own prefix, then /mu0;
    3. **no usable unit attrs** fall back to the mode alone: 'b' (fluxgate
       B is archived in nT) converts, 'h' keeps the legacy unitless
       convention (synthetic fixtures hold A/m already).

    Previously (3) short-circuited before the units were consulted, so a
    fluxgate-mode payload that had been converted to A/m upstream was
    scaled a second time (Z wrong by ~1.26e3).
    """
    _reject_counts(x, comp, units, "Compute FC")
    raw = (units or {}).get(comp, "")
    # _canonical_unit keeps the registry's own casing — normalize here so
    # the tables below are plain lowercase lookups (as _electric_to_vm does)
    canon = _canonical_unit(raw)
    key = (canon if canon is not None else str(raw)).strip().lower()
    factor = _H_UNIT_FACTORS.get(key)
    if factor is not None:
        if b_to_h and factor != 1.0:
            # the mode promised a B payload; the units say otherwise and
            # win — worth a trace for anyone reading the fc attrs later
            _log.debug(
                "channel '%s': mode 'b' but unit attrs %r are already H — "
                "leaving untouched (DESIGN.md §10.5)", comp, raw)
        return x * factor
    factor = _B_UNIT_FACTORS.get(key)
    if factor is None and "nano" in key:
        factor = 1e-9  # raw nano-spelling the registry could not canonicalize
    if factor is not None:
        return x * factor / MU0
    if b_to_h:
        return x * 1e-9 / MU0
    return x


def _canonical_unit(u):
    """Canonical mt_metadata unit name for a unit string, or None.

    Uses mt_metadata's unit registry (``'mV/km'`` -> ``'milliVolt per
    kilometer'``, ``'V/m'`` -> ``'Volt per meter'``); unknown strings come
    back lowercased (the registry answers 'unknown' for them). Lazy import —
    mt_metadata is plugin-tier, never a core dependency.
    """
    if not u:
        return None
    try:
        from mt_metadata.common.units import get_unit_object

        name = str(get_unit_object(str(u)).name)
        if name.lower() != "unknown":
            return name
    except Exception:  # noqa: BLE001, S110 — registry unavailable: raw text
        pass
    return str(u).strip().lower()


def _electric_to_vm(x, comp, e_scale, units):
    """Convert an electric channel to V/m; returns ``(x_scaled, label)``.

    Policy (DESIGN.md §10.5, review 2026-09-12 — modelled on _magnetic_to_h):

    - ``e_scale`` set (``e_dipole_length_m``): the explicit raw-potential
      path, ``x * 1e-3 / dipole`` — the data are assumed to be a raw mV
      potential. A warning is logged when the channel's unit attrs say
      mV/km (that combination double-scales by dipole/1000 — clear the
      dipole param for field data).
    - no dipole: scale according to the channel's unit attrs:
        * ``milliVolt per kilometer`` -> x1e-6 (LEMI archives; legacy path)
        * ``Volt per meter``          -> passthrough (already physical —
          e.g. calibrated data; previously silently mis-scaled x1e6)
        * ``milliVolt per meter``     -> x1e-3
        * ``milliVolt`` (raw potential, no dipole) -> hard error: set the
          dipole length (previously a silent wrong x1e-6)
        * other unknown units -> warning + passthrough (never silently
          scale what cannot be interpreted)
        * no unit attrs -> x1e-6 (legacy unitless behavior: synthetic
          fixtures and core payloads; emptyui's LEMI reader writes mV/km).
          The branch WARNS — missing attrs are indistinguishable from a
          stripped V/m payload (which this assumption would corrupt 1e6)
    """
    u = (_canonical_unit((units or {}).get(comp, "")) or "").lower()
    _reject_counts(x, comp, units, "Compute FC")
    if e_scale is not None:
        if u and "per kilometer" in u:
            _log.warning(
                "electric channel '%s' carries unit attrs '%s' but "
                "e_dipole_length_m is set: the dipole scaling assumes a raw "
                "mV potential and will double-scale mV/km field data by "
                "dipole/1000 — clear e_dipole_length_m for field data",
                comp, (units or {}).get(comp),
            )
        return x * e_scale, "mV potential -> V/m (dipole)"
    if not u:
        _log.warning(
            "electric channel '%s' carries NO unit attrs — the legacy path "
            "assumes mV/km and scales x1e-6 (a payload actually in V/m, "
            "e.g. with stripped attrs, would be wrong by 1e6): set the "
            "channel's units attr to make the scaling explicit",
            comp,
        )
        return x * 1e-6, "legacy (unitless, mV/km assumed)"
    if u == "millivolt per kilometer":
        return x * 1e-6, "mV/km -> V/m"
    if u == "volt per meter":
        return x, "passthrough (V/m)"
    if u == "millivolt per meter":
        return x * 1e-3, "mV/m -> V/m"
    if u == "millivolt":
        raise ValueError(
            f"electric channel '{comp}' carries unit attrs 'milliVolt' — a "
            "raw potential, not a field: set e_dipole_length_m so Compute FC "
            "can convert it to V/m (previously this was silently mis-scaled)"
        )
    _log.warning(
        "electric channel '%s' carries unrecognized unit attrs '%s' — "
        "assuming the data are already in V/m and leaving them untouched",
        comp, (units or {}).get(comp),
    )
    return x, f"passthrough (unrecognized unit {(units or {}).get(comp)!r})"


def _normalize_channels(channels, input_channels, output_channels, b_to_h, e_scale,
                        units=None):
    """Apply instrument unit normalization (B->H, mV->V/m) to a channel dict.

    Returns ``(out, electric_scaling)`` — the normalized dict and the
    electric-scaling label for the fc attrs (DESIGN.md §10.5).
    """
    out = {}
    e_labels = set()
    for comp, x in channels.items():
        x = np.asarray(x, dtype=float)
        if comp in input_channels:
            x = _magnetic_to_h(x, comp, b_to_h, units)
        elif comp in output_channels:
            if is_magnetic_name(comp):  # vertical magnetic output (tipper)
                x = _magnetic_to_h(x, comp, b_to_h, units)  # Bz -> Hz, same as inputs
            elif is_electric_name(comp):
                x, label = _electric_to_vm(x, comp, e_scale, units)
                e_labels.add(label)
        out[comp] = x
    return out, ("; ".join(sorted(e_labels)) if e_labels else "none")


def _decimate_channels(channels, factor):
    """Cascade-decimate every channel by an integer factor (scipy decimate).

    The binary "saturated" auxiliary channel (§24.12 QC) is max-pooled
    instead: a group survives decimation flagged iff any of its samples was
    flagged — scipy's IIR decimate would blur the 0/1 mask and could let a
    saturated block fade below any threshold.
    """
    from scipy.signal import decimate

    factor = max(2, int(factor))
    out = {}
    for k, v in channels.items():
        v = np.asarray(v)
        if k == "saturated":
            pad = (-v.size) % factor
            if pad:
                v = np.pad(v, (0, pad), mode="edge")
            out[k] = v.reshape(-1, factor).max(axis=1)
        else:
            out[k] = np.asarray(decimate(v, factor))
    return out


#: Reference instants accepted by the ``clock_zero`` option (§24.5 P1 v2b).
CLOCK_ZERO_EPOCH = "1970-01-01T00:00:00"


def _clock_zero_head_skip(data_start_abs, sample_rate, nperseg, noverlap,
                          clock_zero):
    """Head-skip so window starts land on a common clock-zero grid.

    Returns ``(n_skip, start_offset_s)``: drop ``n_skip`` samples from the
    series head so the first full window starts at (or after) the data
    start, at the sample nearest the grid ``t0 + k·step`` (step = window
    step in seconds; within half a sample of it, M9). ``start_offset_s`` is
    the sliced series' start relative to the payload record start — the
    offset the fc ``time`` coordinate must be built from, i.e. the time of
    the sample the data really starts on. With ``clock_zero`` "" (default)
    nothing is dropped and the current run-start anchoring is kept.

    The point (§24.5 P1 v2b): remote-reference pairing matches FC rows by
    window position with an identical-grid check, so separately-computed
    runs/stations must share one absolute window grid. A run-independent
    ``t0`` (Unix epoch) guarantees coincidence for any run starts; anchoring
    to a run's own start does not (starts are not whole steps apart).
    Requires an absolute start; without one the anchoring is a warned no-op.
    """
    if clock_zero in ("", None):
        return 0, 0.0
    cz = str(clock_zero).strip()
    try:
        t0 = pd.Timestamp(CLOCK_ZERO_EPOCH if cz.lower() == "epoch" else cz)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"clock_zero {clock_zero!r} is not a valid anchor — use 'epoch', "
            "an ISO datetime, or empty for run-start anchoring") from exc
    if pd.isna(t0):
        raise ValueError(
            f"clock_zero {clock_zero!r} did not parse to a datetime — use "
            "'epoch', an ISO datetime, or empty for run-start anchoring")
    if data_start_abs is None:
        _log.warning(
            "clock_zero=%r requires an absolute start time — the payload "
            "carries none; anchoring skipped (windows anchor at the run "
            "start)", clock_zero)
        return 0, 0.0
    # tz alignment: the epoch is definitionally UTC — localize to match the
    # payload's awareness (naive payloads keep a naive t0)
    if data_start_abs.tzinfo is not None and t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    elif data_start_abs.tzinfo is None and t0.tzinfo is not None:
        # CONVERT to UTC then drop the tz: tz_localize(None) alone keeps
        # the wall time, silently shifting a non-UTC-offset anchor (RR
        # pairing matches by these coordinates; review 2026-09-24)
        t0 = t0.tz_convert("UTC").tz_localize(None)
    dt = (data_start_abs - t0).total_seconds()
    if dt < 0:
        # an anchor LATER than the data would silently degrade to run-start
        # anchoring (k0 <= 0) — exactly the misalignment the feature exists
        # to prevent; warn so the typo'd/overshot anchor is visible
        _log.warning(
            "clock_zero=%r lies AFTER the record start (%.3f s) — the "
            "record has no sample at the anchor; windows anchor at the run "
            "start (head skip disabled)", clock_zero, dt)
        return 0, 0.0
    step_s = (nperseg - noverlap) / float(sample_rate)
    k0 = int(np.ceil(dt / step_s))
    n_skip = max(0, round((k0 * step_s - dt) * sample_rate))
    # M9: the window labels must be the time of the sample the data really
    # starts on. The skip is rounded to whole samples, so the offset is the
    # rounded skip, not the unrounded grid distance (which put the labels up
    # to half a sample off the data). Two records starting off the sample
    # grid therefore differ by up to one sample of label; _pair_remote_fc
    # pairs within one sample to absorb that.
    return n_skip, n_skip / float(sample_rate)


def _stft_channels(channels, sample_rate, num_samples, overlap, window_type,
                   input_channels, output_channels,
                   prewhitening="", per_window_detrend=""):
    """Windowed STFT of a channel dict -> (f, t, {channel: (freq,time) complex}).

    ``prewhitening``: "" (off, default) or ``"first difference"`` — the
    leakage-reduction prewhitening of classical MT processing: a time-domain
    first difference x[n] − x[n−1] per channel before the STFT, inverted
    exactly in the frequency domain by dividing by (1 − e^{−j2πfΔt})
    (recoloring; the DC bin becomes NaN — recoloring is undefined there).
    aurora 0.6.2 defaults to the same first-difference prewhitening but
    recolors with the idealized j2πf, which does not exactly invert its
    central-difference operator (response j·sin(ωΔt)/Δt) and leaves a
    low-frequency amplitude bias — the exact inverse filter is used here
    (history §24).
    ``per_window_detrend``: "" (off, default) or ``"linear"`` — scipy detrend
    per STFT segment before tapering (aurora's per-window default), keeping
    window-level trends out of the low-frequency bins.
    """
    from scipy.signal import stft

    if prewhitening and prewhitening != "first difference":
        raise ValueError(
            f"unknown prewhitening {prewhitening!r} (supported: "
            "'first difference' or '' for off)"
        )
    nperseg = max(8, int(num_samples))
    noverlap = int(nperseg * overlap)
    win = _window(window_type)
    # L-A9: heterogeneous channel lengths (the row-indexed LEMI pathology)
    # produce different window counts per channel — the fc Dataset build
    # cannot align them; crop every processed channel to the common minimum
    # with a loud warning (never mutate the input dict — immutability §7.1)
    used = list(input_channels) + list(output_channels)
    lens = {c: len(channels[c]) for c in used if c in channels}
    if len(set(lens.values())) > 1:
        n_min = min(lens.values())
        _log.warning(
            "Compute FC: channels of unequal length (%s) — cropped to the "
            "shortest (%d samples) so the window grids align",
            sorted(lens.values()), n_min)
        channels = dict(channels)
        for c, n in lens.items():
            if n > n_min:
                channels[c] = np.asarray(channels[c], dtype=float)[:n_min]
    f = t = None
    fc_vars = {}
    for comp in input_channels + output_channels:
        x = np.asarray(channels[comp], dtype=float)
        if prewhitening:
            x = x - np.concatenate(([x[0]], x[:-1]))
        fi, ti, zi = stft(
            x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
            window=win, boundary=None, padded=False,
            detrend=per_window_detrend or None,
        )
        if prewhitening:
            # recolor along the frequency axis only (divisor must broadcast
            # as a column — (nf,) would broadcast along time)
            with np.errstate(divide="ignore", invalid="ignore"):
                zi = zi / (1.0 - np.exp(-2j * np.pi * fi / sample_rate))[:, None]
            zi[fi == 0.0] = np.nan  # DC — recoloring undefined there
        f, t = fi, ti
        fc_vars[comp] = (("frequency", "time"), zi)
    return f, t, fc_vars


def _window(name: str) -> str:
    """Map our window_type vocabulary to scipy window names."""
    return {"hanning": "hann"}.get(name, name)


def _saturated_window_mask(sat, nperseg, noverlap, n_win,
                           saturated_max: float = 0.01) -> np.ndarray:
    """Per-FC-window saturation flags (§24.12 QC).

    ``sat`` is the binary per-sample mask at the CURRENT level's rate; an FC
    window is flagged iff the flagged FRACTION of its sample span exceeds
    ``saturated_max``. The original rule zero-weighted on ANY overlap, which
    is right for a clean recording but orders-of-magnitude over-conservative
    when saturation is frequent: a 73-minute deep window almost always
    overlaps one clipped sample out of 65536, and rejecting it deletes the
    entire long-period tail (the union fix in Build FC exposed this —
    grouped spans had silently dropped the mask channel before). Returns a
    boolean array of length ``n_win`` (the STFT's window count) aligned to
    the FC time axis.
    """
    sat = np.asarray(sat, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(sat > 0)])
    step = max(1, nperseg - noverlap)
    starts = np.arange(n_win) * step
    ends = np.minimum(starts + nperseg, c.size - 1)
    n = np.maximum(ends - starts, 1)
    return (c[ends] - c[starts]) / n > saturated_max


def _calibrate_stft_vars(fc_vars, f, calibration, where: str) -> str:
    """Divide the per-channel FCs by their instrument response (§24.12).

    Frequency-domain calibration inside Compute FC, aurora-style: after the
    STFT, each channel's Fourier coefficients are divided by the chained
    complex response of the filters whose applied flag is ``True`` (aurora
    semantics: ``applied=True`` means the response IS part of the recorded
    data; a prior time-domain Calibrate flips the flags to ``False``, so the
    two paths compose without double application). A DC response of exactly
    zero is treated aurora-style: the divisor becomes 1 there and the DC
    column is set to NaN (undefined calibration at 0 Hz).
    """
    note: list[str] = []
    for comp, pairs in calibration.items():
        if comp not in fc_vars:
            continue  # auxiliary channel — was not STFT'd
        resp = None
        names: list[str] = []
        for obj, applied in pairs:
            if not applied:
                continue
            r = np.asarray(obj.complex_response(f), dtype=complex)
            resp = r if resp is None else resp * r
            names.append(str(obj.name))
        if not names:
            continue
        resp = resp.astype(complex)
        # L-A9: an all-zero response (a broken/padded calibration table)
        # would divide every FC by zero (inf/nan garbage downstream); the
        # DC-only guard below misses it because max(|resp[1:]|) is 0
        if not np.any(np.abs(resp) > 0):
            raise ValueError(
                f"calibration response for '{comp}' is all zero — the "
                "filter table is broken (check the calibration file/chain)")
        # Aurora treats a DC calibration response of exactly 0 as "undefined
        # at DC" (divisor 1, DC column NaN). Induction-coil tables padded
        # with the linear-in-f low-frequency extension extrapolate to ~0 at
        # DC within float error (not exactly 0), so a near-zero check is
        # needed too — otherwise the DC bin would carry ~1e15 garbage.
        if resp[0] == 0 or abs(resp[0]) < 1e-12 * np.max(np.abs(resp[1:])):
            resp[0] = 1.0
            dc_zero = True
        else:
            dc_zero = False
        dims, zi = fc_vars[comp]
        zi = zi / resp[:, None]
        if dc_zero:
            zi[:, 0] = np.nan + 1j * np.nan
        fc_vars[comp] = (dims, zi)
        note.append(f"{comp}={'+'.join(names)}")
        _log.debug("%s: channel '%s' calibrated by %s", where, comp,
                   "+".join(names))
    return "; ".join(note) if note else "none"


def _resolve_estimator(estimator_engine, remote_reference) -> str:
    """Map the user-facing estimator choice + remote_reference to an internal
    aurora engine (``OLS`` / ``RME`` / ``RME_RR``).

    Accepts both the current two-option vocabulary (``Least Squares`` /
    ``Robust``) and the legacy three-value one (``OLS`` / ``RME`` / ``RME_RR``)
    so previously saved workflows keep loading. There is no least-squares +
    remote-reference path by design; requesting one raises a clear error.
    """
    e = str(estimator_engine).strip().lower().replace("_", " ")
    if e in ("ols", "least squares", "ls"):
        if remote_reference:
            raise ValueError(
                "least-squares remote reference is not supported — aurora's "
                "remote-reference estimator is robust (RME_RR). Use estimator "
                "'Robust' with remote reference, or 'Least Squares' without."
            )
        return "OLS"
    if e in ("rme", "rme rr", "robust"):
        return "RME_RR" if remote_reference else "RME"
    raise ValueError(f"unknown estimator engine: {estimator_engine!r}")


def _normalize_fc_layout(fc):
    """Normalize an fc to a ``(frequency, time)`` data layout.

    emptyUI's :func:`compute_fc_levels` produces ``(frequency, time)``; mth5's FC
    persistence stores ``(time, frequency)`` and ``to_xarray`` returns that
    layout. The band loop indexes ``fc[channel][f_idx, :]`` (frequency rows),
    so read-back FCs must be transposed first.
    """
    if "frequency" not in fc.dims or "time" not in fc.dims:
        raise ValueError("fc must have 'frequency' and 'time' dimensions")
    first = next(iter(fc.data_vars))
    if fc[first].dims[0] != "frequency":
        fc = fc.transpose("frequency", "time")
    return fc


def _time_key(t):
    """Comparable int64 key for a window-time array (datetime64 -> ns epoch,
    float seconds as-is)."""
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").astype(np.int64)
    if t.dtype == object:
        # tz-aware fc time coords (the RunTS start carries a UTC offset, so
        # the built coord is a tz-aware DatetimeIndex) expose ``.values`` as
        # object arrays of ``datetime.datetime`` — astype(np.float64) on
        # those raises "float() argument must be a string or a real number".
        # Normalize to UTC epoch ns (naive = UTC, the repo-wide convention).
        idx = pd.DatetimeIndex(t)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        return idx.tz_localize(None).astype("int64")
    return t.astype(np.float64)


def _pair_remote_times(local_t, ref_t, tol_s):
    """One-to-one window-time matching -> (local_idx, ref_idx).

    ``tol_s == 0``: exact-value intersection (sort + searchsorted — the
    epoch-anchored grids of two GPS-locked stations coincide exactly where
    their coverage overlaps). ``tol_s > 0``: greedy nearest matching within
    the tolerance, closest pairs first (GPS-stray clocks shift a station's
    whole grid by its clock error; the closest pairs still pair 1:1).
    Unmatched rows on either side are the caller's business (reported and
    dropped — the SSMT2000 semantics: a slot whose clocks cannot match is
    dropped, not interpolated).
    """
    li, ri = _time_key(local_t), _time_key(ref_t)
    if tol_s and tol_s > 0:
        tol = float(tol_s)
        if np.issubdtype(np.asarray(local_t).dtype, np.datetime64):
            tol_ns = round(tol * 1e9)
        else:
            tol_ns = tol
        # candidate pairs: each local's nearest ref (within tol), greedy
        # assignment closest-diff-first, one-to-one. NB: no int() cast —
        # relative-second axes are float64 and truncating to int produced
        # false pairs (100.2 vs 101.8 paired at tol=1.0) and failed exact
        # matches for sub-second grids
        pos = np.searchsorted(ri, li)
        cand = []
        for k, p in enumerate(pos):
            for j in (p - 1, p):
                if 0 <= j < ri.size:
                    d = abs(li[k] - ri[j])
                    if d <= tol_ns:
                        cand.append((d, k, j))
        cand.sort()
        used_l, used_r, li_out, ri_out = set(), set(), [], []
        for _d, k, j in cand:
            if k in used_l or j in used_r:
                continue
            used_l.add(k)
            used_r.add(j)
            li_out.append(k)
            ri_out.append(j)
        # M11: the greedy match is closest-first, but the caller indexes the
        # pairs by position (Coherence Sort's ensemble is a sliding window over
        # array index), so return them in local time order
        order = np.argsort(np.asarray(li_out, dtype=int), kind="stable")
        return (np.asarray(li_out, dtype=int)[order],
                np.asarray(ri_out, dtype=int)[order])
    # exact intersection
    order = np.argsort(ri, kind="stable")
    rs = ri[order]
    pos = np.searchsorted(rs, li)
    ok = pos < rs.size
    idx = np.clip(pos, 0, max(rs.size - 1, 0))
    ok &= rs[idx] == li
    # L-A12: several local rows with the SAME window time (possible in
    # merged databases) would all map onto the same remote row — reusing
    # one remote window double-counts its information; keep the first
    li_out, ri_out = [], []
    # M22: a boolean mask over the remote rows makes the reuse check O(1); a
    # list membership test per row was quadratic (37 s at 10^5 windows)
    used = np.zeros(ri.size, dtype=bool)
    for k, j in zip(np.nonzero(ok)[0], order[pos[ok]]):
        if used[j]:
            continue
        used[j] = True
        li_out.append(k)
        ri_out.append(j)
    return (np.asarray(li_out, dtype=int), np.asarray(ri_out, dtype=int))


def _one_sample_s(local_fc, ref_fc) -> float:
    """M9: one sample of the level's rate, in seconds — the pairing floor for
    records that start off the sample grid (their labels differ by up to one
    sample). A small epsilon absorbs float rounding at the boundary. Returns
    0.0 when the rate is not recorded on either fc (exact pairing then)."""
    for fc in (local_fc, ref_fc):
        fs = float((getattr(fc, "attrs", None) or {}).get("sample_rate") or 0.0)
        if fs > 0:
            return 1.0 / fs + 1e-6
    return 0.0


def _pair_remote_fc(local_fc, ref_fc, pairing="intersect", tol_s=0.0,
                    level=None):
    """Validate + pair the remote-reference fc to the local window grid.

    Frequency grids stay STRICT (a mismatch is a window-settings/duty-cycle
    inconsistency — the official SSMT2000 refuses to work there too). The
    TIME dimension follows the ``pairing`` mode:

    - ``"intersect"`` (default): match windows by absolute start time
      (exact, or within ``tol_s`` — never less than one sample of the level,
      M9, so sub-sample record offsets still pair — for GPS-stray clocks) and
      drop the
      unmatched rows on BOTH sides with a loud warning naming counts and —
      when the per-window ``run_id`` coordinate exists (Build FC) — the
      affected slots. This is the slot-level pairing made de-facto: the
      coarse layer is the Build FC window (§9.3), the fine layer trims the
      residual coverage differences. Identical grids are a fast no-op.
    - ``"strict"``: the legacy identical-grid requirement (parity
      harnesses; the pre-burst single-rate contract).

    Returns ``(local_fc, ref_fc)`` — both row-reduced to the matched
    windows (the positional row contract requires both sides aligned 1:1).
    """
    if not np.array_equal(np.asarray(ref_fc["frequency"].values),
                          np.asarray(local_fc["frequency"].values)):
        raise ValueError(
            "remote-reference fc has a different frequency grid than the "
            "local fc — compute both with the same window settings")
    lt, rt = local_fc["time"].values, ref_fc["time"].values
    if np.array_equal(lt, rt):
        return local_fc, ref_fc
    if pairing == "strict":
        raise ValueError(
            "remote-reference fc has a different time grid than the local "
            "fc — the runs are not aligned; crop both to their common "
            "acquisition overlap (Align Runs) before computing FCs, or "
            "use pairing='intersect' to match window times with row drops"
        )
    tol_eff = max(float(tol_s or 0.0), _one_sample_s(local_fc, ref_fc))
    li, ri = _pair_remote_times(lt, rt, tol_eff)
    if li.size == 0:
        raise ValueError(
            f"no remote-reference windows match the local time grid "
            f"(0 of {lt.size} local windows)"
            + (f" within {tol_eff:g} s" if tol_eff else "")
            + " — check the clock-zero anchors (both stations must use "
            "'epoch' or the same explicit anchor), the window settings, "
            "or raise 'Pair tolerance (s)' for GPS-stray clocks")
    n_local_drop, n_ref_drop = lt.size - li.size, rt.size - ri.size
    if n_local_drop or n_ref_drop:
        detail = ""
        if "run_id" in local_fc.coords:
            runs = np.asarray(local_fc["run_id"].values)[
                np.setdiff1d(np.arange(lt.size), li)]
            uniq = sorted({str(r) for r in np.unique(runs)})
            if uniq:
                shown = ", ".join(uniq[:5]) + (
                    f" (+{len(uniq) - 5} more)" if len(uniq) > 5 else "")
                detail = f"; affected local slots: {shown}"
        _log.warning(
            "remote-reference pairing%s: kept %d of %d local and %d of %d "
            "remote window(s); dropped %d on each side (unmatched window "
            "times — GPS stray clocks? raise 'Pair tolerance (s)'; or "
            "coverage differences — restrict both stations to the same "
            "Build FC window)%s",
            f" at level {level}" if level is not None else "",
            li.size, lt.size, ri.size, rt.size, n_local_drop, detail,
        )
    return local_fc.isel(time=li), ref_fc.isel(time=ri)


#: Provenance attrs Merge Runs records on the ts.run payload and Compute FC
#: stamps on the fc (M4 #2); every producer/consumer of the chain keys off
#: this single tuple (Compute FC copies them, merge_fc_containers unions
#: them, _warn_remote_gaps reads the first two).
_MERGE_PROVENANCE_ATTRS = (
    "merged_gap_intervals",
    "merged_source_runs",
    "merged_gap_method",
)


def _centre_to_start_s(fc) -> float:
    """Seconds from a window's centre label back to its first sample (M10).

    The fc time labels are scipy segment centres, so a window starts
    ``N / (2 fs)`` before its label (L11 — the centre convention is kept).
    Returns 0.0 when the window size or rate is not recorded on the fc.
    """
    attrs = getattr(fc, "attrs", None) or {}
    n = float(attrs.get("num_samples_window") or 0)
    fs = float(attrs.get("sample_rate") or 0)
    return n / (2.0 * fs) if n > 0 and fs > 0 else 0.0


def _warn_remote_gaps(local_fc, ref_fc) -> None:
    """M4 #2: warn when local windows pair with interpolated remote samples.

    The remote-reference fc's provenance (Merge Runs' ``merged_gap_intervals``
    attr, carried through Align Runs and stamped on the fc by Compute FC)
    records the acquisition gaps its physical merge bridged by interpolation.
    A local run spanning such a gap pairs its windows inside the gap with
    FABRICATED remote samples — the grids still match (that is the point of
    the interpolation), so nothing errors and the estimate silently degrades.

    Burst-mode archives are deliberately out of scope: their slot runs are
    separated by scheduled gaps, the epoch-anchored window grid pairs slots
    across stations by design, and the FC-database route (Build FC / Merge FC)
    never fabricates samples — see DESIGN.md §3 for the rationale.

    M10: a window is flagged when its START lies in a gap. The fc labels are
    window centres (half a window later), so the check shifts them back by
    ``N / (2 fs)`` first; testing the centres missed gaps at the start of a
    window (badly so at decimated levels, where the half window is tens of
    minutes).
    """
    gaps = (getattr(ref_fc, "attrs", None) or {}).get("merged_gap_intervals")
    if not gaps:
        return
    t = np.asarray(local_fc["time"].values)
    if not (np.issubdtype(t.dtype, np.datetime64) or t.dtype == object):
        return  # relative time axes: gap intervals are not interpretable
    try:
        # tz-aware coords (xarray stores pandas tz-aware indexes) expose
        # object arrays of datetime.datetime — DatetimeIndex handles both
        t_idx = pd.DatetimeIndex(t)
    except (ValueError, TypeError):
        return  # object values that are not datetimes
    # naive stamps are assumed UTC (the repo-wide convention); the recorded
    # intervals may be naive while the fc time axis is tz-aware (or vice
    # versa) — normalize both before comparing
    def _as_utc(ts):
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    t_idx = pd.DatetimeIndex(t)
    if t_idx.tz is None:
        t_idx = t_idx.tz_localize("UTC")
    else:
        t_idx = t_idx.tz_convert("UTC")
    # M10: the fc labels are window CENTRES; a window's samples start half a
    # window earlier. Gap membership is tested on the window START (the first
    # sample the window uses), so the centre labels are shifted back first.
    t_idx = t_idx - pd.to_timedelta(_centre_to_start_s(local_fc), unit="s")
    affected = np.zeros(t_idx.size, dtype=bool)
    hits = []
    for g in gaps:
        try:
            gs, ge = _as_utc(pd.Timestamp(g[0])), _as_utc(pd.Timestamp(g[1]))
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "remote-reference fc: malformed gap interval %r in its "
                "provenance attrs (%s) — skipping it, the gap coverage check "
                "is only partial for this level", g, exc)
            continue
        m = (t_idx >= gs) & (t_idx < ge)
        if m.any():
            hits.append(f"[{gs.isoformat()}, {ge.isoformat()})")
            affected |= m
    if not affected.any():
        return
    _log.warning(
        "remote-reference fc carries interpolated gap(s) %s inside the "
        "local run's span — %d of %d local window(s) pair with fabricated "
        "(interpolated) remote samples and silently degrade the estimate. "
        "Use Merge FC (concatenates Fourier coefficients without "
        "fabrication) or exclude the gap intervals via the time-selector "
        "window (M4 #2).",
        ", ".join(hits), int(affected.sum()), t_idx.size,
    )


def _level_ref(ref_levels, level):
    """Remote fc for one decimation level, failing loudly on a partial match.

    ``ref_levels is None`` (nothing connected) is the documented self-reference
    case and is passed through; but a multi-level remote container that lacks
    the local fc's level would silently degrade those levels to a self-reference
    (≈RME) — that is almost always a window/decimation mismatch between the
    two Compute FC nodes, so it must be an error instead.
    """
    if ref_levels is None:
        return None
    if isinstance(ref_levels, dict):
        if level not in ref_levels:
            raise ValueError(
                f"remote-reference fc has no decimation level {level} "
                f"(has {sorted(ref_levels)}) — compute both stations' FCs "
                "with the same decimation/window settings"
            )
        return ref_levels[level]
    return ref_levels


def _simple_coherence(x, y):
    """Pairwise coherence² between two complex series (band ensemble)."""
    denom = np.sum(np.abs(x) ** 2) * np.sum(np.abs(y) ** 2)
    if denom <= 0:
        return 0.0
    return float(np.clip(np.abs(np.sum(x * np.conj(y))) ** 2 / denom, 0.0, 1.0))


def _multiple_coherence(E, H):
    """Per-window multiple coherence² of each output (rows of E) on the
    magnetic inputs (rows of H), over the band's frequency-bin ensemble.

    ``E`` is (n_out, n_bins) and ``H`` is (n_in, n_bins) complex FC samples.
    Standard multiple coherence: coh²_o = (S_oH S_HH⁻¹ S_Ho) / S_oo from the
    bin-ensemble cross-spectral matrices. Bands with too few bins (n_bins <
    n_in + 1) fall back to the minimum pairwise coherence (conservative);
    degenerate (all-zero) outputs score 0.
    """
    n_out, n_bins = E.shape
    n_in = H.shape[0]
    out = np.zeros(n_out)
    if n_bins < n_in + 1:
        for o in range(n_out):
            out[o] = min(_simple_coherence(E[o], H[i]) for i in range(n_in))
        return out
    S_HH_inv = np.linalg.pinv(H @ H.conj().T)
    for o in range(n_out):
        e = E[o]
        S_oo = float(np.real(e @ e.conj()))
        if S_oo <= 0:
            out[o] = 0.0
            continue
        S_Ho = H @ e.conj()
        c = float(np.real(S_Ho.conj() @ (S_HH_inv @ S_Ho)) / S_oo)
        if not np.isfinite(c):
            c = min(_simple_coherence(e, H[i]) for i in range(n_in))
        out[o] = float(np.clip(c, 0.0, 1.0))
    return out


def _coherence_weight(coh2, lo, hi, mode, ramp=0.05):
    """Weight for a window's coherence² outside [lo, hi].

    ``mode``: ``off`` → 1 always; ``drop`` → 0 outside the range; ``weight``
    → smooth trapezoid (linear ramp of width ``ramp`` outside the bounds, so
    borderline windows are down-weighted, not hard-cut). The upper bound
    matters because perfect (≈1) coherence can be a noise/overfit signature.
    """
    if mode == "off":
        return 1.0
    if coh2 < lo:
        if mode == "drop":
            return 0.0
        return float(np.clip((coh2 - (lo - ramp)) / ramp, 0.0, 1.0))
    if coh2 > hi:
        if mode == "drop":
            return 0.0
        return float(np.clip(((hi + ramp) - coh2) / ramp, 0.0, 1.0))
    return 1.0


def _resolve_band_style(band_style):
    """Normalize the ``band_style`` value to one of the two supported
    schemes (DESIGN.md §10.2): ``"l2u"`` (default) — the fixed reference
    template with log-uniform window-starvation coarsening — or
    ``"uniform"`` — the uniform log2-grid partition with merge-based level
    combination (formerly named ``"log"``, legacy alias accepted). The
    retired ``"emtf"`` value is accepted as a legacy alias of ``"l2u"``
    — not because the schemes are identical (the auto-derived disjoint
    EMTF-style table was replaced by the l2u scheme on 2026-09-11), but
    so saved workflows keep running; the retired ``"index"`` mode raises
    a clear error.
    """
    style = str(band_style or "l2u").strip().lower()
    if style == "log":  # legacy alias, identical behavior
        style = "uniform"
    if style == "emtf":  # legacy alias, superseded by l2u (2026-09-11)
        style = "l2u"
    if style not in ("l2u", "uniform", "manual"):
        raise ValueError(
            f"unknown band_style '{band_style}' (expected 'l2u', 'uniform' "
            "or 'manual'; the retired 'index' mode was removed 2026-09-09 — "
            "use 'l2u' for the fixed-template log-uniform scheme)")
    return style


def _per_bin_coherence(local, ref, ins, outs, ensemble_windows=16,
                       smooth_bins=0):
    """Per-frequency multiple coherence squared over a sliding window
    ensemble (history §24.6 - the EMTF-style statistic, M4 #7/E7).

    For every frequency bin f and window w: the multiple coherence squared
    of each output on the magnetic inputs, estimated over the ensemble of
    windows within +-ensemble_windows of w (center included - leave-one-out
    would penalize isolated clean windows), optionally pooled over
    +-smooth_bins adjacent bins first (a FIXED smoothing scale - deliberately
    not the estimate's band table; restores dof at deep levels).
    Adjusted-R2 correction (L-A11 — the implemented identity, not the
    textbook intercept form): ``r2_adj = (r2·(W_seg−1) − (n_in−1)) /
    (W_seg−n_in)`` — the df-corrected form whose null expectation is
    1/(W_seg−n_in) (a small floor at short ensembles; the clip to [0, 1]
    absorbs it).

    Returns ``(coh_e, coh_x)`` - (n_bin, n_win) arrays: coh2 of each output
    on the predicted-H ensemble (remote H when ``ref`` is given), minimum
    across outputs; and the local<->remote magnetic transfer coherence (1.0
    single-FC). NaN samples are excluded from the ensemble (segment counts
    shrink accordingly); segments with <= n_in valid windows score 0.
    """
    import numpy as np

    # L-A9: a ref fc lacking an input channel raised a raw KeyError inside
    # the stack — name the channel(s) and the remedy
    if ref is not None:
        missing = [c for c in ins if c not in ref.data_vars]
        if missing:
            raise ValueError(
                f"remote fc lacks input channel(s) {missing} — compute it "
                "with the same input_channels as the local fc")

    def _stack(ds, names):
        return np.stack([np.asarray(ds[c].values, dtype=complex)
                         for c in names], axis=0)  # (n, F, W)

    E = _stack(local, outs)                            # (O, F, W)
    Hl = _stack(local, ins)                            # (I, F, W)
    Hp = _stack(ref, ins) if ref is not None else Hl   # predicted H
    R = _stack(ref, ins) if ref is not None else None  # transfer target
    F, W = E.shape[1], E.shape[2]

    # window-level validity from the ORIGINAL samples — computed BEFORE any
    # pooling: pooled arrays are 0-filled where invalid (so a NaN bin cannot
    # poison its neighbors), and the validity mask must survive pooling
    ok_e = np.all(np.isfinite(E), axis=0) & np.all(np.isfinite(Hp), axis=0)
    ok_x = (np.all(np.isfinite(Hl), axis=0)
            & (np.all(np.isfinite(R), axis=0) if R is not None else ok_e))

    if smooth_bins > 0:
        def pool(a):  # sum over +-k adjacent bins (bin axis = 1)
            # edge-clamped (no wraparound: the first bins' ensemble must not
            # include the last/Nyquist bins) and NaN-aware: a NaN bin (the
            # DC bin after first-difference recoloring) poisons its
            # neighbors when pooled raw — the whole bin ensemble became
            # NaN -> weight 0 and the adjacent bins silently vanished from
            # the sort. Pooled sums of the VALID samples only (the
            # coherence is a ratio — scale-equivariant, no count needed).
            az = np.where(np.isfinite(a), a, 0)
            out = np.zeros_like(az)
            for k in range(-smooth_bins, smooth_bins + 1):
                b_lo, b_hi = max(0, -k), min(F, F - k)
                if b_hi > b_lo:
                    out[:, b_lo:b_hi, :] += az[:, b_lo + k:b_hi + k, :]
            return out
        E, Hl, Hp = pool(E), pool(Hl), pool(Hp)
        Rp = pool(R) if R is not None else None
    else:
        Rp = R

    w0 = max(0, int(ensemble_windows))

    def _min_coh(Y, X, ok):
        """Per (bin, window) multiple coherence squared of Y's outputs on
        X's inputs over the sliding ensemble; adjusted-R2; min across
        outputs.

        M21: windows are processed in chunks of a size derived from a
        memory budget — each chunk's haloed slice contains every window's
        full ensemble, so the statistic is identical to the unchunked
        computation (a chunk covering all windows IS the pre-M21 path,
        bit-for-bit). The chunk size is derived per call from the channel
        counts, so small inputs (every test fixture) take the single-chunk
        path unchanged.
        """
        n_in = X.shape[0]
        n_out = Y.shape[0]
        # bytes per WINDOW across the chunk's live intermediates
        # (complex128): cs_hh I·I + cs_he I·O + cs_oo O + the S_* mirrors
        # + pinv I·I + t O·I — ~3I² + 3IO + O fields of F·16 bytes
        bytes_per_window = F * 16 * (3 * n_in * n_in + 3 * n_in * n_out + n_out)
        chunk = max(1, int(_COH_BUDGET_BYTES // max(bytes_per_window, 1)))
        if chunk < W:
            _log.info(
                "coherence sort: %d windows exceed the %.0f MiB transient "
                "budget for %d bins x %d inputs x %d outputs — processing "
                "in %d haloed chunk(s) (~%.1f GiB transient avoided)",
                W, _COH_BUDGET_BYTES / 2**20, F, n_in, n_out,
                -(-W // chunk), F * W * 16 * (3 * n_in * n_in
                                              + 3 * n_in * n_out + n_out) / 2**30)
        coh = np.empty((F, W))
        for c0 in range(0, W, chunk):
            c1 = min(c0 + chunk, W)
            # haloed slice: every window in [c0, c1) has its full ensemble
            # (±w0) inside [m0, m1)
            m0, m1 = max(0, c0 - w0), min(W, c1 + w0)
            Xc = X[:, :, m0:m1]
            Yc = Y[:, :, m0:m1]
            okc = ok[:, m0:m1]
            L = m1 - m0
            pad_c = np.zeros((F, 1, n_in, n_in), dtype=complex)
            pad_he = np.zeros((F, 1, n_in, n_out), dtype=complex)
            pad_oo = np.zeros((F, 1, n_out), dtype=complex)
            Xz = np.where(okc[None], Xc, 0)
            Yz = np.where(okc[None], Yc, 0)
            cnt = np.cumsum(np.concatenate(
                [np.zeros((F, 1), dtype=int), okc.astype(int)], axis=1), axis=1)
            cs_hh = np.cumsum(np.concatenate(
                [pad_c, np.einsum("ifw,jfw->fwij", Xz, Xz.conj())], axis=1),
                axis=1)
            cs_he = np.cumsum(np.concatenate(
                [pad_he, np.einsum("ifw,ofw->fwio", Xz, Yz.conj())], axis=1),
                axis=1)
            cs_oo = np.cumsum(np.concatenate(
                [pad_oo, np.einsum("ofw,ofw->fwo", Yz.conj(), Yz)], axis=1),
                axis=1)
            # window j (global) sits at j-m0 in the slice; its ensemble is
            # [j-w0, j+w0+1) ∩ [0, W) — inside the slice by construction
            idx = np.arange(c0, c1) - m0
            lo = np.maximum(idx - w0, 0)
            hi = np.minimum(idx + w0 + 1, L)  # exclusive
            S_hh = cs_hh[:, hi] - cs_hh[:, lo]               # (F, L, I, I)
            S_he = cs_he[:, hi] - cs_he[:, lo]               # (F, L, I, O)
            S_oo = np.real(cs_oo[:, hi] - cs_oo[:, lo])      # (F, L, O)
            W_seg = (cnt[:, hi] - cnt[:, lo]).astype(float)  # (F, L)
            ok_chunk = ok[:, c0:c1]
            usable = (W_seg > n_in) & ok_chunk
            coh_c = np.zeros((F, c1 - c0, n_out))
            if np.any(usable):
                pinv = np.linalg.pinv(
                    S_hh.reshape(-1, n_in, n_in)).reshape(S_hh.shape)
                t = np.einsum("fwio,fwij->fwjo", S_he.conj(), pinv)
                num = np.real(np.einsum("fwjo,fwjo->fwo", t, S_he))
                with np.errstate(divide="ignore", invalid="ignore"):
                    r2 = num / np.maximum(S_oo, 1e-300)
                # adjusted R2: E[r2 | null] = n_in/(W_seg-1) -> debias
                wseg = np.maximum(W_seg, 1.0)[:, :, None]
                r2 = (r2 * (wseg - 1.0) - (n_in - 1.0)) / np.maximum(
                    wseg - n_in, 1.0)
                coh_c = np.clip(np.nan_to_num(r2, nan=0.0), 0.0, 1.0)
            coh_c[~ok_chunk] = 0.0
            coh[:, c0:c1] = coh_c.min(axis=2)
        return coh

    coh_e = _min_coh(E, Hp, ok_e)
    coh_x = np.full((F, W), 1.0) if Rp is None else _min_coh(Hl, Rp, ok_x)
    return coh_e, coh_x


def _coherence_weight_array(coh, lo, hi, mode, ramp=0.05):
    """Vectorized :func:`_coherence_weight` over an array of coherence²."""
    import numpy as np

    coh = np.asarray(coh, dtype=float)
    if mode == "off":
        return np.ones_like(coh)
    w = np.where(coh < lo, np.clip((coh - (lo - ramp)) / ramp, 0.0, 1.0), 1.0)
    w = np.where(coh > hi, np.clip(((hi + ramp) - coh) / ramp, 0.0, 1.0), w)
    if mode == "drop":
        w = np.where((coh < lo) | (coh > hi), 0.0, w)
    return w


def _per_bin_sort(local, ref, local_out, ref_out, ins, outs, *,
                  coherence_min, coherence_max, mode,
                  apply_below_period, ensemble_windows, smooth_bins):
    """The per-frequency sort body (DESIGN.md §10.3): per-(bin, window)
    weights from :func:`_per_bin_coherence` — banding-independent weighting,
    per-frequency-bin diagnostic rows (one row per bin). The period scope
    and NaN-row grammar follow §24.5 P3."""
    import numpy as np

    freq_vals = np.asarray(local["frequency"].values, dtype=float)
    coh_e, coh_x = _per_bin_coherence(
        local, ref, ins, outs, ensemble_windows=ensemble_windows,
        smooth_bins=smooth_bins)
    coh = np.minimum(coh_e, coh_x) if ref is not None else coh_e  # (F, W)
    weights = _coherence_weight_array(coh, coherence_min, coherence_max, mode)
    thr = (float(apply_below_period)
           if apply_below_period not in (None, "")
           and float(apply_below_period) > 0 else None)
    if thr:
        with np.errstate(divide="ignore"):
            bin_periods = np.where(freq_vals > 0, 1.0 / np.maximum(freq_vals, 1e-300),
                                   np.inf)
        weights = np.where(bin_periods[:, None] >= thr, 1.0, weights)
        coh = np.where(bin_periods[:, None] >= thr, np.nan, coh)
    scale = np.sqrt(weights)
    # fully down-weighted samples are excluded (NaN) rather than zeroed —
    # zero rows would make the remote-reference QR singular (same grammar
    # as the band statistic); estimate_tf skips non-finite rows.
    scale[scale <= 0.0] = np.nan
    scale[freq_vals <= 0.0] = np.nan  # DC excluded (recoloring already NaNs it under PW)
    if mode != "off":
        for oc in outs + ins:
            local_out[oc].values[:] = local_out[oc].values * scale
        if ref_out is not None:
            for ic in ins:
                ref_out[ic].values[:] = ref_out[ic].values * scale
    # diagnostics: one row per frequency bin — banding is Estimate TF's
    # concern (§24.9); the sort reports exactly what it weighted
    rows = []
    W = coh.shape[1]
    for i, fv in enumerate(freq_vals):
        if not np.isfinite(fv) or fv <= 0.0:
            continue  # DC bin — no period, never weighted
        rows.append({
            "period": float(1.0 / fv),
            "n_windows": int(W),
            "n_retained": (int(np.count_nonzero(scale[i, :] > 0))
                           if mode != "off" else int(W)),
            "median_coherence_e": (float(np.nanmedian(coh_e[i, :]))
                                   if np.isfinite(coh_e[i, :]).any() else None),
            "median_coherence_transfer": (float(np.nanmedian(coh_x[i, :]))
                                          if ref is not None and
                                          np.isfinite(coh_x[i, :]).any() else None),
            "median_coherence": (float(np.nanmedian(coh[i, :]))
                                 if np.isfinite(coh[i, :]).any() else None),
            "unsorted": bool(thr and 1.0 / fv >= thr),
        })
    return local_out, ref_out, rows


def coherence_sort_fc(local, ref, input_channels, output_channels,
                      coherence_min=0.25, coherence_max=1.0, mode="weight",
                      apply_below_period=None,
                      ensemble_windows=16, smooth_bins=0,
                      pairing="intersect", tol_s=0.0):
    """EMTF-style per-frequency coherence sort of one decimation level's FC
    (DESIGN.md §10.3).

    L-A9: ``ensemble_windows`` < 1 degenerates the sliding ensemble to the
    window itself — the adjusted-R² then zeroes every weight (the whole
    sort silently rejects everything); reject with an actionable message.

    Returns ``(local_out, ref_out, rows)`` — copies of the FCs with the
    per-(bin, window) coherence weights applied (``weight``: multiplied by
    √w; ``drop``: excluded samples set to NaN), plus per-bin diagnostic
    rows. In two-FC mode the criterion is the **conservative minimum** of
    the E–remote-H signal coherence and the local↔remote magnetic transfer
    coherence (aurora's RME_RR signal-covariance estimate degrades when the
    local and remote magnetic fields are uncorrelated), and a single shared
    weight per row is applied to **both** FCs (exact weighted-LS semantics).

    The statistic is the EMTF-style per-frequency window-ensemble coherence
    over a sliding ±``ensemble_windows`` window ensemble (center included)
    with optional ±``smooth_bins`` frequency pooling; the sample-coherence
    bias is adjusted-R²-corrected. Banding is deliberately absent — the
    weights are per (bin, window), so band construction belongs to
    ``Estimate TF`` alone (§24.9): the sort's diagnostics are one row per
    frequency bin. The legacy band-aggregated statistic was removed with
    the per-bin flip (§24.6): no consumer ever weighted with it after the
    default flip, and its banding parameters duplicated the estimator's.
    """
    if int(ensemble_windows) < 1:
        # L-A9: ensemble_windows=0 degenerates the sliding ensemble to the
        # window itself — the adjusted-R² then zeroes EVERY weight (the
        # sort silently rejects everything); reject at the API level
        raise ValueError(
            f"ensemble_windows must be >= 1 — got {ensemble_windows!r} "
            "(the sliding ensemble would degenerate to the window itself, "
            "zeroing every coherence weight)")
    local = _normalize_fc_layout(local)
    ref = _normalize_fc_layout(ref) if ref is not None else None
    if ref is not None:
        local, ref = _pair_remote_fc(local, ref, pairing=pairing, tol_s=tol_s)
        _warn_remote_gaps(local, ref)
    ins = [c for c in input_channels if c in local.data_vars]
    outs = [c for c in output_channels if c in local.data_vars]
    if not ins or not outs:
        raise ValueError(
            f"could not determine input/output channels from the fc — "
            f"data_vars: {list(local.data_vars)}"
        )

    local_out = local.copy(deep=True)
    ref_out = ref.copy(deep=True) if ref is not None else None

    return _per_bin_sort(
        local, ref, local_out, ref_out, ins, outs,
        coherence_min=coherence_min, coherence_max=coherence_max,
        mode=mode, apply_below_period=apply_below_period,
        ensemble_windows=ensemble_windows, smooth_bins=smooth_bins)


def coherence_sort_fcs(levels, ref_levels, input_channels, output_channels,
                       coherence_min=0.25, coherence_max=1.0, mode="weight",
                       apply_below_period=None,
                       ensemble_windows=16, smooth_bins=0,
                       pairing="intersect", tol_s=0.0):
    """Coherence-sort every decimation level (see :func:`coherence_sort_fc`).

    ``levels`` is the multi-level fc container; ``ref_levels`` is an optional
    same-shaped container (or a single fc) for the remote station. Returns
    ``(levels_out, ref_levels_out, rows)`` where the rows carry the
    decimation level for the diagnostics table (one row per frequency bin
    per level — no band geometry: banding belongs to Estimate TF, §24.9).
    """
    out_levels: dict[int, object] = {}
    out_ref: dict[int, object] = {} if ref_levels is not None else None
    rows: list[dict] = []
    for level, fc in sorted(levels.items()):
        ref = _level_ref(ref_levels, level)
        l_out, r_out, level_rows = coherence_sort_fc(
            fc, ref, input_channels, output_channels,
            coherence_min=coherence_min, coherence_max=coherence_max,
            mode=mode, apply_below_period=apply_below_period,
            ensemble_windows=ensemble_windows, smooth_bins=smooth_bins,
            pairing=pairing, tol_s=tol_s,
        )
        out_levels[int(level)] = l_out
        if ref_levels is not None:
            out_ref[int(level)] = r_out
        for r in level_rows:
            r["decimation_level"] = int(level)
            rows.append(r)
    return out_levels, out_ref, rows


def estimate_tf(
    fc,
    input_channels,
    output_channels,
    estimator_engine="RME_RR",
    remote_reference=False,
    periods_per_octave=3,
    ref_fc=None,
    band_style="l2u",
    bands_override=None,
    t_short=None,
    pairing="intersect",
    pair_tolerance_s=0.0,
):
    """MT transfer-function estimate -> (periods, Z[n_out x n_in complex], Z_err).

    Z rows = outputs (e.g. ex,ey and optionally bz/hz), cols = inputs (hx,hy).
    Z_err holds the standard error of each complex element (same orientation).
    With outputs = [ex,ey] this is the 2x2 impedance; adding a vertical
    magnetic output (bz/hz) extends it to a 3x2 whose third row is the
    geomagnetic transfer function (tipper). The fc already
    holds H for the magnetic inputs (B->H done in compute_fc_levels when the
    instrument is fluxgate), so impedance is in ohm (V/m per A/m).

    ``estimator_engine`` selects the estimator (all aurora-backed):
      - ``Least Squares`` : ordinary least squares (aurora ``estimate_ols``).
      - ``Robust``        : robust M-estimator (aurora ``RME``); with
                            ``remote_reference=True`` uses the robust
                            remote-reference estimator (aurora ``RME_RR``).
    There is intentionally **no least-squares + remote-reference** path: aurora
    offers remote reference only through the robust ``RME_RR``, and we keep
    that design (see history §13). For backward compatibility the legacy
    engine values ``OLS`` / ``RME`` / ``RME_RR`` are still accepted.

    Period bands (DESIGN.md §10.2): ``band_style="l2u"`` (default) uses the
    reference-template band table with log-uniform window-starvation
    coarsening (:func:`_l2u_band_table`) — the same fixed template at every
    decimation level, starved levels rebuilt as a coarser uniform grid;
    ``band_style="uniform"`` uses the uniform log2-grid partition
    (:func:`_period_bands`, ``periods_per_octave`` bands per octave,
    merge-based across overlapping levels). The legacy ``band_style="log"``
    value is accepted as an alias of ``"uniform"`` and ``"emtf"`` as an
    alias of ``"l2u"`` (the retired auto-derived EMTF-style table was
    replaced by l2u on 2026-09-11).

    ``ref_fc`` optionally supplies a *second station's* Fourier coefficients
    used as the remote reference (magnetic channels only). When omitted (and
    remote reference is requested), the local input channels are reused as the
    reference — a self-reference (degenerate but well-defined, equals RME).
    This is also what a duplicated/identical "dummy remote station" reduces to.
    The omission is **warned loudly**: an operator must never believe remote
    reference is in effect when it is not (review 2026-09-28, E2).

    ``t_short`` (head clip, §14.9) is honored for BOTH schemes: the primary
    l2u table clips upstream (:func:`_clip_head_bands`), and the uniform /
    l2u-fallback paths clip in :func:`_resolve_period_bands`.

    FCs read back from an mth5 archive carry a ``(time, frequency)`` layout
    (mth5 ``to_xarray``), while :func:`compute_fc_levels` produces
    ``(frequency, time)``; both are normalized to ``(frequency, time)`` here.
    """

    engine = _resolve_estimator(estimator_engine, remote_reference)

    if engine == "RME_RR" and ref_fc is None:
        # E2: the RR topology degenerates to plain RME — the operator must
        # know (each level logs; _level_ref keeps the per-level pairing
        # semantics of the surrounding loop)
        _log.warning(
            "Estimate TF: remote reference is enabled but no remote fc is "
            "connected (ref_fc empty) — estimating against a SELF-REFERENCE, "
            "which is mathematically equivalent to plain Robust RME: connect "
            "the remote station's fc to the ref input for true remote "
            "reference")

    fc = _normalize_fc_layout(fc)
    freq = fc["frequency"].values
    ok = freq > 0  # exclude DC bin (period would be infinite)
    fc = fc.isel(frequency=ok)

    if ref_fc is not None:
        if engine != "RME_RR":
            # pairing row-reduces BOTH sides — doing it for an estimator
            # that never reads the reference silently dropped local rows
            # (the estimate was computed on fewer windows than asked for,
            # as a side effect of wiring a port)
            _log.warning(
                "remote-reference fc connected but the estimator (%s) does "
                "not use it — estimating from the local fc only (remote "
                "reference requires the robust RME_RR engine)", engine)
        else:
            ref_fc = _normalize_fc_layout(ref_fc)
            ref_fc = ref_fc.isel(frequency=ref_fc["frequency"].values > 0)
            fc, ref_fc = _pair_remote_fc(fc, ref_fc, pairing=pairing,
                                         tol_s=pair_tolerance_s)
            _warn_remote_gaps(fc, ref_fc)

    # Realistic, octave-aligned period bands for this decimation level
    # (DESIGN.md §10.2: band_style selects the uniform log2-grid partition
    # or the reference-template table; bands_override carries a precomputed
    # table (frequency pairs from _l2u_band_table — disjoint by design,
    # nothing to merge) when the container-level derivation succeeded
    # upstream).
    if bands_override is not None:
        periods_out, band_bins_list = _bands_from_freq_pairs(fc, bands_override)
    else:
        periods_out, band_bins_list = _resolve_period_bands(
            fc, periods_per_octave, band_style, t_short=t_short)

    nbands = len(periods_out)
    n_in = len(input_channels)
    n_out = len(output_channels)
    # Z rows = outputs (ex,ey,...,bz), cols = inputs (hx,hy): a 2x2 impedance
    # when outputs are ex/ey, extended to a 3x2 (with a bz/hz tipper row) when a
    # vertical channel is included.
    Zout = np.zeros((nbands, n_out, n_in), dtype=complex)
    Zerr = np.zeros((nbands, n_out, n_in), dtype=float)
    skipped: list[str] = []

    for b in range(nbands):
        band_bins = band_bins_list[b]
        if band_bins.size == 0:
            skipped.append(f"T={periods_out[b]:.4g}s (no bins)")
            continue
        # gather samples across selected frequencies and all windows
        H = []  # inputs  -> [n, n_in]
        E = []  # outputs -> [n, n_out]
        R = []  # reference inputs -> [n, n_in]
        # L-A9: a ref fc lacking an input channel raised a raw KeyError —
        # name the channel and the remedy
        if engine == "RME_RR" and ref_fc is not None:
            missing = [ic for ic in input_channels if ic not in ref_fc.data_vars]
            if missing:
                raise ValueError(
                    f"remote fc lacks input channel(s) {missing} — compute "
                    "it with the same input_channels as the local fc")
        for f_idx in band_bins:
            H.append(np.stack([fc[ic].values[f_idx, :] for ic in input_channels], axis=-1))
            E.append(np.stack([fc[oc].values[f_idx, :] for oc in output_channels], axis=-1))
            if engine == "RME_RR":
                if ref_fc is not None:
                    R.append(np.stack(
                        [ref_fc[ic].values[f_idx, :] for ic in input_channels], axis=-1))
                else:  # self-reference (dummy/duplicated remote station)
                    R.append(np.stack(
                        [fc[ic].values[f_idx, :] for ic in input_channels], axis=-1))
        H = np.concatenate(H, axis=0)
        E = np.concatenate(E, axis=0)
        if engine == "RME_RR":
            R_band = np.concatenate(R, axis=0)
        else:
            R_band = None
        # skip non-finite FC samples (coherence-sort drop/trim sets NaN);
        # for remote reference the union of local/remote rows is dropped so
        # the row pairing stays aligned (aurora RME_RR rejects NaN).
        ok = np.all(np.isfinite(H), axis=1) & np.all(np.isfinite(E), axis=1)
        if R_band is not None:
            ok = ok & np.all(np.isfinite(R_band), axis=1)
        if not np.all(ok):
            H = H[ok]
            E = E[ok]
            if R_band is not None:
                R_band = R_band[ok]
        if H.shape[0] <= len(input_channels):
            skipped.append(
                f"T={periods_out[b]:.4g}s (only {H.shape[0]} retained "
                f"sample row(s), need > {len(input_channels)})"
            )
            continue
        Zout[b], Zerr[b] = _solve_band(H, E, R_band, input_channels, output_channels, engine)

    if skipped:
        _log.warning(
            "Estimate TF: %d band(s) had no usable samples and were skipped "
            "— typically every window was down-weighted to zero by coherence "
            "sorting: %s",
            len(skipped), "; ".join(skipped),
        )

    # All estimators now come from aurora and share aurora's impedance
    # convention, so no sign/phase rotation is applied.
    return periods_out, Zout, Zerr


def _period_bands(fc, periods_per_octave=4):
    """Uniform log2-grid period bands -> (band-center periods, edges).

    The ``"uniform"`` band scheme (formerly named ``"log"``): each level's
    highest trusted frequency is **Fs/4** (so the
    shortest period is 4/Fs, avoiding the noisy Nyquist-edge bins) and its
    longest period is **window_length/4** (N/(4·Fs)). Band centers sit on a
    global log2 (octave) grid with ``periods_per_octave`` points per octave
    anchored at 4/Fs. Because decimation factors are powers of 2, consecutive
    levels' grids land on the *same* absolute periods wherever they overlap —
    so the merged TF has consistent frequency points across levels (and can be
    combined by the merge). Returns (periods shortest→longest, log2-frequency
    edges high→low) for the :func:`estimate_tf` band loop.
    """
    fs = float(fc.attrs.get("sample_rate") or 0.0)
    if not fs:
        # fc without attrs (e.g. read back from mth5 before enrichment): the
        # Nyquist bin is fs/2, so the native rate is recoverable from the grid.
        fs = 2.0 * float(fc["frequency"].values.max())
    ppo = max(1, int(periods_per_octave))
    p_short = 4.0 / fs  # Fs/4 as the highest trusted frequency

    N = int(fc.attrs.get("num_samples_window") or 0)
    if not N:
        wl = fc.attrs.get("window_length_s")
        if wl:
            N = round(float(wl) * fs)
    p_long = N / (4.0 * fs) if N else None  # window_length / 4

    longest_present = 1.0 / float(fc["frequency"].values[fc["frequency"].values > 0].min())
    if p_long is None or p_long > longest_present:
        p_long = longest_present
    if p_long <= p_short:
        p_long = longest_present

    M = max(0, int(np.ceil(ppo * np.log2(p_long / p_short))))
    logp = np.log2(p_short) + np.arange(M + 1) / ppo
    periods_out = 2.0 ** logp  # shortest -> longest

    # binning edges in log2 frequency (high -> low), one boundary per band
    logf = -logp
    d = 1.0 / ppo
    edges = np.empty(M + 2)
    edges[0] = logf[0] + d / 2.0
    edges[1:M + 1] = (logf[:-1] + logf[1:]) / 2.0
    edges[M + 1] = logf[-1] - d / 2.0
    return periods_out, edges


def _container_geometry(levels):
    """Common per-level geometry of a multi-level fc container, or ``None``
    when it cannot be derived (missing attrs, non-constant window). Returns
    a list (shallow→deep) of dicts with ``fs`` (level sample rate), ``df``
    (bin spacing), ``f_ceil`` (Fs/4) and ``f_floor`` (window/4); the
    template applies per level through each level's own df (L-A6: the
    cascade ``factor`` field was deleted — levels sort ascending by id,
    so the pairwise rate ratio computed r/s = 1/cascade could never reach
    the >= 2 gate; the field had no consumers)."""
    items = sorted(levels.items())
    infos = []
    window = None
    for _, fc in items:
        fs = float(fc.attrs.get("sample_rate") or 0.0)
        if not fs:
            return None
        n = int(fc.attrs.get("num_samples_window") or 0)
        if not n:
            wl = fc.attrs.get("window_length_s")
            if not wl:
                return None
            n = round(float(wl) * fs)
        if window is None:
            window = n
        elif n != window:  # windows must be constant across levels (D2)
            return None
        infos.append({"fs": fs, "df": fs / n,
                      "f_ceil": fs / 4.0, "f_floor": 4.0 * fs / n})
    return {"levels": [lv for lv, _ in items], "infos": infos,
            "window": window}


#: Reference band template in window-128 native bins (DESIGN.md §10.2):
#: the classic EMTF ladder the official USArray grid repeats verbatim at
#: levels 1-3 (L1-L3 = {[15-16],[12],[9-10],[7-8],[6],[5]}), hand-crafted
#: ~3 bands/octave. Scaled to the container's window (×window/128), the
#: SAME template is used at every decimation level — the periods scale by
#: the cascade factor per level (the official grid's structure).
_L2_TEMPLATE_128 = ((15, 16), (12, 12), (9, 10), (7, 8), (6, 6), (5, 5))


def _l2u_template(window):
    """The reference template scaled to the container's window (×window/128,
    rounded, overlaps merged, clamped to the valid bin range). At window 128
    and 256 this returns the official template family exactly (the official's
    window-256 template is the doubled form); other windows get the closest
    scaled equivalent."""
    scale = float(window) / 128.0
    out = []
    for k1, k2 in _L2_TEMPLATE_128:
        a = int(np.floor(k1 * scale + 0.5))
        b = int(np.floor(k2 * scale + 0.5))
        a = max(a, 1)
        b = min(b, int(window) // 2)
        if a <= b:
            if out and b >= out[-1][0]:  # descending k: overlap check
                out[-1] = (min(out[-1][0], a), max(out[-1][1], b))
            else:
                out.append((a, b))
    return tuple(out)


def _l2u_intervals(lv, template, geom):
    """Map the template's bin indices to frequency intervals for one level
    (same bins at every level; the level's own df scales the periods by the
    cascade factor per level — the official grid's structure)."""
    infos = geom["infos"]
    idx = geom["levels"].index(lv)
    df = infos[idx]["df"]
    return [((k2 + 0.5) * df, (k1 - 0.5) * df) for k1, k2 in template]


def _l2u_band_table(levels, min_bins_per_band=20.0, t_short=None):
    """Reference-template band table with log-uniform window-starvation
    coarsening (DESIGN.md §10.2, the default frequency-grid method).

    The reference template (:func:`_l2u_template`) is applied to EVERY
    decimation level (same bin indices; periods scale by the cascade factor
    per level). Per level, if every band's total samples (n_windows × band
    width) exceed ``min_bins_per_band``, the template is kept unchanged;
    otherwise the level's grid is REBUILT as a coarser uniform table:
    ``n - floor(n/3)`` bands, each widened to the minimal width whose total
    samples exceed the threshold, re-spaced uniformly in log period across
    the template's span (endpoints included), bands sharing bins dropped
    until no overlap or ≤2 bands remain. This reproduces the official
    USArray grid's strategy: the ppo-dense head untouched, the
    window-starved deepest levels coarsened to the official counts. The
    head clip ``t_short`` drops bands whose center period is shorter
    (default None = no clip).

    Returns ``{level: [(f_hi, f_lo), ...]}`` (frequency intervals,
    high→low), or ``None`` when the container lacks the metadata needed to
    derive it (callers fall back to the per-level bin walk).
    """
    geom = _container_geometry(levels)
    if geom is None:
        return None
    template = _l2u_template(geom["window"])
    out = {}
    for lv, fc in sorted(levels.items()):
        n_windows = int(fc.sizes.get("time", 0))
        if n_windows:
            # L-A9: count USABLE windows — a coherence-sort drop NaN'd the
            # dead windows and they must not count toward the starvation
            # threshold (the pre-pairing trim remains unknowable here: the
            # per-level pairing happens inside estimate_tf)
            first = next(iter(fc.data_vars.values()))
            arr = np.asarray(first.values)
            if arr.ndim == 2:
                # reduce across BINS (the non-window axis): layout is
                # (frequency, time) or (time, frequency)
                bin_axis = 0 if arr.shape[1] == n_windows else 1
                usable = int(np.any(np.isfinite(arr), axis=bin_axis).sum())
                if usable < n_windows:
                    _log.warning(
                        "band table: level %d counts %d usable window(s) "
                        "of %d (the rest are coherence-dropped)", lv,
                        usable, n_windows)
                n_windows = usable
        if n_windows <= 0 or len(template) <= 2:
            out[lv] = _l2u_intervals(lv, template, geom)
            continue
        cur = [list(b) for b in template]
        totals = [n_windows * (k2 - k1 + 1) for k1, k2 in cur]
        if not min_bins_per_band or min(totals) > min_bins_per_band:
            out[lv] = _l2u_intervals(lv, template, geom)
            continue
        df = geom["infos"][geom["levels"].index(lv)]["df"]
        p_lo = 1.0 / (np.sqrt(cur[0][0] * cur[0][1]) * df)   # shortest period
        p_hi = 1.0 / (np.sqrt(cur[-1][0] * cur[-1][1]) * df)  # longest period
        k_lo_min = cur[-1][0]  # template span: k descends (short → long)
        k_hi_max = cur[0][1]
        w = int(np.floor(min_bins_per_band / n_windows)) + 1
        # w = floor(m/n)+1 => n*w > m always, so the "cannot satisfy" branch
        # below is unreachable by construction (kept out — dead code)
        n_target = max(2, len(cur) - len(cur) // 3)
        chosen = None
        while n_target >= 2:
            # log-uniform placement covering the template's period span
            # (endpoints included — the coarsened grid keeps the coverage)
            centers = np.geomspace(p_lo, p_hi, n_target)
            bands = []
            for c in centers:
                kc = 1.0 / (c * df)
                k1 = int(np.floor(kc - (w - 1) / 2.0 + 0.5))
                k2 = k1 + w - 1
                # clamp into the template's bin span by SHIFTING (keep the
                # width — truncating would drop below the threshold)
                if k1 < k_lo_min:
                    k1, k2 = k_lo_min, k_lo_min + w - 1
                if k2 > k_hi_max:
                    k2, k1 = k_hi_max, k_hi_max - w + 1
                bands.append((k1, k2))
            bands.sort()
            overlap = any(bands[i][1] >= bands[i + 1][0]
                          for i in range(len(bands) - 1))
            chosen = bands
            if not overlap:
                break
            if n_target == 2:
                # two bands cannot fit disjointly at this width (the clamp
                # shifts both inward — extreme starvation): shrink the width
                # until they tile the span (best effort — the threshold can
                # no longer be met, the documented floor behavior)
                w = min(w, (k_hi_max - k_lo_min + 1) // 2)
                chosen = [(k_hi_max - w + 1, k_hi_max),
                          (k_lo_min, k_lo_min + w - 1)]
                break
            n_target -= 1
        # shortest → longest period (descending k), like the template path
        chosen = sorted(chosen, reverse=True)
        out[lv] = [((k2 + 0.5) * df, (k1 - 0.5) * df) for k1, k2 in chosen]
    out = _resolve_cross_level_claims(out, levels)
    return _clip_head_bands(out, t_short)


def _resolve_cross_level_claims(out, levels):
    """Deepest-rate-first period tiling across decimation levels
    (history §24.8): on aligned single-rate cascades the template's level spans
    already abut (nothing to resolve), but a burst archive's rate families
    are NOT related by the cascade factor (150/15 = 10) — their level spans
    overlap and the same periods would be estimated twice with slightly
    different band centers (the duplicate/overlapping long-period points).
    The same     deepest-wins rule ``merge_tf_levels`` applies, applied here at
    the SOURCE so the sort diagnostics and the estimate share one table: a
    period is estimated by the deepest (lowest-rate) level that covers it;
    shallower levels' bands are clipped to the unclaimed remainder. A band
    straddling a claim can leave SEVERAL unclaimed pieces — every piece is
    kept as its own band (each with its own geometric-mean center): the old
    keep-only-the-center-piece rule turned the other pieces into permanent
    coverage holes estimated by nobody (L-A8)."""
    def rate_of(lv):
        return float(levels[lv].attrs.get("sample_rate", np.inf))

    claimed = []  # merged ascending period intervals
    for lv in sorted(out, key=rate_of):  # deepest (lowest rate) first
        kept = []
        for f_hi, f_lo in out[lv]:
            t_lo, t_hi = 1.0 / f_hi, 1.0 / f_lo  # band's period span
            # subtract the claimed intervals
            seg = [(t_lo, t_hi)]
            for c0, c1 in claimed:
                nxt = []
                for a, b in seg:
                    if c1 <= a or c0 >= b:
                        nxt.append((a, b))
                        continue
                    if c0 > a:
                        nxt.append((a, c0))
                    if c1 < b:
                        nxt.append((c1, b))
                seg = nxt
                if not seg:
                    break
            if not seg:
                continue  # fully claimed by deeper levels
            # L-A8: EVERY unclaimed piece becomes its own band (its own
            # geometric-mean center) — keeping only the piece closest to
            # the original band center left the other pieces as permanent
            # coverage holes between deeper claims
            for piece in sorted(seg, key=lambda p: p[0]):
                kept.append((1.0 / piece[0], 1.0 / piece[1]))
            # claim this band's span (merged with touching claims)
            new = (t_lo, t_hi)
            merged_claims, pending = [], new
            for c0, c1 in claimed:
                if pending[1] < c0 or pending[0] > c1:
                    merged_claims.append((c0, c1))
                else:
                    pending = (min(pending[0], c0), max(pending[1], c1))
            merged_claims.append(pending)
            claimed = sorted(merged_claims)
        out[lv] = kept
    return out


def _clip_head_bands(owned, t_short):
    """Head clip (§14.9): drop bands whose center period is shorter than
    ``t_short`` seconds (no-op for None/0 — legacy behavior)."""
    if not t_short or t_short <= 0:
        return owned
    out = {}
    for lv, pairs in owned.items():
        kept = [(hi, lo) for hi, lo in pairs
                if 1.0 / np.sqrt(hi * lo) >= t_short]
        if kept:
            out[lv] = kept
    return out


def _bands_from_freq_pairs(fc, pairs):
    """Map disjoint ``(f_hi, f_lo)`` intervals to integer bin positions on
    ``fc``'s OWN frequency axis (layout-agnostic: works with or without the
    DC bin, ``(frequency, time)`` or ``(time, frequency)`` — call after
    layout normalization). Bin membership is by frequency value, so bins are
    disjoint across bands automatically; a band with no bin in its interval
    is kept as an empty entry (callers skip empties). Band periods are the
    geometric means of the ACTUAL selected edge bins, so each reported
    frequency is self-consistent with the data that produced it — and since
    every band is owned by exactly one level, no cross-level label alignment
    is needed and the merge is a no-op.

    Returns ``(periods_out, band_bins)`` — periods shortest→longest.
    """
    freq_vals = np.asarray(fc["frequency"].values, dtype=float)
    periods_out = []
    band_bins = []
    for hi, lo in pairs:
        k_lo = int(np.searchsorted(freq_vals, lo, side="right"))
        k_hi = int(np.searchsorted(freq_vals, hi, side="right")) - 1
        if k_hi < k_lo:
            periods_out.append(np.nan)  # no bin of this fc in the interval
            band_bins.append(np.empty(0, dtype=int))
            continue
        periods_out.append(1.0 / np.sqrt(freq_vals[k_lo] * freq_vals[k_hi]))
        band_bins.append(np.arange(k_lo, k_hi + 1))
    return np.array(periods_out), band_bins


def _grid_nearest_keep(reference, other, rtol=1e-3):
    """Boolean mask over ``reference`` bins that have a near-equal partner
    in ``other`` (relative tolerance — float grid drift tolerance)."""
    keep = np.zeros(reference.size, dtype=bool)
    pos = np.searchsorted(other, reference)
    for side in (-1, 0):
        idx = np.clip(pos + side, 0, max(other.size - 1, 0))
        keep |= np.isclose(reference, other[idx], rtol=rtol, atol=0.0)
    return keep


def _union_merge_provenance_attrs(target: dict, parts: list) -> None:
    """Union Merge Runs' provenance across one level's FC contributors.

    The FC merge keeps the FIRST contributor's attrs (``merged.attrs =
    dict(base.attrs)``), so an interpolated gap recorded on contributor 2+
    was dropped — the remote-reference pairing then never learned that those
    remote samples were fabricated (M4 #2). The list-shaped keys union across
    every contributor; the scalar ``merged_gap_method`` keeps the first
    recorded one.
    """
    list_keys = _MERGE_PROVENANCE_ATTRS[:2]
    method_key = _MERGE_PROVENANCE_ATTRS[2]
    for key in list_keys:
        union: list = []
        seen: set[str] = set()
        for part in parts:
            value = (getattr(part, "attrs", None) or {}).get(key)
            if value is None:
                continue
            for item in value:
                if str(item) not in seen:
                    seen.add(str(item))
                    union.append(item)
        if union:
            target[key] = union
    for part in parts:
        method = (getattr(part, "attrs", None) or {}).get(method_key)
        if method is not None:
            target[method_key] = method
            break


def merge_fc_containers(containers):
    """Concatenate per-run FC containers along time, per rate level.

    The FC-database merge (aurora ``merge_stfts``; history §24.5 P1 v2) for
    runs that must not be concatenated in the time domain: physically
    non-adjacent runs (gaps) and burst-mode acquisition (thousands of short
    recurring slots — a merged-TS decimation cascade cannot cross slot
    boundaries, per-run cascades are fine and deep levels starve naturally).

    Each input is a single fc Dataset (level 0) or a ``{level: Dataset}``
    container. Levels are keyed by their ACTUAL sample rate (from the
    ``sample_rate`` attr), not by decimation level index: a burst archive
    mixes rate families (2400 Hz slots + 150 Hz slots + 15 Hz continuous),
    whose cascade levels do not correspond by index — the merged output is
    re-keyed to consecutive ints in DESCENDING rate order (level 0 = the
    highest rate; single-rate inputs reproduce the input indices exactly).
    Contributing inputs at one rate should share the window settings
    (warning — mixed settings change the banding); the frequency axes are
    intersected with a small relative tolerance (fast path: bit-identical),
    channels missing in some inputs are dropped with a warning, and the
    windows are concatenated in time-sorted order (duplicate/overlapping
    window times warn). Levels missing in some inputs merge the rest.

    Returns ``{level: xr.Dataset}`` with the first contributor's attrs,
    plus Merge Runs' gap provenance unioned over all contributors
    (:func:`_union_merge_provenance_attrs`).
    """
    import xarray as xr

    norm: list[dict[int, xr.Dataset]] = []
    for c in containers:
        if c is None:
            continue
        if isinstance(c, dict):
            norm.append({int(lv): ds for lv, ds in c.items()})
        else:
            norm.append({0: c})
    if not norm:
        raise ValueError("no fc to merge")
    # re-key by actual sample rate: the rate — not the decimation level
    # index — is the identity that makes cascades from different rate
    # families commensurable (history §24.8)
    by_rate: dict[float, list[xr.Dataset]] = {}
    for c in norm:
        for ds in c.values():
            by_rate.setdefault(
                round(float(ds.attrs.get("sample_rate", 0.0)), 9), []).append(ds)
    out: dict[int, xr.Dataset] = {}
    for i, rate in enumerate(sorted(by_rate, reverse=True)):
        parts = by_rate[rate]
        base = parts[0]
        for j, p in enumerate(parts[1:], start=2):
            for key in ("num_samples_window", "overlap", "window_type"):
                if str(p.attrs.get(key, "")) != str(base.attrs.get(key, "")):
                    _log.warning(
                        "merge fc: rate %g contributor %d %s=%r differs from "
                        "%r — mixed window settings change the banding",
                        rate, j, key, p.attrs.get(key), base.attrs.get(key))
        vars0 = [str(v) for v in base.data_vars]
        common = set(vars0)
        for p in parts[1:]:
            common &= {str(v) for v in p.data_vars}
        if not common:
            raise ValueError(
                "merge fc: inputs share no channels: "
                f"{[[str(v) for v in p.data_vars] for p in parts]}")
        dropped = [v for v in vars0 if v not in common]
        if dropped:
            _log.warning(
                "merge fc: channels %s are missing in some input — dropped "
                "from the merged FC", dropped)
        keep_vars = [v for v in vars0 if v in common]

        freqs = [np.asarray(p["frequency"].values, dtype=float) for p in parts]
        f0 = freqs[0]
        aligned = all(np.array_equal(f0, g) for g in freqs[1:])
        sliced = []
        if aligned:
            sliced = list(parts)
        else:
            keep = np.ones(f0.size, dtype=bool)
            for g in freqs[1:]:
                keep &= _grid_nearest_keep(f0, g)
            n_kept = int(np.count_nonzero(keep))
            if n_kept == 0:
                raise ValueError(
                    "merge fc: input frequency grids do not intersect at "
                    f"rate {rate}")
            if n_kept < f0.size:
                _log.warning(
                    "merge fc: rate %g frequency grids differ — intersected "
                    "to %d of %d bins (mixed window settings?)", rate, n_kept,
                    f0.size)
            f0 = f0[keep]
            for p, g in zip(parts, freqs):
                # this part's nearest partner bin for every kept reference
                # bin, re-labelled onto the reference grid (concat must see
                # identical frequency coordinates or xarray outer-joins them)
                pos = np.searchsorted(g, f0)
                left = np.clip(pos - 1, 0, max(g.size - 1, 0))
                right = np.clip(pos, 0, max(g.size - 1, 0))
                use_left = np.isclose(f0, g[left], rtol=1e-3, atol=0.0)
                use_right = np.isclose(f0, g[right], rtol=1e-3, atol=0.0)
                pi = np.where(
                    use_left & (~use_right
                                | (np.abs(f0 - g[left])
                                   <= np.abs(f0 - g[right]))),
                    left, right)
                if not np.all(np.isclose(f0, g[pi], rtol=1e-3, atol=0.0)):
                    raise ValueError(
                        "merge fc: input frequency grids do not intersect "
                        f"at rate {rate}")
                sliced.append(
                    p.isel(frequency=pi).assign_coords(frequency=f0))
        merged = xr.concat(sliced, dim="time")
        merged = merged.sortby("time")
        times = merged["time"].values
        if times.size > 1 and np.any(np.diff(times) == np.timedelta64(0)):
            _log.warning(
                "merge fc: rate %g has duplicate/overlapping window times "
                "across inputs", rate)
        merged = merged[keep_vars]
        merged.attrs = dict(base.attrs)
        _union_merge_provenance_attrs(merged.attrs, parts)
        out[i] = merged
    return out


def manual_bands(freqs, counts) -> dict[int, list[tuple[float, float]]]:
    """Per-level ``(f_hi, f_lo)`` band table from band-center frequencies.

    The EMTF band-table reconstruction used by the SDG31 parity harness
    (`exact_official_bands`, history §24.4 item 5 / §24.5 P4): ``freqs`` is
    the flat list of band-center frequencies (any order — normalized to
    descending), ``counts`` the number of bands per decimation level as a
    comma string or sequence (level 0 = the highest frequencies, e.g.
    ``"6,6,6,6,4,2"``). Band edges sit at the geometric midpoints between
    adjacent centers; the outer edges extend by the same geometric ratio.
    Each level needs >= 2 centers (a lone band has no midpoint to anchor
    its edges).

    Returns ``{level: [(f_hi, f_lo), ...]}`` — feed per level into
    ``estimate_tf(..., bands_override=...)``. ``freqs``/``counts`` accept
    comma-separated strings or sequences.
    """
    def _numbers(value, what):
        if isinstance(value, str):
            parts = [x for x in value.split(",") if x.strip()]
        else:
            try:
                parts = list(value)
            except TypeError:
                parts = [value]
        try:
            return np.asarray([float(x) for x in parts], dtype=float)
        except ValueError as exc:
            raise ValueError(f"{what}: cannot parse values ({exc})") from exc

    f = _numbers(freqs, "band_centers")
    if f.size == 0 or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("band_centers must be finite positive frequencies")
    f = np.sort(f)[::-1]  # descending: level 0 = highest frequencies
    if np.any(np.diff(f) == 0):
        raise ValueError("band_centers contains duplicate frequencies")
    c = [int(x) for x in _numbers(counts, "bands_per_level")]
    if not c or any(k < 1 for k in c):
        raise ValueError("bands_per_level must be >= 1 per level")
    if any(k < 2 for k in c):
        raise ValueError(
            f"bands_per_level: every level needs >= 2 band centers to "
            f"reconstruct edges (got {c})")
    if sum(c) != f.size:
        raise ValueError(
            f"bands_per_level sums to {sum(c)} but {f.size} band centers given")
    out: dict[int, list[tuple[float, float]]] = {}
    pos = 0
    for lv, k in enumerate(c):
        cs = f[pos:pos + k]
        pos += k
        r = np.sqrt(cs[0] / cs[1])
        edges = [cs[0] * r]
        for i in range(len(cs) - 1):
            edges.append(np.sqrt(cs[i] * cs[i + 1]))
        edges.append(cs[-1] / np.sqrt(cs[-2] / cs[-1]))
        out[lv] = [(float(edges[i]), float(edges[i + 1]))
                   for i in range(len(edges) - 1)]
    return out


def _period_bands_indexed(fc, periods_per_octave=4):
    """Discrete, bin-snapped period bands (DESIGN.md §10.2) — the
    per-level fallback of the ``"l2u"`` scheme: used when the
    container-level table (:func:`_l2u_band_table`) could not be derived
    (fcs without metadata); each level then walks its own bin grid.

    Note: per-level walks give slightly different band labels per level, so
    a multi-level merge of these fallback bands can interleave
    near-duplicates at level overlaps (the historical "zigzag" defect) —
    only reached in the metadata-less degenerate case.

    Classic EMTF (and aurora's own default ``bs_256_26.cfg`` band-setup
    table) defines bands as explicit, non-overlapping ``index_min``/
    ``index_max`` integer ranges into the raw FFT-bin grid — hand-tuned once
    for a specific window length / decimation factor. :func:`_period_bands`
    instead partitions a *continuous* log2(period) grid and only afterwards
    tests which raw bins fall in each nominal band; because that grid is
    agnostic to the actual (linearly spaced) FFT-bin grid, nominal bands
    near the long-period end of a decimation level can silently collect
    zero raw bins (dropped after estimation), in an outcome that depends on
    floating-point edge coincidences rather than being controlled.

    This function instead walks the *integer* FFT-bin index grid directly,
    from ``index_ceiling`` (the Fs/4 cutoff, same as :func:`_period_bands`'s
    ``p_short``) down to ``index_floor`` (the window-length reliability cap,
    same as ``p_long``, clamped to the data's available range and to never
    include the DC bin), at a target ratio of ``2**(1/periods_per_octave)``
    per step, snapped to whole bins. This guarantees, by construction: every
    raw bin in ``[index_floor, index_ceiling]`` is assigned to exactly one
    band (no gaps, no overlaps, no silently-empty bands), and band width is
    monotonically non-decreasing toward higher frequency — matching the
    shape of aurora's bundled ``bs_256_26.cfg`` table, but derived
    automatically for any sample rate / window length / decimation factor
    instead of requiring a hand-authored config file tied to one specific
    configuration.

    Each band's period label is the geometric mean of the frequencies at its
    *actual* selected ``index_min``/``index_max`` (not a pre-imagined
    continuous grid point), so the reported period is always self-consistent
    with the raw data that produced it.

    Returns ``(periods_out, index_ranges)`` — periods shortest→longest, and
    one ``(index_min, index_max)`` inclusive integer pair per period, into
    the fc's raw (including-DC) frequency axis.
    """
    freq_vals = np.asarray(fc["frequency"].values, dtype=float)
    n_freq = freq_vals.size
    positive_mask = freq_vals > 0
    if not positive_mask.any():
        raise ValueError("fc has no positive-frequency bins to band")
    first_positive_idx = int(np.argmax(positive_mask))

    fs = float(fc.attrs.get("sample_rate") or 0.0)
    if not fs:
        fs = 2.0 * float(freq_vals[positive_mask].max())
    ppo = max(1, int(periods_per_octave))

    if n_freq > 1:
        delta_f = float((freq_vals[-1] - freq_vals[0]) / (n_freq - 1))
    else:
        delta_f = fs / 2.0

    def _index_of(frequency: float) -> int:
        pos = (frequency - freq_vals[0]) / delta_f if delta_f else 0.0
        return int(np.clip(round(pos), first_positive_idx, n_freq - 1))

    p_short = 4.0 / fs  # Fs/4, same reliability cutoff as _period_bands
    N = int(fc.attrs.get("num_samples_window") or 0)
    if not N:
        wl = fc.attrs.get("window_length_s")
        if wl:
            N = round(float(wl) * fs)
    p_long = N / (4.0 * fs) if N else None
    longest_present = 1.0 / float(freq_vals[positive_mask].min())
    if p_long is None or p_long > longest_present:
        p_long = longest_present
    if p_long <= p_short:
        p_long = longest_present

    index_ceiling = _index_of(1.0 / p_short)
    index_floor = _index_of(1.0 / p_long)
    if index_ceiling <= index_floor:
        index_ceiling = min(n_freq - 1, index_floor + 1)
    if index_ceiling <= index_floor:
        # only one usable bin in this level's reliable range
        only = index_floor
        return np.array([1.0 / freq_vals[only]]), [(only, only)]

    ratio = 2.0 ** (1.0 / ppo)
    index_ranges: list[tuple[int, int]] = []
    hi = index_ceiling
    while hi >= index_floor:
        lo = round(hi / ratio)
        lo = min(lo, hi - 1)
        lo = max(lo, index_floor)
        index_ranges.append((lo, hi))
        hi = lo - 1

    periods_out = np.array([
        1.0 / np.sqrt(freq_vals[lo] * freq_vals[hi]) for lo, hi in index_ranges
    ])
    return periods_out, index_ranges


def _resolve_period_bands(fc, periods_per_octave=4, band_style="l2u",
                          t_short=None):
    """Dispatch to the two band-construction schemes (DESIGN.md §10.2):
    the uniform log2-grid bands (:func:`_period_bands`) or the discrete
    bin-snapped partition (:func:`_period_bands_indexed` — the per-level
    fallback used when :func:`_l2u_band_table` could not derive the
    container-level table, e.g. for fcs without metadata; each level then
    walks its own bin grid).

    ``t_short`` (head clip, §14.9) is applied to BOTH schemes here —
    including the l2u fallback — so a shortest-period bound is honored
    identically to the primary l2u table path (which clips upstream via
    :func:`_clip_head_bands`); the criterion is the same geometric-mean
    center period ``>= t_short``.

    Returns ``(periods_out, band_bins)`` — ``band_bins[b]`` is an
    ``np.ndarray`` of integer positions into ``fc``'s frequency axis for
    band ``b`` (may be empty for the uniform style near a level's
    long-period end — callers already handle empty bands by skipping them).
    """
    style = _resolve_band_style(band_style)
    if style == "manual":
        raise ValueError(
            "band_style='manual' requires an explicit per-level band table "
            "(bands_override from manual_bands) — there is no auto table "
            "for manual mode")
    if style == "l2u":
        periods_out, index_ranges = _period_bands_indexed(fc, periods_per_octave)
        band_bins = [np.arange(lo, hi + 1) for lo, hi in index_ranges]
    else:
        periods_out, edges = _period_bands(fc, periods_per_octave)
        with np.errstate(divide="ignore"):
            logf = np.log2(fc["frequency"].values)
        band_bins = []
        for b in range(len(periods_out)):
            sel = (logf <= edges[b]) & (logf > edges[b + 1])
            band_bins.append(np.where(sel)[0])
    if t_short and t_short > 0:
        periods_out = np.asarray(periods_out, dtype=float)
        keep = periods_out >= float(t_short)
        periods_out = periods_out[keep]
        band_bins = [bb for bb, k in zip(band_bins, keep) if k]
    return periods_out, band_bins


def _band_datasets(H, E, R, input_channels, output_channels):
    """Build xarray Datasets for aurora's regression from stacked band samples.

    Returns ``(X, Y, Z)`` where ``Z`` is ``None`` unless a remote reference is
    in use. Arrays are shaped [n_obs, n_in] / [n_obs, n_out]; aurora expects
    channels as data_vars and handles any number of outputs (so a bz/hz output
    is estimated as a tipper row as well as the ex/ey impedance).
    """
    X = _ds({ic: H[:, i] for i, ic in enumerate(input_channels)})
    Y = _ds({oc: E[:, i] for i, oc in enumerate(output_channels)})
    Z = None
    if R is not None:
        Z = _ds({ic: R[:, i] for i, ic in enumerate(input_channels)})
    return X, Y, Z


def _ds(channels):
    import xarray as xr

    return xr.Dataset({name: ("obs", col) for name, col in channels.items()})


def _solve_band(H, E, R, input_channels, output_channels, engine):
    """Estimate one frequency band via an aurora estimator (OLS/RME/RME_RR).

    Returns ``(Z, Z_err)`` — Z[2x2] in emptyUI's (output, input) orientation,
    already in aurora's impedance convention (aurora's b is
    [n_input, n_output]), and the standard error of each complex impedance
    element (same orientation). The errors follow aurora's own
    ``TTFZ.standard_error()``: ``std = sqrt(|cov_ss_inv[i,i] * cov_nn[o,o]|)``
    where ``cov_ss_inv`` is the inverse signal covariance of the inputs and
    ``cov_nn`` the noise covariance of the outputs. RME/RME_RR populate these
    during estimation (``IterControl(return_covariance=True)``, the default);
    OLS does not, so the classical covariance ``(XᴴX)⁻¹·σ²`` is computed from
    the residuals instead.
    """
    from aurora.transfer_function.regression.base import RegressionEstimator
    from aurora.transfer_function.regression.iter_control import IterControl
    from aurora.transfer_function.regression.RME import RME
    from aurora.transfer_function.regression.RME_RR import RME_RR

    X, Y, Z = _band_datasets(H, E, R, input_channels, output_channels)
    if engine == "OLS":
        est = RegressionEstimator(X, Y)
        est.estimate_ols(mode="solve")
        Z_err = _ols_band_errors(est, len(output_channels))
    elif engine == "RME_RR":
        est = RME_RR(X=X, Y=Y, Z=Z, iter_control=IterControl())
        est.estimate()
        Z_err = _estimator_band_errors(est, input_channels, output_channels)
    else:
        est = RME(X=X, Y=Y, iter_control=IterControl())
        est.estimate()
        Z_err = _estimator_band_errors(est, input_channels, output_channels)
    return np.asarray(est.b).T, Z_err


def _cov_diag(values, channel_names, channels):
    """Diagonal of a covariance matrix re-ordered to ``channels`` order.

    aurora's covariance DataArrays use its own channel-name coordinate order;
    the values array is indexed by the *declared* name order, so this maps our
    channel list onto it defensively.
    """
    idx = [channel_names.index(c) for c in channels]
    return np.asarray(values)[np.ix_(idx, idx)]


def _cov_coords(cov):
    """Channel names from an aurora covariance DataArray coordinate."""
    for key in ("input_channel_1", "output_channel_1"):
        if key in cov.coords:
            return list(cov.coords[key].values)
    if cov.dims:
        # the DataArray dim is a NAME (a string) — iterating it produced
        # one bogus letter per character; fall back to the coordinate of
        # the first dim, or the declared order is unknowable
        first = cov.dims[0]
        if first in cov.coords:
            return list(cov.coords[first].values)
    raise ValueError(
        "aurora covariance DataArray carries neither input_channel_1/"
        "output_channel_1 nor a coordinate for its first dimension — "
        "cannot map channel order")


def _std_from_cov(cov_ss_inv, cov_nn, input_channels, output_channels):
    """aurora ``TTFZ.standard_error()``: sqrt(|cov_ss_inv[i,i] * cov_nn[o,o]|)."""
    n_in, n_out = len(input_channels), len(output_channels)
    std = np.zeros((n_out, n_in), dtype=float)
    if cov_ss_inv is None or cov_nn is None:
        return std
    cs = _cov_diag(cov_ss_inv.values, _cov_coords(cov_ss_inv), input_channels)
    cn = _cov_diag(cov_nn.values, _cov_coords(cov_nn), output_channels)
    for o in range(n_out):
        for i in range(n_in):
            std[o, i] = np.sqrt(abs(cs[i, i] * cn[o, o]))
    return std


def _estimator_band_errors(est, input_channels, output_channels):
    """Standard errors from an RME/RME_RR estimator's computed covariances."""
    return _std_from_cov(est.cov_ss_inv, est.cov_nn, input_channels, output_channels)


def _ols_band_errors(est, n_out):
    """Classical OLS standard errors from residuals: (XᴴX)⁻¹ · σ².

    aurora's OLS path does not populate covariances, so the textbook
    covariance is computed here (same form aurora uses for RME):
    ``cov_ss_inv = (XᴴX)⁻¹`` and ``cov_nn = diag(residual SS) / dof``.
    """
    X = np.asarray(est.X, dtype=complex)
    Y = np.asarray(est.Y, dtype=complex)
    b = np.asarray(est.b, dtype=complex)
    n_obs, n_in = X.shape
    dof = n_obs - n_in
    if dof <= 0 or n_out == 0:
        return np.zeros((n_out, n_in), dtype=float)
    cs = np.linalg.inv(X.conj().T @ X)
    res = Y - X @ b  # [n_obs, n_out]
    ssr = (res.conj().T @ res).real  # [n_out, n_out]
    cn = ssr / dof
    std = np.zeros((n_out, n_in), dtype=float)
    for o in range(n_out):
        for i in range(n_in):
            std[o, i] = np.sqrt(abs(cs[i, i].real * cn[o, o]))
    return std


_OUT_COORDS = tuple(STANDARD_OUTPUTS)  # ("ex", "ey", "hz")
_IN_COORDS = tuple(STANDARD_INPUTS)  # ("hx", "hy")


def build_tf(periods, Z, input_channels, output_channels, channel_nomenclature=None,
             Z_err=None, survey_id=None, station_id=None, run_id=None,
             latitude=None, longitude=None, elevation=None, start_datetime=None,
             processing_notes=None):
    """Build an mt_metadata TF carrying the actual channel names plus the
    channel_nomenclature map (aurora-style).

    mt_metadata's TF uses ``channel_nomenclature`` (standard -> actual) to
    define its coordinate names — ``tf.ex``/``tf.hx`` return the *actual*
    instrument names and ``has_impedance``/``impedance``/``tipper`` are all
    nomenclature-aware. The TF is therefore built with the actual names as
    coordinates and the map attached, exactly like aurora's
    ``TF(channel_nomenclature=...)``, so aurora tooling (``TTFZ``,
    ``TransferFunctionCollection``) can consume it directly. Exporters that
    require standard names (EDI/XML) standardize on write; ``ExportTF`` maps
    back for the ZMM writer.

    The actual channel names are resolved through the mt_metadata CHANNEL_MAPS
    reverse mapping for the given nomenclature keyword: lemi12 maps e1/e2 (and
    bx/by/bz), lemi34 maps e3/e4, phoenix123 maps e1/e2/h1/h2/h3, default maps
    ex/ey/hx/hy/hz. When no keyword is given it is inferred from the channel
    names. Both the impedance (ex,ey rows) and the tipper (hz row) are
    populated when the corresponding output channel is present.

    ``Z_err`` (optional, same shape as ``Z``) is stored as
    ``tf.transfer_function_error`` (the std of each complex element) so plots
    and exports (EDI/XML/ZMM) can carry error bars / error lines. The station
    identity (survey/station/run id + lat/lon/elevation) is attached to
    ``tf.survey_metadata``/``tf.station_metadata`` — the single point of truth
    for plot titles and export headers (DESIGN.md §10.6).
    """
    import xarray as xr
    from mt_metadata.transfer_functions.core import TF, Station, Survey

    keyword = _resolve_keyword(
        channel_nomenclature, input_channels, output_channels
    )
    mapping = channel_map(keyword)  # standard -> actual

    out_coords = [mapping[s] for s in _OUT_COORDS]
    in_coords = [mapping[s] for s in _IN_COORDS]
    reverse = {v.lower(): k for k, v in mapping.items()}

    data = np.zeros((len(periods), len(_OUT_COORDS), len(_IN_COORDS)), dtype=complex)
    for ri, oc in enumerate(output_channels):
        std = reverse.get(str(oc).strip().lower())
        if std is None or std not in _OUT_COORDS:
            continue
        for ci, ic in enumerate(input_channels):
            ist = reverse.get(str(ic).strip().lower())
            if ist is None or ist not in _IN_COORDS:
                continue
            data[:, _OUT_COORDS.index(std), _IN_COORDS.index(ist)] = Z[:, ri, ci]

    tf = TF()
    tf.channel_nomenclature = {s: mapping[s] for s in ("ex", "ey", "hx", "hy", "hz")}
    arr = xr.DataArray(
        data,
        dims=("period", "output", "input"),
        coords={
            "period": periods,
            "output": out_coords,
            "input": in_coords,
        },
    )
    # use the property setter (tf.dataset[...] = arr writes to the xarray
    # Dataset only, not the TF's canonical impedance storage)
    tf.transfer_function = arr
    if Z_err is not None:
        err = np.zeros((len(periods), len(_OUT_COORDS), len(_IN_COORDS)), dtype=float)
        for ri, oc in enumerate(output_channels):
            std = reverse.get(str(oc).strip().lower())
            if std is None or std not in _OUT_COORDS:
                continue
            for ci, ic in enumerate(input_channels):
                ist = reverse.get(str(ic).strip().lower())
                if ist is None or ist not in _IN_COORDS:
                    continue
                err[:, _OUT_COORDS.index(std), _IN_COORDS.index(ist)] = Z_err[:, ri, ci]
        tf.transfer_function_error = xr.DataArray(
            err,
            dims=("period", "output", "input"),
            coords={
                "period": periods,
                "output": out_coords,
                "input": in_coords,
            },
        )
    if station_id or survey_id:
        if survey_id:
            surv = Survey()
            surv.id = survey_id
            tf.survey_metadata = surv
        st = Station()
        st.id = station_id or "local"
        if latitude is not None:
            st.location.latitude = float(latitude)
        if longitude is not None:
            st.location.longitude = float(longitude)
        if elevation is not None:
            st.location.elevation = float(elevation)
        tf.station_metadata = st
    if run_id or processing_notes:
        rm = tf.run_metadata
        if run_id:
            rm.id = run_id
        if processing_notes:
            rm.comments.value = processing_notes
    return tf


def estimate_tf_levels(
    levels,
    input_channels,
    output_channels,
    estimator_engine="Robust",
    remote_reference=False,
    periods_per_octave=3,
    ref_levels=None,
    channel_nomenclature=None,
    band_style="l2u",
    min_bins_per_band=20.0,
    t_short=None,
    bands_override=None,
    pairing="intersect",
    pair_tolerance_s=0.0,
):
    """Estimate a per-level TF for every decimation level in a multi-level fc
    container.

    Returns ``{level: {"periods": P, "Z": Z[2x2], "Z_err": Z_err,
    "input_channels": ..., "output_channels": ..., "channel_nomenclature": ...,
    "survey_id"/"station_id"/"run_id"/"station_latitude"/...: ...}}``.
    ``ref_levels`` is an optional same-shaped container (or a single fc) used
    as the remote reference per level. ``band_style`` selects the band
    construction scheme (DESIGN.md §10.2):

    - ``"l2u"`` (default): the reference-template band table with
      log-uniform window-starvation coarsening (:func:`_l2u_band_table`) —
      the same fixed template at every decimation level; starved levels are
      rebuilt as a coarser uniform grid, nothing to merge.
    - ``"uniform"``: uniform log2-grid partition (:func:`_period_bands`) —
      overlapping levels estimate the same nominal bands and
      ``merge_tf_levels`` keeps the deepest.

    For ``"l2u"``, band construction happens once for the whole container;
    if the container's metadata is insufficient (missing attrs, non-uniform
    level ratios) the per-level bin walk is used as a fallback.
    ``min_bins_per_band`` (window-starvation threshold on total samples per
    band) and ``t_short`` (head clip) refine the table per DESIGN.md §10.2;
    ``t_short`` is honored for both schemes (the uniform path clips in
    :func:`_resolve_period_bands`; a fully-clipped level is skipped).
    """
    keyword = _resolve_keyword(channel_nomenclature, input_channels, output_channels)
    style = _resolve_band_style(band_style)
    if bands_override is None and style == "l2u":
        bands_override = _l2u_band_table(
            levels, min_bins_per_band=min_bins_per_band, t_short=t_short)
    result = {}
    for level, fc in sorted(levels.items()):
        if bands_override is not None:
            bands = bands_override.get(level, [])
            if not bands:
                continue  # head clip / coarsening left this level no bands
        else:
            # No container-level table (uniform style, or the l2u fallback
            # for metadata-less fcs): resolve the SAME table estimate_tf
            # will use internally (§23.5 choke point) and pass it
            # explicitly, so the recorded band_intervals align 1:1 with
            # the Z rows — Residual Sort's pooling key (DESIGN.md §10.4).
            fc_n = _normalize_fc_layout(fc)
            fc_n = fc_n.isel(frequency=fc_n["frequency"].values > 0)
            f_n = np.asarray(fc_n["frequency"].values, dtype=float)
            _, bins_r = _resolve_period_bands(
                fc_n, periods_per_octave, band_style, t_short=t_short)
            # Pairs from the bins' actual extent, widened by one ULP on
            # both ends: _bands_from_freq_pairs maps intervals with
            # searchsorted(side="right"), which would EXCLUDE a bin lying
            # exactly on the interval edge — a single-bin band (common at
            # the uniform partition's long-period end) would come back
            # empty (NaN period).
            bands = [
                (float(np.nextafter(f_n[b].max(), np.inf)),
                 float(np.nextafter(f_n[b].min(), -np.inf)))
                for b in bins_r if b.size]
        ref_fc = _level_ref(ref_levels, level)
        periods, Z, Z_err = estimate_tf(
            fc, input_channels, output_channels,
            estimator_engine=estimator_engine,
            remote_reference=remote_reference,
            periods_per_octave=periods_per_octave,
            ref_fc=ref_fc,
            band_style=band_style,
            bands_override=bands,
            t_short=t_short,
            pairing=pairing,
            pair_tolerance_s=pair_tolerance_s,
        )
        if periods.size == 0:
            continue  # fully head-clipped — this level contributes nothing
        entry = {
            "periods": periods,
            "Z": Z,
            "Z_err": Z_err,
            "input_channels": list(input_channels),
            "output_channels": list(output_channels),
            "channel_nomenclature": keyword,
            "periods_per_octave": periods_per_octave,
            # the band geometry this level's estimates were built from —
            # Residual Sort pools the FC bins per band to predict each
            # window's output from the band's Z (DESIGN.md §10.4)
            "band_intervals": [tuple(b) for b in bands] if bands else None,
        }
        entry.update(_identity_attrs(fc))
        result[int(level)] = entry
    return result


def _identity_attrs(fc):
    """Station identity attrs carried on an fc (set by ComputeFC)."""
    return {
        k: fc.attrs[k]
        for k in ("survey_id", "station_id", "run_id", "station_latitude",
                  "station_longitude", "station_elevation", "start_datetime")
        if fc.attrs.get(k) not in (None, "")
    }


def merge_tf_levels(per_level):
    """Merge per-decimation-level TF estimates into a single broadband
    (periods, Z, Z_err).

    Levels overlap (each covers from Fs/4 up to window/4), so the same
    period can be estimated by several levels. Because a deeper level has
    a longer window (more cycles of a long period), for each period we
    keep the estimate from the **deepest** level that covers it;
    degenerate (all-zero) bands are dropped. Dedup is tolerance-based on
    the shared octave grid (within half the level's center spacing,
    1/(2·ppo) octaves): aligned cascades collide exactly (legacy exact
    dedup), while a burst archive's misaligned rate families interleave
    near-duplicate centers that must also resolve to one estimate
    (history §24.8). The resulting grid is strictly ascending and free of
    duplicates.
    """
    merged: dict[float, np.ndarray] = {}
    merged_err: dict[float, np.ndarray] = {}
    for lv in sorted(per_level, reverse=True):  # deepest level wins
        P = np.asarray(per_level[lv]["periods"])
        if P.size == 0:
            continue  # a level may legitimately contribute nothing (clipped)
        Z = np.asarray(per_level[lv]["Z"])
        err = per_level[lv].get("Z_err")
        if err is None:
            # a hand-built level payload may carry no errors — treat as zeros
            err = np.zeros(len(Z))
        else:
            err = np.asarray(err, dtype=float)
            if err.ndim == 0:
                err = np.full(len(Z), float(err))
        ok = ~np.all(np.abs(Z).reshape(len(Z), -1) == 0, axis=1)
        ppo = float(per_level[lv].get("periods_per_octave", 3) or 3)
        half_w = 0.5 / ppo  # half the level's center spacing, in octaves
        degenerate = P[~ok]
        if degenerate.size:
            _log.warning(
                "Merge TF levels: level %d contributed no estimate at %d "
                "band(s) (all-zero/degenerate — usually the coherence sort "
                "removed every window, sometimes a band interval fell "
                "outside this level's frequency grid): %s",
                lv, degenerate.size,
                "; ".join(f"T={p:.4g}s" for p in degenerate[:8]),
            )
        for p, z, e in zip(P[ok], Z[ok], err[ok]):
            lp = float(np.log2(p))
            if any(abs(lp - k) < half_w for k in merged):
                continue  # already estimated by a deeper level
            merged[round(lp, 8)] = z
            merged_err[round(lp, 8)] = e
    if not merged:
        raise ValueError("no non-degenerate TF bands to merge")
    keys = sorted(merged)
    P = 2.0 ** np.asarray(keys)
    Z = np.stack([merged[k] for k in keys])
    Z_err = np.stack([merged_err[k] for k in keys])
    return P, Z, Z_err
