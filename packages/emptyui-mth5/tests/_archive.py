"""Tiny synthetic mth5 archives shared by executor-route tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _runts(start: str, rid: str, n: int = 32, sr: float = 1.0,
           comps=("hx", "hy", "ex", "ey")):
    """A real mth5 RunTS (standard names) from noise."""
    import warnings

    warnings.filterwarnings("ignore")
    from mt_timeseries import ChannelTS, RunTS

    rng = np.random.default_rng(abs(hash(rid)) % 2**32)
    chs = []
    for comp in comps:
        typ = "magnetic" if comp[0] in ("b", "h") else "electric"
        c = ChannelTS(typ, data=rng.standard_normal(n), start=start,
                      sample_rate=sr)
        c.component = comp
        chs.append(c)
    runts = RunTS(array_list=chs)
    runts.run_metadata.id = rid
    return runts


def multi_run_archive(path: Path, rids: list[str], starts: list[str],
                      rates: list[float] | None = None,
                      survey: str = "4P", station: str = "SDX"):
    """An mth5 archive with one run per (rid, start[, rate]); opened
    read-only on return."""
    import warnings

    warnings.filterwarnings("ignore")
    from mth5.mth5 import MTH5

    rates = rates or [1.0] * len(rids)
    m = MTH5()
    m.open_mth5(str(path), "w")
    sg = m.add_survey(survey)
    stg = sg.stations_group.add_station(station)
    for rid, start, sr in zip(rids, starts, rates):
        rg = stg.add_run(rid)
        rg.from_runts(_runts(start, rid, sr=sr))
    stg.write_metadata()
    sg.update_metadata()
    m.close_mth5()
    archive = MTH5()
    archive.open_mth5(str(path), "r")
    return archive
