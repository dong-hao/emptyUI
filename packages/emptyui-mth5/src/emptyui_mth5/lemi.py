"""Raw-instrument loaders (LEMI-424)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emptyui import Node, Output, Param


class LoadLemi(Node):
    type = "mth5.io.lemi"
    title = "Load LEMI-424"
    tooltip = "Convert raw LEMI-424 .INF/.TXT data into an MTH5 archive and open it."
    description = (
        "Convert raw LEMI-424 long-period data (.INF/.TXT) into an MTH5 "
        "archive and open it as the data source. The channel nomenclature "
        "selects which electric pair to read: lemi12 uses e1/e2, lemi34 uses "
        "e3/e4 (4-channel instruments)."
    )
    category = "Data I/O"
    inputs = ()
    outputs = (Output("mth5", "mth5"),)
    params = (
        Param("data_path", "directory", label="LEMI data directory", path_kind="in"),
        Param("survey_id", "string", default="survey"),
        Param("station_id", "string", default="station"),
        Param("mth5_filename", "string", label="Output file name", default="from_lemi424.mth5"),
        Param("save_path", "directory", label="Output directory", default="",
              optional=True, path_kind="out"),
        Param("channel_nomenclature", "select", label="Channel nomenclature",
              options=("lemi12", "lemi34"), default="lemi12",
              description="mt_metadata CHANNEL_MAPS keyword: lemi12 = electric "
                          "pair e1/e2 (default), lemi34 = pair e3/e4"),
        Param("calibration_path", "string", label="Calibration file/dir",
              default="", optional=True, path_kind="in",
              description="Manufacturer calibration (.chn/.def FAP or mt_io "
                          ".json). A directory is searched for a matching "
                          "file; empty auto-pickups *.chn/*.def/*.json in the "
                          "data directory (station-pair files preferred)."),
        Param("extend_flat", "boolean", label="Flat below table", default=True,
              description="Treat the FAP response as constant (first row, "
                          "zero phase) below the measured frequency range."),
    )

    def execute(self, ctx, data_path: str, survey_id: str = "survey",
                station_id: str = "station", mth5_filename: str = "from_lemi424.mth5",
                save_path: str = "",
                channel_nomenclature: str = "lemi12",
                calibration_path: str = "",
                extend_flat: bool = True) -> dict[str, Any]:
        from mth5 import read_file
        from mth5.clients.lemi424 import LEMI424Client
        from mth5.mth5 import MTH5

        data_path = str(Path(data_path).expanduser().resolve())
        if not Path(data_path).exists():
            raise FileNotFoundError(f"LEMI data dir not found: {data_path}")
        save = str(Path(save_path).expanduser().resolve()) if save_path else None

        client = LEMI424Client(
            data_path=data_path,
            save_path=save,
            mth5_filename=mth5_filename,
        )
        # A previous run that was force-stopped mid-write can leave a corrupt /
        # locked MTH5 at the target path; open_mth5(mode="a") then fails on the
        # leftover file. Overwrite any existing file so we always start fresh.
        target = Path(client.save_path)
        if target.exists():
            ctx.logger.warning("overwriting existing MTH5 at %s", target)
            target.unlink()

        # LEMI-424 raw files carry four electric channels (e1..e4); the
        # LEMI424Client hardcodes e1/e2, so the conversion loop below mirrors
        # it but selects the pair via the mt_metadata nomenclature.
        e_channels = (
            ["e3", "e4"] if channel_nomenclature == "lemi34" else ["e1", "e2"]
        )
        client.collection.survey_id = survey_id
        client.collection.station_id = station_id
        # Manufacturer calibration (DESIGN §16.1): explicit file/dir wins,
        # otherwise auto-pickup in the data directory; the filters are
        # attached to each run BEFORE it is written, so the association is
        # archived and survives Extract Run / MTH5 round trips.
        from .calibration import (
            attach_filters,
            find_calibration_file,
            load_calibration,
        )

        cal_filters = None
        cal_src = calibration_path.strip()
        if not cal_src:
            found = find_calibration_file(data_path, station_id=station_id)
            if found is not None:
                cal_src = str(found)
        if cal_src:
            cal_path = Path(cal_src).expanduser().resolve()
            if cal_path.is_dir():
                found = find_calibration_file(cal_path, station_id=station_id)
                if found is None:
                    ctx.logger.warning(
                        "no calibration file (.chn/.def/.json) in %s", cal_path)
                else:
                    cal_path = found
            if cal_path.is_file():
                cal_filters = load_calibration(cal_path, extend_flat=extend_flat)
                ctx.logger.info(
                    "using calibration %s for %s channels",
                    cal_path.name, sorted(cal_filters))

        runs = client.get_run_dict()
        if not runs:
            raise ValueError(
                f"no LEMI runs found in the data directory — no readable "
                f"time-series files (looked for the instrument's .TXT "
                f"files under {data_path})")

        with MTH5(**client.h5_kwargs) as m:
            m.open_mth5(str(target), client.mth5_file_mode)
            survey_group = m.add_survey(client.collection.survey_id)
            for st_id, runs_by_id in runs.items():
                station_group = survey_group.stations_group.add_station(st_id)
                station_runts = None
                for run_id, run_df in runs_by_id.items():
                    run_group = station_group.add_run(run_id)
                    run_ts = read_file(
                        run_df.fn.to_list(),
                        e_channels=e_channels,
                        calibration_dict=client.collection.calibration_dict,
                    )
                    if cal_filters is not None:
                        attach_filters(run_ts, cal_filters)
                    run_ts.run_metadata.id = run_id
                    run_group.from_runts(run_ts)
                    station_runts = run_ts
                # the LAST run's station metadata (a loop-variable leak when
                # runs_by_id is empty raised NameError before)
                if station_runts is not None:
                    station_group.metadata.update(station_runts.station_metadata)
                station_group.write_metadata()
            survey_group.update_metadata()

        archive = MTH5()
        archive.open_mth5(str(target), mode="a")
        ctx.progress(1.0, f"loaded LEMI -> {target.name}")
        return {"mth5": archive}


