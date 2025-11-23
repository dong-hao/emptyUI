"""Raw-instrument loaders (LEMI-424)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emptyui import Node, Output, Param


class LoadLemi(Node):
    type = "mth5.io.lemi"
    title = "Load LEMI-424"
    description = (
        "Convert raw LEMI-424 long-period data (.INF/.TXT) into an MTH5 "
        "archive and open it as the data source."
    )
    category = "Data I/O"
    inputs = ()
    outputs = (Output("mth5", "mth5"),)
    params = (
        Param("data_path", "directory", label="LEMI data directory"),
        Param("survey_id", "string", default="survey"),
        Param("station_id", "string", default="station"),
        Param("mth5_filename", "string", label="Output file name", default="from_lemi424.mth5"),
        Param("save_path", "directory", label="Output directory", default="",
              optional=True),
    )

    def execute(self, ctx, data_path: str, survey_id: str = "survey",
                station_id: str = "station", mth5_filename: str = "from_lemi424.mth5",
                save_path: str = "") -> dict[str, Any]:
        from mth5.clients.lemi424 import LEMI424Client
        from mth5.mth5 import MTH5

        data_path = str(Path(data_path).expanduser().resolve())
        if not Path(data_path).exists():
            raise FileNotFoundError(f"LEMI data dir not found: {data_path}")
        save = str(Path(save_path).expanduser().resolve()) if save_path else None

        # Use LEMI424Client directly: mth5 0.6.8's MakeMTH5.from_lemi424 has a
        # duplicate sample_rates kwarg bug; LEMI424Client hardcodes sample_rates=[1].
        client = LEMI424Client(
            data_path=data_path,
            save_path=save,
            mth5_filename=mth5_filename,
        )
        created = client.make_mth5_from_lemi424(survey_id, station_id)

        archive = MTH5()
        archive.open_mth5(created, mode="a")
        ctx.progress(1.0, f"loaded LEMI -> {Path(created).name}")
        return {"mth5": archive}


def register(registry) -> None:
    registry.add(LoadLemi)
