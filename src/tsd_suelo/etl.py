from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .flatfiles import load_events, load_record_flatfile, load_stations
from .geometry import add_geometry
from .h5_targets import build_h5_targets, list_h5_files
from .utils import ensure_dir, write_json, write_parquet


def build_inventory(records_dir: Path, flatfiles_dir: Path, output_dir: Path, max_h5: int | None = None) -> dict[str, object]:
    h5_files = list_h5_files(records_dir, max_h5=max_h5)
    events = load_events(flatfiles_dir)
    records = load_record_flatfile(flatfiles_dir)
    stations = load_stations(flatfiles_dir)
    inventory = {
        "records_dir": str(records_dir),
        "flatfiles_dir": str(flatfiles_dir),
        "h5_count": len(h5_files),
        "event_count_flatfile": int(events["event_id"].nunique()),
        "record_count_flatfile": int(records.shape[0]),
        "station_count_flatfile": int(stations["station_id"].nunique()),
        "h5_files": [path.name for path in h5_files],
    }
    write_json(output_dir / "observed_inventory.json", inventory)
    return inventory


def build_waveform_targets(records_dir: Path, output_dir: Path, max_h5: int | None = None, damping: float = 0.05) -> pd.DataFrame:
    targets = build_h5_targets(records_dir, max_h5=max_h5, damping=damping)
    write_parquet(targets, output_dir / "waveform_targets_observed.parquet")
    return targets


def build_geo_targets(
    waveform_targets: pd.DataFrame,
    flatfiles_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = load_events(flatfiles_dir)
    records = load_record_flatfile(flatfiles_dir)
    stations = load_stations(flatfiles_dir)

    observed = waveform_targets.merge(records, on=["event_id", "station_id"], how="left", suffixes=("", "_record"))
    observed = observed.merge(events, on="event_id", how="left", suffixes=("", "_event"))
    observed = observed.merge(stations, on="station_id", how="left", suffixes=("", "_station"))

    for column in ("event_latitude_deg", "event_longitude_deg", "event_depth_km", "mw", "tectonic_type"):
        event_col = f"{column}_event"
        if event_col in observed.columns:
            observed[column] = observed[column].where(observed[column].notna(), observed[event_col])
    for column in ("station_latitude_deg", "station_longitude_deg"):
        station_col = f"{column}_station"
        if station_col in observed.columns:
            observed[column] = observed[column].where(observed[column].notna(), observed[station_col])
    if "vs30_m_s" in observed.columns and "vs30_m_s_record" in observed.columns:
        observed["vs30_m_s"] = observed["vs30_m_s"].where(observed["vs30_m_s"].notna(), observed["vs30_m_s_record"])

    observed = add_geometry(observed)
    observed["source3d_id"] = observed["source_cell_j3"].fillna(observed["event_id"])
    observed["receiver_id"] = observed["station_id"].fillna(observed["receiver_cell_j3"])
    observed["route_id"] = observed["route_id_j3"]
    observed["has_known_site"] = observed[["vs30_m_s", "f0_hvsr_hz", "a0_hvsr"]].notna().any(axis=1)

    geometry_cols = [
        "record_observed_id",
        "event_id",
        "station_id",
        "event_latitude_deg",
        "event_longitude_deg",
        "event_depth_km",
        "station_latitude_deg",
        "station_longitude_deg",
        "distance_km",
        "repi_km_calc",
        "rhyp_km_calc",
        "azimuth_deg",
        "backazimuth_deg",
        "incidence_angle_deg",
        "direction_bin_30deg",
        "source_cell_j1",
        "source_cell_j2",
        "source_cell_j3",
        "source_cell_j4",
        "receiver_cell_j1",
        "receiver_cell_j2",
        "receiver_cell_j3",
        "receiver_cell_j4",
        "route_id_j1",
        "route_id_j2",
        "route_id_j3",
        "route_id_j4",
    ]
    geometry = observed[[c for c in geometry_cols if c in observed.columns]].copy()
    receiver_index = (
        observed.groupby("station_id", dropna=False)
        .agg(
            station_latitude_deg=("station_latitude_deg", "first"),
            station_longitude_deg=("station_longitude_deg", "first"),
            vs30_m_s=("vs30_m_s", "first"),
            f0_hvsr_hz=("f0_hvsr_hz", "first"),
            a0_hvsr=("a0_hvsr", "first"),
            geology=("geology", "first"),
            n_records=("record_observed_id", "count"),
        )
        .reset_index()
    )
    source_index = (
        observed.groupby("event_id", dropna=False)
        .agg(
            event_latitude_deg=("event_latitude_deg", "first"),
            event_longitude_deg=("event_longitude_deg", "first"),
            event_depth_km=("event_depth_km", "first"),
            mw=("mw", "first"),
            tectonic_type=("tectonic_type", "first"),
            source_cell_j3=("source_cell_j3", "first"),
            n_records=("record_observed_id", "count"),
        )
        .reset_index()
    )

    write_parquet(geometry, output_dir / "record_geometry.parquet")
    write_parquet(receiver_index, output_dir / "receiver_index.parquet")
    write_parquet(source_index, output_dir / "source3d_index.parquet")
    write_parquet(observed, output_dir / "geo_targets_observed.parquet")
    return observed, geometry, receiver_index, source_index

