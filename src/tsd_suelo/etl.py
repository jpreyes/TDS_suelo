from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

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


def build_waveform_targets(
    records_dir: Path,
    output_dir: Path,
    max_h5: int | None = None,
    damping: float = 0.05,
    compute_psa: bool = True,
    workers: int = 1,
    progress_every: int = 500,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    mode_name = f"psa_damping_{str(damping).replace('.', 'p')}" if compute_psa else "no_psa"
    checkpoint_dir = output_dir / "_h5_target_batches" / mode_name
    targets = build_h5_targets(
        records_dir,
        max_h5=max_h5,
        damping=damping,
        compute_psa=compute_psa,
        workers=workers,
        progress_every=progress_every,
        log=log,
        checkpoint_dir=checkpoint_dir,
    )
    h5_errors = 0
    if "read_ok" in targets.columns:
        errors = targets[targets["read_ok"] == False].copy()  # noqa: E712
        h5_errors = int(errors.shape[0])
        if not errors.empty:
            errors.to_csv(output_dir / "waveform_targets_errors.csv", index=False)
            if log:
                log(f"H5 con error={errors.shape[0]} -> {output_dir / 'waveform_targets_errors.csv'}")
        targets = targets[targets["read_ok"] != False].copy()  # noqa: E712
    if "record_observed_id" in targets.columns:
        targets = targets.sort_values("record_observed_id").reset_index(drop=True)
    write_parquet(targets, output_dir / "waveform_targets_observed.parquet")
    write_json(
        output_dir / "waveform_targets_observed.meta.json",
        {
            "records_dir": str(records_dir),
            "max_h5": max_h5,
            "damping": damping,
            "compute_psa": compute_psa,
            "workers": workers,
            "h5_processed": int(targets.shape[0]),
            "h5_errors": h5_errors,
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    return targets


def _record_key(df: pd.DataFrame) -> pd.Series:
    return df["event_id"].astype(str) + "_" + df["station_id"].astype(str)


def _flatfile_waveform_targets(records: pd.DataFrame, existing_keys: set[str]) -> pd.DataFrame:
    keys = _record_key(records)
    flat = records.loc[~keys.isin(existing_keys)].copy()
    if flat.empty:
        return pd.DataFrame(columns=["record_observed_id", "event_id", "station_id", "observed_source"])
    out = pd.DataFrame(index=flat.index)
    out["record_observed_id"] = _record_key(flat)
    out["event_id"] = flat["event_id"]
    out["station_id"] = flat["station_id"]
    out["observed_source"] = "flatfile"
    out["h5_file"] = None
    out["h5_name"] = None
    out["dt_s"] = np.nan
    out["sample_rate_hz"] = flat.get("sample_rate_hz_flatfile", np.nan)
    out["pga_e_g"] = flat.get("flat_pga_g_e", np.nan)
    out["pga_n_g"] = flat.get("flat_pga_g_n", np.nan)
    out["pga_z_g"] = flat.get("flat_pga_g_z", np.nan)
    out["pga_h_g"] = flat.get("flat_pga_hmax_g", np.nan)
    out["arias_e_m_s"] = flat.get("flat_arias_m_s_e", np.nan)
    out["arias_n_m_s"] = flat.get("flat_arias_m_s_n", np.nan)
    out["arias_z_m_s"] = flat.get("flat_arias_m_s_z", np.nan)
    out["arias_h_m_s"] = out[["arias_e_m_s", "arias_n_m_s"]].max(axis=1, skipna=True)
    out["duration_5_75_e_s"] = flat.get("flat_duration_5_75_s_e", np.nan)
    out["duration_5_75_n_s"] = flat.get("flat_duration_5_75_s_n", np.nan)
    out["duration_5_75_z_s"] = flat.get("flat_duration_5_75_s_z", np.nan)
    out["duration_5_95_e_s"] = flat.get("flat_duration_5_95_s_e", np.nan)
    out["duration_5_95_n_s"] = flat.get("flat_duration_5_95_s_n", np.nan)
    out["duration_5_95_z_s"] = flat.get("flat_duration_5_95_s_z", np.nan)
    out["duration_5_75_h_s"] = out[["duration_5_75_e_s", "duration_5_75_n_s"]].mean(axis=1, skipna=True)
    out["duration_5_95_h_s"] = out[["duration_5_95_e_s", "duration_5_95_n_s"]].mean(axis=1, skipna=True)
    out["horizontal_to_vertical_pga"] = out["pga_h_g"] / out["pga_z_g"].replace(0, np.nan)
    out["east_to_north_pga"] = out["pga_e_g"] / out["pga_n_g"].replace(0, np.nan)
    for period in (0.1, 0.2, 0.5, 1.0, 2.0):
        out[f"psa_t{str(period).replace('.', 'p')}_g"] = flat.get(f"flat_psa_t{str(period).replace('.', 'p')}_g", np.nan)
    return out.reset_index(drop=True)


def build_geo_targets(
    waveform_targets: pd.DataFrame,
    flatfiles_dir: Path,
    output_dir: Path,
    include_flatfile_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = load_events(flatfiles_dir)
    records = load_record_flatfile(flatfiles_dir)
    stations = load_stations(flatfiles_dir)

    observed_targets = waveform_targets.copy()
    observed_targets["observed_source"] = "h5"
    if include_flatfile_only:
        existing_keys = set(_record_key(observed_targets))
        flat_targets = _flatfile_waveform_targets(records, existing_keys)
        observed_targets = pd.concat([observed_targets, flat_targets], ignore_index=True, sort=False)

    observed = observed_targets.merge(records, on=["event_id", "station_id"], how="left", suffixes=("", "_record"))
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
