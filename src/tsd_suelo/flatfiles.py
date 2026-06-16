from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import coalesce_columns, numeric_series, read_csv_str


EVENTS_CSV = "Flatfile_chileanEvents_v11.csv"
RECORDS_CSV = "Flatfile_chileanRecord_v8.csv"
STATIONS_CSV = "Flatfile_chileanSiteStation_v7.csv"


def _resolve_csv(flatfiles_dir: Path, filename: str) -> Path:
    path = flatfiles_dir / filename
    if path.exists():
        return path
    sibling = flatfiles_dir.parent / filename
    if sibling.exists():
        return sibling
    raise FileNotFoundError(f"No se encontro {filename} en {flatfiles_dir}")


def load_events(flatfiles_dir: Path) -> pd.DataFrame:
    df = read_csv_str(_resolve_csv(flatfiles_dir, EVENTS_CSV))
    out = pd.DataFrame(index=df.index)
    out["event_id"] = df["EventID"].astype(str).str.strip()
    out["event_number"] = numeric_series(df.get("EventNumber", pd.Series(index=df.index)))
    out["event_latitude_deg"] = numeric_series(df.get("Latitude_deg", pd.Series(index=df.index)))
    out["event_longitude_deg"] = numeric_series(df.get("Longitude_deg", pd.Series(index=df.index)))
    out["event_depth_km"] = numeric_series(df.get("Depth_km", pd.Series(index=df.index)))
    out["mw"] = numeric_series(df.get("Mw", pd.Series(index=df.index)))
    out["tectonic_type"] = df.get("TectonicType", pd.Series(index=df.index, dtype=str))
    out["mechanism"] = df.get("Mechanism", pd.Series(index=df.index, dtype=str))
    out["rupture_model_source"] = df.get("RuptureModelSource", pd.Series(index=df.index, dtype=str))
    out["rupture_points_raw"] = df.get("Rupture_Lat_Lon_Dep_deg_deg_km", pd.Series(index=df.index, dtype=str))
    out["mt_latitude_deg"] = numeric_series(df.get("MT_Latitude_deg", pd.Series(index=df.index)))
    out["mt_longitude_deg"] = numeric_series(df.get("MT_Longitude_deg", pd.Series(index=df.index)))
    out["mt_depth_km"] = numeric_series(df.get("MT_Depth_km", pd.Series(index=df.index)))
    out["source_latitude_deg"] = out["mt_latitude_deg"].where(out["mt_latitude_deg"].notna(), out["event_latitude_deg"])
    out["source_longitude_deg"] = out["mt_longitude_deg"].where(out["mt_longitude_deg"].notna(), out["event_longitude_deg"])
    out["source_depth_km"] = out["mt_depth_km"].where(out["mt_depth_km"].notna(), out["event_depth_km"])
    return out.drop_duplicates("event_id")


def load_stations(flatfiles_dir: Path) -> pd.DataFrame:
    df = read_csv_str(_resolve_csv(flatfiles_dir, STATIONS_CSV))
    out = pd.DataFrame(index=df.index)
    out["station_id"] = df["CodeSta"].astype(str).str.strip()
    out["station_latitude_deg"] = numeric_series(df.get("Latitude", pd.Series(index=df.index)))
    out["station_longitude_deg"] = numeric_series(df.get("Longitude", pd.Series(index=df.index)))
    out["station_altitude_m"] = numeric_series(df.get("Altitude", pd.Series(index=df.index)))
    out["station_name"] = df.get("StaName", pd.Series(index=df.index, dtype=str))
    out["network"] = df.get("Network", pd.Series(index=df.index, dtype=str))
    out["geology"] = df.get("Geology", pd.Series(index=df.index, dtype=str))
    out["vs30_m_s"] = coalesce_columns(df, ["PreferedVs30", "MeasuredVs30", "Vs30_fCSN", "InferedVs30"])
    out["f0_hvsr_hz"] = numeric_series(df.get("f0_HVSR", pd.Series(index=df.index)))
    out["a0_hvsr"] = numeric_series(df.get("A0_HVSR", pd.Series(index=df.index)))
    out["kappa0"] = numeric_series(df.get("Kappa0", pd.Series(index=df.index)))
    out["topographic_slope"] = numeric_series(df.get("TopographicSlope", pd.Series(index=df.index)))
    out["channels"] = df.get("Channels", pd.Series(index=df.index, dtype=str))
    return out.drop_duplicates("station_id")


def load_record_flatfile(flatfiles_dir: Path) -> pd.DataFrame:
    df = read_csv_str(_resolve_csv(flatfiles_dir, RECORDS_CSV))
    out = pd.DataFrame(index=df.index)
    out["flat_record_id"] = df["RecordID"].astype(str).str.strip()
    out["event_id"] = df["EventID"].astype(str).str.strip()
    out["station_id"] = df["StationID"].astype(str).str.strip()
    out["event_latitude_deg"] = numeric_series(df.get("EventLatitude_deg", pd.Series(index=df.index)))
    out["event_longitude_deg"] = numeric_series(df.get("EventLongitude_deg", pd.Series(index=df.index)))
    out["event_depth_km"] = numeric_series(df.get("EventDepth_km", pd.Series(index=df.index)))
    out["mw"] = numeric_series(df.get("Mw", pd.Series(index=df.index)))
    out["tectonic_type"] = df.get("TectonicType", pd.Series(index=df.index, dtype=str))
    out["station_latitude_deg"] = numeric_series(df.get("StationLatitude_deg", pd.Series(index=df.index)))
    out["station_longitude_deg"] = numeric_series(df.get("StationLongitude_deg", pd.Series(index=df.index)))
    out["vs30_m_s_record"] = numeric_series(df.get("StationVs30_m_s", pd.Series(index=df.index)))
    out["repi_km_flatfile"] = numeric_series(df.get("Repi_km_", pd.Series(index=df.index)))
    out["rhyp_km_flatfile"] = numeric_series(df.get("Rhyp_km_", pd.Series(index=df.index)))
    out["rjb_km_flatfile"] = numeric_series(df.get("Rjb_km_", pd.Series(index=df.index)))
    out["rrup_km_flatfile"] = numeric_series(df.get("Rrup_km_", pd.Series(index=df.index)))
    out["sample_rate_hz_flatfile"] = numeric_series(df.get("Samp_Rate_Hz_", pd.Series(index=df.index)))
    out["flat_pga_g_n"] = numeric_series(df.get("PGA_g_N", pd.Series(index=df.index)))
    out["flat_pga_g_e"] = numeric_series(df.get("PGA_g_E", pd.Series(index=df.index)))
    out["flat_pga_g_z"] = numeric_series(df.get("PGA_g_Z", pd.Series(index=df.index)))
    out["flat_arias_m_s_n"] = numeric_series(df.get("Ia_ms_N", pd.Series(index=df.index)))
    out["flat_arias_m_s_e"] = numeric_series(df.get("Ia_ms_E", pd.Series(index=df.index)))
    out["flat_arias_m_s_z"] = numeric_series(df.get("Ia_ms_Z", pd.Series(index=df.index)))
    out["flat_duration_5_95_s_n"] = numeric_series(df.get("Ds_5_to_95_N", pd.Series(index=df.index)))
    out["flat_duration_5_95_s_e"] = numeric_series(df.get("Ds_5_to_95_E", pd.Series(index=df.index)))
    out["flat_duration_5_95_s_z"] = numeric_series(df.get("Ds_5_to_95_Z", pd.Series(index=df.index)))
    out["flat_pga_hmax_g"] = np.nanmax(
        out[["flat_pga_g_n", "flat_pga_g_e"]].to_numpy(dtype=float),
        axis=1,
    )
    return out.drop_duplicates(["event_id", "station_id"])


def load_flatfiles(flatfiles_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_events(flatfiles_dir),
        load_record_flatfile(flatfiles_dir),
        load_stations(flatfiles_dir),
    )

