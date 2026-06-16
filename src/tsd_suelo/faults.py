from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .atlas import write_geojson, write_kmz
from .graph import mode_columns
from .utils import write_parquet


def _merge_modes(geo_targets: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    if geo_targets.empty or modes.empty:
        return geo_targets.copy()
    keep = ["record_observed_id"] + mode_columns(modes)
    return geo_targets.merge(modes[keep], on="record_observed_id", how="left")


def _route_level(geo_targets: pd.DataFrame) -> int | None:
    for level in (3, 4, 2, 1):
        if f"route_id_j{level}" in geo_targets.columns:
            return level
    return None


def _rank01(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    return numeric.fillna(numeric.median()).rank(pct=True).astype(float)


def _quantile90(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(0.9)) if not numeric.empty else float("nan")


def _strike_from_backazimuth(backazimuth_deg: pd.Series) -> pd.Series:
    return pd.to_numeric(backazimuth_deg, errors="coerce").mod(180.0)


def build_fault_candidates(
    geo_targets: pd.DataFrame,
    modes: pd.DataFrame,
    kozyrev_fields: pd.DataFrame,
    top_n: int | None = None,
    min_records: int = 1,
) -> pd.DataFrame:
    """Build observed fault/lineament candidates from route anomalies.

    These are not named catalog faults. They are observed corridors where latent
    residual modes and Kozyrev deltas concentrate after source/distance/site
    residualization.
    """

    level = _route_level(geo_targets)
    required = {
        "record_observed_id",
        "event_latitude_deg",
        "event_longitude_deg",
        "station_latitude_deg",
        "station_longitude_deg",
    }
    if level is None or geo_targets.empty or not required.issubset(geo_targets.columns):
        return pd.DataFrame()

    route_col = f"route_id_j{level}"
    observed = _merge_modes(geo_targets, modes)
    mode_cols = mode_columns(observed)
    if mode_cols:
        observed["mode_anomaly_score"] = np.sqrt(np.square(observed[mode_cols].fillna(0.0)).sum(axis=1))
    else:
        observed["mode_anomaly_score"] = np.nan

    agg: dict[str, Any] = {
        "record_observed_id": "count",
        "event_latitude_deg": "mean",
        "event_longitude_deg": "mean",
        "station_latitude_deg": "mean",
        "station_longitude_deg": "mean",
        "distance_km": "mean",
        "backazimuth_deg": "mean",
        "pga_h_g": "mean",
        "arias_h_m_s": "mean",
        "mode_anomaly_score": ["mean", _quantile90],
    }
    if "route_in_chile_mask" in observed.columns:
        agg["route_in_chile_mask"] = "mean"
    if "receiver_in_chile_mask" in observed.columns:
        agg["receiver_in_chile_mask"] = "mean"

    agg = {column: spec for column, spec in agg.items() if column in observed.columns}
    work_cols = [route_col] + [column for column in agg if column in observed.columns]
    grouped = observed[work_cols].dropna(subset=[route_col]).groupby(route_col, dropna=False).agg(agg)
    if grouped.empty:
        return pd.DataFrame()
    grouped.columns = ["_".join([part for part in column if part]) if isinstance(column, tuple) else column for column in grouped.columns]
    grouped = grouped.reset_index().rename(
        columns={
            route_col: "route_id",
            "record_observed_id_count": "n_records",
            "mode_anomaly_score_mean": "mode_anomaly_mean",
            "mode_anomaly_score__quantile90": "mode_anomaly_p90",
            "route_in_chile_mask_mean": "route_in_chile_mask_fraction",
            "receiver_in_chile_mask_mean": "receiver_in_chile_mask_fraction",
        }
    )
    grouped = grouped[grouped["n_records"] >= min_records].copy()
    if grouped.empty:
        return grouped

    grouped["node_id"] = "route:" + grouped["route_id"].astype(str)
    route_fields = kozyrev_fields[
        (kozyrev_fields.get("node_type") == "route") & (kozyrev_fields.get("level") == level)
    ].copy() if not kozyrev_fields.empty and {"node_type", "level", "node_id"}.issubset(kozyrev_fields.columns) else pd.DataFrame()
    if not route_fields.empty:
        keep = [column for column in ["node_id", "kozyrev_delta_norm", "mode_norm"] if column in route_fields.columns]
        grouped = grouped.merge(route_fields[keep], on="node_id", how="left")
    else:
        grouped["kozyrev_delta_norm"] = np.nan
        grouped["mode_norm"] = np.nan

    for column in ("mode_anomaly_p90", "kozyrev_delta_norm", "pga_h_g_mean"):
        if column not in grouped.columns:
            grouped[column] = np.nan

    grouped["midpoint_latitude_deg"] = (grouped["event_latitude_deg_mean"] + grouped["station_latitude_deg_mean"]) / 2.0
    grouped["midpoint_longitude_deg"] = (grouped["event_longitude_deg_mean"] + grouped["station_longitude_deg_mean"]) / 2.0
    grouped["strike_deg"] = _strike_from_backazimuth(grouped["backazimuth_deg_mean"])
    grouped["fault_candidate_score"] = (
        0.42 * _rank01(grouped["mode_anomaly_p90"])
        + 0.36 * _rank01(grouped["kozyrev_delta_norm"])
        + 0.12 * _rank01(grouped["pga_h_g_mean"])
        + 0.10 * _rank01(np.log1p(grouped["n_records"]))
    )
    grouped["fault_probability_pct"] = (100.0 * grouped["fault_candidate_score"]).clip(0.0, 100.0)
    grouped["probability_basis"] = "empirical_route_percentile"
    grouped = grouped.sort_values("fault_candidate_score", ascending=False).reset_index(drop=True)
    if top_n is not None:
        grouped = grouped.head(top_n).reset_index(drop=True)
    grouped.insert(0, "candidate_id", [f"fault_candidate_{idx:06d}" for idx in range(1, len(grouped) + 1)])
    grouped.insert(1, "priority_rank", np.arange(1, len(grouped) + 1))

    high_cut = grouped["fault_candidate_score"].quantile(0.9) if len(grouped) > 1 else grouped["fault_candidate_score"].max()
    medium_cut = grouped["fault_candidate_score"].quantile(0.75) if len(grouped) > 1 else grouped["fault_candidate_score"].max()
    grouped["confidence"] = np.select(
        [
            (grouped["fault_candidate_score"] >= high_cut) & (grouped["n_records"] >= 5),
            grouped["fault_candidate_score"] >= medium_cut,
        ],
        ["high", "medium"],
        default="screening",
    )
    grouped["interpretation"] = "lineamiento_candidato_no_catalogado"
    grouped["route_level"] = level
    return grouped


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def fault_candidate_features(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    features = []
    if candidates.empty:
        return features
    for _, row in candidates.iterrows():
        lon1 = row.get("event_longitude_deg_mean")
        lat1 = row.get("event_latitude_deg_mean")
        lon2 = row.get("station_longitude_deg_mean")
        lat2 = row.get("station_latitude_deg_mean")
        if not all(np.isfinite([lon1, lat1, lon2, lat2])):
            continue
        props = {key: _clean(value) for key, value in row.items() if not key.endswith("_deg_mean")}
        props["feature_type"] = "fault_candidate"
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(lon1), float(lat1)], [float(lon2), float(lat2)]],
                },
                "properties": props,
            }
        )
    return features


def write_fault_products(candidates: pd.DataFrame, output_dir: Path, top_n: int = 200) -> None:
    write_parquet(candidates, output_dir / "fault_candidates.parquet")
    candidates.head(top_n).to_csv(output_dir / "top_fault_candidates.csv", index=False)
    features = fault_candidate_features(candidates)
    write_geojson(features, output_dir / "fault_candidates.geojson")
    write_kmz(features, output_dir / "fault_candidates.kmz")
