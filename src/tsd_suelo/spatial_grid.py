from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .atlas import write_geojson, write_kmz
from .graph import mode_columns
from .utils import ensure_dir, write_parquet


DEFAULT_SPATIAL_LEVELS = tuple(range(1, 13))
DEFAULT_BASE_STEP_DEG = 4.0


def _rank01(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    filled = numeric.fillna(numeric.median())
    if float(filled.max()) <= 0.0 and float(filled.min()) >= 0.0:
        return pd.Series(0.0, index=values.index)
    return filled.rank(pct=True).astype(float)


def _rank_pct_by_level(frame: pd.DataFrame, value_col: str) -> pd.Series:
    if frame.empty or value_col not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)
    return 100.0 * frame.groupby("level", group_keys=False)[value_col].apply(_rank01)


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def _merge_modes(geo_targets: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    if geo_targets.empty or modes.empty:
        return geo_targets.copy()
    keep = ["record_observed_id"] + mode_columns(modes)
    return geo_targets.merge(modes[keep], on="record_observed_id", how="left")


def _cell_indices(lat: pd.Series, lon: pd.Series, level: int, base_step_deg: float) -> tuple[pd.Series, pd.Series, float]:
    step = base_step_deg / (2 ** (level - 1))
    x = np.floor((pd.to_numeric(lon, errors="coerce") + 180.0) / step).astype("Int64")
    y = np.floor((pd.to_numeric(lat, errors="coerce") + 90.0) / step).astype("Int64")
    return x, y, step


def _sample_points(geo_targets: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    observed = _merge_modes(geo_targets, modes)
    mode_cols = mode_columns(observed)
    base_cols = [
        "record_observed_id",
        "event_id",
        "station_id",
        "distance_km",
        "backazimuth_deg",
        "pga_h_g",
        "arias_h_m_s",
    ] + mode_cols
    base_cols = [column for column in base_cols if column in observed.columns]
    frames = []

    if {"station_latitude_deg", "station_longitude_deg"}.issubset(observed.columns):
        receiver = observed[base_cols + ["station_latitude_deg", "station_longitude_deg"]].copy()
        receiver = receiver.rename(columns={"station_latitude_deg": "sample_latitude_deg", "station_longitude_deg": "sample_longitude_deg"})
        receiver["sample_role"] = "receiver"
        frames.append(receiver)

    if {"event_latitude_deg", "event_longitude_deg", "station_latitude_deg", "station_longitude_deg"}.issubset(observed.columns):
        route = observed[base_cols + ["event_latitude_deg", "event_longitude_deg", "station_latitude_deg", "station_longitude_deg"]].copy()
        route["sample_latitude_deg"] = (route["event_latitude_deg"] + route["station_latitude_deg"]) / 2.0
        route["sample_longitude_deg"] = (route["event_longitude_deg"] + route["station_longitude_deg"]) / 2.0
        route["sample_role"] = "route_midpoint"
        route = route.drop(columns=["event_latitude_deg", "event_longitude_deg", "station_latitude_deg", "station_longitude_deg"])
        frames.append(route)

    samples = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if samples.empty:
        return samples
    samples = samples.dropna(subset=["sample_latitude_deg", "sample_longitude_deg"]).copy()
    if mode_cols:
        samples["sample_mode_norm"] = np.sqrt(np.square(samples[mode_cols].fillna(0.0)).sum(axis=1))
    else:
        samples["sample_mode_norm"] = np.nan
    return samples


def build_spatial_grid_nodes(
    geo_targets: pd.DataFrame,
    modes: pd.DataFrame,
    levels: tuple[int, ...] = DEFAULT_SPATIAL_LEVELS,
    base_step_deg: float = DEFAULT_BASE_STEP_DEG,
) -> pd.DataFrame:
    samples = _sample_points(geo_targets, modes)
    if samples.empty:
        return pd.DataFrame()

    mode_cols = mode_columns(samples)
    frames = []
    for level in levels:
        work = samples.copy()
        work["level"] = level
        work["grid_step_deg"] = base_step_deg / (2 ** (level - 1))
        work["grid_x"], work["grid_y"], step = _cell_indices(work["sample_latitude_deg"], work["sample_longitude_deg"], level, base_step_deg)
        work = work[work["grid_x"].notna() & work["grid_y"].notna()].copy()
        work["grid_x"] = work["grid_x"].astype(int)
        work["grid_y"] = work["grid_y"].astype(int)
        work["cell_id"] = "J" + str(level) + ":x" + work["grid_x"].astype(str) + ":y" + work["grid_y"].astype(str)
        work["node_id"] = "spatial:" + work["cell_id"]
        work["lon_min_deg"] = work["grid_x"] * step - 180.0
        work["lon_max_deg"] = work["lon_min_deg"] + step
        work["lat_min_deg"] = work["grid_y"] * step - 90.0
        work["lat_max_deg"] = work["lat_min_deg"] + step
        work["center_longitude_deg"] = work["lon_min_deg"] + step / 2.0
        work["center_latitude_deg"] = work["lat_min_deg"] + step / 2.0
        if level > 1:
            work["parent_cell_id"] = (
                "J"
                + str(level - 1)
                + ":x"
                + (work["grid_x"] // 2).astype(str)
                + ":y"
                + (work["grid_y"] // 2).astype(str)
            )
        else:
            work["parent_cell_id"] = None

        agg: dict[str, Any] = {
            "record_observed_id": pd.Series.nunique,
            "event_id": pd.Series.nunique if "event_id" in work.columns else "size",
            "station_id": pd.Series.nunique if "station_id" in work.columns else "size",
            "sample_role": lambda values: ",".join(sorted(set(values.dropna().astype(str)))),
            "sample_mode_norm": "mean",
            "pga_h_g": "mean" if "pga_h_g" in work.columns else "size",
            "arias_h_m_s": "mean" if "arias_h_m_s" in work.columns else "size",
            "distance_km": "mean" if "distance_km" in work.columns else "size",
            "backazimuth_deg": "mean" if "backazimuth_deg" in work.columns else "size",
            "grid_step_deg": "first",
            "grid_x": "first",
            "grid_y": "first",
            "lon_min_deg": "first",
            "lon_max_deg": "first",
            "lat_min_deg": "first",
            "lat_max_deg": "first",
            "center_longitude_deg": "first",
            "center_latitude_deg": "first",
            "parent_cell_id": "first",
        }
        for column in mode_cols:
            agg[column] = "mean"
        agg = {column: spec for column, spec in agg.items() if column in work.columns}
        grouped = (
            work.groupby(["level", "cell_id", "node_id"], dropna=False)
            .agg(agg)
            .rename(
                columns={
                    "record_observed_id": "n_records",
                    "event_id": "n_events",
                    "station_id": "n_stations",
                    "sample_mode_norm": "mode_norm",
                    "pga_h_g": "pga_h_g_mean",
                    "arias_h_m_s": "arias_h_m_s_mean",
                    "distance_km": "distance_km_mean",
                    "backazimuth_deg": "backazimuth_deg_mean",
                }
            )
            .reset_index()
        )
        grouped["node_type"] = "spatial_cell"
        frames.append(grouped)

    nodes = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if nodes.empty:
        return nodes
    for column in ("mode_norm", "pga_h_g_mean", "arias_h_m_s_mean", "n_records"):
        if column not in nodes.columns:
            nodes[column] = np.nan
    nodes["support_probability_pct"] = _rank_pct_by_level(nodes.assign(log_support=np.log1p(pd.to_numeric(nodes["n_records"], errors="coerce"))), "log_support")
    nodes["mode_probability_pct"] = _rank_pct_by_level(nodes, "mode_norm")
    nodes["pga_probability_pct"] = _rank_pct_by_level(nodes, "pga_h_g_mean")
    nodes["arias_probability_pct"] = _rank_pct_by_level(nodes, "arias_h_m_s_mean")
    nodes["intensity_probability_pct"] = nodes[["pga_probability_pct", "arias_probability_pct"]].max(axis=1, skipna=True).fillna(0.0)
    nodes["anomaly_probability_pct"] = (
        0.55 * nodes["mode_probability_pct"].fillna(0.0)
        + 0.25 * nodes["intensity_probability_pct"].fillna(0.0)
        + 0.20 * nodes["support_probability_pct"].fillna(0.0)
    ).clip(0.0, 100.0)
    nodes["probability_basis"] = "empirical_spatial_grid_percentile"
    return nodes.sort_values(["level", "grid_y", "grid_x"]).reset_index(drop=True)


def _neighbor_pairs(nodes_level: pd.DataFrame) -> list[tuple[int, int, str, float]]:
    lookup = {(int(row.grid_x), int(row.grid_y)): idx for idx, row in nodes_level.iterrows()}
    offsets = [
        (1, 0, "east_west", 90.0),
        (0, 1, "north_south", 0.0),
        (1, 1, "diagonal_ne_sw", 45.0),
        (1, -1, "diagonal_nw_se", 135.0),
    ]
    pairs = []
    for idx, row in nodes_level.iterrows():
        x = int(row.grid_x)
        y = int(row.grid_y)
        for dx, dy, orientation, azimuth in offsets:
            other = lookup.get((x + dx, y + dy))
            if other is not None:
                pairs.append((idx, other, orientation, azimuth))
    return pairs


def build_spatial_grid_edges(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame()
    mode_cols = mode_columns(nodes)
    edge_rows = []
    for level, nodes_level in nodes.groupby("level", sort=True):
        nodes_level = nodes_level.reset_index(drop=True)
        for idx_a, idx_b, orientation, azimuth in _neighbor_pairs(nodes_level):
            a = nodes_level.loc[idx_a]
            b = nodes_level.loc[idx_b]
            if mode_cols:
                diff = a[mode_cols].astype(float).fillna(0.0).to_numpy() - b[mode_cols].astype(float).fillna(0.0).to_numpy()
                mode_jump = float(np.sqrt(np.square(diff).sum()))
            else:
                mode_jump = float(abs(float(a.get("mode_norm", 0.0) or 0.0) - float(b.get("mode_norm", 0.0) or 0.0)))
            edge_rows.append(
                {
                    "level": int(level),
                    "edge_id": f"spatial_edge:J{int(level)}:{a.cell_id}->{b.cell_id}",
                    "from_node": a.node_id,
                    "to_node": b.node_id,
                    "from_cell_id": a.cell_id,
                    "to_cell_id": b.cell_id,
                    "neighbor_orientation": orientation,
                    "neighbor_azimuth_deg": azimuth,
                    "neighbor_kind": "diagonal" if "diagonal" in orientation else "rook",
                    "from_longitude_deg": float(a.center_longitude_deg),
                    "from_latitude_deg": float(a.center_latitude_deg),
                    "to_longitude_deg": float(b.center_longitude_deg),
                    "to_latitude_deg": float(b.center_latitude_deg),
                    "from_anomaly_probability_pct": float(a.anomaly_probability_pct),
                    "to_anomaly_probability_pct": float(b.anomaly_probability_pct),
                    "anomaly_probability_jump_pct": float(abs(a.anomaly_probability_pct - b.anomaly_probability_pct)),
                    "mode_jump_norm": mode_jump,
                    "pga_jump": float(abs((a.get("pga_h_g_mean") or 0.0) - (b.get("pga_h_g_mean") or 0.0))),
                    "arias_jump": float(abs((a.get("arias_h_m_s_mean") or 0.0) - (b.get("arias_h_m_s_mean") or 0.0))),
                    "min_n_records": int(min(a.n_records, b.n_records)),
                    "mean_n_records": float((a.n_records + b.n_records) / 2.0),
                }
            )
    edges = pd.DataFrame(edge_rows)
    if edges.empty:
        return edges
    edges["mode_jump_probability_pct"] = _rank_pct_by_level(edges, "mode_jump_norm")
    edges["anomaly_jump_probability_pct"] = _rank_pct_by_level(edges, "anomaly_probability_jump_pct")
    edges["support_probability_pct"] = _rank_pct_by_level(edges.assign(log_support=np.log1p(pd.to_numeric(edges["min_n_records"], errors="coerce"))), "log_support")
    edges["fault_probability_pct"] = (
        0.62 * edges["mode_jump_probability_pct"].fillna(0.0)
        + 0.25 * edges["anomaly_jump_probability_pct"].fillna(0.0)
        + 0.13 * edges["support_probability_pct"].fillna(0.0)
    ).clip(0.0, 100.0)
    edges["edge_probability_pct"] = edges["fault_probability_pct"]
    edges["probability_basis"] = "empirical_spatial_neighbor_jump_percentile"
    edges["edge_family"] = "spatial_neighbor_grid"
    return edges.sort_values(["level", "fault_probability_pct"], ascending=[True, False]).reset_index(drop=True)


def _display_level(nodes: pd.DataFrame, edges: pd.DataFrame) -> int | None:
    if not edges.empty:
        counts = edges.groupby("level").size()
        usable = counts[counts >= 3]
        if not usable.empty:
            return int(usable.index.max())
        return int(counts.index.max())
    if not nodes.empty:
        return int(pd.to_numeric(nodes["level"], errors="coerce").max())
    return None


def _node_feature(row: pd.Series) -> dict[str, Any] | None:
    required = ["lon_min_deg", "lat_min_deg", "lon_max_deg", "lat_max_deg"]
    if not all(np.isfinite([row.get(column) for column in required])):
        return None
    props = {key: _clean(value) for key, value in row.items() if key not in required}
    props["feature_type"] = "spatial_grid_node"
    lon_min, lat_min, lon_max, lat_max = [float(row[column]) for column in required]
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min],
                ]
            ],
        },
        "properties": props,
    }


def _edge_feature(row: pd.Series) -> dict[str, Any] | None:
    coords = [row.get("from_longitude_deg"), row.get("from_latitude_deg"), row.get("to_longitude_deg"), row.get("to_latitude_deg")]
    if not all(np.isfinite(coords)):
        return None
    props = {
        key: _clean(value)
        for key, value in row.items()
        if key not in {"from_longitude_deg", "from_latitude_deg", "to_longitude_deg", "to_latitude_deg"}
    }
    props["feature_type"] = "spatial_grid_edge"
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(coords[0]), float(coords[1])], [float(coords[2]), float(coords[3])]],
        },
        "properties": props,
    }


def spatial_grid_features(nodes: pd.DataFrame, edges: pd.DataFrame, level: int | None = None) -> list[dict[str, Any]]:
    selected_level = _display_level(nodes, edges) if level is None else level
    if selected_level is None:
        return []
    node_layer = nodes[pd.to_numeric(nodes.get("level"), errors="coerce") == selected_level].copy() if not nodes.empty else pd.DataFrame()
    edge_layer = edges[pd.to_numeric(edges.get("level"), errors="coerce") == selected_level].copy() if not edges.empty else pd.DataFrame()
    features = []
    features.extend(feature for _, row in node_layer.iterrows() if (feature := _node_feature(row)) is not None)
    features.extend(feature for _, row in edge_layer.iterrows() if (feature := _edge_feature(row)) is not None)
    for feature in features:
        feature["properties"]["display_level"] = selected_level
    return features


def write_spatial_grid_products(nodes: pd.DataFrame, edges: pd.DataFrame, output_dir: Path) -> None:
    write_parquet(nodes, output_dir / "spatial_grid_nodes.parquet")
    write_parquet(edges, output_dir / "spatial_grid_edges.parquet")
    level = _display_level(nodes, edges)
    node_layer = nodes[pd.to_numeric(nodes.get("level"), errors="coerce") == level].copy() if level is not None and not nodes.empty else pd.DataFrame()
    edge_layer = edges[pd.to_numeric(edges.get("level"), errors="coerce") == level].copy() if level is not None and not edges.empty else pd.DataFrame()
    write_geojson([feature for _, row in node_layer.iterrows() if (feature := _node_feature(row)) is not None], output_dir / "spatial_anomaly_nodes.geojson")
    write_geojson([feature for _, row in edge_layer.iterrows() if (feature := _edge_feature(row)) is not None], output_dir / "spatial_fault_edges.geojson")
    heatmap = spatial_grid_features(nodes, edges, level=level)
    write_geojson(heatmap, output_dir / "spatial_probability_heatmap.geojson")
    write_kmz(heatmap, output_dir / "spatial_probability_heatmap.kmz")
    ensure_dir(output_dir)
    (output_dir / "spatial_grid_summary.json").write_text(
        json.dumps(
            {
                "display_level": level,
                "nodes": int(nodes.shape[0]),
                "edges": int(edges.shape[0]),
                "probability_basis": "empirical spatial-grid percentiles by level",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
