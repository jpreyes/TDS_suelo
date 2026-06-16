from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .atlas import write_geojson, write_kmz
from .graph import mode_columns
from .utils import write_parquet


def _rank01(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    filled = numeric.fillna(numeric.median())
    if float(filled.max()) <= 0.0 and float(filled.min()) >= 0.0:
        return pd.Series(0.0, index=values.index)
    return filled.rank(pct=True).astype(float)


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def _geometry_rows(geo_targets: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for level in (1, 2, 3, 4):
        specs = [
            (
                "source3d",
                f"source_cell_j{level}",
                "event_latitude_deg",
                "event_longitude_deg",
                None,
                None,
            ),
            (
                "receiver",
                f"receiver_cell_j{level}",
                "station_latitude_deg",
                "station_longitude_deg",
                None,
                None,
            ),
            (
                "route",
                f"route_id_j{level}",
                None,
                None,
                ("event_latitude_deg", "event_longitude_deg"),
                ("station_latitude_deg", "station_longitude_deg"),
            ),
        ]
        for node_type, node_col, lat_col, lon_col, start_cols, end_cols in specs:
            if node_col not in geo_targets.columns:
                continue
            work = geo_targets[geo_targets[node_col].notna()].copy()
            if work.empty:
                continue
            if node_type == "route":
                agg = {
                    "record_observed_id": "count",
                    start_cols[0]: "mean",
                    start_cols[1]: "mean",
                    end_cols[0]: "mean",
                    end_cols[1]: "mean",
                    "distance_km": "mean",
                    "backazimuth_deg": "mean",
                }
                agg = {column: fn for column, fn in agg.items() if column in work.columns}
                grouped = work.groupby(node_col, dropna=False).agg(agg).reset_index()
                grouped = grouped.rename(columns={node_col: "raw_node_id", "record_observed_id": "n_records_geometry"})
                grouped["centroid_latitude_deg_geometry"] = (
                    grouped["event_latitude_deg"] + grouped["station_latitude_deg"]
                ) / 2.0
                grouped["centroid_longitude_deg_geometry"] = (
                    grouped["event_longitude_deg"] + grouped["station_longitude_deg"]
                ) / 2.0
                grouped = grouped.rename(
                    columns={
                        "event_latitude_deg": "line_start_latitude_deg",
                        "event_longitude_deg": "line_start_longitude_deg",
                        "station_latitude_deg": "line_end_latitude_deg",
                        "station_longitude_deg": "line_end_longitude_deg",
                    }
                )
            else:
                agg = {
                    "record_observed_id": "count",
                    lat_col: "mean",
                    lon_col: "mean",
                    "distance_km": "mean",
                    "backazimuth_deg": "mean",
                }
                agg = {column: fn for column, fn in agg.items() if column in work.columns}
                grouped = work.groupby(node_col, dropna=False).agg(agg).reset_index()
                grouped = grouped.rename(
                    columns={
                        node_col: "raw_node_id",
                        "record_observed_id": "n_records_geometry",
                        lat_col: "centroid_latitude_deg_geometry",
                        lon_col: "centroid_longitude_deg_geometry",
                    }
                )
            grouped.insert(0, "node_type", node_type)
            grouped.insert(1, "level", level)
            grouped["node_id"] = grouped["node_type"] + ":" + grouped["raw_node_id"].astype(str)
            frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_ultrametric_nodes(geo_targets: pd.DataFrame, kozyrev_fields: pd.DataFrame) -> pd.DataFrame:
    geometry = _geometry_rows(geo_targets)
    if kozyrev_fields.empty:
        nodes = geometry.copy()
        nodes["parent_node_id"] = None
    else:
        nodes = kozyrev_fields.copy()
        if not geometry.empty:
            nodes = nodes.merge(
                geometry.drop(columns=["n_records_geometry"], errors="ignore"),
                on=["node_type", "level", "node_id"],
                how="left",
                suffixes=("", "_geometry"),
            )

    if nodes.empty:
        return nodes

    if "centroid_latitude_deg_geometry" in nodes.columns:
        nodes["centroid_latitude_deg"] = nodes["centroid_latitude_deg_geometry"].where(
            nodes["centroid_latitude_deg_geometry"].notna(),
            nodes.get("centroid_latitude_deg"),
        )
    if "centroid_longitude_deg_geometry" in nodes.columns:
        nodes["centroid_longitude_deg"] = nodes["centroid_longitude_deg_geometry"].where(
            nodes["centroid_longitude_deg_geometry"].notna(),
            nodes.get("centroid_longitude_deg"),
        )

    for column in ("kozyrev_delta_norm", "mode_norm", "n_records"):
        if column not in nodes.columns:
            nodes[column] = np.nan
    nodes["support_probability_pct"] = 100.0 * _rank01(np.log1p(pd.to_numeric(nodes["n_records"], errors="coerce")))
    nodes["delta_probability_pct"] = 100.0 * _rank01(nodes["kozyrev_delta_norm"])
    nodes["mode_probability_pct"] = 100.0 * _rank01(nodes["mode_norm"])
    nodes["failure_probability_pct"] = (
        0.48 * nodes["delta_probability_pct"]
        + 0.37 * nodes["mode_probability_pct"]
        + 0.15 * nodes["support_probability_pct"]
    ).clip(0.0, 100.0)
    nodes["probability_basis"] = "empirical_ultrametric_percentile"
    nodes["graph_node_family"] = "kozyrev_ultrametric_node"
    return nodes.sort_values(["level", "node_type", "node_id"]).reset_index(drop=True)


def _node_lookup(nodes: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "node_id",
        "node_type",
        "level",
        "centroid_latitude_deg",
        "centroid_longitude_deg",
        "failure_probability_pct",
        "mode_norm",
        "kozyrev_delta_norm",
        "n_records",
    ]
    return nodes[[column for column in keep if column in nodes.columns]].drop_duplicates("node_id")


def _parent_child_edges(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty or "parent_node_id" not in nodes.columns:
        return pd.DataFrame()
    child = nodes[nodes["parent_node_id"].notna()].copy()
    if child.empty:
        return pd.DataFrame()
    parent = _node_lookup(nodes).rename(
        columns={
            "node_id": "from_node",
            "centroid_latitude_deg": "from_latitude_deg",
            "centroid_longitude_deg": "from_longitude_deg",
            "failure_probability_pct": "from_failure_probability_pct",
        }
    )
    edges = child.rename(
        columns={
            "parent_node_id": "from_node",
            "node_id": "to_node",
            "centroid_latitude_deg": "to_latitude_deg",
            "centroid_longitude_deg": "to_longitude_deg",
            "failure_probability_pct": "to_failure_probability_pct",
        }
    )
    edges = edges.merge(parent[["from_node", "from_latitude_deg", "from_longitude_deg", "from_failure_probability_pct"]], on="from_node", how="left")
    edges["edge_family"] = "ultrametric_parent_child"
    edges["edge_type"] = "parent_to_child"
    edges["from_level"] = edges["level"] - 1
    edges["to_level"] = edges["level"]
    edges["edge_probability_pct"] = (
        0.75 * pd.to_numeric(edges["to_failure_probability_pct"], errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(edges["from_failure_probability_pct"], errors="coerce").fillna(0.0)
    ).clip(0.0, 100.0)
    return edges


def _propagation_edges(route_graph: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    if route_graph.empty or nodes.empty:
        return pd.DataFrame()
    lookup = _node_lookup(nodes)
    from_lookup = lookup.rename(
        columns={
            "node_id": "from_node",
            "centroid_latitude_deg": "from_latitude_deg",
            "centroid_longitude_deg": "from_longitude_deg",
            "failure_probability_pct": "from_failure_probability_pct",
        }
    )
    to_lookup = lookup.rename(
        columns={
            "node_id": "to_node",
            "centroid_latitude_deg": "to_latitude_deg",
            "centroid_longitude_deg": "to_longitude_deg",
            "failure_probability_pct": "to_failure_probability_pct",
        }
    )
    edges = route_graph.copy()
    edges = edges.merge(from_lookup[["from_node", "from_latitude_deg", "from_longitude_deg", "from_failure_probability_pct"]], on="from_node", how="left")
    edges = edges.merge(to_lookup[["to_node", "to_latitude_deg", "to_longitude_deg", "to_failure_probability_pct"]], on="to_node", how="left")
    edges["edge_family"] = "source_route_receiver"
    edges["from_level"] = edges["level"]
    edges["to_level"] = edges["level"]
    edge_anomaly = 100.0 * _rank01(edges.get("mode_anomaly_score", pd.Series(np.nan, index=edges.index)))
    edges["edge_probability_pct"] = (
        0.46 * edge_anomaly
        + 0.27 * pd.to_numeric(edges["from_failure_probability_pct"], errors="coerce").fillna(0.0)
        + 0.27 * pd.to_numeric(edges["to_failure_probability_pct"], errors="coerce").fillna(0.0)
    ).clip(0.0, 100.0)
    return edges


def build_ultrametric_edges(route_graph: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    frames = [_parent_child_edges(nodes), _propagation_edges(route_graph, nodes)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    edges = pd.concat(frames, ignore_index=True, sort=False)
    edges["probability_basis"] = "empirical_ultrametric_percentile"
    return edges.sort_values(["edge_family", "edge_type", "from_node", "to_node"]).reset_index(drop=True)


def _node_feature(row: pd.Series) -> dict[str, Any] | None:
    props = {key: _clean(value) for key, value in row.items() if key not in {"centroid_longitude_deg", "centroid_latitude_deg"}}
    props["feature_type"] = "kozyrev_ultrametric_node"
    if row.get("node_type") == "route" and all(
        pd.notna(row.get(column))
        for column in ("line_start_longitude_deg", "line_start_latitude_deg", "line_end_longitude_deg", "line_end_latitude_deg")
    ):
        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(row["line_start_longitude_deg"]), float(row["line_start_latitude_deg"])],
                    [float(row["line_end_longitude_deg"]), float(row["line_end_latitude_deg"])],
                ],
            },
            "properties": props,
        }
    lon = row.get("centroid_longitude_deg")
    lat = row.get("centroid_latitude_deg")
    if not all(np.isfinite([lon, lat])):
        return None
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
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
    props["feature_type"] = "kozyrev_ultrametric_edge"
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(coords[0]), float(coords[1])], [float(coords[2]), float(coords[3])]],
        },
        "properties": props,
    }


def ultrametric_node_features(nodes: pd.DataFrame) -> list[dict[str, Any]]:
    return [feature for _, row in nodes.iterrows() if (feature := _node_feature(row)) is not None]


def ultrametric_edge_features(edges: pd.DataFrame) -> list[dict[str, Any]]:
    return [feature for _, row in edges.iterrows() if (feature := _edge_feature(row)) is not None]


def kozyrev_heatmap_features(nodes: pd.DataFrame, edges: pd.DataFrame) -> list[dict[str, Any]]:
    features = []
    if not nodes.empty:
        max_level = int(pd.to_numeric(nodes["level"], errors="coerce").max()) if "level" in nodes.columns else None
        route_nodes = nodes[(nodes.get("node_type") == "route") & (nodes.get("level") == max_level)].copy()
        source_receiver = nodes[nodes.get("node_type").isin(["source3d", "receiver"]) & (nodes.get("level") == max_level)].copy()
        features.extend(ultrametric_node_features(route_nodes))
        features.extend(ultrametric_node_features(source_receiver))
    if not edges.empty:
        max_level = int(pd.to_numeric(edges["to_level"], errors="coerce").max()) if "to_level" in edges.columns else None
        edge_layer = edges[pd.to_numeric(edges.get("to_level"), errors="coerce") == max_level].copy()
        features.extend(ultrametric_edge_features(edge_layer))
    for feature in features:
        feature["properties"]["feature_type"] = "kozyrev_probability_heatmap"
    return features


def write_ultrametric_products(nodes: pd.DataFrame, edges: pd.DataFrame, output_dir: Path) -> None:
    write_parquet(nodes, output_dir / "kozyrev_ultrametric_nodes.parquet")
    write_parquet(edges, output_dir / "kozyrev_ultrametric_edges.parquet")
    write_geojson(ultrametric_node_features(nodes), output_dir / "kozyrev_ultrametric_nodes.geojson")
    write_geojson(ultrametric_edge_features(edges), output_dir / "kozyrev_ultrametric_edges.geojson")
    heatmap = kozyrev_heatmap_features(nodes, edges)
    write_geojson(heatmap, output_dir / "kozyrev_heatmap.geojson")
    write_kmz(heatmap, output_dir / "kozyrev_heatmap.kmz")
