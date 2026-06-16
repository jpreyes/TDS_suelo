from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .utils import write_parquet


MODE_RE = re.compile(r"^mode_\d+$")


def mode_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if MODE_RE.match(column)]


def _merge_modes(geo_targets: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    if modes.empty:
        return geo_targets.copy()
    keep = ["record_observed_id"] + mode_columns(modes)
    return geo_targets.merge(modes[keep], on="record_observed_id", how="left")


def _edge_rows(observed: pd.DataFrame, level: int) -> pd.DataFrame:
    source_col = f"source_cell_j{level}"
    route_col = f"route_id_j{level}"
    receiver_col = f"receiver_cell_j{level}"
    rows = []
    for row in observed.itertuples(index=False):
        source = getattr(row, source_col, None)
        route = getattr(row, route_col, None)
        receiver = getattr(row, receiver_col, None)
        record_id = getattr(row, "record_observed_id", None)
        if pd.notna(source) and pd.notna(route):
            rows.append(
                {
                    "level": level,
                    "edge_type": "source3d_to_route",
                    "from_node": f"source3d:{source}",
                    "to_node": f"route:{route}",
                    "record_observed_id": record_id,
                }
            )
        if pd.notna(route) and pd.notna(receiver):
            rows.append(
                {
                    "level": level,
                    "edge_type": "route_to_receiver",
                    "from_node": f"route:{route}",
                    "to_node": f"receiver:{receiver}",
                    "record_observed_id": record_id,
                }
            )
    return pd.DataFrame(rows)


def build_route_graph(geo_targets: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    observed = _merge_modes(geo_targets, modes)
    mode_cols = mode_columns(observed)
    edge_frames = [_edge_rows(observed, level) for level in (1, 2, 3, 4)]
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    if edges.empty:
        return edges
    attrs = [
        "record_observed_id",
        "event_id",
        "station_id",
        "distance_km",
        "backazimuth_deg",
        "pga_h_g",
        "arias_h_m_s",
    ] + mode_cols
    attrs = [column for column in attrs if column in observed.columns]
    edges = edges.merge(observed[attrs], on="record_observed_id", how="left")
    agg_spec = {
        "record_observed_id": "count",
        "distance_km": "mean",
        "backazimuth_deg": "mean",
        "pga_h_g": "mean",
        "arias_h_m_s": "mean",
    }
    for column in mode_cols:
        agg_spec[column] = "mean"
    agg_spec = {k: v for k, v in agg_spec.items() if k in edges.columns}
    graph = (
        edges.groupby(["level", "edge_type", "from_node", "to_node"], dropna=False)
        .agg(agg_spec)
        .rename(columns={"record_observed_id": "n_records"})
        .reset_index()
    )
    if mode_cols:
        graph["mode_anomaly_score"] = np.sqrt(np.square(graph[mode_cols].fillna(0.0)).sum(axis=1))
    return graph


def _field_group(observed: pd.DataFrame, node_type: str, level: int) -> pd.DataFrame:
    if node_type == "source3d":
        node_col = f"source_cell_j{level}"
        parent_col = f"source_cell_j{level - 1}" if level > 1 else None
        lat_col, lon_col = "event_latitude_deg", "event_longitude_deg"
    elif node_type == "receiver":
        node_col = f"receiver_cell_j{level}"
        parent_col = f"receiver_cell_j{level - 1}" if level > 1 else None
        lat_col, lon_col = "station_latitude_deg", "station_longitude_deg"
    else:
        node_col = f"route_id_j{level}"
        parent_col = f"route_id_j{level - 1}" if level > 1 else None
        lat_col, lon_col = "station_latitude_deg", "station_longitude_deg"

    if node_col not in observed.columns:
        return pd.DataFrame()
    mode_cols = mode_columns(observed)
    work_cols = [node_col, "record_observed_id", "distance_km", "backazimuth_deg", lat_col, lon_col] + mode_cols
    if parent_col:
        work_cols.append(parent_col)
    work_cols = [column for column in work_cols if column in observed.columns]
    work = observed[work_cols].copy()
    work = work[work[node_col].notna()]
    if work.empty:
        return pd.DataFrame()

    agg = {
        "record_observed_id": "count",
        "distance_km": "mean",
        "backazimuth_deg": "mean",
        lat_col: "mean",
        lon_col: "mean",
    }
    for column in mode_cols:
        agg[column] = "mean"
    if parent_col:
        agg[parent_col] = "first"
    grouped = work.groupby(node_col, dropna=False).agg(agg).reset_index()
    grouped = grouped.rename(
        columns={
            node_col: "node_id",
            "record_observed_id": "n_records",
            lat_col: "centroid_latitude_deg",
            lon_col: "centroid_longitude_deg",
            parent_col or "": "parent_node_id",
        }
    )
    grouped.insert(0, "node_type", node_type)
    grouped.insert(1, "level", level)
    grouped["node_id"] = grouped["node_type"] + ":" + grouped["node_id"].astype(str)
    if parent_col and "parent_node_id" in grouped.columns:
        grouped["parent_node_id"] = grouped["node_type"] + ":" + grouped["parent_node_id"].astype(str)
    else:
        grouped["parent_node_id"] = None
    return grouped


def build_kozyrev_fields(geo_targets: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    observed = _merge_modes(geo_targets, modes)
    frames = []
    for level in (1, 2, 3, 4):
        for node_type in ("source3d", "route", "receiver"):
            frames.append(_field_group(observed, node_type, level))
    fields = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if fields.empty:
        return fields
    mode_cols = mode_columns(fields)
    parent_lookup = fields.set_index("node_id")[mode_cols].to_dict(orient="index") if mode_cols else {}
    for column in mode_cols:
        deltas = []
        for row in fields.itertuples(index=False):
            parent = getattr(row, "parent_node_id")
            current = getattr(row, column)
            parent_value = parent_lookup.get(parent, {}).get(column) if parent else 0.0
            deltas.append(current - parent_value if pd.notna(current) and parent_value is not None else np.nan)
        fields[f"kozyrev_delta_{column}"] = deltas
    if mode_cols:
        delta_cols = [f"kozyrev_delta_{column}" for column in mode_cols]
        fields["kozyrev_delta_norm"] = np.sqrt(np.square(fields[delta_cols].fillna(0.0)).sum(axis=1))
        fields["mode_norm"] = np.sqrt(np.square(fields[mode_cols].fillna(0.0)).sum(axis=1))
    return fields


def write_graph_products(route_graph: pd.DataFrame, kozyrev_fields: pd.DataFrame, output_dir) -> None:
    write_parquet(route_graph, output_dir / "route_graph_observed.parquet")
    write_parquet(kozyrev_fields, output_dir / "kozyrev_graph_fields.parquet")

