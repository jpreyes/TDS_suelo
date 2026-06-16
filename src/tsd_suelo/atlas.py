from __future__ import annotations

import json
import math
import zipfile
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graph import mode_columns
from .mask import GeoMask, feature_in_mask, mask_feature
from .utils import ensure_dir


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def _props(row: pd.Series, exclude: set[str]) -> dict[str, Any]:
    return {key: _clean(value) for key, value in row.items() if key not in exclude}


def _point_feature(lon: float, lat: float, properties: dict[str, Any]) -> dict[str, Any] | None:
    if not np.isfinite(lon) or not np.isfinite(lat):
        return None
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "properties": properties,
    }


def _line_feature(lon1: float, lat1: float, lon2: float, lat2: float, properties: dict[str, Any]) -> dict[str, Any] | None:
    if not all(np.isfinite([lon1, lat1, lon2, lat2])):
        return None
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(lon1), float(lat1)], [float(lon2), float(lat2)]],
        },
        "properties": properties,
    }


def _merge_modes(geo_targets: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    if modes.empty:
        return geo_targets.copy()
    keep = ["record_observed_id"] + mode_columns(modes)
    return geo_targets.merge(modes[keep], on="record_observed_id", how="left")


def build_atlas_features(
    geo_targets: pd.DataFrame,
    modes: pd.DataFrame,
    kozyrev_fields: pd.DataFrame,
    geo_mask: GeoMask | None = None,
) -> list[dict[str, Any]]:
    observed = _merge_modes(geo_targets, modes)
    mode_cols = mode_columns(observed)
    if mode_cols:
        observed["mode_anomaly_score"] = np.sqrt(np.square(observed[mode_cols].fillna(0.0)).sum(axis=1))
    else:
        observed["mode_anomaly_score"] = np.nan

    features: list[dict[str, Any]] = [mask_feature(geo_mask)] if geo_mask else []
    station_agg = {
        "station_latitude_deg": "first",
        "station_longitude_deg": "first",
        "vs30_m_s": "first",
        "f0_hvsr_hz": "first",
        "a0_hvsr": "first",
        "geology": "first",
        "record_observed_id": "count",
        "pga_h_g": "mean",
        "mode_anomaly_score": "mean",
    }
    for column in mode_cols:
        station_agg[column] = "mean"
    station_nodes = (
        observed.groupby("station_id", dropna=False)
        .agg(station_agg)
        .rename(columns={"record_observed_id": "n_records"})
        .reset_index()
    )
    for _, row in station_nodes.iterrows():
        feature = _point_feature(
            row["station_longitude_deg"],
            row["station_latitude_deg"],
            {"feature_type": "receiver", **_props(row, {"station_longitude_deg", "station_latitude_deg"})},
        )
        if feature:
            features.append(feature)

    source_agg = {
        "event_latitude_deg": "first",
        "event_longitude_deg": "first",
        "event_depth_km": "first",
        "mw": "first",
        "tectonic_type": "first",
        "record_observed_id": "count",
        "pga_h_g": "mean",
        "mode_anomaly_score": "mean",
    }
    for column in mode_cols:
        source_agg[column] = "mean"
    source_nodes = (
        observed.groupby("event_id", dropna=False)
        .agg(source_agg)
        .rename(columns={"record_observed_id": "n_records"})
        .reset_index()
    )
    for _, row in source_nodes.iterrows():
        feature = _point_feature(
            row["event_longitude_deg"],
            row["event_latitude_deg"],
            {"feature_type": "source3d", **_props(row, {"event_longitude_deg", "event_latitude_deg"})},
        )
        if feature:
            features.append(feature)

    route_agg = {
        "event_latitude_deg": "mean",
        "event_longitude_deg": "mean",
        "station_latitude_deg": "mean",
        "station_longitude_deg": "mean",
        "record_observed_id": "count",
        "distance_km": "mean",
        "backazimuth_deg": "mean",
        "pga_h_g": "mean",
        "mode_anomaly_score": "mean",
    }
    for column in mode_cols:
        route_agg[column] = "mean"
    route_nodes = (
        observed.groupby("route_id_j3", dropna=False)
        .agg(route_agg)
        .rename(columns={"record_observed_id": "n_records"})
        .reset_index()
    )
    for _, row in route_nodes.iterrows():
        feature = _line_feature(
            row["event_longitude_deg"],
            row["event_latitude_deg"],
            row["station_longitude_deg"],
            row["station_latitude_deg"],
            {
                "feature_type": "route",
                **_props(row, {"event_longitude_deg", "event_latitude_deg", "station_longitude_deg", "station_latitude_deg"}),
            },
        )
        if feature:
            features.append(feature)

    if not kozyrev_fields.empty and {"centroid_longitude_deg", "centroid_latitude_deg"}.issubset(kozyrev_fields.columns):
        fields = kozyrev_fields.copy()
        if "kozyrev_delta_norm" in fields.columns:
            fields = fields.sort_values("kozyrev_delta_norm", ascending=False).head(250)
        for _, row in fields.iterrows():
            feature = _point_feature(
                row["centroid_longitude_deg"],
                row["centroid_latitude_deg"],
                {"feature_type": "kozyrev_field", **_props(row, {"centroid_longitude_deg", "centroid_latitude_deg"})},
            )
            if feature:
                features.append(feature)

    if geo_mask:
        mask = features[:1]
        masked = [feature for feature in features[1:] if feature_in_mask(feature, geo_mask)]
        return mask + masked
    return features


def write_geojson(features: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _kml_for_features(features: list[dict[str, Any]]) -> str:
    placemarks = []
    for feature in features:
        props = feature.get("properties", {})
        name = escape(str(props.get("station_id") or props.get("event_id") or props.get("node_id") or props.get("route_id_j3") or "feature"))
        description = escape(json.dumps(props, ensure_ascii=False, default=str))
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "Point":
            lon, lat = geometry["coordinates"][:2]
            geom_xml = f"<Point><coordinates>{lon},{lat},0</coordinates></Point>"
        elif geometry.get("type") == "LineString":
            coords = " ".join(f"{lon},{lat},0" for lon, lat in geometry["coordinates"])
            geom_xml = f"<LineString><coordinates>{coords}</coordinates></LineString>"
        elif geometry.get("type") == "Polygon":
            ring = geometry["coordinates"][0]
            coords = " ".join(f"{lon},{lat},0" for lon, lat, *_ in ring)
            geom_xml = f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
        elif geometry.get("type") == "MultiPolygon":
            parts = []
            for polygon in geometry["coordinates"]:
                if not polygon:
                    continue
                coords = " ".join(f"{lon},{lat},0" for lon, lat, *_ in polygon[0])
                parts.append(f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>")
            geom_xml = "<MultiGeometry>" + "".join(parts) + "</MultiGeometry>"
        else:
            continue
        placemarks.append(f"<Placemark><name>{name}</name><description>{description}</description>{geom_xml}</Placemark>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        + "".join(placemarks)
        + "</Document></kml>"
    )


def write_kmz(features: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    kml = _kml_for_features(features)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr("doc.kml", kml)


def write_atlas_products(
    geo_targets: pd.DataFrame,
    modes: pd.DataFrame,
    kozyrev_fields: pd.DataFrame,
    output_dir: Path,
    geo_mask: GeoMask | None = None,
) -> None:
    features = build_atlas_features(geo_targets, modes, kozyrev_fields, geo_mask=geo_mask)
    write_geojson(features, output_dir / "atlas_geologico.geojson")
    write_kmz(features, output_dir / "atlas_geologico.kmz")
