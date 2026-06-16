from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_dir


BUILTIN_CHILE_MASK = {
    "type": "FeatureCollection",
    "name": "builtin_coarse_chile_mask",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "name": "Chile coarse continental mask",
                "source": "built-in approximate operational mask; replace with --mask-geojson for official boundaries",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-75.8, -56.5],
                        [-74.8, -52.5],
                        [-75.2, -47.0],
                        [-74.6, -42.0],
                        [-73.6, -37.0],
                        [-72.5, -32.0],
                        [-71.5, -27.0],
                        [-70.6, -22.0],
                        [-70.0, -18.0],
                        [-68.3, -17.2],
                        [-67.3, -22.0],
                        [-68.1, -27.0],
                        [-69.3, -32.0],
                        [-70.4, -37.5],
                        [-71.6, -43.0],
                        [-72.4, -49.0],
                        [-72.2, -54.5],
                        [-75.8, -56.5],
                    ]
                ],
            },
        }
    ],
}


@dataclass(frozen=True)
class GeoMask:
    name: str
    polygons: list[list[tuple[float, float]]]
    geojson: dict[str, Any]

    def contains(self, lon: float, lat: float) -> bool:
        if not np.isfinite(lon) or not np.isfinite(lat):
            return False
        return any(_point_in_polygon(lon, lat, polygon) for polygon in self.polygons)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        points = [point for polygon in self.polygons for point in polygon]
        lons = [point[0] for point in points]
        lats = [point[1] for point in points]
        return min(lons), min(lats), max(lons), max(lats)


def load_chile_mask(mask_geojson: Path | None = None) -> GeoMask:
    if mask_geojson:
        payload = json.loads(mask_geojson.read_text(encoding="utf-8"))
        name = mask_geojson.stem
    else:
        payload = BUILTIN_CHILE_MASK
        name = "builtin_coarse_chile_mask"
    polygons = _extract_polygons(payload)
    if not polygons:
        raise ValueError("La mascara GeoJSON no contiene poligonos validos.")
    return GeoMask(name=name, polygons=polygons, geojson=payload)


def write_mask(mask: GeoMask, output_dir: Path) -> None:
    ensure_dir(output_dir)
    (output_dir / "chile_mask.geojson").write_text(
        json.dumps(mask.geojson, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def annotate_geo_targets(geo_targets: pd.DataFrame, mask: GeoMask) -> pd.DataFrame:
    out = geo_targets.copy()
    out["receiver_in_chile_mask"] = [
        mask.contains(lon, lat)
        for lon, lat in zip(out.get("station_longitude_deg", pd.Series(np.nan, index=out.index)), out.get("station_latitude_deg", pd.Series(np.nan, index=out.index)))
    ]
    out["source_in_chile_mask"] = [
        mask.contains(lon, lat)
        for lon, lat in zip(out.get("event_longitude_deg", pd.Series(np.nan, index=out.index)), out.get("event_latitude_deg", pd.Series(np.nan, index=out.index)))
    ]
    out["route_in_chile_mask"] = out["receiver_in_chile_mask"] | out["source_in_chile_mask"]
    return out


def feature_in_mask(feature: dict[str, Any], mask: GeoMask) -> bool:
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates")
    if geometry.get("type") == "Point" and coords:
        lon, lat = coords[:2]
        return mask.contains(float(lon), float(lat))
    if geometry.get("type") == "LineString" and coords:
        return any(mask.contains(float(lon), float(lat)) for lon, lat, *_ in coords)
    if geometry.get("type") in {"Polygon", "MultiPolygon"}:
        return True
    return False


def mask_feature(mask: GeoMask) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": mask.geojson["features"][0]["geometry"] if mask.geojson.get("type") == "FeatureCollection" else mask.geojson.get("geometry"),
        "properties": {
            "feature_type": "chile_mask",
            "mask_name": mask.name,
        },
    }


def _extract_polygons(payload: dict[str, Any]) -> list[list[tuple[float, float]]]:
    geometries: list[dict[str, Any]] = []
    if payload.get("type") == "FeatureCollection":
        geometries = [(feature.get("geometry") or {}) for feature in payload.get("features", [])]
    elif payload.get("type") == "Feature":
        geometries = [payload.get("geometry") or {}]
    else:
        geometries = [payload]

    polygons: list[list[tuple[float, float]]] = []
    for geometry in geometries:
        if geometry.get("type") == "Polygon":
            for ring in geometry.get("coordinates", [])[:1]:
                polygons.append([(float(lon), float(lat)) for lon, lat, *_ in ring])
        elif geometry.get("type") == "MultiPolygon":
            for polygon in geometry.get("coordinates", []):
                if polygon:
                    polygons.append([(float(lon), float(lat)) for lon, lat, *_ in polygon[0]])
    return polygons


def _point_in_polygon(lon: float, lat: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    if len(polygon) < 3:
        return False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if ((y1 > lat) != (y2 > lat)) and (lon < (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-12) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside

