from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .utils import finite_or_nan


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if not all(np.isfinite([lat1, lon1, lat2, lon2])):
        return math.nan
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return finite_or_nan(2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a)))


def azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if not all(np.isfinite([lat1, lon1, lat2, lon2])):
        return math.nan
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return finite_or_nan((math.degrees(math.atan2(x, y)) + 360.0) % 360.0)


def backazimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return azimuth_deg(lat2, lon2, lat1, lon1)


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    if not all(np.isfinite([lat, lon, bearing_deg, distance_km])):
        return math.nan, math.nan
    angular_distance = distance_km / EARTH_RADIUS_KM
    bearing = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    sin_phi2 = (
        math.sin(phi1) * math.cos(angular_distance)
        + math.cos(phi1) * math.sin(angular_distance) * math.cos(bearing)
    )
    phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))
    lambda2 = lambda1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(phi1),
        math.cos(angular_distance) - math.sin(phi1) * math.sin(phi2),
    )
    out_lon = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0
    return finite_or_nan(math.degrees(phi2)), finite_or_nan(out_lon)


def incidence_angle_deg(epicentral_km: float, depth_km: float) -> float:
    if not all(np.isfinite([epicentral_km, depth_km])) or epicentral_km < 0 or depth_km < 0:
        return math.nan
    if depth_km == 0 and epicentral_km == 0:
        return 0.0
    return finite_or_nan(math.degrees(math.atan2(epicentral_km, depth_km)))


def direction_bin(angle_deg: float, width_deg: int = 30) -> str | None:
    if not np.isfinite(angle_deg):
        return None
    center = int((round(angle_deg / width_deg) * width_deg) % 360)
    return f"{center:03d}"


def cell_id(lat: float, lon: float, depth: float | None = None, level: int = 2) -> str | None:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None
    step = {1: 2.0, 2: 1.0, 3: 0.5, 4: 0.25}.get(level, 1.0)
    lat_bin = math.floor(lat / step) * step
    lon_bin = math.floor(lon / step) * step
    if depth is None or not np.isfinite(depth):
        return f"J{level}:lat{lat_bin:.2f}:lon{lon_bin:.2f}"
    depth_step = {1: 50.0, 2: 25.0, 3: 10.0, 4: 5.0}.get(level, 25.0)
    depth_bin = math.floor(depth / depth_step) * depth_step
    return f"J{level}:lat{lat_bin:.2f}:lon{lon_bin:.2f}:dep{depth_bin:.1f}"


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rows: list[dict[str, float | str | None]] = []
    for row in out.itertuples(index=False):
        event_lat = getattr(row, "event_latitude_deg", math.nan)
        event_lon = getattr(row, "event_longitude_deg", math.nan)
        event_depth = getattr(row, "event_depth_km", math.nan)
        station_lat = getattr(row, "station_latitude_deg", math.nan)
        station_lon = getattr(row, "station_longitude_deg", math.nan)
        repi = haversine_km(event_lat, event_lon, station_lat, station_lon)
        rhyp = math.sqrt(repi * repi + event_depth * event_depth) if np.isfinite(repi) and np.isfinite(event_depth) else math.nan
        az = azimuth_deg(event_lat, event_lon, station_lat, station_lon)
        baz = backazimuth_deg(event_lat, event_lon, station_lat, station_lon)
        rows.append(
            {
                "repi_km_calc": repi,
                "rhyp_km_calc": finite_or_nan(rhyp),
                "azimuth_deg": az,
                "backazimuth_deg": baz,
                "incidence_angle_deg": incidence_angle_deg(repi, event_depth),
                "direction_bin_30deg": direction_bin(baz, 30),
                "source_cell_j1": cell_id(event_lat, event_lon, event_depth, 1),
                "source_cell_j2": cell_id(event_lat, event_lon, event_depth, 2),
                "source_cell_j3": cell_id(event_lat, event_lon, event_depth, 3),
                "source_cell_j4": cell_id(event_lat, event_lon, event_depth, 4),
                "receiver_cell_j1": cell_id(station_lat, station_lon, None, 1),
                "receiver_cell_j2": cell_id(station_lat, station_lon, None, 2),
                "receiver_cell_j3": cell_id(station_lat, station_lon, None, 3),
                "receiver_cell_j4": cell_id(station_lat, station_lon, None, 4),
            }
        )
    geom = pd.DataFrame(rows, index=out.index)
    out = pd.concat([out, geom], axis=1)
    out["distance_km"] = out.get("rrup_km_flatfile", pd.Series(np.nan, index=out.index))
    for candidate in ("rhyp_km_h5", "rhyp_km_flatfile", "rhyp_km_calc", "repi_km_h5", "repi_km_flatfile", "repi_km_calc"):
        if candidate in out.columns:
            out["distance_km"] = out["distance_km"].where(out["distance_km"].notna(), out[candidate])
    for level in (1, 2, 3, 4):
        source = out[f"source_cell_j{level}"]
        receiver = out[f"receiver_cell_j{level}"]
        out[f"route_id_j{level}"] = np.where(
            source.notna() & receiver.notna(),
            source.astype(str) + "->" + receiver.astype(str),
            None,
        )
    return out
