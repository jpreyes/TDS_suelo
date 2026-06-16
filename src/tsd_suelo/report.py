from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graph import mode_columns
from .mask import GeoMask, load_chile_mask
from .utils import ensure_dir


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_results_report(output_dir: Path, mask_geojson: Path | None = None, top_n: int = 50) -> dict[str, Any]:
    ensure_dir(output_dir)
    geo = _read_optional_parquet(output_dir / "geo_targets_observed.parquet")
    modes = _read_optional_parquet(output_dir / "latent_modes.parquet")
    kozyrev = _read_optional_parquet(output_dir / "kozyrev_graph_fields.parquet")
    route_graph = _read_optional_parquet(output_dir / "route_graph_observed.parquet")
    attribution = _read_optional_csv(output_dir / "target_level_attribution.csv")
    mask = load_chile_mask(mask_geojson if mask_geojson else _maybe_existing_mask(output_dir))

    geo_modes = _merge_modes(geo, modes)
    geo_modes = _add_mode_norm(geo_modes)
    kozyrev_top = _top_kozyrev(kozyrev, top_n)
    receiver_top = _top_receivers(geo_modes, top_n)
    route_top = _top_routes(route_graph, top_n)

    kozyrev_top.to_csv(output_dir / "top_kozyrev_anomalies.csv", index=False)
    receiver_top.to_csv(output_dir / "top_receiver_anomalies.csv", index=False)
    route_top.to_csv(output_dir / "top_route_anomalies.csv", index=False)

    summary = {
        "geo_targets": int(geo.shape[0]),
        "h5_records": int((geo.get("observed_source") == "h5").sum()) if "observed_source" in geo else 0,
        "flatfile_records": int((geo.get("observed_source") == "flatfile").sum()) if "observed_source" in geo else 0,
        "receivers": int(geo["station_id"].nunique()) if "station_id" in geo else 0,
        "events": int(geo["event_id"].nunique()) if "event_id" in geo else 0,
        "route_edges": int(route_graph.shape[0]),
        "kozyrev_nodes": int(kozyrev.shape[0]),
        "mask_name": mask.name,
        "receiver_in_chile_mask": int(geo.get("receiver_in_chile_mask", pd.Series(dtype=bool)).fillna(False).sum()),
        "route_in_chile_mask": int(geo.get("route_in_chile_mask", pd.Series(dtype=bool)).fillna(False).sum()),
    }

    html_text = _render_html(
        summary=summary,
        attribution=attribution.head(top_n),
        kozyrev_top=kozyrev_top,
        receiver_top=receiver_top,
        route_top=route_top,
        geo_modes=geo_modes,
        mask=mask,
    )
    (output_dir / "results_report.html").write_text(html_text, encoding="utf-8")
    (output_dir / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(output_dir: Path, top_n: int = 10) -> str:
    manifest_path = output_dir / "pipeline_manifest.json"
    summary_path = output_dir / "results_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    kozyrev = _read_optional_csv(output_dir / "top_kozyrev_anomalies.csv").head(top_n)
    receivers = _read_optional_csv(output_dir / "top_receiver_anomalies.csv").head(top_n)

    lines = ["Resumen TSD-Suelo", ""]
    if manifest.get("rows"):
        for key, value in manifest["rows"].items():
            lines.append(f"{key}: {value}")
    if summary:
        lines.extend(
            [
                "",
                f"H5 usados: {summary.get('h5_records', 0)}",
                f"Flatfile-only usados: {summary.get('flatfile_records', 0)}",
                f"Receptores dentro mascara Chile: {summary.get('receiver_in_chile_mask', 0)}",
                f"Rutas dentro mascara Chile: {summary.get('route_in_chile_mask', 0)}",
            ]
        )
    if not kozyrev.empty:
        lines.extend(["", "Top Kozyrev:"])
        for row in kozyrev.itertuples(index=False):
            lines.append(f"- {getattr(row, 'node_id', '')}: delta={getattr(row, 'kozyrev_delta_norm', np.nan):.3f}, n={getattr(row, 'n_records', 0)}")
    if not receivers.empty:
        lines.extend(["", "Top receptores:"])
        for row in receivers.itertuples(index=False):
            lines.append(f"- {getattr(row, 'station_id', '')}: mode_norm={getattr(row, 'mode_norm', np.nan):.3f}, n={getattr(row, 'n_records', 0)}")
    lines.extend(["", f"HTML: {output_dir / 'results_report.html'}"])
    return "\n".join(lines)


def _maybe_existing_mask(output_dir: Path) -> Path | None:
    path = output_dir / "chile_mask.geojson"
    return path if path.exists() else None


def _merge_modes(geo: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    if geo.empty or modes.empty:
        return geo.copy()
    keep = ["record_observed_id"] + mode_columns(modes)
    return geo.merge(modes[keep], on="record_observed_id", how="left")


def _add_mode_norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = mode_columns(out)
    out["mode_norm"] = np.sqrt(np.square(out[cols].fillna(0.0)).sum(axis=1)) if cols else np.nan
    return out


def _top_kozyrev(kozyrev: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if kozyrev.empty or "kozyrev_delta_norm" not in kozyrev.columns:
        return pd.DataFrame()
    cols = [
        "node_type",
        "level",
        "node_id",
        "parent_node_id",
        "n_records",
        "centroid_latitude_deg",
        "centroid_longitude_deg",
        "kozyrev_delta_norm",
        "mode_norm",
    ]
    return kozyrev.sort_values("kozyrev_delta_norm", ascending=False)[[c for c in cols if c in kozyrev.columns]].head(top_n)


def _top_receivers(geo_modes: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if geo_modes.empty:
        return pd.DataFrame()
    grouped = (
        geo_modes.groupby("station_id", dropna=False)
        .agg(
            n_records=("record_observed_id", "count"),
            station_latitude_deg=("station_latitude_deg", "first"),
            station_longitude_deg=("station_longitude_deg", "first"),
            receiver_in_chile_mask=("receiver_in_chile_mask", "first") if "receiver_in_chile_mask" in geo_modes.columns else ("record_observed_id", "size"),
            vs30_m_s=("vs30_m_s", "first") if "vs30_m_s" in geo_modes.columns else ("record_observed_id", "size"),
            f0_hvsr_hz=("f0_hvsr_hz", "first") if "f0_hvsr_hz" in geo_modes.columns else ("record_observed_id", "size"),
            pga_h_g=("pga_h_g", "mean") if "pga_h_g" in geo_modes.columns else ("record_observed_id", "size"),
            mode_norm=("mode_norm", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values("mode_norm", ascending=False).head(top_n)


def _top_routes(route_graph: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if route_graph.empty:
        return pd.DataFrame()
    score = "mode_anomaly_score" if "mode_anomaly_score" in route_graph.columns else "n_records"
    cols = ["level", "edge_type", "from_node", "to_node", "n_records", "distance_km", "backazimuth_deg", score]
    return route_graph.sort_values(score, ascending=False)[[c for c in cols if c in route_graph.columns]].head(top_n)


def _render_html(
    summary: dict[str, Any],
    attribution: pd.DataFrame,
    kozyrev_top: pd.DataFrame,
    receiver_top: pd.DataFrame,
    route_top: pd.DataFrame,
    geo_modes: pd.DataFrame,
    mask: GeoMask,
) -> str:
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>TSD-Suelo Results</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1d252c; }}
h1, h2 {{ margin: 0.8rem 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin: 18px 0; }}
.metric {{ border: 1px solid #d5dde5; border-radius: 6px; padding: 10px; background: #f8fafc; }}
.metric strong {{ display: block; font-size: 1.4rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 0.88rem; }}
th, td {{ border: 1px solid #d9e0e7; padding: 6px 8px; text-align: left; }}
th {{ background: #edf2f7; }}
svg {{ width: 100%; max-width: 920px; height: 760px; border: 1px solid #d5dde5; background: #f9fbfd; }}
.note {{ color: #53606d; }}
</style>
</head>
<body>
<h1>TSD-Suelo Results</h1>
<p class="note">Mascara: {html.escape(str(summary.get("mask_name", "")))}. Reporte autonomo generado desde parquets locales.</p>
{_summary_grid(summary)}
<h2>Mapa Estatico</h2>
{_svg_map(geo_modes, receiver_top, mask)}
<h2>Top Kozyrev</h2>
{_table_html(kozyrev_top)}
<h2>Top Receptores</h2>
{_table_html(receiver_top)}
<h2>Top Rutas</h2>
{_table_html(route_top)}
<h2>Atribucion Por Target</h2>
{_table_html(attribution)}
</body>
</html>"""


def _summary_grid(summary: dict[str, Any]) -> str:
    labels = {
        "geo_targets": "Geo targets",
        "h5_records": "H5",
        "flatfile_records": "Flatfile-only",
        "receivers": "Receptores",
        "events": "Eventos",
        "route_edges": "Aristas ruta",
        "kozyrev_nodes": "Nodos Kozyrev",
        "receiver_in_chile_mask": "Receptores en mascara",
    }
    cards = []
    for key, label in labels.items():
        cards.append(f"<div class='metric'><span>{html.escape(label)}</span><strong>{summary.get(key, 0)}</strong></div>")
    return "<div class='grid'>" + "".join(cards) + "</div>"


def _table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='note'>Sin datos.</p>"
    return df.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4g}")


def _svg_map(geo_modes: pd.DataFrame, receiver_top: pd.DataFrame, mask: GeoMask) -> str:
    min_lon, min_lat, max_lon, max_lat = mask.bounds
    pad_lon = (max_lon - min_lon) * 0.08
    pad_lat = (max_lat - min_lat) * 0.04
    min_lon -= pad_lon
    max_lon += pad_lon
    min_lat -= pad_lat
    max_lat += pad_lat
    width, height = 760, 760

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = (lon - min_lon) / (max_lon - min_lon) * width
        y = height - (lat - min_lat) / (max_lat - min_lat) * height
        return x, y

    mask_paths = []
    for polygon in mask.polygons:
        coords = [project(lon, lat) for lon, lat in polygon]
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in coords) + " Z"
        mask_paths.append(f"<path d='{d}' fill='#e6f2ed' stroke='#315f50' stroke-width='1.5'/>")

    route_lines = []
    if not geo_modes.empty:
        if "route_in_chile_mask" in geo_modes.columns:
            sample = geo_modes[geo_modes["route_in_chile_mask"].fillna(True)]
        else:
            sample = geo_modes
        sample = sample.dropna(
            subset=["event_longitude_deg", "event_latitude_deg", "station_longitude_deg", "station_latitude_deg"]
        )
        if len(sample) > 500:
            sample = sample.sample(500, random_state=7)
        for row in sample.itertuples(index=False):
            x1, y1 = project(getattr(row, "event_longitude_deg"), getattr(row, "event_latitude_deg"))
            x2, y2 = project(getattr(row, "station_longitude_deg"), getattr(row, "station_latitude_deg"))
            route_lines.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='#8aa0b5' stroke-opacity='0.22' stroke-width='0.8'/>")

    receiver_points = []
    if not receiver_top.empty:
        max_score = float(receiver_top["mode_norm"].max()) if "mode_norm" in receiver_top else 1.0
        max_score = max(max_score, 1e-9)
        for row in receiver_top.itertuples(index=False):
            lon = getattr(row, "station_longitude_deg")
            lat = getattr(row, "station_latitude_deg")
            if not np.isfinite(lon) or not np.isfinite(lat):
                continue
            x, y = project(lon, lat)
            score = getattr(row, "mode_norm", 0.0)
            r = 3.5 + 8.0 * min(float(score) / max_score, 1.0)
            label = html.escape(str(getattr(row, "station_id", "")))
            receiver_points.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' fill='#be4b38' fill-opacity='0.75'><title>{label}</title></circle>")

    return f"<svg viewBox='0 0 {width} {height}' role='img'>{''.join(mask_paths)}{''.join(route_lines)}{''.join(receiver_points)}</svg>"
