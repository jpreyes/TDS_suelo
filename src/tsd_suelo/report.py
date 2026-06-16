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
    ultrametric_nodes = _read_optional_parquet(output_dir / "kozyrev_ultrametric_nodes.parquet")
    ultrametric_edges = _read_optional_parquet(output_dir / "kozyrev_ultrametric_edges.parquet")
    faults = _read_optional_parquet(output_dir / "fault_candidates.parquet")
    compatible = _read_optional_parquet(output_dir / "compatible_dynamics.parquet")
    profiles = _read_optional_parquet(output_dir / "forward_conditioning_profiles.parquet")
    attribution = _read_optional_csv(output_dir / "target_level_attribution.csv")
    mask = load_chile_mask(mask_geojson if mask_geojson else _maybe_existing_mask(output_dir))

    geo_modes = _merge_modes(geo, modes)
    geo_modes = _add_mode_norm(geo_modes)
    kozyrev_top = _top_kozyrev(kozyrev, top_n)
    receiver_top = _top_receivers(geo_modes, top_n)
    route_top = _top_routes(route_graph, top_n)
    fault_top = _top_faults(faults, top_n)
    ultrametric_node_top = _top_ultrametric_nodes(ultrametric_nodes, top_n)
    ultrametric_edge_top = _top_ultrametric_edges(ultrametric_edges, top_n)

    kozyrev_top.to_csv(output_dir / "top_kozyrev_anomalies.csv", index=False)
    receiver_top.to_csv(output_dir / "top_receiver_anomalies.csv", index=False)
    route_top.to_csv(output_dir / "top_route_anomalies.csv", index=False)
    fault_top.to_csv(output_dir / "top_fault_candidates.csv", index=False)

    summary = {
        "geo_targets": int(geo.shape[0]),
        "h5_records": int((geo.get("observed_source") == "h5").sum()) if "observed_source" in geo else 0,
        "flatfile_records": int((geo.get("observed_source") == "flatfile").sum()) if "observed_source" in geo else 0,
        "receivers": int(geo["station_id"].nunique()) if "station_id" in geo else 0,
        "events": int(geo["event_id"].nunique()) if "event_id" in geo else 0,
        "route_edges": int(route_graph.shape[0]),
        "kozyrev_nodes": int(kozyrev.shape[0]),
        "ultrametric_nodes": int(ultrametric_nodes.shape[0]),
        "ultrametric_edges": int(ultrametric_edges.shape[0]),
        "fault_candidates": int(faults.shape[0]),
        "compatible_dynamics": int(compatible.shape[0]),
        "forward_profiles": int(profiles.shape[0]),
        "mask_name": mask.name,
        "receiver_in_chile_mask": int(geo.get("receiver_in_chile_mask", pd.Series(dtype=bool)).fillna(False).sum()),
        "route_in_chile_mask": int(geo.get("route_in_chile_mask", pd.Series(dtype=bool)).fillna(False).sum()),
    }

    html_text = _render_html(
        summary=summary,
        attribution=attribution.head(top_n),
        kozyrev_top=kozyrev_top,
        ultrametric_node_top=ultrametric_node_top,
        ultrametric_edge_top=ultrametric_edge_top,
        fault_top=fault_top,
        receiver_top=receiver_top,
        route_top=route_top,
        geo_modes=geo_modes,
        ultrametric_nodes=ultrametric_nodes,
        ultrametric_edges=ultrametric_edges,
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
    faults = _read_optional_csv(output_dir / "top_fault_candidates.csv").head(top_n)
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
                f"Nodos ultrametricos Kozyrev: {summary.get('ultrametric_nodes', 0)}",
                f"Aristas ultrametricas Kozyrev: {summary.get('ultrametric_edges', 0)}",
            ]
        )
    if not kozyrev.empty:
        lines.extend(["", "Top Kozyrev:"])
        for row in kozyrev.itertuples(index=False):
            lines.append(f"- {getattr(row, 'node_id', '')}: delta={getattr(row, 'kozyrev_delta_norm', np.nan):.3f}, n={getattr(row, 'n_records', 0)}")
    if not faults.empty:
        lines.extend(["", "Top candidatos de falla:"])
        for row in faults.itertuples(index=False):
            lines.append(
                "- "
                f"{getattr(row, 'candidate_id', '')}: "
                f"score={getattr(row, 'fault_candidate_score', np.nan):.3f}, "
                f"prob={getattr(row, 'fault_probability_pct', np.nan):.1f}%, "
                f"strike={getattr(row, 'strike_deg', np.nan):.1f}, "
                f"n={getattr(row, 'n_records', 0)}"
            )
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


def _top_faults(faults: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if faults.empty:
        return pd.DataFrame()
    score = "fault_candidate_score" if "fault_candidate_score" in faults.columns else "n_records"
    cols = [
        "candidate_id",
        "priority_rank",
        "confidence",
        "route_id",
        "n_records",
        "strike_deg",
        "midpoint_latitude_deg",
        "midpoint_longitude_deg",
        "mode_anomaly_p90",
        "kozyrev_delta_norm",
        "pga_h_g_mean",
        "fault_probability_pct",
        score,
        "interpretation",
    ]
    return faults.sort_values(score, ascending=False)[[c for c in cols if c in faults.columns]].head(top_n)


def _top_ultrametric_nodes(nodes: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if nodes.empty or "failure_probability_pct" not in nodes.columns:
        return pd.DataFrame()
    cols = [
        "node_type",
        "level",
        "node_id",
        "parent_node_id",
        "n_records",
        "centroid_latitude_deg",
        "centroid_longitude_deg",
        "failure_probability_pct",
        "delta_probability_pct",
        "mode_probability_pct",
        "support_probability_pct",
    ]
    return nodes.sort_values("failure_probability_pct", ascending=False)[[c for c in cols if c in nodes.columns]].head(top_n)


def _top_ultrametric_edges(edges: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if edges.empty or "edge_probability_pct" not in edges.columns:
        return pd.DataFrame()
    cols = [
        "edge_family",
        "edge_type",
        "from_level",
        "to_level",
        "from_node",
        "to_node",
        "n_records",
        "edge_probability_pct",
        "from_failure_probability_pct",
        "to_failure_probability_pct",
    ]
    return edges.sort_values("edge_probability_pct", ascending=False)[[c for c in cols if c in edges.columns]].head(top_n)


def _render_html(
    summary: dict[str, Any],
    attribution: pd.DataFrame,
    kozyrev_top: pd.DataFrame,
    ultrametric_node_top: pd.DataFrame,
    ultrametric_edge_top: pd.DataFrame,
    fault_top: pd.DataFrame,
    receiver_top: pd.DataFrame,
    route_top: pd.DataFrame,
    geo_modes: pd.DataFrame,
    ultrametric_nodes: pd.DataFrame,
    ultrametric_edges: pd.DataFrame,
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
<h2>Mapa De Calor Kozyrev</h2>
<p class="note">Color por probabilidad empirica observada compatible. Azul bajo, amarillo medio, rojo alto. Los GeoJSON/parquets contienen todos los nodos y aristas.</p>
{_svg_probability_map(ultrametric_nodes, ultrametric_edges, mask)}
<h2>Candidatos De Falla</h2>
<p class="note">Lineamientos observados por concentracion de modos residuales y saltos Kozyrev. No son nombres oficiales de fallas.</p>
{_table_html(fault_top)}
<h2>Nodos Ultrametricos Kozyrev</h2>
{_table_html(ultrametric_node_top)}
<h2>Aristas Ultrametricas Kozyrev</h2>
{_table_html(ultrametric_edge_top)}
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
        "ultrametric_nodes": "Nodos ultra",
        "ultrametric_edges": "Aristas ultra",
        "fault_candidates": "Candidatos falla",
        "compatible_dynamics": "Dinamica compatible",
        "forward_profiles": "Perfiles forward",
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


def _probability_color(probability_pct: float) -> str:
    p = min(max(float(probability_pct) if np.isfinite(probability_pct) else 0.0, 0.0), 100.0) / 100.0
    if p <= 0.5:
        t = p / 0.5
        start = np.array([44, 123, 182], dtype=float)
        end = np.array([255, 255, 191], dtype=float)
    else:
        t = (p - 0.5) / 0.5
        start = np.array([255, 255, 191], dtype=float)
        end = np.array([215, 25, 28], dtype=float)
    rgb = np.round(start + t * (end - start)).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _svg_probability_map(nodes: pd.DataFrame, edges: pd.DataFrame, mask: GeoMask) -> str:
    if nodes.empty:
        return "<p class='note'>Sin grafo ultrametrico probabilistico. Regenera el build.</p>"
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
        mask_paths.append(f"<path d='{d}' fill='#eef4ef' stroke='#315f50' stroke-width='1.3'/>")

    levels = pd.to_numeric(nodes.get("level"), errors="coerce") if "level" in nodes else pd.Series(dtype=float)
    max_level = int(levels.max()) if levels.notna().any() else 4
    node_type = nodes.get("node_type", pd.Series("", index=nodes.index)).astype(str)
    route_nodes = nodes[(node_type == "route") & (pd.to_numeric(nodes.get("level"), errors="coerce") == max_level)].copy()
    route_lines = []
    required = ["line_start_longitude_deg", "line_start_latitude_deg", "line_end_longitude_deg", "line_end_latitude_deg"]
    if set(required).issubset(route_nodes.columns):
        for row in route_nodes.itertuples(index=False):
            values = [getattr(row, column) for column in required]
            if not all(np.isfinite(values)):
                continue
            p = float(getattr(row, "failure_probability_pct", 0.0))
            x1, y1 = project(values[0], values[1])
            x2, y2 = project(values[2], values[3])
            color = _probability_color(p)
            opacity = 0.10 + 0.70 * min(max(p, 0.0), 100.0) / 100.0
            width_px = 0.45 + 2.4 * min(max(p, 0.0), 100.0) / 100.0
            title = html.escape(f"{getattr(row, 'node_id', '')} | {p:.1f}%")
            route_lines.append(
                f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                f"stroke='{color}' stroke-opacity='{opacity:.3f}' stroke-width='{width_px:.2f}'><title>{title}</title></line>"
            )

    point_nodes = nodes[
        node_type.isin(["source3d", "receiver"])
        & (pd.to_numeric(nodes.get("level"), errors="coerce") == max_level)
    ].copy()
    points = []
    for row in point_nodes.itertuples(index=False):
        lon = getattr(row, "centroid_longitude_deg", np.nan)
        lat = getattr(row, "centroid_latitude_deg", np.nan)
        if not np.isfinite(lon) or not np.isfinite(lat):
            continue
        p = float(getattr(row, "failure_probability_pct", 0.0))
        x, y = project(lon, lat)
        color = _probability_color(p)
        radius = 1.8 + 5.5 * min(max(p, 0.0), 100.0) / 100.0
        title = html.escape(f"{getattr(row, 'node_id', '')} | {p:.1f}%")
        points.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' fill='{color}' fill-opacity='0.75'><title>{title}</title></circle>")

    legend = (
        "<g transform='translate(24,24)'>"
        "<rect x='0' y='0' width='214' height='54' fill='white' fill-opacity='0.86' stroke='#cfd8df'/>"
        "<text x='10' y='18' font-size='12' fill='#1d252c'>Probabilidad Kozyrev (%)</text>"
        "<rect x='10' y='28' width='55' height='12' fill='rgb(44,123,182)'/>"
        "<rect x='65' y='28' width='55' height='12' fill='rgb(255,255,191)'/>"
        "<rect x='120' y='28' width='55' height='12' fill='rgb(215,25,28)'/>"
        "<text x='10' y='50' font-size='10'>0</text><text x='89' y='50' font-size='10'>50</text><text x='158' y='50' font-size='10'>100</text>"
        "</g>"
    )
    return f"<svg viewBox='0 0 {width} {height}' role='img'>{''.join(mask_paths)}{''.join(route_lines)}{''.join(points)}{legend}</svg>"
