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


def _read_optional_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_results_report(output_dir: Path, mask_geojson: Path | None = None, top_n: int = 50) -> dict[str, Any]:
    ensure_dir(output_dir)
    geo = _read_optional_parquet(output_dir / "geo_targets_observed.parquet")
    modes = _read_optional_parquet(output_dir / "latent_modes.parquet")
    kozyrev = _read_optional_parquet(output_dir / "kozyrev_graph_fields.parquet")
    route_graph = _read_optional_parquet(output_dir / "route_graph_observed.parquet")
    spatial_nodes = _read_optional_parquet(output_dir / "spatial_grid_nodes.parquet")
    spatial_edges = _read_optional_parquet(output_dir / "spatial_grid_edges.parquet")
    spectral_nodes = _read_optional_parquet(output_dir / "spectral_node_dynamics.parquet")
    spectral_edges = _read_optional_parquet(output_dir / "spectral_edge_transmissibility.parquet")
    spectral_modes = _read_optional_parquet(output_dir / "spectral_dynamic_modes.parquet")
    ultrametric_nodes = _read_optional_parquet(output_dir / "kozyrev_ultrametric_nodes.parquet")
    ultrametric_edges = _read_optional_parquet(output_dir / "kozyrev_ultrametric_edges.parquet")
    faults = _read_optional_parquet(output_dir / "fault_candidates.parquet")
    compatible = _read_optional_parquet(output_dir / "compatible_dynamics.parquet")
    profiles = _read_optional_parquet(output_dir / "forward_conditioning_profiles.parquet")
    scenario_result = _read_optional_parquet(output_dir / "forward_scenario_result.parquet")
    scenario_analogs = _read_optional_csv(output_dir / "forward_scenario_analogs.csv")
    scenario_faults = _read_optional_csv(output_dir / "forward_scenario_faults.csv")
    scenario_input = _read_optional_json(output_dir / "forward_scenario_input.json")
    attribution = _read_optional_csv(output_dir / "target_level_attribution.csv")
    mask = load_chile_mask(mask_geojson if mask_geojson else _maybe_existing_mask(output_dir))

    geo_modes = _merge_modes(geo, modes)
    geo_modes = _add_mode_norm(geo_modes)
    kozyrev_top = _top_kozyrev(kozyrev, top_n)
    receiver_top = _top_receivers(geo_modes, top_n)
    route_top = _top_routes(route_graph, top_n)
    spatial_node_top = _top_spatial_nodes(spatial_nodes, top_n)
    spatial_edge_top = _top_spatial_edges(spatial_edges, top_n)
    spectral_node_top = _top_spectral_nodes(spectral_nodes, top_n)
    spectral_edge_top = _top_spectral_edges(spectral_edges, top_n)
    fault_top = _top_faults(faults, top_n)
    ultrametric_node_top = _top_ultrametric_nodes(ultrametric_nodes, top_n)
    ultrametric_edge_top = _top_ultrametric_edges(ultrametric_edges, top_n)
    forward_top = _top_compatible_dynamics(compatible, top_n)
    forward_profile_top = _top_forward_profiles(profiles, top_n)
    scenario_target_top = _top_scenario_targets(scenario_result, top_n)
    scenario_analog_top = scenario_analogs.head(top_n)
    scenario_fault_top = scenario_faults.head(top_n)

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
        "spatial_grid_nodes": int(spatial_nodes.shape[0]),
        "spatial_grid_edges": int(spatial_edges.shape[0]),
        "spectral_node_dynamics": int(spectral_nodes.shape[0]),
        "spectral_edge_transmissibility": int(spectral_edges.shape[0]),
        "spectral_dynamic_modes": int(spectral_modes.shape[0]),
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
        output_dir=output_dir,
        summary=summary,
        attribution=attribution.head(top_n),
        kozyrev_top=kozyrev_top,
        spatial_node_top=spatial_node_top,
        spatial_edge_top=spatial_edge_top,
        spectral_node_top=spectral_node_top,
        spectral_edge_top=spectral_edge_top,
        ultrametric_node_top=ultrametric_node_top,
        ultrametric_edge_top=ultrametric_edge_top,
        forward_top=forward_top,
        forward_profile_top=forward_profile_top,
        scenario_input=scenario_input,
        scenario_target_top=scenario_target_top,
        scenario_analog_top=scenario_analog_top,
        scenario_fault_top=scenario_fault_top,
        fault_top=fault_top,
        receiver_top=receiver_top,
        route_top=route_top,
        geo_modes=geo_modes,
        spatial_nodes=spatial_nodes,
        spatial_edges=spatial_edges,
        spectral_nodes=spectral_nodes,
        spectral_edges=spectral_edges,
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
                f"Nodos grilla espacial: {summary.get('spatial_grid_nodes', 0)}",
                f"Aristas grilla espacial: {summary.get('spatial_grid_edges', 0)}",
                f"Nodos dinamica espectral: {summary.get('spectral_node_dynamics', 0)}",
                f"Aristas dinamica espectral: {summary.get('spectral_edge_transmissibility', 0)}",
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


def _top_spatial_nodes(nodes: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if nodes.empty or "anomaly_probability_pct" not in nodes.columns:
        return pd.DataFrame()
    cols = [
        "level",
        "cell_id",
        "n_records",
        "n_stations",
        "n_events",
        "center_latitude_deg",
        "center_longitude_deg",
        "sample_role",
        "mode_norm",
        "anomaly_probability_pct",
        "mode_probability_pct",
        "intensity_probability_pct",
        "support_probability_pct",
    ]
    return nodes.sort_values("anomaly_probability_pct", ascending=False)[[c for c in cols if c in nodes.columns]].head(top_n)


def _top_spatial_edges(edges: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if edges.empty or "fault_probability_pct" not in edges.columns:
        return pd.DataFrame()
    cols = [
        "level",
        "from_cell_id",
        "to_cell_id",
        "neighbor_orientation",
        "neighbor_kind",
        "min_n_records",
        "mode_jump_norm",
        "anomaly_probability_jump_pct",
        "fault_probability_pct",
        "mode_jump_probability_pct",
        "anomaly_jump_probability_pct",
        "support_probability_pct",
    ]
    return edges.sort_values("fault_probability_pct", ascending=False)[[c for c in cols if c in edges.columns]].head(top_n)


def _top_spectral_nodes(nodes: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if nodes.empty or "spectral_dynamic_probability_pct" not in nodes.columns:
        return pd.DataFrame()
    cols = [
        "level",
        "cell_id",
        "n_records",
        "n_stations",
        "n_events",
        "center_latitude_deg",
        "center_longitude_deg",
        "spectral_dynamic_probability_pct",
        "spectral_shape_probability_pct",
        "spectral_energy_probability_pct",
        "support_probability_pct",
    ]
    return nodes.sort_values("spectral_dynamic_probability_pct", ascending=False)[[c for c in cols if c in nodes.columns]].head(top_n)


def _top_spectral_edges(edges: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if edges.empty or "spectral_transfer_probability_pct" not in edges.columns:
        return pd.DataFrame()
    cols = [
        "level",
        "from_cell_id",
        "to_cell_id",
        "neighbor_orientation",
        "neighbor_kind",
        "min_n_records",
        "spectral_jump_norm",
        "spectral_transfer_log_mean",
        "spectral_transfer_log_std",
        "spectral_transfer_probability_pct",
        "spectral_jump_probability_pct",
        "support_probability_pct",
    ]
    return edges.sort_values("spectral_transfer_probability_pct", ascending=False)[[c for c in cols if c in edges.columns]].head(top_n)


def _top_compatible_dynamics(compatible: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if compatible.empty or "dynamic_anomaly_score" not in compatible.columns:
        return pd.DataFrame()
    cols = [
        "record_observed_id",
        "observed_source",
        "event_id",
        "station_id",
        "route_id",
        "distance_km",
        "mw",
        "dynamic_anomaly_score",
        "forward_support_weight",
        "fault_probability_pct",
        "fault_candidate_id",
        "compatible_dynamics_status",
        "target_coverage_fraction",
    ]
    return compatible.sort_values("dynamic_anomaly_score", ascending=False)[[c for c in cols if c in compatible.columns]].head(top_n)


def _top_forward_profiles(profiles: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame()
    sort_col = "dynamic_anomaly_score_p90" if "dynamic_anomaly_score_p90" in profiles.columns else "n_records"
    cols = [
        "context_type",
        "level",
        "context_id",
        "n_records",
        "centroid_latitude_deg",
        "centroid_longitude_deg",
        "dynamic_anomaly_score_mean",
        "dynamic_anomaly_score_p90",
        "forward_support_weight_mean",
        "forward_support_weight_p90",
        "fault_candidate_score_mean",
        "fault_candidate_score_p90",
    ]
    return profiles.sort_values(sort_col, ascending=False)[[c for c in cols if c in profiles.columns]].head(top_n)


def _top_scenario_targets(result: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if result.empty:
        return pd.DataFrame()
    cols = [
        "scenario_name",
        "target",
        "forward_p16",
        "forward_p50",
        "forward_p84",
        "forward_weighted_mean",
        "n_analogs",
        "mw",
        "vs30_m_s",
        "source_distance_km",
        "source_bearing_from_receiver_deg",
        "nearest_fault_candidate_id",
        "nearest_fault_probability_pct",
        "method",
    ]
    return result[[c for c in cols if c in result.columns]].head(top_n)


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
    output_dir: Path,
    summary: dict[str, Any],
    attribution: pd.DataFrame,
    kozyrev_top: pd.DataFrame,
    spatial_node_top: pd.DataFrame,
    spatial_edge_top: pd.DataFrame,
    spectral_node_top: pd.DataFrame,
    spectral_edge_top: pd.DataFrame,
    ultrametric_node_top: pd.DataFrame,
    ultrametric_edge_top: pd.DataFrame,
    forward_top: pd.DataFrame,
    forward_profile_top: pd.DataFrame,
    scenario_input: dict[str, Any],
    scenario_target_top: pd.DataFrame,
    scenario_analog_top: pd.DataFrame,
    scenario_fault_top: pd.DataFrame,
    fault_top: pd.DataFrame,
    receiver_top: pd.DataFrame,
    route_top: pd.DataFrame,
    geo_modes: pd.DataFrame,
    spatial_nodes: pd.DataFrame,
    spatial_edges: pd.DataFrame,
    spectral_nodes: pd.DataFrame,
    spectral_edges: pd.DataFrame,
    ultrametric_nodes: pd.DataFrame,
    ultrametric_edges: pd.DataFrame,
    mask: GeoMask,
) -> str:
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TSD-Suelo Results</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root {{
  --ink: #17212b;
  --muted: #617180;
  --line: #d7e0e7;
  --panel: #ffffff;
  --soft: #f4f7fa;
  --brand: #205c6b;
  --brand-strong: #163f4b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background: #eef3f6;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
}}
a {{ color: var(--brand); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.report-shell {{ max-width: 1280px; margin: 0 auto; padding: 22px; }}
.hero {{
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 24px;
  align-items: end;
  padding: 26px;
  color: #ffffff;
  background: #102832;
  border-bottom: 4px solid #2f7d8c;
}}
.eyebrow {{ margin: 0 0 8px; color: #a8d6df; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }}
h1 {{ margin: 0; font-size: clamp(1.75rem, 2.4vw, 2.45rem); }}
.hero .note {{ color: #d4e8ed; max-width: 860px; }}
.hero-actions {{ display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; }}
.button-link {{ display: inline-flex; align-items: center; min-height: 36px; padding: 8px 11px; border: 1px solid #7bb8c4; border-radius: 6px; color: #ffffff; background: rgba(255,255,255,0.08); font-weight: 650; }}
.button-link:hover {{ text-decoration: none; background: rgba(255,255,255,0.14); }}
.subnav {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }}
.subnav a {{ padding: 7px 10px; border: 1px solid var(--line); border-radius: 999px; background: #ffffff; color: var(--brand-strong); font-size: 0.9rem; }}
h2 {{ margin: 0; font-size: 1.2rem; }}
.section-head {{ display: flex; justify-content: space-between; gap: 14px; align-items: baseline; margin-bottom: 12px; }}
.panel {{ margin: 18px 0; padding: 18px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); box-shadow: 0 1px 2px rgba(16,40,50,0.05); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 12px; margin: 18px 0; }}
.metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 13px; background: var(--panel); box-shadow: 0 1px 2px rgba(16,40,50,0.04); }}
.metric span {{ display: block; color: var(--muted); font-size: 0.82rem; }}
.metric strong {{ display: block; margin-top: 6px; font-size: 1.42rem; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #ffffff; }}
table {{ border-collapse: collapse; width: 100%; margin: 0; font-size: 0.84rem; }}
th, td {{ border-bottom: 1px solid #e3e9ee; padding: 7px 9px; text-align: left; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #edf3f6; color: #263845; z-index: 1; }}
tr:nth-child(even) td {{ background: #fafcfd; }}
svg {{ width: 100%; max-width: 100%; height: 760px; border: 1px solid var(--line); background: #f9fbfd; border-radius: 8px; }}
.map-shell {{ max-width: 100%; margin: 14px 0 0; border: 1px solid var(--line); background: #f8fafc; border-radius: 8px; overflow: hidden; }}
#interactive-map {{ width: 100%; height: min(78vh, 760px); min-height: 520px; }}
.map-controls {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 10px 12px; border-top: 1px solid #d9e0e7; background: #ffffff; }}
.map-controls label {{ white-space: nowrap; font-size: 0.9rem; }}
.map-status {{ padding: 8px 12px; color: #53606d; font-size: 0.86rem; border-top: 1px solid #d9e0e7; background: #ffffff; }}
.leaflet-popup-content table {{ margin: 6px 0 0; font-size: 0.78rem; }}
.leaflet-popup-content th, .leaflet-popup-content td {{ padding: 3px 5px; }}
.downloads {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin: 12px 0 0; padding: 0; list-style: none; }}
.downloads a {{ display: block; padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--soft); color: var(--brand-strong); font-size: 0.88rem; }}
.downloads a:hover {{ text-decoration: none; background: #e7eff3; }}
.note {{ color: #53606d; }}
@media (max-width: 760px) {{
  .hero {{ grid-template-columns: 1fr; padding: 20px; }}
  .hero-actions {{ justify-content: flex-start; }}
  .report-shell {{ padding: 14px; }}
}}
</style>
</head>
<body>
<header class="hero">
  <div>
    <p class="eyebrow">TSD-Suelo observado</p>
    <h1>Atlas dinamico fuente-ruta-receptor</h1>
    <p class="note">Mascara: {html.escape(str(summary.get("mask_name", "")))}. Reporte autonomo generado desde parquets locales observados.</p>
  </div>
  <div class="hero-actions">
    <a class="button-link" href="/admin">Admin</a>
    <a class="button-link" href="results_summary.json" download>Resumen JSON</a>
    <a class="button-link" href="run.log">run.log</a>
  </div>
</header>
<main class="report-shell">
  <nav class="subnav">
    <a href="#mapa">Mapa</a>
    <a href="#forward">Forward</a>
    <a href="#escenario">Escenario</a>
    <a href="#fallas">Fallas</a>
    <a href="#espectral">Espectral</a>
    <a href="#espacial">Espacial</a>
    <a href="#kozyrev">Kozyrev</a>
    <a href="#descargas">Descargas</a>
  </nav>
  {_summary_grid(summary)}

  <section class="panel" id="mapa">
    <div class="section-head">
      <h2>Mapa Interactivo Chile</h2>
      <span class="note">Leaflet + OpenStreetMap, sin API key</span>
    </div>
    <p class="note">Carga GeoJSON locales desde esta misma carpeta. Activa o desactiva capas para inspeccionar fallas, dinamica espectral, anomalias espaciales y atlas.</p>
    {_interactive_leaflet_map(output_dir)}
  </section>

  <section class="panel" id="forward">
    <div class="section-head">
      <h2>Forward Condicionado</h2>
      <span class="note">Dinamica compatible reutilizable</span>
    </div>
    <p class="note">Productos observados para condicionamiento posterior. No son acelerogramas sinteticos; son correcciones y perfiles multiescala listos para un forward condicionado validado.</p>
    <h3>Registros con mayor dinamica compatible</h3>
    {_table_html(forward_top)}
    <h3>Perfiles de condicionamiento</h3>
    {_table_html(forward_profile_top)}
  </section>

  {_scenario_section(scenario_input, scenario_target_top, scenario_analog_top, scenario_fault_top)}

  <section class="panel" id="fallas">
    <div class="section-head">
      <h2>Candidatos De Falla</h2>
      <span class="note">Lineamientos no catalogados oficialmente</span>
    </div>
    {_table_html(fault_top)}
  </section>

  <section class="panel" id="espectral">
    <div class="section-head">
      <h2>Mapa Dinamico Espectral</h2>
      <span class="note">Celdas por respuesta, lineas por transmisibilidad</span>
    </div>
    <p class="note">Red estructural equivalente en frecuencia usando todas las frecuencias de la grilla espectral simultaneamente.</p>
    {_svg_spectral_probability_map(spectral_nodes, spectral_edges, mask)}
    <h3>Nodos dinamicos espectrales</h3>
    {_table_html(spectral_node_top)}
    <h3>Aristas de transmisibilidad espectral</h3>
    {_table_html(spectral_edge_top)}
  </section>

  <section class="panel" id="espacial">
    <div class="section-head">
      <h2>Mapa De Calor Espacial</h2>
      <span class="note">Grilla rectangular de anomalias</span>
    </div>
    <p class="note">Celdas rectangulares por probabilidad de anomalia y aristas vecinas por probabilidad de falla. Azul bajo, amarillo medio, rojo alto.</p>
    {_svg_spatial_probability_map(spatial_nodes, spatial_edges, mask)}
    <h3>Celdas espaciales anomalas</h3>
    {_table_html(spatial_node_top)}
    <h3>Aristas espaciales con salto</h3>
    {_table_html(spatial_edge_top)}
  </section>

  <section class="panel" id="kozyrev">
    <div class="section-head">
      <h2>Kozyrev Y Rutas</h2>
      <span class="note">Referencia fuente-ruta-receptor anterior</span>
    </div>
    {_svg_probability_map(ultrametric_nodes, ultrametric_edges, mask)}
    <h3>Nodos ultrametricos Kozyrev</h3>
    {_table_html(ultrametric_node_top)}
    <h3>Aristas ultrametricas Kozyrev</h3>
    {_table_html(ultrametric_edge_top)}
    <h3>Top Kozyrev</h3>
    {_table_html(kozyrev_top)}
    <h3>Top receptores</h3>
    {_table_html(receiver_top)}
    <h3>Top rutas</h3>
    {_table_html(route_top)}
  </section>

  <section class="panel">
    <div class="section-head">
      <h2>Atribucion Por Target</h2>
      <span class="note">Calidad residual por familia fisica</span>
    </div>
    {_table_html(attribution)}
  </section>

  <section class="panel" id="descargas">
    <div class="section-head">
      <h2>Descargas</h2>
      <span class="note">Productos reproducibles</span>
    </div>
    {_download_links(output_dir)}
  </section>
</main>
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
        "spatial_grid_nodes": "Celdas espaciales",
        "spatial_grid_edges": "Aristas espaciales",
        "spectral_node_dynamics": "Nodos espectrales",
        "spectral_edge_transmissibility": "Aristas espectrales",
        "spectral_dynamic_modes": "Modos espectrales",
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
    table = df.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4g}")
    return f"<div class='table-wrap'>{table}</div>"


def _scenario_section(
    scenario_input: dict[str, Any],
    scenario_target_top: pd.DataFrame,
    scenario_analog_top: pd.DataFrame,
    scenario_fault_top: pd.DataFrame,
) -> str:
    if not scenario_input and scenario_target_top.empty:
        return ""
    summary_items = [
        ("Escenario", scenario_input.get("scenario_name")),
        ("Mw", scenario_input.get("mw")),
        ("Vs30 m/s", scenario_input.get("vs30_m_s")),
        ("Distancia km", scenario_input.get("source_distance_km")),
        ("Direccion", scenario_input.get("source_direction")),
        ("Azimut fuente", scenario_input.get("source_bearing_from_receiver_deg")),
        ("Fuente lat", scenario_input.get("source_latitude_deg")),
        ("Fuente lon", scenario_input.get("source_longitude_deg")),
        ("Analogos", scenario_input.get("n_analogs")),
    ]
    cards = "".join(
        f"<div class='metric'><span>{html.escape(str(label))}</span><strong>{html.escape(str(value if value is not None else '-'))}</strong></div>"
        for label, value in summary_items
    )
    return f"""
  <section class="panel" id="escenario">
    <div class="section-head">
      <h2>Escenario Forward</h2>
      <span class="note">Estimacion condicionada por analogos observados</span>
    </div>
    <div class="grid">{cards}</div>
    <h3>Targets estimados</h3>
    {_table_html(scenario_target_top)}
    <h3>Analogos observados usados</h3>
    {_table_html(scenario_analog_top)}
    <h3>Fallas candidatas cercanas</h3>
    {_table_html(scenario_fault_top)}
  </section>
"""


def _download_links(output_dir: Path) -> str:
    products = [
        ("Reporte HTML", "results_report.html"),
        ("Resumen JSON", "results_summary.json"),
        ("Mapa calor espacial GeoJSON", "spatial_probability_heatmap.geojson"),
        ("Mapa calor espacial KMZ", "spatial_probability_heatmap.kmz"),
        ("Celdas espaciales GeoJSON", "spatial_anomaly_nodes.geojson"),
        ("Aristas espaciales GeoJSON", "spatial_fault_edges.geojson"),
        ("Celdas espaciales parquet", "spatial_grid_nodes.parquet"),
        ("Aristas espaciales parquet", "spatial_grid_edges.parquet"),
        ("Resumen grilla espacial JSON", "spatial_grid_summary.json"),
        ("Mapa dinamico espectral GeoJSON", "spectral_dynamic_heatmap.geojson"),
        ("Mapa dinamico espectral KMZ", "spectral_dynamic_heatmap.kmz"),
        ("Firmas espectrales por registro parquet", "spectral_record_signatures.parquet"),
        ("Nodos dinamicos espectrales parquet", "spectral_node_dynamics.parquet"),
        ("Aristas transmisibilidad espectral parquet", "spectral_edge_transmissibility.parquet"),
        ("Modos dinamicos espectrales parquet", "spectral_dynamic_modes.parquet"),
        ("Componentes modos espectrales CSV", "spectral_mode_components.csv"),
        ("Grilla frecuencias espectral JSON", "spectral_frequency_grid.json"),
        ("Mapa calor Kozyrev GeoJSON", "kozyrev_heatmap.geojson"),
        ("Mapa calor Kozyrev KMZ", "kozyrev_heatmap.kmz"),
        ("Nodos ultrametricos GeoJSON", "kozyrev_ultrametric_nodes.geojson"),
        ("Aristas ultrametricas GeoJSON", "kozyrev_ultrametric_edges.geojson"),
        ("Fallas candidatas GeoJSON", "fault_candidates.geojson"),
        ("Fallas candidatas KMZ", "fault_candidates.kmz"),
        ("Atlas geologico GeoJSON", "atlas_geologico.geojson"),
        ("Atlas geologico KMZ", "atlas_geologico.kmz"),
        ("Top fallas candidatas CSV", "top_fault_candidates.csv"),
        ("Top Kozyrev CSV", "top_kozyrev_anomalies.csv"),
        ("Dinamica compatible parquet", "compatible_dynamics.parquet"),
        ("Perfiles forward parquet", "forward_conditioning_profiles.parquet"),
        ("Manifest forward JSON", "forward_manifest.json"),
        ("Escenario forward JSON", "forward_scenario_input.json"),
        ("Escenario forward CSV", "forward_scenario_result.csv"),
        ("Escenario forward parquet", "forward_scenario_result.parquet"),
        ("Escenario forward GeoJSON", "forward_scenario.geojson"),
        ("Analogos escenario CSV", "forward_scenario_analogs.csv"),
        ("Fallas escenario CSV", "forward_scenario_faults.csv"),
        ("Nodos ultrametricos parquet", "kozyrev_ultrametric_nodes.parquet"),
        ("Aristas ultrametricas parquet", "kozyrev_ultrametric_edges.parquet"),
        ("Log build", "run.log"),
    ]
    links = []
    for label, filename in products:
        if (output_dir / filename).exists():
            links.append(f"<li><a href='{html.escape(filename)}' download>{html.escape(label)}</a></li>")
    if not links:
        return "<p class='note'>Sin archivos de descarga todavia.</p>"
    return "<ul class='downloads'>" + "".join(links) + "</ul>"


def _interactive_leaflet_map(output_dir: Path) -> str:
    layer_defs = [
        {
            "id": "mask",
            "label": "Chile",
            "file": "chile_mask.geojson",
            "kind": "mask",
            "checked": True,
        },
        {
            "id": "spectral",
            "label": "Dinamica espectral",
            "file": "spectral_dynamic_heatmap.geojson",
            "kind": "spectral",
            "checked": True,
        },
        {
            "id": "faults",
            "label": "Fallas candidatas",
            "file": "fault_candidates.geojson",
            "kind": "faults",
            "checked": True,
        },
        {
            "id": "scenario",
            "label": "Escenario forward",
            "file": "forward_scenario.geojson",
            "kind": "scenario",
            "checked": True,
        },
        {
            "id": "spatial",
            "label": "Anomalia espacial",
            "file": "spatial_probability_heatmap.geojson",
            "kind": "spatial",
            "checked": False,
        },
        {
            "id": "kozyrev",
            "label": "Kozyrev",
            "file": "kozyrev_heatmap.geojson",
            "kind": "kozyrev",
            "checked": False,
        },
        {
            "id": "atlas",
            "label": "Atlas geologico",
            "file": "atlas_geologico.geojson",
            "kind": "atlas",
            "checked": False,
        },
    ]
    layers = [layer for layer in layer_defs if (output_dir / layer["file"]).exists()]
    if not layers:
        return "<p class='note'>Sin GeoJSON para mapa interactivo.</p>"

    template = """
<div class="map-shell">
  <div id="interactive-map"></div>
  <div class="map-controls" id="interactive-map-controls"></div>
  <div class="map-status" id="interactive-map-status">Preparando mapa...</div>
</div>
<script>
(function () {
  const layerConfigs = __LAYER_CONFIGS__;
  const status = document.getElementById("interactive-map-status");
  const controls = document.getElementById("interactive-map-controls");
  const container = document.getElementById("interactive-map");
  if (!window.L) {
    status.textContent = "Leaflet no cargo. Revisa conexion a internet o usa los mapas SVG/GeoJSON descargables.";
    return;
  }

  const map = L.map(container, { preferCanvas: true }).setView([-30.5, -71.0], 5);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);

  const layerState = new Map();
  const probabilityFields = [
    "spectral_dynamic_probability_pct",
    "spectral_transfer_probability_pct",
    "fault_probability_pct",
    "anomaly_probability_pct",
    "failure_probability_pct",
    "edge_probability_pct",
    "mode_probability_pct",
    "intensity_probability_pct",
    "support_probability_pct"
  ];

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function asNumber(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
  }

  function probability(props) {
    for (const field of probabilityFields) {
      const value = asNumber(props[field]);
      if (value !== null) {
        return clamp(value, 0, 100);
      }
    }
    return 50;
  }

  function probabilityColor(value) {
    const p = clamp(value, 0, 100) / 100;
    const a = p <= 0.5 ? [44, 123, 182] : [255, 255, 191];
    const b = p <= 0.5 ? [255, 255, 191] : [215, 25, 28];
    const t = p <= 0.5 ? p / 0.5 : (p - 0.5) / 0.5;
    const rgb = a.map((start, i) => Math.round(start + t * (b[i] - start)));
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }

  function styleFeature(feature, config) {
    const props = feature.properties || {};
    const geometryType = feature.geometry ? feature.geometry.type : "";
    if (config.kind === "mask") {
      return {
        color: "#255f49",
        weight: 2,
        opacity: 0.95,
        fillColor: "#9bd3ae",
        fillOpacity: 0.18
      };
    }
    const p = probability(props);
    const color = probabilityColor(p);
    if (geometryType.includes("Polygon")) {
      return {
        color: "#24333d",
        weight: config.kind === "spectral" ? 0.7 : 0.45,
        opacity: 0.55,
        fillColor: color,
        fillOpacity: 0.20 + 0.62 * p / 100
      };
    }
    if (geometryType.includes("LineString")) {
      return {
        color,
        weight: 0.8 + 4.2 * p / 100,
        opacity: 0.22 + 0.74 * p / 100
      };
    }
    return {
      color,
      weight: 1.2,
      fillColor: color,
      fillOpacity: 0.72
    };
  }

  function pointLayer(feature, latlng, config) {
    const p = probability(feature.properties || {});
    return L.circleMarker(latlng, {
      radius: 3 + 7 * p / 100,
      color: "#24333d",
      weight: 0.5,
      fillColor: probabilityColor(p),
      fillOpacity: 0.78
    });
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (char) {
      return {"&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"}[char];
    });
  }

  function popupHtml(feature, config) {
    const props = feature.properties || {};
    const preferred = [
      "feature_type",
      "cell_id",
      "node_id",
      "candidate_id",
      "level",
      "display_level",
      "n_records",
      "n_events",
      "n_stations",
      "spectral_dynamic_probability_pct",
      "spectral_transfer_probability_pct",
      "fault_probability_pct",
      "anomaly_probability_pct",
      "failure_probability_pct",
      "edge_probability_pct",
      "strike_deg",
      "center_latitude_deg",
      "center_longitude_deg",
      "midpoint_latitude_deg",
      "midpoint_longitude_deg"
    ];
    const rows = [];
    const used = new Set();
    for (const key of preferred) {
      if (props[key] !== undefined && props[key] !== null) {
        rows.push([key, props[key]]);
        used.add(key);
      }
    }
    for (const key of Object.keys(props)) {
      if (rows.length >= 14) {
        break;
      }
      if (!used.has(key) && props[key] !== null && props[key] !== undefined) {
        rows.push([key, props[key]]);
      }
    }
    const body = rows.map(([key, value]) => (
      `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(value)}</td></tr>`
    )).join("");
    return `<strong>${escapeHtml(config.label)}</strong><table>${body}</table>`;
  }

  async function loadLayer(config) {
    status.textContent = `Cargando ${config.label}...`;
    const response = await fetch(config.file, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`${config.file}: HTTP ${response.status}`);
    }
    const data = await response.json();
    const layer = L.geoJSON(data, {
      style: function (feature) {
        return styleFeature(feature, config);
      },
      pointToLayer: function (feature, latlng) {
        return pointLayer(feature, latlng, config);
      },
      onEachFeature: function (feature, layerItem) {
        layerItem.bindPopup(popupHtml(feature, config), { maxHeight: 320 });
      }
    });
    layerState.set(config.id, layer);
    return layer;
  }

  async function setLayer(config, checked) {
    try {
      let layer = layerState.get(config.id);
      if (checked) {
        if (!layer) {
          layer = await loadLayer(config);
        }
        layer.addTo(map);
        if (config.kind === "mask" || config.kind === "spectral") {
          const bounds = layer.getBounds();
          if (bounds.isValid()) {
            map.fitBounds(bounds.pad(0.06));
          }
        }
      } else if (layer) {
        map.removeLayer(layer);
      }
      status.textContent = "Mapa listo. Activa capas segun necesites; las capas grandes se cargan al seleccionarlas.";
    } catch (error) {
      status.textContent = `No se pudo cargar ${config.label}: ${error.message}`;
    }
  }

  for (const config of layerConfigs) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(config.checked);
    input.addEventListener("change", function () {
      setLayer(config, input.checked);
    });
    label.appendChild(input);
    label.appendChild(document.createTextNode(" " + config.label));
    controls.appendChild(label);
  }

  Promise.all(layerConfigs.filter((config) => config.checked).map((config) => setLayer(config, true)))
    .then(function () {
      status.textContent = "Mapa listo. Activa capas segun necesites; las capas grandes se cargan al seleccionarlas.";
    });
})();
</script>
"""
    return template.replace("__LAYER_CONFIGS__", json.dumps(layers, ensure_ascii=False))


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


def _spatial_display_level(nodes: pd.DataFrame, edges: pd.DataFrame) -> int | None:
    if not edges.empty and "level" in edges.columns:
        counts = edges.groupby("level").size()
        usable = counts[counts >= 3]
        if not usable.empty:
            return int(usable.index.max())
        if not counts.empty:
            return int(counts.index.max())
    if not nodes.empty and "level" in nodes.columns:
        levels = pd.to_numeric(nodes["level"], errors="coerce").dropna()
        if not levels.empty:
            return int(levels.max())
    return None


def _visible_grid_display_level(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
    *,
    min_cell_px: float = 3.0,
    min_edges: int = 3,
) -> int | None:
    if nodes.empty or "level" not in nodes.columns:
        return None
    required = {"lon_min_deg", "lat_min_deg", "lon_max_deg", "lat_max_deg"}
    if not required.issubset(nodes.columns):
        return _spatial_display_level(nodes, edges)

    min_lon, min_lat, max_lon, max_lat = bounds
    width, height = canvas_size
    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)
    work = nodes.copy()
    work["level_numeric"] = pd.to_numeric(work["level"], errors="coerce")
    lon_min = pd.to_numeric(work["lon_min_deg"], errors="coerce")
    lon_max = pd.to_numeric(work["lon_max_deg"], errors="coerce")
    lat_min = pd.to_numeric(work["lat_min_deg"], errors="coerce")
    lat_max = pd.to_numeric(work["lat_max_deg"], errors="coerce")
    work["cell_px_w"] = (lon_max - lon_min).abs() / lon_span * width
    work["cell_px_h"] = (lat_max - lat_min).abs() / lat_span * height
    work["cell_px_min"] = work[["cell_px_w", "cell_px_h"]].min(axis=1)
    work = work[np.isfinite(work["level_numeric"]) & np.isfinite(work["cell_px_min"])]
    if work.empty:
        return _spatial_display_level(nodes, edges)

    levels = [int(level) for level in sorted(work["level_numeric"].unique(), reverse=True)]
    edge_counts = (
        edges.groupby("level").size()
        if not edges.empty and "level" in edges.columns
        else pd.Series(dtype=int)
    )
    visible_levels = []
    for level in levels:
        layer = work[work["level_numeric"] == level]
        if layer.empty:
            continue
        if float(layer["cell_px_min"].median()) < min_cell_px:
            continue
        visible_levels.append(level)
        if int(edge_counts.get(level, 0)) >= min_edges:
            return level
    if visible_levels:
        return visible_levels[0]
    return _spatial_display_level(nodes, edges)


def _svg_spatial_probability_map(nodes: pd.DataFrame, edges: pd.DataFrame, mask: GeoMask) -> str:
    if nodes.empty:
        return "<p class='note'>Sin grilla espacial. Regenera el build.</p>"
    display_level = _spatial_display_level(nodes, edges)
    if display_level is None:
        return "<p class='note'>Sin nivel espacial disponible.</p>"

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

    node_layer = nodes[pd.to_numeric(nodes.get("level"), errors="coerce") == display_level].copy()
    edge_layer = edges[pd.to_numeric(edges.get("level"), errors="coerce") == display_level].copy() if not edges.empty else pd.DataFrame()

    cells = []
    required = ["lon_min_deg", "lat_min_deg", "lon_max_deg", "lat_max_deg"]
    if set(required).issubset(node_layer.columns):
        for row in node_layer.itertuples(index=False):
            values = [getattr(row, column) for column in required]
            if not all(np.isfinite(values)):
                continue
            p = float(getattr(row, "anomaly_probability_pct", 0.0))
            x1, y1 = project(values[0], values[1])
            x2, y2 = project(values[2], values[3])
            x = min(x1, x2)
            y = min(y1, y2)
            cell_w = max(abs(x2 - x1), 0.6)
            cell_h = max(abs(y2 - y1), 0.6)
            color = _probability_color(p)
            opacity = 0.10 + 0.55 * min(max(p, 0.0), 100.0) / 100.0
            title = html.escape(f"{getattr(row, 'cell_id', '')} | anomalia {p:.1f}% | n={getattr(row, 'n_records', 0)}")
            cells.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{cell_w:.2f}' height='{cell_h:.2f}' "
                f"fill='{color}' fill-opacity='{opacity:.3f}' stroke='none'><title>{title}</title></rect>"
            )

    edge_lines = []
    edge_required = ["from_longitude_deg", "from_latitude_deg", "to_longitude_deg", "to_latitude_deg"]
    if not edge_layer.empty and set(edge_required).issubset(edge_layer.columns):
        for row in edge_layer.itertuples(index=False):
            values = [getattr(row, column) for column in edge_required]
            if not all(np.isfinite(values)):
                continue
            p = float(getattr(row, "fault_probability_pct", 0.0))
            x1, y1 = project(values[0], values[1])
            x2, y2 = project(values[2], values[3])
            color = _probability_color(p)
            opacity = 0.18 + 0.74 * min(max(p, 0.0), 100.0) / 100.0
            width_px = 0.35 + 2.7 * min(max(p, 0.0), 100.0) / 100.0
            title = html.escape(f"{getattr(row, 'from_cell_id', '')} -> {getattr(row, 'to_cell_id', '')} | falla {p:.1f}%")
            edge_lines.append(
                f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                f"stroke='{color}' stroke-opacity='{opacity:.3f}' stroke-width='{width_px:.2f}'><title>{title}</title></line>"
            )

    legend = (
        "<g transform='translate(24,24)'>"
        "<rect x='0' y='0' width='258' height='70' fill='white' fill-opacity='0.88' stroke='#cfd8df'/>"
        f"<text x='10' y='18' font-size='12' fill='#1d252c'>Grilla espacial nivel J{display_level}</text>"
        "<text x='10' y='35' font-size='10' fill='#53606d'>celdas=anomalia, lineas=falla</text>"
        "<rect x='10' y='44' width='55' height='12' fill='rgb(44,123,182)'/>"
        "<rect x='65' y='44' width='55' height='12' fill='rgb(255,255,191)'/>"
        "<rect x='120' y='44' width='55' height='12' fill='rgb(215,25,28)'/>"
        "<text x='10' y='66' font-size='10'>0</text><text x='89' y='66' font-size='10'>50</text><text x='158' y='66' font-size='10'>100</text>"
        "</g>"
    )
    return f"<svg viewBox='0 0 {width} {height}' role='img'>{''.join(mask_paths)}{''.join(cells)}{''.join(edge_lines)}{legend}</svg>"


def _svg_spectral_probability_map(nodes: pd.DataFrame, edges: pd.DataFrame, mask: GeoMask) -> str:
    if nodes.empty:
        return "<p class='note'>Sin red dinamica espectral. Ejecuta build con analysis-mode=spectral o both.</p>"
    min_lon, min_lat, max_lon, max_lat = mask.bounds
    pad_lon = (max_lon - min_lon) * 0.08
    pad_lat = (max_lat - min_lat) * 0.04
    min_lon -= pad_lon
    max_lon += pad_lon
    min_lat -= pad_lat
    max_lat += pad_lat
    width, height = 760, 760
    display_level = _visible_grid_display_level(nodes, edges, (min_lon, min_lat, max_lon, max_lat), (width, height))
    if display_level is None:
        return "<p class='note'>Sin nivel espectral disponible.</p>"

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = (lon - min_lon) / (max_lon - min_lon) * width
        y = height - (lat - min_lat) / (max_lat - min_lat) * height
        return x, y

    mask_paths = []
    for polygon in mask.polygons:
        coords = [project(lon, lat) for lon, lat in polygon]
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in coords) + " Z"
        mask_paths.append(f"<path d='{d}' fill='#eef4ef' stroke='#315f50' stroke-width='1.3'/>")

    node_layer = nodes[pd.to_numeric(nodes.get("level"), errors="coerce") == display_level].copy()
    edge_layer = edges[pd.to_numeric(edges.get("level"), errors="coerce") == display_level].copy() if not edges.empty else pd.DataFrame()

    cells = []
    required = ["lon_min_deg", "lat_min_deg", "lon_max_deg", "lat_max_deg"]
    if set(required).issubset(node_layer.columns):
        for row in node_layer.itertuples(index=False):
            values = [getattr(row, column) for column in required]
            if not all(np.isfinite(values)):
                continue
            p = float(getattr(row, "spectral_dynamic_probability_pct", 0.0))
            x1, y1 = project(values[0], values[1])
            x2, y2 = project(values[2], values[3])
            color = _probability_color(p)
            opacity = 0.24 + 0.70 * min(max(p, 0.0), 100.0) / 100.0
            title = html.escape(f"{getattr(row, 'cell_id', '')} | dinamica espectral {p:.1f}% | n={getattr(row, 'n_records', 0)}")
            cells.append(
                f"<rect x='{min(x1, x2):.1f}' y='{min(y1, y2):.1f}' width='{max(abs(x2-x1), 0.6):.2f}' height='{max(abs(y2-y1), 0.6):.2f}' "
                f"fill='{color}' fill-opacity='{opacity:.3f}' stroke='#22313c' stroke-opacity='0.34' "
                f"stroke-width='0.22'><title>{title}</title></rect>"
            )

    edge_lines = []
    edge_required = ["from_longitude_deg", "from_latitude_deg", "to_longitude_deg", "to_latitude_deg"]
    if not edge_layer.empty and set(edge_required).issubset(edge_layer.columns):
        for row in edge_layer.itertuples(index=False):
            values = [getattr(row, column) for column in edge_required]
            if not all(np.isfinite(values)):
                continue
            p = float(getattr(row, "spectral_transfer_probability_pct", 0.0))
            x1, y1 = project(values[0], values[1])
            x2, y2 = project(values[2], values[3])
            color = _probability_color(p)
            opacity = 0.18 + 0.74 * min(max(p, 0.0), 100.0) / 100.0
            width_px = 0.35 + 2.7 * min(max(p, 0.0), 100.0) / 100.0
            title = html.escape(f"{getattr(row, 'from_cell_id', '')} -> {getattr(row, 'to_cell_id', '')} | transferencia {p:.1f}%")
            edge_lines.append(
                f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                f"stroke='{color}' stroke-opacity='{opacity:.3f}' stroke-width='{width_px:.2f}'><title>{title}</title></line>"
            )

    legend = (
        "<g transform='translate(24,24)'>"
        "<rect x='0' y='0' width='286' height='70' fill='white' fill-opacity='0.88' stroke='#cfd8df'/>"
        f"<text x='10' y='18' font-size='12' fill='#1d252c'>Red espectral nivel J{display_level} auto</text>"
        "<text x='10' y='35' font-size='10' fill='#53606d'>celdas=respuesta, lineas=transmisibilidad</text>"
        "<rect x='10' y='44' width='55' height='12' fill='rgb(44,123,182)'/>"
        "<rect x='65' y='44' width='55' height='12' fill='rgb(255,255,191)'/>"
        "<rect x='120' y='44' width='55' height='12' fill='rgb(215,25,28)'/>"
        "<text x='10' y='66' font-size='10'>0</text><text x='89' y='66' font-size='10'>50</text><text x='158' y='66' font-size='10'>100</text>"
        "</g>"
    )
    return f"<svg viewBox='0 0 {width} {height}' role='img'>{''.join(mask_paths)}{''.join(cells)}{''.join(edge_lines)}{legend}</svg>"


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
