from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .atlas import write_atlas_products
from .config import PipelineConfig
from .etl import build_geo_targets, build_inventory, build_waveform_targets
from .forward import write_forward_template
from .graph import build_kozyrev_fields, build_route_graph, write_graph_products
from .latent import discover_latent_modes, write_latent_products
from .mask import annotate_geo_targets, load_chile_mask, write_mask
from .report import build_results_report
from .residuals import residualize_targets, write_residual_products
from .utils import ensure_dir, write_json, write_parquet


LogFn = Callable[[str], None]


def _log(log: LogFn | None, message: str) -> None:
    if log:
        log(message)


def run_inventory(config: PipelineConfig, log: LogFn | None = None) -> dict[str, Any]:
    cfg = config.resolved()
    ensure_dir(cfg.output_dir)
    _log(log, "F00 inventario observado")
    return build_inventory(cfg.records_dir, cfg.flatfiles_dir, cfg.output_dir, max_h5=cfg.max_h5)


def run_targets(config: PipelineConfig, log: LogFn | None = None):
    cfg = config.resolved()
    ensure_dir(cfg.output_dir)
    target_path = cfg.output_dir / "waveform_targets_observed.parquet"
    if cfg.reuse_targets and target_path.exists():
        _log(log, f"Reusando targets H5 existentes: {target_path}")
        import pandas as pd

        return pd.read_parquet(target_path)
    _log(log, "F05 targets fisicos observados desde H5")
    return build_waveform_targets(
        cfg.records_dir,
        cfg.output_dir,
        max_h5=cfg.max_h5,
        damping=cfg.damping,
        compute_psa=cfg.compute_psa,
        workers=cfg.workers,
    )


def run_build(config: PipelineConfig, log: LogFn | None = None) -> dict[str, Any]:
    cfg = config.resolved()
    ensure_dir(cfg.output_dir)

    inventory = run_inventory(cfg, log=log)
    waveform_targets = run_targets(cfg, log=log)

    _log(log, "F01-F06 geometria, indices fuente/receptor y geo_targets_observed")
    geo_targets, geometry, receivers, sources = build_geo_targets(
        waveform_targets,
        cfg.flatfiles_dir,
        cfg.output_dir,
        include_flatfile_only=cfg.include_flatfile_only,
    )
    geo_mask = None
    if cfg.use_chile_mask:
        _log(log, "Aplicando mascara de Chile")
        geo_mask = load_chile_mask(cfg.mask_geojson)
        write_mask(geo_mask, cfg.output_dir)
        geo_targets = annotate_geo_targets(geo_targets, geo_mask)
        write_parquet(geo_targets, cfg.output_dir / "geo_targets_observed.parquet")

    _log(log, "F07 residualizacion por fuente/distancia/sitio conocido")
    residuals, attribution = residualize_targets(geo_targets)
    write_residual_products(residuals, attribution, cfg.output_dir)

    _log(log, "F08 descubrimiento de modos latentes")
    modes, components = discover_latent_modes(residuals)
    write_latent_products(modes, components, cfg.output_dir)

    _log(log, "F09 grafo Kozyrev fuente 3D -> ruta -> receptor")
    route_graph = build_route_graph(geo_targets, modes)
    kozyrev_fields = build_kozyrev_fields(geo_targets, modes)
    write_graph_products(route_graph, kozyrev_fields, cfg.output_dir)

    _log(log, "F10 atlas geologico observado")
    write_atlas_products(geo_targets, modes, kozyrev_fields, cfg.output_dir, geo_mask=geo_mask)

    _log(log, "Contrato de forward condicionado posterior")
    write_forward_template(geo_targets, modes, cfg.output_dir)

    _log(log, "Reporte de resultados")
    build_results_report(cfg.output_dir, mask_geojson=cfg.mask_geojson)

    manifest = {
        "output_dir": str(cfg.output_dir),
        "inventory": inventory,
        "rows": {
            "waveform_targets_observed": int(waveform_targets.shape[0]),
            "flatfile_records_available": int(max(0, geo_targets.shape[0] - waveform_targets.shape[0])),
            "record_geometry": int(geometry.shape[0]),
            "receiver_index": int(receivers.shape[0]),
            "source3d_index": int(sources.shape[0]),
            "geo_targets_observed": int(geo_targets.shape[0]),
            "geo_residuals": int(residuals.shape[0]),
            "latent_modes": int(modes.shape[0]),
            "route_graph_observed": int(route_graph.shape[0]),
            "kozyrev_graph_fields": int(kozyrev_fields.shape[0]),
        },
        "products": [
            "observed_inventory.json",
            "waveform_targets_observed.parquet",
            "waveform_targets_observed.meta.json",
            "record_geometry.parquet",
            "receiver_index.parquet",
            "source3d_index.parquet",
            "geo_targets_observed.parquet",
            "geo_residuals.parquet",
            "target_level_attribution.csv",
            "latent_modes.parquet",
            "latent_mode_components.csv",
            "route_graph_observed.parquet",
            "kozyrev_graph_fields.parquet",
            "atlas_geologico.geojson",
            "atlas_geologico.kmz",
            "chile_mask.geojson",
            "forward_conditioning_template.json",
            "results_report.html",
            "results_summary.json",
            "top_kozyrev_anomalies.csv",
            "top_receiver_anomalies.csv",
            "top_route_anomalies.csv",
        ],
    }
    write_json(cfg.output_dir / "pipeline_manifest.json", manifest)
    return manifest
