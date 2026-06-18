from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from .atlas import write_atlas_products
from .config import PipelineConfig
from .etl import build_geo_targets, build_inventory, build_waveform_targets
from .faults import build_fault_candidates, fault_candidate_features, write_fault_products
from .forward import write_forward_products
from .graph import build_kozyrev_fields, build_route_graph, write_graph_products
from .latent import discover_latent_modes, write_latent_products
from .logging_utils import PhaseTimer, RunLogger, format_seconds
from .mask import annotate_geo_targets, load_chile_mask, write_mask
from .report import build_results_report
from .residuals import residualize_targets, write_residual_products
from .spatial_grid import build_spatial_grid_edges, build_spatial_grid_nodes, spatial_grid_features, write_spatial_grid_products
from .spectral import (
    build_spectral_edge_transmissibility,
    build_spectral_modes,
    build_spectral_node_dynamics,
    build_spectral_record_signatures,
    spectral_heatmap_features,
    write_spectral_products,
)
from .ultrametric import build_ultrametric_edges, build_ultrametric_nodes, kozyrev_heatmap_features, write_ultrametric_products
from .utils import ensure_dir, write_json, write_parquet


LogFn = Callable[[str], None]


def _log(log: LogFn | None, message: str) -> None:
    if log:
        log(message)


def _all_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def _empty_frame():
    import pandas as pd

    return pd.DataFrame()


def _parquet_has_columns(path: Path, columns: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        frame = _read_parquet(path)
    except Exception:
        return False
    return columns.issubset(frame.columns)


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
        meta_path = cfg.output_dir / "waveform_targets_observed.meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cfg.max_h5 is None and meta.get("max_h5") is not None:
                raise ValueError(
                    "No se reutilizan targets parciales para una corrida completa. "
                    f"Meta max_h5={meta.get('max_h5')} en {meta_path}. "
                    "Usa un output-dir completo o repite con el mismo --max-h5."
                )
            if cfg.max_h5 is not None and meta.get("max_h5") not in (None, cfg.max_h5):
                raise ValueError(
                    "El max_h5 solicitado no coincide con los targets existentes. "
                    f"Solicitado={cfg.max_h5}, meta={meta.get('max_h5')}."
                )
            if cfg.compute_psa and meta.get("compute_psa") is False:
                raise ValueError(
                    "Los targets existentes fueron calculados con --skip-psa. "
                    "Para reutilizarlos usa tambien --skip-psa o recalcula sin --reuse-targets."
                )
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
        progress_every=cfg.progress_every,
        log=log,
    )


def run_build(config: PipelineConfig, log: LogFn | None = None) -> dict[str, Any]:
    cfg = config.resolved()
    ensure_dir(cfg.output_dir)
    close_logger = False
    run_logger = None
    if log is None:
        run_logger = RunLogger(cfg.log_file or (cfg.output_dir / "run.log"), verbose=not cfg.quiet)
        log = run_logger
        close_logger = True

    import time

    build_start = time.perf_counter()
    try:
        _log(log, f"Output dir: {cfg.output_dir}")
        _log(log, f"Log file: {cfg.log_file or (cfg.output_dir / 'run.log')}")
        _log(
            log,
            "Config "
            f"records={cfg.records_dir} flatfiles={cfg.flatfiles_dir} "
            f"workers={cfg.workers} compute_psa={cfg.compute_psa} "
            f"reuse_targets={cfg.reuse_targets} reuse_products={cfg.reuse_products} "
            f"analysis_mode={cfg.analysis_mode} progress_every={cfg.progress_every}",
        )

        with PhaseTimer(log, "F00 inventario observado"):
            inventory = run_inventory(cfg, log=None)
            _log(
                log,
                "Inventario "
                f"H5={inventory['h5_count']} "
                f"records_flatfile={inventory['record_count_flatfile']} "
                f"eventos={inventory['event_count_flatfile']} "
                f"estaciones={inventory['station_count_flatfile']}",
            )

        with PhaseTimer(log, "F05 targets fisicos observados desde H5"):
            waveform_targets = run_targets(cfg, log=log)
            _log(log, f"Targets H5 filas={waveform_targets.shape[0]}")

        with PhaseTimer(log, "F01-F06 geometria, indices fuente/receptor y geo_targets_observed"):
            geo_paths = [
                cfg.output_dir / "geo_targets_observed.parquet",
                cfg.output_dir / "record_geometry.parquet",
                cfg.output_dir / "receiver_index.parquet",
                cfg.output_dir / "source3d_index.parquet",
            ]
            if cfg.reuse_products and _all_exist(geo_paths):
                _log(log, "Reusando geo_targets/geometry/indices existentes")
                geo_targets = _read_parquet(geo_paths[0])
                geometry = _read_parquet(geo_paths[1])
                receivers = _read_parquet(geo_paths[2])
                sources = _read_parquet(geo_paths[3])
            else:
                geo_targets, geometry, receivers, sources = build_geo_targets(
                    waveform_targets,
                    cfg.flatfiles_dir,
                    cfg.output_dir,
                    include_flatfile_only=cfg.include_flatfile_only,
                )
            _log(
                log,
                f"Geo targets={geo_targets.shape[0]} "
                f"geometria={geometry.shape[0]} receptores={receivers.shape[0]} fuentes={sources.shape[0]}",
            )

        geo_mask = None
        if cfg.use_chile_mask:
            with PhaseTimer(log, "Mascara de Chile"):
                geo_mask = load_chile_mask(cfg.mask_geojson)
                write_mask(geo_mask, cfg.output_dir)
                if {"receiver_in_chile_mask", "route_in_chile_mask"}.issubset(geo_targets.columns):
                    _log(log, "Mascara ya presente en geo_targets; se conserva")
                else:
                    geo_targets = annotate_geo_targets(geo_targets, geo_mask)
                    write_parquet(geo_targets, cfg.output_dir / "geo_targets_observed.parquet")
                _log(
                    log,
                    f"Mask={geo_mask.name} "
                    f"receiver_in_mask={int(geo_targets['receiver_in_chile_mask'].sum())} "
                    f"route_in_mask={int(geo_targets['route_in_chile_mask'].sum())}",
                )

        with PhaseTimer(log, "F07 residualizacion por fuente/distancia/sitio conocido"):
            residual_paths = [
                cfg.output_dir / "geo_residuals.parquet",
                cfg.output_dir / "target_level_attribution.csv",
            ]
            if cfg.reuse_products and _all_exist(residual_paths):
                _log(log, "Reusando geo_residuals/target_level_attribution existentes")
                residuals = _read_parquet(residual_paths[0])
                attribution = _read_csv(residual_paths[1])
            else:
                residuals, attribution = residualize_targets(geo_targets)
                write_residual_products(residuals, attribution, cfg.output_dir)
            _log(log, f"Residuals filas={residuals.shape[0]} targets={attribution.shape[0]}")

        with PhaseTimer(log, "F08 descubrimiento de modos latentes"):
            latent_paths = [
                cfg.output_dir / "latent_modes.parquet",
                cfg.output_dir / "latent_mode_components.csv",
            ]
            if cfg.reuse_products and _all_exist(latent_paths):
                _log(log, "Reusando latent_modes/latent_mode_components existentes")
                modes = _read_parquet(latent_paths[0])
                components = _read_csv(latent_paths[1])
            else:
                modes, components = discover_latent_modes(residuals)
                write_latent_products(modes, components, cfg.output_dir)
            _log(log, f"Modes filas={modes.shape[0]} components={components.shape[0]}")

        spatial_nodes = _empty_frame()
        spatial_edges = _empty_frame()
        if cfg.analysis_mode in {"spatial", "both"}:
            with PhaseTimer(log, "F08b grilla espacial jerarquica de anomalias"):
                spatial_paths = [
                    cfg.output_dir / "spatial_grid_nodes.parquet",
                    cfg.output_dir / "spatial_grid_edges.parquet",
                    cfg.output_dir / "spatial_anomaly_nodes.geojson",
                    cfg.output_dir / "spatial_fault_edges.geojson",
                    cfg.output_dir / "spatial_probability_heatmap.geojson",
                    cfg.output_dir / "spatial_probability_heatmap.kmz",
                ]
                can_reuse_spatial = (
                    cfg.reuse_products
                    and _all_exist(spatial_paths)
                    and _parquet_has_columns(spatial_paths[0], {"anomaly_probability_pct"})
                    and _parquet_has_columns(spatial_paths[1], {"fault_probability_pct"})
                )
                if can_reuse_spatial:
                    _log(log, "Reusando grilla espacial existente")
                    spatial_nodes = _read_parquet(spatial_paths[0])
                    spatial_edges = _read_parquet(spatial_paths[1])
                else:
                    spatial_nodes = build_spatial_grid_nodes(geo_targets, modes)
                    spatial_edges = build_spatial_grid_edges(spatial_nodes)
                    write_spatial_grid_products(spatial_nodes, spatial_edges, cfg.output_dir)
                _log(log, f"Spatial grid nodes={spatial_nodes.shape[0]} edges={spatial_edges.shape[0]}")
        else:
            _log(log, "F08b grilla espacial omitida por analysis_mode=spectral")

        spectral_records = _empty_frame()
        spectral_nodes = _empty_frame()
        spectral_edges = _empty_frame()
        spectral_modes = _empty_frame()
        spectral_components = _empty_frame()
        if cfg.analysis_mode in {"spectral", "both"}:
            with PhaseTimer(log, "F08c red dinamica espectral equivalente"):
                spectral_paths = [
                    cfg.output_dir / "spectral_record_signatures.parquet",
                    cfg.output_dir / "spectral_node_dynamics.parquet",
                    cfg.output_dir / "spectral_edge_transmissibility.parquet",
                    cfg.output_dir / "spectral_dynamic_modes.parquet",
                    cfg.output_dir / "spectral_mode_components.csv",
                    cfg.output_dir / "spectral_dynamic_heatmap.geojson",
                    cfg.output_dir / "spectral_dynamic_heatmap.kmz",
                    cfg.output_dir / "spectral_frequency_grid.json",
                ]
                can_reuse_spectral = (
                    cfg.reuse_products
                    and _all_exist(spectral_paths)
                    and _parquet_has_columns(spectral_paths[1], {"spectral_dynamic_probability_pct"})
                    and _parquet_has_columns(spectral_paths[2], {"spectral_transfer_probability_pct"})
                )
                if can_reuse_spectral:
                    _log(log, "Reusando red dinamica espectral existente")
                    spectral_records = _read_parquet(spectral_paths[0])
                    spectral_nodes = _read_parquet(spectral_paths[1])
                    spectral_edges = _read_parquet(spectral_paths[2])
                    spectral_modes = _read_parquet(spectral_paths[3])
                    spectral_components = _read_csv(spectral_paths[4])
                else:
                    spectral_records = build_spectral_record_signatures(
                        waveform_targets,
                        cfg.records_dir,
                        workers=cfg.workers,
                        progress_every=cfg.progress_every,
                        log=log,
                    )
                    spectral_nodes = build_spectral_node_dynamics(geo_targets, spectral_records)
                    spectral_edges = build_spectral_edge_transmissibility(spectral_nodes)
                    spectral_modes, spectral_components = build_spectral_modes(spectral_nodes)
                    write_spectral_products(
                        spectral_records,
                        spectral_nodes,
                        spectral_edges,
                        spectral_modes,
                        spectral_components,
                        cfg.output_dir,
                    )
                _log(
                    log,
                    f"Spectral records={spectral_records.shape[0]} "
                    f"nodes={spectral_nodes.shape[0]} edges={spectral_edges.shape[0]} "
                    f"modes={spectral_modes.shape[0]}",
                )
        else:
            _log(log, "F08c red dinamica espectral omitida por analysis_mode=spatial")

        with PhaseTimer(log, "F09 grafo Kozyrev fuente 3D -> ruta -> receptor"):
            graph_paths = [
                cfg.output_dir / "route_graph_observed.parquet",
                cfg.output_dir / "kozyrev_graph_fields.parquet",
            ]
            if cfg.reuse_products and _all_exist(graph_paths):
                _log(log, "Reusando route_graph/kozyrev_graph_fields existentes")
                route_graph = _read_parquet(graph_paths[0])
                kozyrev_fields = _read_parquet(graph_paths[1])
            else:
                route_graph = build_route_graph(geo_targets, modes)
                kozyrev_fields = build_kozyrev_fields(geo_targets, modes)
                write_graph_products(route_graph, kozyrev_fields, cfg.output_dir)
            _log(log, f"Route graph filas={route_graph.shape[0]} Kozyrev fields filas={kozyrev_fields.shape[0]}")

        with PhaseTimer(log, "F09a grafo ultrametrico Kozyrev probabilistico"):
            ultrametric_paths = [
                cfg.output_dir / "kozyrev_ultrametric_nodes.parquet",
                cfg.output_dir / "kozyrev_ultrametric_edges.parquet",
                cfg.output_dir / "kozyrev_ultrametric_nodes.geojson",
                cfg.output_dir / "kozyrev_ultrametric_edges.geojson",
                cfg.output_dir / "kozyrev_heatmap.geojson",
                cfg.output_dir / "kozyrev_heatmap.kmz",
            ]
            can_reuse_ultrametric = (
                cfg.reuse_products
                and _all_exist(ultrametric_paths)
                and _parquet_has_columns(ultrametric_paths[0], {"failure_probability_pct"})
                and _parquet_has_columns(ultrametric_paths[1], {"edge_probability_pct"})
            )
            if can_reuse_ultrametric:
                _log(log, "Reusando grafo ultrametrico Kozyrev existente")
                ultrametric_nodes = _read_parquet(ultrametric_paths[0])
                ultrametric_edges = _read_parquet(ultrametric_paths[1])
            else:
                ultrametric_nodes = build_ultrametric_nodes(geo_targets, kozyrev_fields)
                ultrametric_edges = build_ultrametric_edges(route_graph, ultrametric_nodes)
                write_ultrametric_products(ultrametric_nodes, ultrametric_edges, cfg.output_dir)
            _log(
                log,
                f"Ultrametric nodes={ultrametric_nodes.shape[0]} "
                f"edges={ultrametric_edges.shape[0]}",
            )

        with PhaseTimer(log, "F09b candidatos de falla observados"):
            fault_paths = [
                cfg.output_dir / "fault_candidates.parquet",
                cfg.output_dir / "top_fault_candidates.csv",
                cfg.output_dir / "fault_candidates.geojson",
                cfg.output_dir / "fault_candidates.kmz",
            ]
            if cfg.reuse_products and _all_exist(fault_paths) and _parquet_has_columns(fault_paths[0], {"fault_probability_pct"}):
                _log(log, "Reusando fault_candidates existentes")
                fault_candidates = _read_parquet(fault_paths[0])
            else:
                fault_candidates = build_fault_candidates(geo_targets, modes, kozyrev_fields)
                write_fault_products(fault_candidates, cfg.output_dir)
            _log(log, f"Fault candidates filas={fault_candidates.shape[0]}")

        with PhaseTimer(log, "F10 atlas geologico observado"):
            write_atlas_products(
                geo_targets,
                modes,
                kozyrev_fields,
                cfg.output_dir,
                geo_mask=geo_mask,
                extra_features=spatial_grid_features(spatial_nodes, spatial_edges)
                + spectral_heatmap_features(spectral_nodes, spectral_edges)
                + kozyrev_heatmap_features(ultrametric_nodes, ultrametric_edges)
                + fault_candidate_features(fault_candidates),
            )

        with PhaseTimer(log, "F11 dinamica compatible para forward condicionado"):
            forward_paths = [
                cfg.output_dir / "compatible_dynamics.parquet",
                cfg.output_dir / "forward_conditioning_profiles.parquet",
                cfg.output_dir / "forward_conditioning_template.json",
            ]
            if (
                cfg.reuse_products
                and _all_exist(forward_paths)
                and _parquet_has_columns(forward_paths[0], {"dynamic_anomaly_score", "forward_support_weight", "fault_probability_pct"})
            ):
                _log(log, "Reusando compatible_dynamics/forward_conditioning_profiles existentes")
                compatible_dynamics = _read_parquet(forward_paths[0])
                forward_profiles = _read_parquet(forward_paths[1])
            else:
                compatible_dynamics, forward_profiles = write_forward_products(
                    geo_targets,
                    residuals,
                    modes,
                    kozyrev_fields,
                    fault_candidates,
                    cfg.output_dir,
                )
            _log(
                log,
                f"Compatible dynamics filas={compatible_dynamics.shape[0]} "
                f"profiles={forward_profiles.shape[0]}",
            )

        with PhaseTimer(log, "Reporte de resultados"):
            build_results_report(cfg.output_dir, mask_geojson=cfg.mask_geojson)

        manifest = {
            "output_dir": str(cfg.output_dir),
            "log_file": str(cfg.log_file or (cfg.output_dir / "run.log")),
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
                "spatial_grid_nodes": int(spatial_nodes.shape[0]),
                "spatial_grid_edges": int(spatial_edges.shape[0]),
                "spectral_record_signatures": int(spectral_records.shape[0]),
                "spectral_node_dynamics": int(spectral_nodes.shape[0]),
                "spectral_edge_transmissibility": int(spectral_edges.shape[0]),
                "spectral_dynamic_modes": int(spectral_modes.shape[0]),
                "route_graph_observed": int(route_graph.shape[0]),
                "kozyrev_graph_fields": int(kozyrev_fields.shape[0]),
                "kozyrev_ultrametric_nodes": int(ultrametric_nodes.shape[0]),
                "kozyrev_ultrametric_edges": int(ultrametric_edges.shape[0]),
                "fault_candidates": int(fault_candidates.shape[0]),
                "compatible_dynamics": int(compatible_dynamics.shape[0]),
                "forward_conditioning_profiles": int(forward_profiles.shape[0]),
            },
            "products": [
                "run.log",
                "observed_inventory.json",
                "waveform_targets_observed.parquet",
                "waveform_targets_observed.meta.json",
                "waveform_targets_errors.csv",
                "record_geometry.parquet",
                "receiver_index.parquet",
                "source3d_index.parquet",
                "geo_targets_observed.parquet",
                "geo_residuals.parquet",
                "target_level_attribution.csv",
                "latent_modes.parquet",
                "latent_mode_components.csv",
                "spatial_grid_nodes.parquet",
                "spatial_grid_edges.parquet",
                "spatial_anomaly_nodes.geojson",
                "spatial_fault_edges.geojson",
                "spatial_probability_heatmap.geojson",
                "spatial_probability_heatmap.kmz",
                "spatial_grid_summary.json",
                "spectral_record_signatures.parquet",
                "spectral_node_dynamics.parquet",
                "spectral_edge_transmissibility.parquet",
                "spectral_dynamic_modes.parquet",
                "spectral_mode_components.csv",
                "spectral_dynamic_heatmap.geojson",
                "spectral_dynamic_heatmap.kmz",
                "spectral_frequency_grid.json",
                "route_graph_observed.parquet",
                "kozyrev_graph_fields.parquet",
                "kozyrev_ultrametric_nodes.parquet",
                "kozyrev_ultrametric_edges.parquet",
                "kozyrev_ultrametric_nodes.geojson",
                "kozyrev_ultrametric_edges.geojson",
                "kozyrev_heatmap.geojson",
                "kozyrev_heatmap.kmz",
                "fault_candidates.parquet",
                "top_fault_candidates.csv",
                "fault_candidates.geojson",
                "fault_candidates.kmz",
                "compatible_dynamics.parquet",
                "forward_conditioning_profiles.parquet",
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
        _log(log, f"BUILD COMPLETO ({format_seconds(time.perf_counter() - build_start)})")
        return manifest
    finally:
        if close_logger and run_logger is not None:
            run_logger.close()
