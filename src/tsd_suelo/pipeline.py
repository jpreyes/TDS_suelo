from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from .atlas import write_atlas_products
from .config import PipelineConfig
from .etl import build_geo_targets, build_inventory, build_waveform_targets
from .forward import write_forward_template
from .graph import build_kozyrev_fields, build_route_graph, write_graph_products
from .latent import discover_latent_modes, write_latent_products
from .logging_utils import PhaseTimer, RunLogger, format_seconds
from .mask import annotate_geo_targets, load_chile_mask, write_mask
from .report import build_results_report
from .residuals import residualize_targets, write_residual_products
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
            f"progress_every={cfg.progress_every}",
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

        with PhaseTimer(log, "F10 atlas geologico observado"):
            write_atlas_products(geo_targets, modes, kozyrev_fields, cfg.output_dir, geo_mask=geo_mask)

        with PhaseTimer(log, "Contrato de forward condicionado posterior"):
            write_forward_template(geo_targets, modes, cfg.output_dir)

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
                "route_graph_observed": int(route_graph.shape[0]),
                "kozyrev_graph_fields": int(kozyrev_fields.shape[0]),
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
        _log(log, f"BUILD COMPLETO ({format_seconds(time.perf_counter() - build_start)})")
        return manifest
    finally:
        if close_logger and run_logger is not None:
            run_logger.close()
