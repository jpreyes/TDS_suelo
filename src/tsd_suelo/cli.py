from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_FLATFILES_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_RECORDS_DIR, PipelineConfig
from .logging_utils import RunLogger
from .pipeline import run_build, run_inventory, run_targets
from .report import build_results_report, print_summary


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsd-suelo",
        description="Pipeline TSD-Suelo observado desde H5 y flatfiles primarios.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "targets", "build", "report", "summary", "serve"):
        cmd = subparsers.add_parser(name)
        cmd.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
        cmd.add_argument("--flatfiles-dir", type=Path, default=DEFAULT_FLATFILES_DIR)
        cmd.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        cmd.add_argument("--max-h5", type=int, default=None, help="Limita cantidad de H5 para pruebas rapidas.")
        cmd.add_argument("--damping", type=float, default=0.05, help="Amortiguamiento PSA, por defecto 5%%.")
        cmd.add_argument("--h5-only", action="store_true", help="No agrega registros flatfile-only al geo_targets.")
        cmd.add_argument("--no-chile-mask", action="store_true", help="No aplica la mascara de Chile.")
        cmd.add_argument("--mask-geojson", type=Path, default=None, help="GeoJSON local de mascara de Chile o region de estudio.")
        cmd.add_argument("--top-n", type=int, default=50, help="Cantidad de filas en reportes/resumen.")
        cmd.add_argument("--workers", type=int, default=1, help="Procesos paralelos para leer H5.")
        cmd.add_argument("--skip-psa", action="store_true", help="Omite PSA desde H5 para primera corrida rapida.")
        cmd.add_argument("--reuse-targets", action="store_true", help="Reusa output waveform_targets_observed.parquet si existe.")
        cmd.add_argument("--reuse-products", action="store_true", help="Reusa todos los parquets intermedios existentes.")
        cmd.add_argument("--analysis-mode", choices=("spatial", "spectral", "both"), default="both", help="Modo de analisis: grilla espacial, red espectral o ambos.")
        cmd.add_argument("--log-file", type=Path, default=None, help="Archivo de log. Por defecto outputs/run.log.")
        cmd.add_argument("--progress-every", type=int, default=500, help="Reporta progreso H5 cada N archivos.")
        cmd.add_argument("--quiet", action="store_true", help="Escribe log sin imprimir progreso en pantalla.")
        if name == "serve":
            cmd.add_argument("--host", default="127.0.0.1", help="Host HTTP, por defecto 127.0.0.1.")
            cmd.add_argument("--port", type=int, default=8787, help="Puerto HTTP, por defecto 8787.")
            cmd.add_argument("--repo-dir", type=Path, default=Path("."), help="Directorio del repo para git pull/install.")
            cmd.add_argument("--admin-token", default=None, help="Token para controles admin. Alternativa: TSD_SUELO_ADMIN_TOKEN.")
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        records_dir=args.records_dir,
        flatfiles_dir=args.flatfiles_dir,
        output_dir=args.output_dir,
        max_h5=args.max_h5,
        damping=args.damping,
        include_flatfile_only=not args.h5_only,
        use_chile_mask=not args.no_chile_mask,
        mask_geojson=args.mask_geojson,
        workers=args.workers,
        compute_psa=not args.skip_psa,
        reuse_targets=args.reuse_targets or args.reuse_products,
        reuse_products=args.reuse_products,
        analysis_mode=args.analysis_mode,
        log_file=args.log_file,
        progress_every=args.progress_every,
        quiet=args.quiet,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _base_parser()
    args = parser.parse_args(argv)
    cfg = _config_from_args(args)
    if args.command == "inventory":
        with RunLogger(cfg.log_file or (cfg.output_dir.expanduser().resolve() / "run.log"), verbose=not cfg.quiet) as log:
            manifest = run_inventory(cfg, log=log)
        print(f"H5: {manifest['h5_count']} | eventos: {manifest['event_count_flatfile']} | estaciones: {manifest['station_count_flatfile']}")
        return 0
    if args.command == "targets":
        with RunLogger(cfg.log_file or (cfg.output_dir.expanduser().resolve() / "run.log"), verbose=not cfg.quiet) as log:
            targets = run_targets(cfg, log=log)
        print(f"waveform_targets_observed.parquet filas={targets.shape[0]}")
        return 0
    if args.command == "build":
        manifest = run_build(cfg, log=None)
        print(f"Build completo en {manifest['output_dir']}")
        print(f"Log: {manifest['log_file']}")
        for name, rows in manifest["rows"].items():
            print(f"  {name}: {rows}")
        return 0
    if args.command == "report":
        summary = build_results_report(cfg.output_dir.expanduser().resolve(), mask_geojson=args.mask_geojson, top_n=args.top_n)
        print(f"Reporte escrito en {cfg.output_dir.expanduser().resolve() / 'results_report.html'}")
        print(f"Geo targets: {summary['geo_targets']} | H5: {summary['h5_records']} | flatfile-only: {summary['flatfile_records']}")
        return 0
    if args.command == "summary":
        print(print_summary(cfg.output_dir.expanduser().resolve(), top_n=args.top_n))
        return 0
    if args.command == "serve":
        from .server import serve

        return serve(
            cfg,
            host=args.host,
            port=args.port,
            repo_dir=args.repo_dir,
            admin_token=args.admin_token,
        )
    parser.error(f"Comando no soportado: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
