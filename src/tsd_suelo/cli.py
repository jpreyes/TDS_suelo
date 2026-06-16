from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_FLATFILES_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_RECORDS_DIR, PipelineConfig
from .pipeline import run_build, run_inventory, run_targets


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsd-suelo",
        description="Pipeline TSD-Suelo observado desde H5 y flatfiles primarios.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "targets", "build"):
        cmd = subparsers.add_parser(name)
        cmd.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
        cmd.add_argument("--flatfiles-dir", type=Path, default=DEFAULT_FLATFILES_DIR)
        cmd.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        cmd.add_argument("--max-h5", type=int, default=None, help="Limita cantidad de H5 para pruebas rapidas.")
        cmd.add_argument("--damping", type=float, default=0.05, help="Amortiguamiento PSA, por defecto 5%.")
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        records_dir=args.records_dir,
        flatfiles_dir=args.flatfiles_dir,
        output_dir=args.output_dir,
        max_h5=args.max_h5,
        damping=args.damping,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _base_parser()
    args = parser.parse_args(argv)
    cfg = _config_from_args(args)
    log = lambda message: print(message, flush=True)
    if args.command == "inventory":
        manifest = run_inventory(cfg, log=log)
        print(f"H5: {manifest['h5_count']} | eventos: {manifest['event_count_flatfile']} | estaciones: {manifest['station_count_flatfile']}")
        return 0
    if args.command == "targets":
        targets = run_targets(cfg, log=log)
        print(f"waveform_targets_observed.parquet filas={targets.shape[0]}")
        return 0
    if args.command == "build":
        manifest = run_build(cfg, log=log)
        print(f"Build completo en {manifest['output_dir']}")
        for name, rows in manifest["rows"].items():
            print(f"  {name}: {rows}")
        return 0
    parser.error(f"Comando no soportado: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

