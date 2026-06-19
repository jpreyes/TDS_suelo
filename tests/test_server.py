from __future__ import annotations

from pathlib import Path

from tsd_suelo.config import PipelineConfig
from tsd_suelo.server import _build_command, _forward_command, _scenario_command


def test_build_command_uses_configured_external_paths(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        records_dir=tmp_path / "records",
        flatfiles_dir=tmp_path / "flatfiles",
        output_dir=tmp_path / "outputs_precomputed",
        workers=4,
        progress_every=250,
    )
    command = _build_command(
        {
            "records_dir": "../records",
            "flatfiles_dir": "../flatfiles",
            "output_dir": "outputs_precomputed",
            "workers": "8",
            "progress_every": "500",
            "reuse_products": "on",
            "analysis_mode": "spectral",
        },
        cfg,
    )
    assert "build" in command
    assert command[command.index("--records-dir") + 1] == "../records"
    assert command[command.index("--flatfiles-dir") + 1] == "../flatfiles"
    assert command[command.index("--output-dir") + 1] == "outputs_precomputed"
    assert command[command.index("--workers") + 1] == "8"
    assert command[command.index("--analysis-mode") + 1] == "spectral"
    assert "--reuse-products" in command


def test_forward_command_uses_existing_output_dir(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        records_dir=tmp_path / "records",
        flatfiles_dir=tmp_path / "flatfiles",
        output_dir=tmp_path / "outputs_precomputed",
    )
    command = _forward_command(
        {
            "output_dir": "outputs_precomputed",
            "top_n": "80",
        },
        cfg,
    )
    assert "forward" in command
    assert command[command.index("--output-dir") + 1] == "outputs_precomputed"
    assert command[command.index("--top-n") + 1] == "80"


def test_scenario_command_defaults_to_santiago_case(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        records_dir=tmp_path / "records",
        flatfiles_dir=tmp_path / "flatfiles",
        output_dir=tmp_path / "outputs_precomputed",
    )
    command = _scenario_command({"output_dir": "outputs_precomputed"}, cfg)
    assert "scenario" in command
    assert command[command.index("--output-dir") + 1] == "outputs_precomputed"
    assert command[command.index("--source-distance-km") + 1] == "100"
    assert command[command.index("--source-direction") + 1] == "suroeste"
    assert command[command.index("--mw") + 1] == "7.5"
    assert command[command.index("--vs30") + 1] == "600"
