from __future__ import annotations

from pathlib import Path

from tsd_suelo.config import PipelineConfig
from tsd_suelo.server import _build_command


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
        },
        cfg,
    )
    assert "build" in command
    assert command[command.index("--records-dir") + 1] == "../records"
    assert command[command.index("--flatfiles-dir") + 1] == "../flatfiles"
    assert command[command.index("--output-dir") + 1] == "outputs_precomputed"
    assert command[command.index("--workers") + 1] == "8"
    assert "--reuse-products" in command
