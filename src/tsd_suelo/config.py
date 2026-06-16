from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_RECORDS_DIR = Path(r"C:\Respaldos\records")
DEFAULT_FLATFILES_DIR = DEFAULT_RECORDS_DIR / "flatfiles"
DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class PipelineConfig:
    records_dir: Path = DEFAULT_RECORDS_DIR
    flatfiles_dir: Path = DEFAULT_FLATFILES_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    max_h5: int | None = None
    damping: float = 0.05
    acceleration_unit: str = "cm_s2"

    def resolved(self) -> "PipelineConfig":
        return PipelineConfig(
            records_dir=self.records_dir.expanduser().resolve(),
            flatfiles_dir=self.flatfiles_dir.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            max_h5=self.max_h5,
            damping=self.damping,
            acceleration_unit=self.acceleration_unit,
        )

