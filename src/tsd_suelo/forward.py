from __future__ import annotations

from pathlib import Path

import pandas as pd

from .graph import mode_columns
from .residuals import DEFAULT_TARGET_COLUMNS
from .utils import write_json


def write_forward_template(geo_targets: pd.DataFrame, modes: pd.DataFrame, output_dir: Path) -> None:
    template = {
        "status": "prepared_contract_only",
        "purpose": "Forward condicionado posterior, no generacion sintetica inicial.",
        "primary_sources": ["records/*.h5", "records/flatfiles/*.csv"],
        "conditioning_inputs": {
            "geometry": [
                "event_latitude_deg",
                "event_longitude_deg",
                "event_depth_km",
                "station_latitude_deg",
                "station_longitude_deg",
                "distance_km",
                "azimuth_deg",
                "backazimuth_deg",
                "incidence_angle_deg",
            ],
            "known_site": ["vs30_m_s", "f0_hvsr_hz", "a0_hvsr", "kappa0", "geology"],
            "latent_modes": mode_columns(modes),
            "targets_available": [column for column in DEFAULT_TARGET_COLUMNS if column in geo_targets.columns],
            "graph_fields": ["route_id", "source3d_id", "receiver_id", "kozyrev_delta_mode_*"],
        },
        "guards": [
            "No usar parquets historicos como fuente primaria.",
            "No importar dependencias internas de GMPE, Modelo E ni TSD estructural.",
            "Ajustar forward solo despues de validar atlas y residuos observados.",
        ],
    }
    write_json(output_dir / "forward_conditioning_template.json", template)

