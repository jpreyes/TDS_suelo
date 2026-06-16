from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .utils import write_parquet


def _standardize_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    values = frame.astype(float).copy()
    medians = values.median(axis=0, skipna=True).fillna(0.0)
    values = values.fillna(medians)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    return (values - means) / stds, means, stds


def discover_latent_modes(
    residuals: pd.DataFrame,
    n_modes: int = 6,
    value_column: str = "residual_known_site_log",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if residuals.empty:
        return pd.DataFrame(), pd.DataFrame()
    pivot = residuals.pivot_table(
        index="record_observed_id",
        columns="target",
        values=value_column,
        aggfunc="mean",
    )
    pivot = pivot.dropna(axis=1, how="all")
    if pivot.shape[0] < 2 or pivot.shape[1] < 1:
        return pd.DataFrame(), pd.DataFrame()
    x, _, _ = _standardize_matrix(pivot)
    u, s, vt = np.linalg.svd(x.to_numpy(dtype=float), full_matrices=False)
    rank = int(min(n_modes, len(s), pivot.shape[1], pivot.shape[0]))
    if rank == 0:
        return pd.DataFrame(), pd.DataFrame()
    scores = u[:, :rank] * s[:rank]
    total_var = float(np.sum(s * s))
    explained = (s[:rank] * s[:rank] / total_var) if total_var > 0 else np.full(rank, math.nan)

    mode_frame = pd.DataFrame({"record_observed_id": pivot.index.to_numpy()})
    for idx in range(rank):
        mode_frame[f"mode_{idx + 1}"] = scores[:, idx]
        mode_frame[f"mode_{idx + 1}_explained_variance"] = float(explained[idx])

    identity_cols = [
        "record_observed_id",
        "event_id",
        "station_id",
        "source3d_id",
        "receiver_id",
        "route_id",
        "distance_km",
        "backazimuth_deg",
        "direction_bin_30deg",
    ]
    identities = residuals[[c for c in identity_cols if c in residuals.columns]].drop_duplicates("record_observed_id")
    mode_frame = mode_frame.merge(identities, on="record_observed_id", how="left")

    components = []
    for mode_idx in range(rank):
        for target_idx, target in enumerate(pivot.columns):
            components.append(
                {
                    "mode": f"mode_{mode_idx + 1}",
                    "target": target,
                    "loading": float(vt[mode_idx, target_idx]),
                    "explained_variance": float(explained[mode_idx]),
                }
            )
    return mode_frame, pd.DataFrame(components)


def write_latent_products(modes: pd.DataFrame, components: pd.DataFrame, output_dir) -> None:
    write_parquet(modes, output_dir / "latent_modes.parquet")
    components.to_csv(output_dir / "latent_mode_components.csv", index=False)

