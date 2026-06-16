from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .utils import log_positive, numeric_series, write_parquet


DEFAULT_TARGET_COLUMNS = [
    "pga_h_g",
    "pga_e_g",
    "pga_n_g",
    "pga_z_g",
    "arias_h_m_s",
    "cav_h_m_s",
    "duration_5_75_h_s",
    "duration_5_95_h_s",
    "dominant_freq_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "energy_0p1_1_hz",
    "energy_1_3_hz",
    "energy_3_8_hz",
    "energy_8_20_hz",
    "energy_20_plus_hz",
    "horizontal_to_vertical_pga",
    "east_to_north_pga",
    "psa_t0p1_g",
    "psa_t0p2_g",
    "psa_t0p5_g",
    "psa_t1p0_g",
    "psa_t2p0_g",
]


def available_targets(df: pd.DataFrame, candidates: list[str] | None = None) -> list[str]:
    targets = []
    for column in candidates or DEFAULT_TARGET_COLUMNS:
        if column not in df.columns:
            continue
        values = numeric_series(df[column])
        if int((values > 0).sum()) >= 3:
            targets.append(column)
    return targets


def _standardized_numeric(series: pd.Series) -> pd.Series:
    values = numeric_series(series)
    median = float(values.median()) if values.notna().any() else 0.0
    values = values.fillna(median)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=series.index)
    return (values - float(values.mean())) / std


def _design_matrix(df: pd.DataFrame, include_site: bool) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    distance = numeric_series(df.get("distance_km", pd.Series(np.nan, index=df.index)))
    distance = distance.where(distance > 0)
    features["log_distance"] = _standardized_numeric(np.log1p(distance))
    for column in ("mw", "event_depth_km"):
        if column in df.columns:
            features[column] = _standardized_numeric(df[column])

    if "tectonic_type" in df.columns:
        tectonic = df["tectonic_type"].fillna("unknown").astype(str)
        dummies = pd.get_dummies(tectonic, prefix="tectonic", dtype=float)
        if dummies.shape[1] > 1:
            features = pd.concat([features, dummies.iloc[:, 1:]], axis=1)

    if include_site:
        if "vs30_m_s" in df.columns:
            vs30 = numeric_series(df["vs30_m_s"]).where(lambda s: s > 0)
            features["log_vs30"] = _standardized_numeric(np.log(vs30))
        if "f0_hvsr_hz" in df.columns:
            f0 = numeric_series(df["f0_hvsr_hz"]).where(lambda s: s > 0)
            features["log_f0_hvsr"] = _standardized_numeric(np.log(f0))
        for column in ("a0_hvsr", "kappa0", "topographic_slope"):
            if column in df.columns:
                features[column] = _standardized_numeric(df[column])

    features.insert(0, "intercept", 1.0)
    return features.astype(float)


def _group_effect(values: pd.Series, group: pd.Series | None) -> pd.Series:
    if group is None:
        return pd.Series(0.0, index=values.index)
    group_key = group.fillna("unknown").astype(str)
    counts = group_key.map(group_key.value_counts())
    effects = values.groupby(group_key).transform("mean")
    effects = effects.where(counts >= 2, 0.0)
    return effects.fillna(0.0)


def _fit_predict(x: pd.DataFrame, y: pd.Series, group: pd.Series | None = None) -> tuple[pd.Series, float, int]:
    mask = y.notna() & np.isfinite(y)
    pred = pd.Series(np.nan, index=y.index, dtype=float)
    n = int(mask.sum())
    if n < 3:
        return pred, math.nan, n
    x_fit = x.loc[mask].copy()
    y_fit = y.loc[mask].astype(float)
    keep = []
    for column in x_fit.columns:
        values = x_fit[column].to_numpy(dtype=float)
        if column == "intercept" or np.nanstd(values) > 1e-12:
            keep.append(column)
    x_fit = x_fit[keep]
    try:
        beta, *_ = np.linalg.lstsq(x_fit.to_numpy(dtype=float), y_fit.to_numpy(dtype=float), rcond=None)
        pred.loc[mask] = x_fit.to_numpy(dtype=float) @ beta
        source_effect = _group_effect(y.loc[mask] - pred.loc[mask], group.loc[mask] if group is not None else None)
        pred.loc[mask] = pred.loc[mask] + source_effect
    except np.linalg.LinAlgError:
        return pred, math.nan, n
    ss_res = float(np.sum((y_fit.to_numpy(dtype=float) - pred.loc[mask].to_numpy(dtype=float)) ** 2))
    ss_tot = float(np.sum((y_fit.to_numpy(dtype=float) - float(y_fit.mean())) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return pred, r2, n


def residualize_targets(
    geo_targets: pd.DataFrame,
    target_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = available_targets(geo_targets, target_columns)
    base_x = _design_matrix(geo_targets, include_site=False)
    site_x = _design_matrix(geo_targets, include_site=True)
    source_group = geo_targets["event_id"] if "event_id" in geo_targets.columns else None
    id_columns = [
        "record_observed_id",
        "event_id",
        "station_id",
        "source3d_id",
        "receiver_id",
        "route_id",
        "distance_km",
        "backazimuth_deg",
        "direction_bin_30deg",
        "vs30_m_s",
        "f0_hvsr_hz",
        "a0_hvsr",
        "geology",
    ]
    present_id_columns = [column for column in id_columns if column in geo_targets.columns]
    residual_rows = []
    attribution_rows = []
    for target in targets:
        y = log_positive(geo_targets[target])
        pred_base, r2_base, n = _fit_predict(base_x, y, group=source_group)
        pred_site, r2_site, _ = _fit_predict(site_x, y, group=source_group)
        residual = geo_targets[present_id_columns].copy()
        residual["target"] = target
        residual["target_value"] = numeric_series(geo_targets[target])
        residual["target_log"] = y
        residual["pred_source_distance_log"] = pred_base
        residual["pred_known_site_log"] = pred_site
        residual["residual_source_distance_log"] = y - pred_base
        residual["residual_known_site_log"] = y - pred_site
        residual_rows.append(residual)
        attribution_rows.append(
            {
                "target": target,
                "n": n,
                "r2_source_distance": r2_base,
                "r2_source_distance_site": r2_site,
                "site_incremental_r2": r2_site - r2_base if np.isfinite(r2_site) and np.isfinite(r2_base) else math.nan,
                "residual_std_known_site": float((y - pred_site).std(skipna=True)),
            }
        )
    residuals = pd.concat(residual_rows, ignore_index=True) if residual_rows else pd.DataFrame()
    attribution = pd.DataFrame(attribution_rows)
    return residuals, attribution


def write_residual_products(
    residuals: pd.DataFrame,
    attribution: pd.DataFrame,
    output_dir,
) -> None:
    write_parquet(residuals, output_dir / "geo_residuals.parquet")
    attribution.to_csv(output_dir / "target_level_attribution.csv", index=False)
