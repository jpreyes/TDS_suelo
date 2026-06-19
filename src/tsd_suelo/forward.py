from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd

from .atlas import write_geojson
from .geometry import add_geometry, azimuth_deg, destination_point, haversine_km
from .graph import mode_columns
from .residuals import DEFAULT_TARGET_COLUMNS
from .utils import write_json, write_parquet


BASE_FORWARD_COLUMNS = [
    "record_observed_id",
    "observed_source",
    "event_id",
    "station_id",
    "source3d_id",
    "receiver_id",
    "route_id",
    "event_latitude_deg",
    "event_longitude_deg",
    "event_depth_km",
    "mw",
    "tectonic_type",
    "station_latitude_deg",
    "station_longitude_deg",
    "distance_km",
    "azimuth_deg",
    "backazimuth_deg",
    "incidence_angle_deg",
    "direction_bin_30deg",
    "vs30_m_s",
    "f0_hvsr_hz",
    "a0_hvsr",
    "kappa0",
    "geology",
    "receiver_in_chile_mask",
    "route_in_chile_mask",
]


DIRECTION_BEARINGS = {
    "n": 0.0,
    "norte": 0.0,
    "ne": 45.0,
    "noreste": 45.0,
    "e": 90.0,
    "este": 90.0,
    "se": 135.0,
    "sureste": 135.0,
    "s": 180.0,
    "sur": 180.0,
    "sw": 225.0,
    "so": 225.0,
    "suroeste": 225.0,
    "sudoeste": 225.0,
    "w": 270.0,
    "o": 270.0,
    "oeste": 270.0,
    "nw": 315.0,
    "no": 315.0,
    "noroeste": 315.0,
}


def _rank01(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    return numeric.fillna(numeric.median()).rank(pct=True).astype(float)


def direction_to_bearing(direction: str | None, bearing_deg: float | None = None) -> float:
    if bearing_deg is not None and np.isfinite(bearing_deg):
        return float(bearing_deg) % 360.0
    key = str(direction or "").strip().lower()
    if key in DIRECTION_BEARINGS:
        return DIRECTION_BEARINGS[key]
    try:
        return float(key) % 360.0
    except ValueError as exc:
        raise ValueError(f"Direccion no reconocida: {direction!r}") from exc


def _angular_diff_deg(a: pd.Series | float, b: float) -> pd.Series:
    numeric = pd.to_numeric(a, errors="coerce")
    return ((numeric - b + 180.0) % 360.0 - 180.0).abs()


def _weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    weight_values = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = numeric.notna() & weight_values.gt(0.0)
    if not mask.any():
        numeric = numeric.dropna()
        return float(numeric.quantile(q)) if not numeric.empty else float("nan")
    order = np.argsort(numeric[mask].to_numpy(dtype=float))
    sorted_values = numeric[mask].to_numpy(dtype=float)[order]
    sorted_weights = weight_values[mask].to_numpy(dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = q * cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, cutoff, side="left")])


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    weight_values = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = numeric.notna() & weight_values.gt(0.0)
    if not mask.any():
        return float(numeric.mean()) if numeric.notna().any() else float("nan")
    return float(np.average(numeric[mask].to_numpy(dtype=float), weights=weight_values[mask].to_numpy(dtype=float)))


def _q90(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(0.9)) if not numeric.empty else float("nan")


def _present_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _merge_modes(geo_targets: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    out = geo_targets.copy()
    if modes.empty:
        return out
    keep = ["record_observed_id"] + mode_columns(modes)
    keep = [column for column in keep if column in modes.columns]
    return out.merge(modes[keep], on="record_observed_id", how="left")


def _residual_wide(residuals: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if residuals.empty or not {"record_observed_id", "target"}.issubset(residuals.columns):
        return pd.DataFrame(columns=["record_observed_id"]), []

    targets = [target for target in DEFAULT_TARGET_COLUMNS if target in set(residuals["target"].astype(str))]
    frames = []
    value_specs = {
        "observed": "target_value",
        "observed_log": "target_log",
        "baseline_known_site_log": "pred_known_site_log",
        "dynamic_correction_log": "residual_known_site_log",
    }
    for prefix, source_col in value_specs.items():
        if source_col not in residuals.columns:
            continue
        pivot = residuals.pivot_table(
            index="record_observed_id",
            columns="target",
            values=source_col,
            aggfunc="mean",
        )
        pivot = pivot[[target for target in targets if target in pivot.columns]]
        pivot = pivot.rename(columns={target: f"{prefix}_{target}" for target in pivot.columns})
        frames.append(pivot)

    if not frames:
        return pd.DataFrame(columns=["record_observed_id"]), targets
    wide = pd.concat(frames, axis=1).reset_index()
    for target in targets:
        baseline = f"baseline_known_site_log_{target}"
        correction = f"dynamic_correction_log_{target}"
        if baseline in wide.columns and correction in wide.columns:
            compatible_log = f"compatible_log_{target}"
            wide[compatible_log] = wide[baseline] + wide[correction]
            wide[f"compatible_{target}"] = np.exp(wide[compatible_log])
    return wide, targets


def _node_column(node_type: str, level: int) -> str:
    if node_type == "source3d":
        return f"source_cell_j{level}"
    if node_type == "receiver":
        return f"receiver_cell_j{level}"
    return f"route_id_j{level}"


def _merge_best_kozyrev(out: pd.DataFrame, kozyrev_fields: pd.DataFrame, node_type: str) -> pd.DataFrame:
    if kozyrev_fields.empty or not {"node_type", "level", "node_id"}.issubset(kozyrev_fields.columns):
        out[f"{node_type}_kozyrev_level"] = np.nan
        out[f"{node_type}_kozyrev_n_records"] = np.nan
        out[f"{node_type}_kozyrev_delta_norm"] = np.nan
        out[f"{node_type}_mode_norm"] = np.nan
        return out

    for dest in (
        f"{node_type}_kozyrev_level",
        f"{node_type}_kozyrev_n_records",
        f"{node_type}_kozyrev_delta_norm",
        f"{node_type}_mode_norm",
    ):
        out[dest] = np.nan

    for level in (4, 3, 2, 1):
        node_col = _node_column(node_type, level)
        if node_col not in out.columns:
            continue
        fields = kozyrev_fields[
            (kozyrev_fields["node_type"].astype(str) == node_type)
            & (pd.to_numeric(kozyrev_fields["level"], errors="coerce") == level)
        ].copy()
        if fields.empty:
            continue
        keep = [column for column in ["node_id", "n_records", "kozyrev_delta_norm", "mode_norm"] if column in fields.columns]
        fields = fields[keep].drop_duplicates("node_id")
        keys = pd.DataFrame(
            {
                "_row_index": out.index,
                "_node_id": node_type + ":" + out[node_col].astype(str),
            }
        )
        merged = keys.merge(fields, left_on="_node_id", right_on="node_id", how="left").set_index("_row_index")
        has_match = merged.get("node_id", pd.Series(index=merged.index, dtype=object)).notna()
        fill_mask = has_match & out.loc[merged.index, f"{node_type}_kozyrev_delta_norm"].isna()
        if not fill_mask.any():
            continue
        idx = merged.index[fill_mask]
        out.loc[idx, f"{node_type}_kozyrev_level"] = level
        if "n_records" in merged:
            out.loc[idx, f"{node_type}_kozyrev_n_records"] = pd.to_numeric(merged.loc[idx, "n_records"], errors="coerce")
        if "kozyrev_delta_norm" in merged:
            out.loc[idx, f"{node_type}_kozyrev_delta_norm"] = pd.to_numeric(merged.loc[idx, "kozyrev_delta_norm"], errors="coerce")
        if "mode_norm" in merged:
            out.loc[idx, f"{node_type}_mode_norm"] = pd.to_numeric(merged.loc[idx, "mode_norm"], errors="coerce")
    return out


def _merge_fault_candidates(out: pd.DataFrame, fault_candidates: pd.DataFrame) -> pd.DataFrame:
    if fault_candidates.empty or not {"route_id", "route_level"}.issubset(fault_candidates.columns):
        for column in ("fault_candidate_id", "fault_candidate_score", "fault_candidate_confidence", "fault_strike_deg"):
            out[column] = np.nan
        return out

    level = int(pd.to_numeric(fault_candidates["route_level"], errors="coerce").dropna().mode().iloc[0])
    route_col = f"route_id_j{level}"
    if route_col not in out.columns:
        return _merge_fault_candidates(out, pd.DataFrame())
    keep = [
        "candidate_id",
        "route_id",
        "fault_candidate_score",
        "fault_probability_pct",
        "confidence",
        "strike_deg",
        "n_records",
        "midpoint_latitude_deg",
        "midpoint_longitude_deg",
    ]
    faults = fault_candidates[[column for column in keep if column in fault_candidates.columns]].drop_duplicates("route_id")
    faults = faults.rename(
        columns={
            "candidate_id": "fault_candidate_id",
            "confidence": "fault_candidate_confidence",
            "strike_deg": "fault_strike_deg",
            "n_records": "fault_candidate_n_records",
            "midpoint_latitude_deg": "fault_midpoint_latitude_deg",
            "midpoint_longitude_deg": "fault_midpoint_longitude_deg",
        }
    )
    return out.merge(faults, left_on=route_col, right_on="route_id", how="left", suffixes=("", "_fault"))


def build_compatible_dynamics(
    geo_targets: pd.DataFrame,
    residuals: pd.DataFrame,
    modes: pd.DataFrame,
    kozyrev_fields: pd.DataFrame,
    fault_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if geo_targets.empty:
        return pd.DataFrame(), []

    base_cols = _present_columns(geo_targets, BASE_FORWARD_COLUMNS)
    for level in (1, 2, 3, 4):
        base_cols.extend(_present_columns(geo_targets, [f"source_cell_j{level}", f"route_id_j{level}", f"receiver_cell_j{level}"]))
    base_cols = list(dict.fromkeys(base_cols))
    out = _merge_modes(geo_targets[base_cols].copy(), modes)

    mode_cols = mode_columns(out)
    if mode_cols:
        out["mode_norm"] = np.sqrt(np.square(out[mode_cols].fillna(0.0)).sum(axis=1))
    else:
        out["mode_norm"] = np.nan

    residual_frame, targets = _residual_wide(residuals)
    if not residual_frame.empty:
        out = out.merge(residual_frame, on="record_observed_id", how="left")

    for node_type in ("source3d", "route", "receiver"):
        out = _merge_best_kozyrev(out, kozyrev_fields, node_type)
    out = _merge_fault_candidates(out, fault_candidates)

    if {"event_latitude_deg", "station_latitude_deg"}.issubset(out.columns):
        out["route_midpoint_latitude_deg"] = (out["event_latitude_deg"] + out["station_latitude_deg"]) / 2.0
    if {"event_longitude_deg", "station_longitude_deg"}.issubset(out.columns):
        out["route_midpoint_longitude_deg"] = (out["event_longitude_deg"] + out["station_longitude_deg"]) / 2.0

    observed_target_cols = [f"observed_{target}" for target in targets if f"observed_{target}" in out.columns]
    out["target_coverage_count"] = out[observed_target_cols].notna().sum(axis=1) if observed_target_cols else 0
    out["target_coverage_fraction"] = out["target_coverage_count"] / max(1, len(targets))

    for column in ("mode_norm", "route_kozyrev_delta_norm", "fault_candidate_score"):
        if column not in out.columns:
            out[column] = np.nan
    if "fault_probability_pct" not in out.columns:
        out["fault_probability_pct"] = np.nan
    out["dynamic_anomaly_score"] = (
        0.45 * _rank01(out["mode_norm"])
        + 0.35 * _rank01(out["route_kozyrev_delta_norm"])
        + 0.20 * _rank01(out["fault_candidate_score"])
    )

    support_counts = []
    for column in ("source3d_kozyrev_n_records", "route_kozyrev_n_records", "receiver_kozyrev_n_records", "fault_candidate_n_records"):
        if column in out.columns:
            support_counts.append(pd.to_numeric(out[column], errors="coerce"))
    support = pd.concat(support_counts, axis=1).max(axis=1) if support_counts else pd.Series(0.0, index=out.index)
    support = np.log1p(support.fillna(0.0))
    support_norm = support / support.max() if float(support.max()) > 0 else pd.Series(0.0, index=out.index)
    mask_score = pd.Series(1.0, index=out.index)
    if "route_in_chile_mask" in out.columns:
        mask_score = pd.to_numeric(out["route_in_chile_mask"], errors="coerce").fillna(0.0).astype(float)
    out["forward_support_weight"] = (
        0.50 * support_norm.astype(float)
        + 0.30 * out["target_coverage_fraction"].fillna(0.0).astype(float)
        + 0.20 * mask_score.clip(0.0, 1.0)
    ).clip(0.0, 1.0)

    source = out.get("observed_source", pd.Series("", index=out.index)).fillna("").astype(str)
    out["compatible_dynamics_status"] = np.select(
        [source.eq("h5"), source.eq("flatfile")],
        ["observed_h5_calibrated", "flatfile_conditioned"],
        default="observed_conditioned",
    )
    return out, targets


def _profile_context(
    compatible: pd.DataFrame,
    context_type: str,
    context_id_col: str,
    level: int | None,
    lat_col: str | None,
    lon_col: str | None,
) -> pd.DataFrame:
    if context_id_col not in compatible.columns:
        return pd.DataFrame()
    work = compatible[compatible[context_id_col].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    named_aggs: dict[str, tuple[str, str | Any]] = {"n_records": ("record_observed_id", "count")}
    if lat_col and lat_col in work.columns:
        named_aggs["centroid_latitude_deg"] = (lat_col, "mean")
    if lon_col and lon_col in work.columns:
        named_aggs["centroid_longitude_deg"] = (lon_col, "mean")
    for column in ("dynamic_anomaly_score", "forward_support_weight", "fault_candidate_score"):
        if column in work.columns:
            named_aggs[f"{column}_mean"] = (column, "mean")
            named_aggs[f"{column}_p90"] = (column, _q90)
    for column in mode_columns(work):
        named_aggs[f"{column}_mean"] = (column, "mean")
    for column in [c for c in work.columns if c.startswith("dynamic_correction_log_")]:
        named_aggs[f"{column}_mean"] = (column, "mean")
        named_aggs[f"{column}_std"] = (column, "std")

    profiles = work.groupby(context_id_col, dropna=False).agg(**named_aggs).reset_index()
    profiles = profiles.rename(columns={context_id_col: "context_id"})
    profiles.insert(0, "context_type", context_type)
    profiles.insert(1, "level", level)
    return profiles


def build_forward_conditioning_profiles(compatible: pd.DataFrame) -> pd.DataFrame:
    if compatible.empty:
        return pd.DataFrame()
    frames = []
    for level in (2, 3, 4):
        frames.append(_profile_context(compatible, "source3d", f"source_cell_j{level}", level, "event_latitude_deg", "event_longitude_deg"))
        frames.append(_profile_context(compatible, "route", f"route_id_j{level}", level, "route_midpoint_latitude_deg", "route_midpoint_longitude_deg"))
        frames.append(_profile_context(compatible, "receiver", f"receiver_cell_j{level}", level, "station_latitude_deg", "station_longitude_deg"))
    frames.append(_profile_context(compatible, "fault_candidate", "fault_candidate_id", None, "fault_midpoint_latitude_deg", "fault_midpoint_longitude_deg"))
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _forward_schema(targets: list[str], compatible: pd.DataFrame, profiles: pd.DataFrame) -> dict[str, Any]:
    dynamic_cols = [column for column in compatible.columns if column.startswith("dynamic_correction_log_")]
    return {
        "status": "observed_conditioning_ready",
        "purpose": "Dinamica compatible fuente-ruta-receptor para forward condicionado posterior.",
        "primary_sources": ["records/*.h5", "records/flatfiles/*.csv"],
        "products": {
            "compatible_dynamics": "compatible_dynamics.parquet",
            "conditioning_profiles": "forward_conditioning_profiles.parquet",
            "fault_candidates": "fault_candidates.parquet",
        },
        "forward_equation_log_space": (
            "log(target_forward) = baseline_source_distance_site_log "
            "+ dynamic_correction_log(context) + optional latent/kozyrev/fault adjustment"
        ),
        "conditioning_inputs": {
            "geometry": [column for column in BASE_FORWARD_COLUMNS if column in compatible.columns],
            "latent_modes": mode_columns(compatible),
            "kozyrev_fields": [
                column
                for column in compatible.columns
                if column.endswith("_kozyrev_delta_norm") or column.endswith("_mode_norm")
            ],
            "fault_candidates": [
                column
                for column in [
                    "fault_candidate_id",
                    "fault_candidate_score",
                    "fault_probability_pct",
                    "fault_candidate_confidence",
                    "fault_strike_deg",
                ]
                if column in compatible.columns
            ],
            "target_dynamics": dynamic_cols,
            "targets_available": targets,
        },
        "rows": {
            "compatible_dynamics": int(compatible.shape[0]),
            "conditioning_profiles": int(profiles.shape[0]),
        },
        "guards": [
            "No usar parquets historicos como fuente primaria.",
            "No importar dependencias internas de GMPE, Modelo E ni TSD estructural.",
            "Usar los perfiles como condicionantes observados, no como catalogo oficial de fallas.",
            "Validar fuera de muestra antes de convertirlo en simulador predictivo.",
        ],
    }


def write_forward_products(
    geo_targets: pd.DataFrame,
    residuals: pd.DataFrame,
    modes: pd.DataFrame,
    kozyrev_fields: pd.DataFrame,
    fault_candidates: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    compatible, targets = build_compatible_dynamics(geo_targets, residuals, modes, kozyrev_fields, fault_candidates)
    profiles = build_forward_conditioning_profiles(compatible)
    write_parquet(compatible, output_dir / "compatible_dynamics.parquet")
    write_parquet(profiles, output_dir / "forward_conditioning_profiles.parquet")
    write_json(output_dir / "forward_conditioning_template.json", _forward_schema(targets, compatible, profiles))
    return compatible, profiles


def _scenario_frame(
    scenario_name: str,
    receiver_latitude_deg: float,
    receiver_longitude_deg: float,
    source_distance_km: float,
    source_bearing_deg: float,
    mw: float,
    vs30_m_s: float,
    depth_km: float,
    tectonic_type: str,
    source_latitude_deg: float | None = None,
    source_longitude_deg: float | None = None,
) -> pd.DataFrame:
    if source_latitude_deg is not None and source_longitude_deg is not None:
        source_lat = float(source_latitude_deg)
        source_lon = float(source_longitude_deg)
        source_distance_km = haversine_km(source_lat, source_lon, receiver_latitude_deg, receiver_longitude_deg)
        source_bearing_deg = azimuth_deg(receiver_latitude_deg, receiver_longitude_deg, source_lat, source_lon)
    else:
        source_lat, source_lon = destination_point(
            receiver_latitude_deg,
            receiver_longitude_deg,
            source_bearing_deg,
            source_distance_km,
        )
    rhyp = math.sqrt(source_distance_km * source_distance_km + depth_km * depth_km)
    base = pd.DataFrame(
        [
            {
                "record_observed_id": scenario_name,
                "observed_source": "scenario_forward",
                "event_id": scenario_name,
                "station_id": f"{scenario_name}_receiver",
                "event_latitude_deg": source_lat,
                "event_longitude_deg": source_lon,
                "event_depth_km": depth_km,
                "mw": mw,
                "tectonic_type": tectonic_type,
                "station_latitude_deg": receiver_latitude_deg,
                "station_longitude_deg": receiver_longitude_deg,
                "vs30_m_s": vs30_m_s,
                "rrup_km_flatfile": rhyp,
                "rhyp_km_flatfile": rhyp,
                "repi_km_flatfile": source_distance_km,
                "receiver_in_chile_mask": True,
                "route_in_chile_mask": True,
            }
        ]
    )
    scenario = add_geometry(base)
    scenario["source3d_id"] = "source3d:" + scenario["source_cell_j4"].astype(str)
    scenario["receiver_id"] = "receiver:" + scenario["receiver_cell_j4"].astype(str)
    scenario["route_id"] = scenario["route_id_j4"]
    return scenario


def _scenario_analogs(scenario: pd.Series, compatible: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    if compatible.empty:
        return pd.DataFrame()
    work = compatible.copy()
    work["scenario_distance_score"] = (
        pd.to_numeric(work.get("distance_km"), errors="coerce").fillna(float(scenario.distance_km))
        .add(1.0)
        .pipe(np.log)
        .sub(math.log(float(scenario.distance_km) + 1.0))
        .abs()
        / 0.55
    )
    if "mw" in work.columns:
        work["scenario_mw_score"] = (pd.to_numeric(work["mw"], errors="coerce").fillna(float(scenario.mw)) - float(scenario.mw)).abs() / 1.0
    else:
        work["scenario_mw_score"] = 0.0
    if "vs30_m_s" in work.columns:
        vs30 = pd.to_numeric(work["vs30_m_s"], errors="coerce").fillna(float(scenario.vs30_m_s)).clip(lower=80.0)
        work["scenario_vs30_score"] = (np.log(vs30) - math.log(float(scenario.vs30_m_s))).abs() / 0.55
    else:
        work["scenario_vs30_score"] = 0.0
    if "backazimuth_deg" in work.columns:
        work["scenario_direction_score"] = _angular_diff_deg(work["backazimuth_deg"], float(scenario.backazimuth_deg)).fillna(90.0) / 90.0
    else:
        work["scenario_direction_score"] = 0.0

    source_bonus = pd.Series(0.0, index=work.index)
    receiver_bonus = pd.Series(0.0, index=work.index)
    route_bonus = pd.Series(0.0, index=work.index)
    for level in (4, 3, 2, 1):
        source_col = f"source_cell_j{level}"
        receiver_col = f"receiver_cell_j{level}"
        route_col = f"route_id_j{level}"
        if source_col in work.columns and source_col in scenario:
            source_bonus = source_bonus.where(~work[source_col].astype(str).eq(str(scenario[source_col])), 1.0 / level)
        if receiver_col in work.columns and receiver_col in scenario:
            receiver_bonus = receiver_bonus.where(~work[receiver_col].astype(str).eq(str(scenario[receiver_col])), 1.0 / level)
        if route_col in work.columns and route_col in scenario:
            route_bonus = route_bonus.where(~work[route_col].astype(str).eq(str(scenario[route_col])), 1.0 / level)

    support = pd.to_numeric(work.get("forward_support_weight", pd.Series(0.5, index=work.index)), errors="coerce").fillna(0.5)
    work["scenario_similarity_score"] = (
        0.34 * work["scenario_distance_score"].fillna(1.0)
        + 0.24 * work["scenario_mw_score"].fillna(1.0)
        + 0.18 * work["scenario_vs30_score"].fillna(1.0)
        + 0.16 * work["scenario_direction_score"].fillna(1.0)
        - 0.05 * source_bonus
        - 0.05 * receiver_bonus
        - 0.08 * route_bonus
    )
    work["scenario_analog_weight"] = np.exp(-work["scenario_similarity_score"].clip(lower=0.0)) * (0.20 + 0.80 * support.clip(0.0, 1.0))
    sort_cols = ["scenario_analog_weight", "forward_support_weight"] if "forward_support_weight" in work.columns else ["scenario_analog_weight"]
    return work.sort_values(sort_cols, ascending=False).head(top_n).reset_index(drop=True)


def _scenario_result_rows(scenario: pd.Series, analogs: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    target_cols = [
        column
        for column in analogs.columns
        if column.startswith("compatible_")
        and not column.startswith("compatible_log_")
        and column != "compatible_dynamics_status"
    ]
    nearest_fault_probability = float("nan")
    nearest_fault_id = None
    if not faults.empty:
        nearest_fault_probability = float(pd.to_numeric(faults.iloc[0].get("fault_probability_pct"), errors="coerce"))
        nearest_fault_id = faults.iloc[0].get("candidate_id")

    rows = []
    for column in target_cols:
        target = column.replace("compatible_", "", 1)
        values = pd.to_numeric(analogs[column], errors="coerce")
        weights = pd.to_numeric(analogs.get("scenario_analog_weight", pd.Series(1.0, index=analogs.index)), errors="coerce")
        if values.notna().sum() == 0:
            continue
        rows.append(
            {
                "scenario_name": scenario.record_observed_id,
                "target": target,
                "forward_p16": _weighted_quantile(values, weights, 0.16),
                "forward_p50": _weighted_quantile(values, weights, 0.50),
                "forward_p84": _weighted_quantile(values, weights, 0.84),
                "forward_weighted_mean": _weighted_mean(values, weights),
                "n_analogs": int(values.notna().sum()),
                "analog_weight_sum": float(weights.fillna(0.0).sum()),
                "source_distance_km": float(scenario.repi_km_calc),
                "hypocentral_distance_km": float(scenario.distance_km),
                "mw": float(scenario.mw),
                "vs30_m_s": float(scenario.vs30_m_s),
                "source_bearing_from_receiver_deg": float(scenario.backazimuth_deg),
                "nearest_fault_candidate_id": nearest_fault_id,
                "nearest_fault_probability_pct": nearest_fault_probability,
                "method": "weighted_observed_analogs_conditioned_by_geometry_mw_vs30_direction",
            }
        )
    return pd.DataFrame(rows)


def _nearest_faults(scenario: pd.Series, fault_candidates: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if fault_candidates.empty:
        return pd.DataFrame()
    required = {"midpoint_latitude_deg", "midpoint_longitude_deg"}
    if not required.issubset(fault_candidates.columns):
        return pd.DataFrame()
    faults = fault_candidates.copy()
    scenario_mid_lat = (float(scenario.event_latitude_deg) + float(scenario.station_latitude_deg)) / 2.0
    scenario_mid_lon = (float(scenario.event_longitude_deg) + float(scenario.station_longitude_deg)) / 2.0
    faults["scenario_midpoint_distance_km"] = [
        haversine_km(scenario_mid_lat, scenario_mid_lon, lat, lon)
        for lat, lon in zip(faults["midpoint_latitude_deg"], faults["midpoint_longitude_deg"])
    ]
    if "strike_deg" in faults.columns:
        faults["scenario_strike_delta_deg"] = _angular_diff_deg(faults["strike_deg"], float(scenario.backazimuth_deg))
    else:
        faults["scenario_strike_delta_deg"] = np.nan
    probability = pd.to_numeric(faults.get("fault_probability_pct", pd.Series(0.0, index=faults.index)), errors="coerce").fillna(0.0)
    distance_score = pd.to_numeric(faults["scenario_midpoint_distance_km"], errors="coerce").fillna(9999.0) / 250.0
    strike_score = pd.to_numeric(faults["scenario_strike_delta_deg"], errors="coerce").fillna(90.0) / 90.0
    faults["scenario_fault_match_score"] = distance_score + 0.35 * strike_score - 0.012 * probability
    return faults.sort_values("scenario_fault_match_score").head(top_n).reset_index(drop=True)


def _scenario_geojson(scenario: pd.Series, faults: pd.DataFrame) -> list[dict[str, Any]]:
    source = [float(scenario.event_longitude_deg), float(scenario.event_latitude_deg)]
    receiver = [float(scenario.station_longitude_deg), float(scenario.station_latitude_deg)]
    features: list[dict[str, Any]] = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": source},
            "properties": {
                "feature_type": "scenario_source",
                "scenario_name": scenario.record_observed_id,
                "mw": float(scenario.mw),
                "depth_km": float(scenario.event_depth_km),
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": receiver},
            "properties": {
                "feature_type": "scenario_receiver",
                "scenario_name": scenario.record_observed_id,
                "vs30_m_s": float(scenario.vs30_m_s),
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [source, receiver]},
            "properties": {
                "feature_type": "scenario_route",
                "scenario_name": scenario.record_observed_id,
                "distance_km": float(scenario.distance_km),
                "backazimuth_deg": float(scenario.backazimuth_deg),
            },
        },
    ]
    for _, row in faults.head(10).iterrows():
        coords = [
            row.get("event_longitude_deg_mean"),
            row.get("event_latitude_deg_mean"),
            row.get("station_longitude_deg_mean"),
            row.get("station_latitude_deg_mean"),
        ]
        if not all(np.isfinite(coords)):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[float(coords[0]), float(coords[1])], [float(coords[2]), float(coords[3])]]},
                "properties": {
                    "feature_type": "scenario_nearest_fault_candidate",
                    "candidate_id": row.get("candidate_id"),
                    "fault_probability_pct": float(row.get("fault_probability_pct", np.nan)),
                    "scenario_fault_match_score": float(row.get("scenario_fault_match_score", np.nan)),
                },
            }
        )
    return features


def write_forward_scenario(
    output_dir: Path,
    scenario_name: str = "santiago_sw_m75",
    receiver_latitude_deg: float = -33.4489,
    receiver_longitude_deg: float = -70.6693,
    source_distance_km: float = 100.0,
    source_direction: str | None = "suroeste",
    source_bearing_deg: float | None = None,
    source_latitude_deg: float | None = None,
    source_longitude_deg: float | None = None,
    mw: float = 7.5,
    vs30_m_s: float = 600.0,
    depth_km: float = 30.0,
    tectonic_type: str = "scenario",
    analog_top_n: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    compatible_path = output_dir / "compatible_dynamics.parquet"
    profiles_path = output_dir / "forward_conditioning_profiles.parquet"
    faults_path = output_dir / "fault_candidates.parquet"
    if not compatible_path.exists():
        raise FileNotFoundError(f"Falta {compatible_path}. Ejecuta primero tsd-suelo forward o build.")
    if (source_latitude_deg is None) != (source_longitude_deg is None):
        raise ValueError("Para fuente directa debes entregar --source-lat y --source-lon juntos.")
    compatible = pd.read_parquet(compatible_path)
    profiles = pd.read_parquet(profiles_path) if profiles_path.exists() else pd.DataFrame()
    fault_candidates = pd.read_parquet(faults_path) if faults_path.exists() else pd.DataFrame()

    has_direct_source = source_latitude_deg is not None and source_longitude_deg is not None
    if has_direct_source and source_bearing_deg is None and not str(source_direction or "").strip():
        bearing = 0.0
    else:
        bearing = direction_to_bearing(source_direction, source_bearing_deg)
    scenario_frame = _scenario_frame(
        scenario_name=scenario_name,
        receiver_latitude_deg=receiver_latitude_deg,
        receiver_longitude_deg=receiver_longitude_deg,
        source_distance_km=source_distance_km,
        source_bearing_deg=bearing,
        mw=mw,
        vs30_m_s=vs30_m_s,
        depth_km=depth_km,
        tectonic_type=tectonic_type,
        source_latitude_deg=source_latitude_deg,
        source_longitude_deg=source_longitude_deg,
    )
    scenario = scenario_frame.iloc[0]
    analogs = _scenario_analogs(scenario, compatible, top_n=analog_top_n)
    nearest_faults = _nearest_faults(scenario, fault_candidates)
    result = _scenario_result_rows(scenario, analogs, nearest_faults)
    matching_profiles = profiles[
        profiles.get("context_id", pd.Series(dtype=object)).astype(str).isin(
            {
                str(scenario.get("source_cell_j2")),
                str(scenario.get("source_cell_j3")),
                str(scenario.get("source_cell_j4")),
                str(scenario.get("receiver_cell_j2")),
                str(scenario.get("receiver_cell_j3")),
                str(scenario.get("receiver_cell_j4")),
                str(scenario.get("route_id_j2")),
                str(scenario.get("route_id_j3")),
                str(scenario.get("route_id_j4")),
            }
        )
    ].copy() if not profiles.empty and "context_id" in profiles.columns else pd.DataFrame()

    write_parquet(result, output_dir / "forward_scenario_result.parquet")
    result.to_csv(output_dir / "forward_scenario_result.csv", index=False)
    analog_cols = [
        column
        for column in [
            "record_observed_id",
            "observed_source",
            "event_id",
            "station_id",
            "distance_km",
            "mw",
            "vs30_m_s",
            "backazimuth_deg",
            "dynamic_anomaly_score",
            "forward_support_weight",
            "scenario_analog_weight",
            "scenario_similarity_score",
        ]
        if column in analogs.columns
    ]
    analogs[analog_cols].to_csv(output_dir / "forward_scenario_analogs.csv", index=False)
    nearest_faults.to_csv(output_dir / "forward_scenario_faults.csv", index=False)
    if not matching_profiles.empty:
        matching_profiles.to_csv(output_dir / "forward_scenario_profiles.csv", index=False)
    write_geojson(_scenario_geojson(scenario, nearest_faults), output_dir / "forward_scenario.geojson")

    scenario_input = {
        "scenario_name": scenario_name,
        "receiver_latitude_deg": receiver_latitude_deg,
        "receiver_longitude_deg": receiver_longitude_deg,
        "source_latitude_deg": float(scenario.event_latitude_deg),
        "source_longitude_deg": float(scenario.event_longitude_deg),
        "source_distance_km": float(scenario.repi_km_calc),
        "source_distance_km_requested": float(source_distance_km),
        "hypocentral_distance_km": float(scenario.distance_km),
        "source_direction": source_direction,
        "source_bearing_from_receiver_deg": float(scenario.backazimuth_deg),
        "mw": mw,
        "vs30_m_s": vs30_m_s,
        "depth_km": depth_km,
        "tectonic_type": tectonic_type,
        "route_id_j4": str(scenario.route_id_j4),
        "receiver_cell_j4": str(scenario.receiver_cell_j4),
        "source_cell_j4": str(scenario.source_cell_j4),
        "n_analogs": int(analogs.shape[0]),
        "n_matching_profiles": int(matching_profiles.shape[0]),
        "n_nearest_faults": int(nearest_faults.shape[0]),
        "method": "weighted observed analogs plus compatible dynamics products",
    }
    write_json(output_dir / "forward_scenario_input.json", scenario_input)
    manifest = {
        "mode": "scenario_forward",
        "scenario": scenario_input,
        "rows": {
            "result_targets": int(result.shape[0]),
            "analogs": int(analogs.shape[0]),
            "nearest_faults": int(nearest_faults.shape[0]),
            "matching_profiles": int(matching_profiles.shape[0]),
        },
        "products": [
            "forward_scenario_input.json",
            "forward_scenario_result.parquet",
            "forward_scenario_result.csv",
            "forward_scenario_analogs.csv",
            "forward_scenario_faults.csv",
            "forward_scenario_profiles.csv",
            "forward_scenario.geojson",
        ],
    }
    write_json(output_dir / "forward_scenario_manifest.json", manifest)
    return result, analogs, nearest_faults, manifest


def write_forward_template(geo_targets: pd.DataFrame, modes: pd.DataFrame, output_dir: Path) -> None:
    template = {
        "status": "prepared_contract_only",
        "purpose": "Forward condicionado posterior, no generacion sintetica inicial.",
        "primary_sources": ["records/*.h5", "records/flatfiles/*.csv"],
        "conditioning_inputs": {
            "geometry": [column for column in BASE_FORWARD_COLUMNS if column in geo_targets.columns],
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
