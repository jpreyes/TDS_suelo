from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def _rank01(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    return numeric.fillna(numeric.median()).rank(pct=True).astype(float)


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
