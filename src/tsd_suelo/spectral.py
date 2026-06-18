from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import math
import time

import h5py
import numpy as np
import pandas as pd

from .atlas import write_geojson, write_kmz
from .geometry import cell_id
from .logging_utils import format_seconds
from .spatial_grid import DEFAULT_BASE_STEP_DEG, _cell_indices
from .utils import as_clean_str, ensure_dir, safe_float, write_json, write_parquet


DEFAULT_FREQ_GRID_HZ = tuple(float(v) for v in np.geomspace(0.1, 50.0, 64))
SPECTRAL_LEVELS = tuple(range(1, 13))


def spectral_columns(freq_count: int = len(DEFAULT_FREQ_GRID_HZ)) -> list[str]:
    return [f"log_amp_f{idx:03d}" for idx in range(freq_count)]


def _rank01(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    filled = numeric.fillna(numeric.median())
    if float(filled.max()) <= 0.0 and float(filled.min()) >= 0.0:
        return pd.Series(0.0, index=values.index)
    return filled.rank(pct=True).astype(float)


def _rank_pct_by_level(frame: pd.DataFrame, value_col: str) -> pd.Series:
    if frame.empty or value_col not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)
    return 100.0 * frame.groupby("level", group_keys=False)[value_col].apply(_rank01)


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def _metadata_scalar(h5: h5py.File, group: str, name: str) -> Any:
    path = f"metadata/{group}/{name}"
    if path not in h5:
        return None
    value = h5[path][()]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    return value


def _component_acc(h5: h5py.File, component: str) -> np.ndarray:
    for root in ("Processed_data", "Unprocessed_data"):
        path = f"{root}/{component}_acc"
        if path in h5:
            data = np.asarray(h5[path][()], dtype=float).reshape(-1)
            return data[np.isfinite(data)]
    return np.asarray([], dtype=float)


def _horizontal(e_acc: np.ndarray, n_acc: np.ndarray) -> np.ndarray:
    size = min(e_acc.size, n_acc.size)
    if size == 0:
        return np.asarray([], dtype=float)
    return np.sqrt(np.square(e_acc[:size]) + np.square(n_acc[:size]))


def _parse_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "_" not in stem:
        return stem, ""
    event_id, station_id = stem.split("_", 1)
    return event_id, station_id


def _spectral_signature(acc: np.ndarray, dt: float, freq_grid: np.ndarray) -> np.ndarray:
    if acc.size < 8 or not np.isfinite(dt) or dt <= 0:
        return np.full(freq_grid.shape, np.nan, dtype=float)
    signal = acc.astype(float) - float(np.nanmean(acc))
    signal = signal * np.hanning(signal.size)
    freqs = np.fft.rfftfreq(signal.size, d=dt)
    amp = np.abs(np.fft.rfft(signal)) / max(signal.size, 1)
    keep = (freqs > 0.0) & np.isfinite(amp) & (amp >= 0.0)
    if int(keep.sum()) < 3:
        return np.full(freq_grid.shape, np.nan, dtype=float)
    freqs = freqs[keep]
    log_amp = np.log1p(amp[keep])
    interp = np.interp(freq_grid, freqs, log_amp, left=np.nan, right=np.nan)
    nyquist = 0.5 / dt
    interp[freq_grid > nyquist] = np.nan
    return interp


def _read_spectral_record(args: tuple[str, tuple[float, ...]]) -> dict[str, Any]:
    path = Path(args[0])
    freq_grid = np.asarray(args[1], dtype=float)
    event_from_name, station_from_name = _parse_filename(path)
    try:
        with h5py.File(path, "r") as h5:
            dt = safe_float(_metadata_scalar(h5, "record", "dt"))
            station_id = as_clean_str(_metadata_scalar(h5, "record", "Sta_Name")) or station_from_name
            event_id_h5 = as_clean_str(_metadata_scalar(h5, "event", "EventID_BM16"))
            e_acc = _component_acc(h5, "E")
            n_acc = _component_acc(h5, "N")
            h_acc = _horizontal(e_acc, n_acc)
        signature = _spectral_signature(h_acc, dt, freq_grid)
        row: dict[str, Any] = {
            "h5_file": str(path),
            "h5_name": path.name,
            "record_observed_id": f"{event_from_name}_{station_from_name}",
            "event_id": event_from_name,
            "h5_event_id_bm16": event_id_h5,
            "station_id": station_id,
            "dt_s": dt,
            "sample_rate_hz": float(1.0 / dt) if dt and dt > 0 else math.nan,
            "spectral_read_ok": True,
            "spectral_read_error": None,
        }
        for idx, value in enumerate(signature):
            row[f"log_amp_f{idx:03d}"] = float(value) if np.isfinite(value) else math.nan
        return row
    except Exception as exc:
        row = {
            "h5_file": str(path),
            "h5_name": path.name,
            "record_observed_id": f"{event_from_name}_{station_from_name}",
            "event_id": event_from_name,
            "station_id": station_from_name,
            "spectral_read_ok": False,
            "spectral_read_error": f"{type(exc).__name__}: {exc}",
        }
        for idx in range(len(freq_grid)):
            row[f"log_amp_f{idx:03d}"] = math.nan
        return row


def _resolve_h5_paths(waveform_targets: pd.DataFrame, records_dir: Path) -> list[Path]:
    if waveform_targets.empty:
        return []
    paths = []
    for row in waveform_targets.itertuples(index=False):
        candidate = Path(str(getattr(row, "h5_file", "")))
        if candidate.exists():
            paths.append(candidate)
            continue
        h5_name = getattr(row, "h5_name", None)
        if h5_name:
            fallback = records_dir / str(h5_name)
            if fallback.exists():
                paths.append(fallback)
    return sorted(set(paths))


def build_spectral_record_signatures(
    waveform_targets: pd.DataFrame,
    records_dir: Path,
    workers: int = 1,
    progress_every: int = 500,
    log: Any | None = None,
    freq_grid_hz: tuple[float, ...] = DEFAULT_FREQ_GRID_HZ,
) -> pd.DataFrame:
    paths = _resolve_h5_paths(waveform_targets, records_dir)
    if not paths:
        return pd.DataFrame()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    total = len(paths)
    if log:
        log(f"Spectral H5 total={total} workers={workers} freq_bins={len(freq_grid_hz)}")
    if workers <= 1:
        for idx, path in enumerate(paths, start=1):
            rows.append(_read_spectral_record((str(path), freq_grid_hz)))
            if log and (idx % progress_every == 0 or idx == total):
                elapsed = time.perf_counter() - started
                rate = idx / max(elapsed, 1e-9)
                eta = (total - idx) / rate if rate > 0 else math.nan
                log(f"Spectral progreso {idx}/{total} ({100*idx/total:.1f}%) | {rate:.2f} H5/s | ETA {format_seconds(eta)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_read_spectral_record, (str(path), freq_grid_hz)) for path in paths]
            for idx, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if log and (idx % progress_every == 0 or idx == total):
                    elapsed = time.perf_counter() - started
                    rate = idx / max(elapsed, 1e-9)
                    eta = (total - idx) / rate if rate > 0 else math.nan
                    log(f"Spectral progreso {idx}/{total} ({100*idx/total:.1f}%) | {rate:.2f} H5/s | ETA {format_seconds(eta)}")
    out = pd.DataFrame(rows)
    if "record_observed_id" in out.columns:
        out = out.sort_values("record_observed_id").reset_index(drop=True)
    return out


def _cell_frame(samples: pd.DataFrame, level: int, base_step_deg: float) -> pd.DataFrame:
    work = samples.copy()
    work["level"] = level
    work["grid_step_deg"] = base_step_deg / (2 ** (level - 1))
    work["grid_x"], work["grid_y"], step = _cell_indices(work["sample_latitude_deg"], work["sample_longitude_deg"], level, base_step_deg)
    work = work[work["grid_x"].notna() & work["grid_y"].notna()].copy()
    work["grid_x"] = work["grid_x"].astype(int)
    work["grid_y"] = work["grid_y"].astype(int)
    work["cell_id"] = "J" + str(level) + ":x" + work["grid_x"].astype(str) + ":y" + work["grid_y"].astype(str)
    work["node_id"] = "spectral:" + work["cell_id"]
    work["lon_min_deg"] = work["grid_x"] * step - 180.0
    work["lon_max_deg"] = work["lon_min_deg"] + step
    work["lat_min_deg"] = work["grid_y"] * step - 90.0
    work["lat_max_deg"] = work["lat_min_deg"] + step
    work["center_longitude_deg"] = work["lon_min_deg"] + step / 2.0
    work["center_latitude_deg"] = work["lat_min_deg"] + step / 2.0
    return work


def build_spectral_node_dynamics(
    geo_targets: pd.DataFrame,
    spectral_records: pd.DataFrame,
    levels: tuple[int, ...] = SPECTRAL_LEVELS,
    base_step_deg: float = DEFAULT_BASE_STEP_DEG,
) -> pd.DataFrame:
    if geo_targets.empty or spectral_records.empty:
        return pd.DataFrame()
    freq_cols = [column for column in spectral_columns() if column in spectral_records.columns]
    if not freq_cols:
        return pd.DataFrame()
    keep = ["record_observed_id", "event_id", "station_id", "station_latitude_deg", "station_longitude_deg"]
    keep = [column for column in keep if column in geo_targets.columns]
    samples = geo_targets[keep].merge(spectral_records[["record_observed_id"] + freq_cols], on="record_observed_id", how="inner")
    if not {"station_latitude_deg", "station_longitude_deg"}.issubset(samples.columns):
        return pd.DataFrame()
    samples = samples.rename(columns={"station_latitude_deg": "sample_latitude_deg", "station_longitude_deg": "sample_longitude_deg"})
    samples = samples.dropna(subset=["sample_latitude_deg", "sample_longitude_deg"])
    if samples.empty:
        return pd.DataFrame()

    frames = []
    for level in levels:
        work = _cell_frame(samples, level, base_step_deg)
        agg: dict[str, Any] = {
            "record_observed_id": pd.Series.nunique,
            "event_id": pd.Series.nunique if "event_id" in work.columns else "size",
            "station_id": pd.Series.nunique if "station_id" in work.columns else "size",
            "grid_step_deg": "first",
            "grid_x": "first",
            "grid_y": "first",
            "lon_min_deg": "first",
            "lon_max_deg": "first",
            "lat_min_deg": "first",
            "lat_max_deg": "first",
            "center_longitude_deg": "first",
            "center_latitude_deg": "first",
        }
        for column in freq_cols:
            agg[column] = "mean"
        grouped = (
            work.groupby(["level", "cell_id", "node_id"], dropna=False)
            .agg(agg)
            .rename(columns={"record_observed_id": "n_records", "event_id": "n_events", "station_id": "n_stations"})
            .reset_index()
        )
        grouped["node_type"] = "spectral_cell"
        frames.append(grouped)
    nodes = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if nodes.empty:
        return nodes
    values = nodes[freq_cols].astype(float)
    nodes["spectral_energy_mean"] = values.mean(axis=1, skipna=True)
    nodes["spectral_energy_std"] = values.std(axis=1, skipna=True)
    nodes["spectral_shape_norm"] = np.sqrt(np.square(values.subtract(values.mean(axis=1), axis=0).fillna(0.0)).sum(axis=1))
    nodes["support_probability_pct"] = _rank_pct_by_level(nodes.assign(log_support=np.log1p(pd.to_numeric(nodes["n_records"], errors="coerce"))), "log_support")
    nodes["spectral_energy_probability_pct"] = _rank_pct_by_level(nodes, "spectral_energy_mean")
    nodes["spectral_shape_probability_pct"] = _rank_pct_by_level(nodes, "spectral_shape_norm")
    nodes["spectral_dynamic_probability_pct"] = (
        0.50 * nodes["spectral_shape_probability_pct"].fillna(0.0)
        + 0.30 * nodes["spectral_energy_probability_pct"].fillna(0.0)
        + 0.20 * nodes["support_probability_pct"].fillna(0.0)
    ).clip(0.0, 100.0)
    nodes["probability_basis"] = "empirical_full_spectrum_grid_percentile"
    return nodes.sort_values(["level", "grid_y", "grid_x"]).reset_index(drop=True)


def _neighbor_pairs(nodes_level: pd.DataFrame) -> list[tuple[int, int, str, float]]:
    lookup = {(int(row.grid_x), int(row.grid_y)): idx for idx, row in nodes_level.iterrows()}
    offsets = [(1, 0, "east_west", 90.0), (0, 1, "north_south", 0.0), (1, 1, "diagonal_ne_sw", 45.0), (1, -1, "diagonal_nw_se", 135.0)]
    pairs = []
    for idx, row in nodes_level.iterrows():
        x = int(row.grid_x)
        y = int(row.grid_y)
        for dx, dy, orientation, azimuth in offsets:
            other = lookup.get((x + dx, y + dy))
            if other is not None:
                pairs.append((idx, other, orientation, azimuth))
    return pairs


def build_spectral_edge_transmissibility(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame()
    freq_cols = [column for column in spectral_columns() if column in nodes.columns]
    if not freq_cols:
        return pd.DataFrame()
    rows = []
    for level, nodes_level in nodes.groupby("level", sort=True):
        nodes_level = nodes_level.reset_index(drop=True)
        for idx_a, idx_b, orientation, azimuth in _neighbor_pairs(nodes_level):
            a = nodes_level.loc[idx_a]
            b = nodes_level.loc[idx_b]
            av = a[freq_cols].astype(float)
            bv = b[freq_cols].astype(float)
            diff = av.fillna(av.median()) - bv.fillna(bv.median())
            jump = float(np.sqrt(np.square(diff.fillna(0.0)).sum()))
            ratio = bv - av
            row = {
                "level": int(level),
                "edge_id": f"spectral_edge:J{int(level)}:{a.cell_id}->{b.cell_id}",
                "from_node": a.node_id,
                "to_node": b.node_id,
                "from_cell_id": a.cell_id,
                "to_cell_id": b.cell_id,
                "neighbor_orientation": orientation,
                "neighbor_azimuth_deg": azimuth,
                "neighbor_kind": "diagonal" if "diagonal" in orientation else "rook",
                "from_longitude_deg": float(a.center_longitude_deg),
                "from_latitude_deg": float(a.center_latitude_deg),
                "to_longitude_deg": float(b.center_longitude_deg),
                "to_latitude_deg": float(b.center_latitude_deg),
                "from_spectral_dynamic_probability_pct": float(a.spectral_dynamic_probability_pct),
                "to_spectral_dynamic_probability_pct": float(b.spectral_dynamic_probability_pct),
                "spectral_jump_norm": jump,
                "spectral_transfer_log_mean": float(ratio.mean(skipna=True)),
                "spectral_transfer_log_std": float(ratio.std(skipna=True)),
                "min_n_records": int(min(a.n_records, b.n_records)),
                "mean_n_records": float((a.n_records + b.n_records) / 2.0),
            }
            for idx, column in enumerate(freq_cols):
                row[f"transfer_log_f{idx:03d}"] = float(ratio[column]) if np.isfinite(ratio[column]) else math.nan
            rows.append(row)
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    edges["spectral_jump_probability_pct"] = _rank_pct_by_level(edges, "spectral_jump_norm")
    edges["support_probability_pct"] = _rank_pct_by_level(edges.assign(log_support=np.log1p(pd.to_numeric(edges["min_n_records"], errors="coerce"))), "log_support")
    edges["spectral_transfer_probability_pct"] = (
        0.78 * edges["spectral_jump_probability_pct"].fillna(0.0)
        + 0.22 * edges["support_probability_pct"].fillna(0.0)
    ).clip(0.0, 100.0)
    edges["edge_probability_pct"] = edges["spectral_transfer_probability_pct"]
    edges["probability_basis"] = "empirical_full_spectrum_neighbor_jump_percentile"
    edges["edge_family"] = "spectral_equivalent_structure"
    return edges.sort_values(["level", "spectral_transfer_probability_pct"], ascending=[True, False]).reset_index(drop=True)


def build_spectral_modes(nodes: pd.DataFrame, n_modes: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    if nodes.empty:
        return pd.DataFrame(), pd.DataFrame()
    freq_cols = [column for column in spectral_columns() if column in nodes.columns]
    work = nodes[nodes["level"] == nodes["level"].max()].copy()
    if work.shape[0] < 2 or not freq_cols:
        return pd.DataFrame(), pd.DataFrame()
    x = work[freq_cols].astype(float)
    med = x.median(axis=0, skipna=True).fillna(0.0)
    x = x.fillna(med)
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    z = (x - means) / stds
    u, s, vt = np.linalg.svd(z.to_numpy(dtype=float), full_matrices=False)
    rank = int(min(n_modes, len(s), z.shape[0], z.shape[1]))
    if rank == 0:
        return pd.DataFrame(), pd.DataFrame()
    scores = u[:, :rank] * s[:rank]
    total_var = float(np.sum(s * s))
    explained = (s[:rank] * s[:rank] / total_var) if total_var > 0 else np.full(rank, math.nan)
    mode_frame = work[["node_id", "cell_id", "level", "center_latitude_deg", "center_longitude_deg", "n_records"]].copy()
    for idx in range(rank):
        mode_frame[f"spectral_mode_{idx + 1}"] = scores[:, idx]
        mode_frame[f"spectral_mode_{idx + 1}_explained_variance"] = float(explained[idx])
    components = []
    for mode_idx in range(rank):
        for freq_idx, column in enumerate(freq_cols):
            components.append(
                {
                    "mode": f"spectral_mode_{mode_idx + 1}",
                    "frequency_hz": DEFAULT_FREQ_GRID_HZ[freq_idx],
                    "spectral_column": column,
                    "loading": float(vt[mode_idx, freq_idx]),
                    "explained_variance": float(explained[mode_idx]),
                }
            )
    return mode_frame, pd.DataFrame(components)


def _display_level(nodes: pd.DataFrame, edges: pd.DataFrame) -> int | None:
    if not edges.empty:
        counts = edges.groupby("level").size()
        usable = counts[counts >= 3]
        if not usable.empty:
            return int(usable.index.max())
        if not counts.empty:
            return int(counts.index.max())
    if not nodes.empty:
        return int(pd.to_numeric(nodes["level"], errors="coerce").max())
    return None


def _node_feature(row: pd.Series) -> dict[str, Any] | None:
    required = ["lon_min_deg", "lat_min_deg", "lon_max_deg", "lat_max_deg"]
    if not all(np.isfinite([row.get(column) for column in required])):
        return None
    props = {key: _clean(value) for key, value in row.items() if key not in required}
    props["feature_type"] = "spectral_dynamic_node"
    lon_min, lat_min, lon_max, lat_max = [float(row[column]) for column in required]
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]],
        },
        "properties": props,
    }


def _edge_feature(row: pd.Series) -> dict[str, Any] | None:
    coords = [row.get("from_longitude_deg"), row.get("from_latitude_deg"), row.get("to_longitude_deg"), row.get("to_latitude_deg")]
    if not all(np.isfinite(coords)):
        return None
    props = {
        key: _clean(value)
        for key, value in row.items()
        if key not in {"from_longitude_deg", "from_latitude_deg", "to_longitude_deg", "to_latitude_deg"}
    }
    props["feature_type"] = "spectral_transfer_edge"
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[float(coords[0]), float(coords[1])], [float(coords[2]), float(coords[3])]]},
        "properties": props,
    }


def spectral_heatmap_features(nodes: pd.DataFrame, edges: pd.DataFrame, level: int | None = None) -> list[dict[str, Any]]:
    selected_level = _display_level(nodes, edges) if level is None else level
    if selected_level is None:
        return []
    node_layer = nodes[pd.to_numeric(nodes.get("level"), errors="coerce") == selected_level].copy() if not nodes.empty else pd.DataFrame()
    edge_layer = edges[pd.to_numeric(edges.get("level"), errors="coerce") == selected_level].copy() if not edges.empty else pd.DataFrame()
    features = []
    features.extend(feature for _, row in node_layer.iterrows() if (feature := _node_feature(row)) is not None)
    features.extend(feature for _, row in edge_layer.iterrows() if (feature := _edge_feature(row)) is not None)
    for feature in features:
        feature["properties"]["display_level"] = selected_level
    return features


def write_spectral_products(
    record_signatures: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    modes: pd.DataFrame,
    components: pd.DataFrame,
    output_dir: Path,
    freq_grid_hz: tuple[float, ...] = DEFAULT_FREQ_GRID_HZ,
) -> None:
    ensure_dir(output_dir)
    write_parquet(record_signatures, output_dir / "spectral_record_signatures.parquet")
    write_parquet(nodes, output_dir / "spectral_node_dynamics.parquet")
    write_parquet(edges, output_dir / "spectral_edge_transmissibility.parquet")
    write_parquet(modes, output_dir / "spectral_dynamic_modes.parquet")
    components.to_csv(output_dir / "spectral_mode_components.csv", index=False)
    features = spectral_heatmap_features(nodes, edges)
    write_geojson(features, output_dir / "spectral_dynamic_heatmap.geojson")
    write_kmz(features, output_dir / "spectral_dynamic_heatmap.kmz")
    write_json(
        output_dir / "spectral_frequency_grid.json",
        {
            "frequency_hz": list(freq_grid_hz),
            "n_frequencies": len(freq_grid_hz),
            "basis": "log-spaced full-spectrum simultaneous grid",
        },
    )
