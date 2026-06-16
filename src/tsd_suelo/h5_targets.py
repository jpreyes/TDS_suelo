from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd

from .utils import as_clean_str, finite_or_nan, safe_float


G_CM_S2 = 980.665
G_M_S2 = 9.80665
DEFAULT_PERIODS_S = (0.1, 0.2, 0.5, 1.0, 2.0)
ENERGY_BANDS_HZ = {
    "energy_0p1_1_hz": (0.1, 1.0),
    "energy_1_3_hz": (1.0, 3.0),
    "energy_3_8_hz": (3.0, 8.0),
    "energy_8_20_hz": (8.0, 20.0),
    "energy_20_plus_hz": (20.0, math.inf),
}


def list_h5_files(records_dir: Path, max_h5: int | None = None) -> list[Path]:
    files = sorted(records_dir.glob("*.h5"))
    return files[:max_h5] if max_h5 is not None else files


def _dataset_scalar(h5: h5py.File, path: str) -> Any:
    if path not in h5:
        return None
    value = h5[path][()]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    return value


def _metadata_scalar(h5: h5py.File, group: str, name: str) -> Any:
    return _dataset_scalar(h5, f"metadata/{group}/{name}")


def _component_acc(h5: h5py.File, component: str) -> np.ndarray:
    for root in ("Processed_data", "Unprocessed_data"):
        path = f"{root}/{component}_acc"
        if path in h5:
            data = np.asarray(h5[path][()], dtype=float).reshape(-1)
            return data[np.isfinite(data)]
    return np.asarray([], dtype=float)


def _duration_between_energy(acc_cm_s2: np.ndarray, dt: float, start_frac: float, end_frac: float) -> float:
    if acc_cm_s2.size == 0 or not np.isfinite(dt) or dt <= 0:
        return math.nan
    energy = np.cumsum(np.square(acc_cm_s2.astype(float)))
    total = float(energy[-1]) if energy.size else 0.0
    if total <= 0:
        return math.nan
    frac = energy / total
    times = np.arange(acc_cm_s2.size, dtype=float) * dt
    t0 = float(np.interp(start_frac, frac, times))
    t1 = float(np.interp(end_frac, frac, times))
    return finite_or_nan(t1 - t0)


def _arias_intensity_m_s(acc_cm_s2: np.ndarray, dt: float) -> float:
    if acc_cm_s2.size == 0 or not np.isfinite(dt) or dt <= 0:
        return math.nan
    acc_m_s2 = acc_cm_s2 * 0.01
    return finite_or_nan((math.pi / (2.0 * G_M_S2)) * float(np.sum(acc_m_s2 * acc_m_s2) * dt))


def _spectral_metrics(acc_cm_s2: np.ndarray, dt: float) -> dict[str, float]:
    metrics = {
        "dominant_freq_hz": math.nan,
        "spectral_centroid_hz": math.nan,
        "spectral_bandwidth_hz": math.nan,
    }
    metrics.update({name: math.nan for name in ENERGY_BANDS_HZ})
    if acc_cm_s2.size < 4 or not np.isfinite(dt) or dt <= 0:
        return metrics
    signal = acc_cm_s2.astype(float) - float(np.nanmean(acc_cm_s2))
    freqs = np.fft.rfftfreq(signal.size, d=dt)
    power = np.square(np.abs(np.fft.rfft(signal)))
    if freqs.size <= 1 or float(np.sum(power)) <= 0:
        return metrics
    freqs = freqs[1:]
    power = power[1:]
    total = float(np.sum(power))
    centroid = float(np.sum(freqs * power) / total)
    metrics["spectral_centroid_hz"] = finite_or_nan(centroid)
    metrics["spectral_bandwidth_hz"] = finite_or_nan(math.sqrt(float(np.sum(power * (freqs - centroid) ** 2) / total)))
    metrics["dominant_freq_hz"] = finite_or_nan(float(freqs[int(np.argmax(power))]))
    for name, (lo, hi) in ENERGY_BANDS_HZ.items():
        if math.isinf(hi):
            mask = freqs >= lo
        else:
            mask = (freqs >= lo) & (freqs < hi)
        metrics[name] = finite_or_nan(float(np.sum(power[mask]) / total))
    return metrics


def _polarization_angle_deg(e_acc: np.ndarray, n_acc: np.ndarray) -> float:
    size = min(e_acc.size, n_acc.size)
    if size < 4:
        return math.nan
    en = np.vstack([e_acc[:size], n_acc[:size]])
    cov = np.cov(en)
    if not np.all(np.isfinite(cov)):
        return math.nan
    vals, vecs = np.linalg.eigh(cov)
    vec = vecs[:, int(np.argmax(vals))]
    angle = math.degrees(math.atan2(vec[0], vec[1]))
    return finite_or_nan((angle + 360.0) % 180.0)


def _horizontal(e_acc: np.ndarray, n_acc: np.ndarray) -> np.ndarray:
    size = min(e_acc.size, n_acc.size)
    if size == 0:
        return np.asarray([], dtype=float)
    return np.sqrt(np.square(e_acc[:size]) + np.square(n_acc[:size]))


def pseudo_spectral_acceleration_g(
    acc_cm_s2: np.ndarray,
    dt: float,
    periods_s: tuple[float, ...] = DEFAULT_PERIODS_S,
    damping: float = 0.05,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if acc_cm_s2.size < 4 or not np.isfinite(dt) or dt <= 0:
        for period in periods_s:
            values[f"psa_t{str(period).replace('.', 'p')}_g"] = math.nan
        return values
    try:
        from scipy.signal import cont2discrete
    except Exception:
        for period in periods_s:
            values[f"psa_t{str(period).replace('.', 'p')}_g"] = math.nan
        return values

    acc = np.asarray(acc_cm_s2, dtype=float)
    for period in periods_s:
        key = f"psa_t{str(period).replace('.', 'p')}_g"
        if period <= 0:
            values[key] = math.nan
            continue
        omega = 2.0 * math.pi / period
        a_matrix = np.array([[0.0, 1.0], [-omega * omega, -2.0 * damping * omega]], dtype=float)
        b_matrix = np.array([[0.0], [-1.0]], dtype=float)
        c_matrix = np.array([[1.0, 0.0]], dtype=float)
        d_matrix = np.array([[0.0]], dtype=float)
        ad, bd, _, _, _ = cont2discrete((a_matrix, b_matrix, c_matrix, d_matrix), dt)
        state = np.zeros(2, dtype=float)
        max_rel_disp = 0.0
        forcing = bd[:, 0]
        for sample in acc:
            state = ad @ state + forcing * sample
            abs_u = abs(float(state[0]))
            if abs_u > max_rel_disp:
                max_rel_disp = abs_u
        values[key] = finite_or_nan((omega * omega * max_rel_disp) / G_CM_S2)
    return values


def read_h5_observation(path: Path, damping: float = 0.05) -> dict[str, Any]:
    return _read_h5_observation(path, damping=damping, compute_psa=True)


def _read_h5_observation(path: Path, damping: float = 0.05, compute_psa: bool = True) -> dict[str, Any]:
    event_from_name, station_from_name = _parse_filename(path)
    with h5py.File(path, "r") as h5:
        record_id_h5 = as_clean_str(_metadata_scalar(h5, "record", "RecordID"))
        event_id_h5 = as_clean_str(_metadata_scalar(h5, "event", "EventID_BM16"))
        station_id_h5 = as_clean_str(_metadata_scalar(h5, "record", "Sta_Name"))
        dt = safe_float(_metadata_scalar(h5, "record", "dt"))
        e_acc = _component_acc(h5, "E")
        n_acc = _component_acc(h5, "N")
        z_acc = _component_acc(h5, "Z")
        h_acc = _horizontal(e_acc, n_acc)

        row: dict[str, Any] = {
            "h5_file": str(path),
            "h5_name": path.name,
            "record_observed_id": f"{event_from_name}_{station_from_name}",
            "h5_record_id": record_id_h5,
            "event_id": event_from_name,
            "h5_event_id_bm16": event_id_h5,
            "station_id": station_id_h5 or station_from_name,
            "filename_event_id": event_from_name,
            "filename_station_id": station_from_name,
            "dt_s": dt,
            "sample_rate_hz": finite_or_nan(1.0 / dt) if dt and dt > 0 else math.nan,
            "n_samples_e": int(e_acc.size),
            "n_samples_n": int(n_acc.size),
            "n_samples_z": int(z_acc.size),
            "duration_h5_s": safe_float(_metadata_scalar(h5, "record", "Dtot")),
            "t_p_rec_s": safe_float(_metadata_scalar(h5, "record", "tP_rec")),
            "t_s_rec_s": safe_float(_metadata_scalar(h5, "record", "tS_rec")),
            "p_duration_s": safe_float(_metadata_scalar(h5, "record", "P_duration")),
            "s_duration_s": safe_float(_metadata_scalar(h5, "record", "S_duration")),
            "s_end_s": safe_float(_metadata_scalar(h5, "record", "S_end")),
            "coda_start_s": safe_float(_metadata_scalar(h5, "record", "CodaStart")),
            "coda_end_s": safe_float(_metadata_scalar(h5, "record", "CodaEnd")),
            "noise_start_s": safe_float(_metadata_scalar(h5, "record", "NoiseStart")),
            "repi_km_h5": safe_float(_metadata_scalar(h5, "record", "Repi")),
            "rhyp_km_h5": safe_float(_metadata_scalar(h5, "record", "Rhyp")),
            "utc_event": as_clean_str(_metadata_scalar(h5, "event", "UTC_event")),
            "utc_record_e": as_clean_str(_metadata_scalar(h5, "record", "UTC_record_E")),
            "processed_flag": as_clean_str(_metadata_scalar(h5, "record", "IfProcessed_Rec")),
        }

    for component, acc in (("e", e_acc), ("n", n_acc), ("z", z_acc), ("h", h_acc)):
        pga_cm_s2 = float(np.nanmax(np.abs(acc))) if acc.size else math.nan
        row[f"pga_{component}_cm_s2"] = finite_or_nan(pga_cm_s2)
        row[f"pga_{component}_g"] = finite_or_nan(pga_cm_s2 / G_CM_S2)
        row[f"arias_{component}_m_s"] = _arias_intensity_m_s(acc, dt)
        row[f"cav_{component}_m_s"] = finite_or_nan(float(np.sum(np.abs(acc * 0.01)) * dt)) if acc.size and dt > 0 else math.nan
        row[f"duration_5_75_{component}_s"] = _duration_between_energy(acc, dt, 0.05, 0.75)
        row[f"duration_5_95_{component}_s"] = _duration_between_energy(acc, dt, 0.05, 0.95)

    row["horizontal_to_vertical_pga"] = finite_or_nan(row["pga_h_g"] / row["pga_z_g"]) if row["pga_z_g"] else math.nan
    row["east_to_north_pga"] = finite_or_nan(row["pga_e_g"] / row["pga_n_g"]) if row["pga_n_g"] else math.nan
    row["polarization_angle_deg"] = _polarization_angle_deg(e_acc, n_acc)
    row.update(_spectral_metrics(h_acc, dt))
    if compute_psa:
        row.update(pseudo_spectral_acceleration_g(h_acc, dt, damping=damping))
    else:
        for period in DEFAULT_PERIODS_S:
            row[f"psa_t{str(period).replace('.', 'p')}_g"] = math.nan
    return row


def _parse_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "_" not in stem:
        return stem, ""
    event_id, station_id = stem.split("_", 1)
    return event_id, station_id


def _read_h5_worker(args: tuple[str, float, bool]) -> dict[str, Any]:
    path, damping, compute_psa = args
    return _read_h5_observation(Path(path), damping=damping, compute_psa=compute_psa)


def build_h5_targets(
    records_dir: Path,
    max_h5: int | None = None,
    damping: float = 0.05,
    compute_psa: bool = True,
    workers: int = 1,
    progress_every: int = 500,
) -> pd.DataFrame:
    files = list_h5_files(records_dir, max_h5=max_h5)
    if workers <= 1 or len(files) <= 1:
        rows = [_read_h5_observation(path, damping=damping, compute_psa=compute_psa) for path in files]
        return pd.DataFrame(rows)

    rows: list[dict[str, Any]] = []
    tasks = [(str(path), damping, compute_psa) for path in files]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_read_h5_worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if progress_every and index % progress_every == 0:
                print(f"H5 procesados: {index}/{len(files)}", flush=True)
    return pd.DataFrame(rows)
