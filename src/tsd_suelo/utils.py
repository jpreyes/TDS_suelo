from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


NULL_STRINGS = {"", "nan", "none", "null", "-999", "-999.0", "na", "n/a"}
FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_clean_str(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return as_clean_str(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return as_clean_str(value[0])
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in NULL_STRINGS:
        return None
    if text.startswith("b'") and text.endswith("'"):
        text = text[2:-1]
    return text


def safe_float(value: Any) -> float:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return math.nan
        return safe_float(value.reshape(-1)[0])
    if value is None:
        return math.nan
    if isinstance(value, str) and value.strip().lower() in NULL_STRINGS:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(list(NULL_STRINGS), np.nan), errors="coerce")


def coalesce_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    values = pd.Series(np.nan, index=df.index, dtype="float64")
    for column in columns:
        if column in df.columns:
            values = values.where(values.notna(), numeric_series(df[column]))
    return values


def parse_numeric_vector(value: Any) -> list[float]:
    text = as_clean_str(value)
    if not text:
        return []
    return [float(match.group(0)) for match in FLOAT_RE.finditer(text)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_csv_str(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else math.nan


def log_positive(series: pd.Series) -> pd.Series:
    values = numeric_series(series)
    values = values.where(values > 0)
    return np.log(values)

