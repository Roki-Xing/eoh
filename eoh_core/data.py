"""
Data access helpers for price series and generic CSV ingestion.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .utils import log, normalize_list

DEFAULT_TIME_COLUMNS = (
    "date",
    "datetime",
    "timestamp",
    "time",
)

DEFAULT_CLOSE_COLUMNS = (
    "adj close",
    "adj_close",
    "close",
    "price",
)


def _normalise_column_candidates(
    candidates: Optional[Iterable[str]],
    fallbacks: Iterable[str],
) -> list[str]:
    if candidates:
        return normalize_list([c.strip().lower() for c in candidates if c])
    return [c.strip().lower() for c in fallbacks]


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the first datetime-like column into the index."""
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            return frame.set_index(col).sort_index()
    # attempt to parse any column that looks like a date
    for col in frame.columns:
        if any(token in col.lower() for token in DEFAULT_TIME_COLUMNS):
            parsed = pd.to_datetime(frame[col], errors="coerce", utc=False)
            frame = frame.loc[~parsed.isna()].copy()
            frame.index = parsed[~parsed.isna()]
            frame.index.name = col
            return frame.sort_index()
    # fall back to the first column or index
    if frame.index.name and any(token in frame.index.name.lower() for token in DEFAULT_TIME_COLUMNS):
        frame.index = pd.to_datetime(frame.index, errors="coerce", utc=False)
        frame = frame.loc[~frame.index.isna()].copy()
        return frame.sort_index()
    first = frame.columns[0]
    parsed = pd.to_datetime(frame[first], errors="coerce", utc=False)
    frame = frame.loc[~parsed.isna()].copy()
    frame.index = parsed[~parsed.isna()]
    frame.index.name = first
    return frame.sort_index()


def robust_read_csv(
    csv_path: str,
    *,
    close_column_candidates: Optional[Iterable[str]] = None,
    time_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV into a pandas DataFrame with a datetime index and a `Close` column.

    The function accepts a flexible set of time/close column aliases and will raise
    `FileNotFoundError` or `ValueError` when required information is missing.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"{csv_path} contains no data")

    time_aliases = _normalise_column_candidates(time_columns, DEFAULT_TIME_COLUMNS)
    close_aliases = _normalise_column_candidates(close_column_candidates, DEFAULT_CLOSE_COLUMNS)

    # ensure datetime index
    frame = _ensure_datetime_index(frame)

    # standardise column names once
    normalised = {col: str(col).strip() for col in frame.columns}
    frame = frame.rename(columns=normalised)

    # locate the price column
    lowered_map = {col.lower(): col for col in frame.columns}
    chosen_col = None
    for alias in close_aliases:
        if alias in lowered_map:
            chosen_col = lowered_map[alias]
            break

    if chosen_col is None:
        raise ValueError("CSV缺少收盘价列(如 Close/Adj Close)")

    frame["Close"] = frame[chosen_col].astype(float)
    frame = frame.loc[~frame["Close"].isna()].copy()
    frame.index.name = frame.index.name or "Date"
    return frame.sort_index()


def slice_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return a copy of *df* filtered by the inclusive (start, end) window."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


PriceProvider = Callable[[str, str, str], Optional[pd.DataFrame]]


@dataclass
class PriceDataLoader:
    """
    Combine a list of price providers and deliver the first successful result.

    Providers receive (symbol, start, end) and may return None to signal failure.
    """

    providers: tuple[PriceProvider, ...]

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        for provider in self.providers:
            try:
                data = provider(symbol, start, end)
            except Exception as exc:  # pragma: no cover - defensive logging
                log(f"[WARN] provider {provider.__name__} failed for {symbol}: {exc}")
                continue
            if data is not None and not data.empty:
                return data.sort_index()
        raise RuntimeError(f"No price provider produced data for {symbol} between {start} and {end}")


def synthetic_price_series(symbol: str, start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic business-day OHLCV series for development/testing.
    """
    idx = pd.date_range(start=start, end=end, freq="B")
    rng = np.random.default_rng(seed)
    px = 100.0
    rows = []
    for dt in idx:
        ret = rng.normal(0, 0.01)
        px = max(1.0, px * (1 + ret))
        open_px = px * (1 + rng.normal(0, 0.002))
        high_px = max(open_px, px) * (1 + abs(rng.normal(0, 0.003)))
        low_px = min(open_px, px) * (1 - abs(rng.normal(0, 0.003)))
        volume = max(0, int(abs(rng.normal(1e6, 2e5))))
        rows.append(
            {
                "Open": open_px,
                "High": high_px,
                "Low": low_px,
                "Close": px,
                "Volume": volume,
            }
        )
    frame = pd.DataFrame(rows, index=idx)
    frame.index.name = "Date"
    log(f"[INFO] generated synthetic prices for {symbol}: rows={len(frame)}")
    return frame
