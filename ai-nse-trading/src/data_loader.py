"""
data_loader.py
==============
Downloads NSE data via yfinance, cleans it, and splits it
into train / validation / test sets WITHOUT shuffling (time-order preserved).
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "THANGAMAYL.NS"]
DATA_DIR = "data"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns produced by yfinance when multiple tickers are fetched."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col]).strip("_") for col in df.columns]
    df.columns = [c.lower() for c in df.columns]
    return df


def _clean_ohlcv(df: pd.DataFrame, tic: str) -> pd.DataFrame:
    """Ensure 1-D float columns, drop NaN rows, add ticker column."""
    required = ["open", "high", "low", "close", "volume"]

    # Keep only required columns (handle yfinance v0.2 naming)
    rename_map = {}
    for col in df.columns:
        for r in required:
            if r in col.lower():
                rename_map[col] = r
    df = df.rename(columns=rename_map)

    # Drop extra columns
    df = df[[c for c in required if c in df.columns]].copy()

    # Flatten to 1-D (squeeze if needed)
    for col in df.columns:
        if hasattr(df[col], "squeeze"):
            df[col] = df[col].squeeze()

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df["tic"] = tic

    # Rename index
    df.index.name = "date"
    df = df.reset_index()

    print(f"  [{tic}] {len(df)} rows after cleaning.")
    return df


# ─── Main Download Functions ──────────────────────────────────────────────────

def download_daily(
    tickers: list = DEFAULT_TICKERS,
    period: str = "5y",
    save: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV for NSE tickers."""
    frames = []
    for tic in tickers:
        print(f"Downloading daily data for {tic} ...")
        raw = yf.download(tic, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  WARNING: No data returned for {tic}")
            continue
        raw = _flatten_columns(raw)
        df = _clean_ohlcv(raw, tic)
        frames.append(df)

    if not frames:
        raise RuntimeError("No data downloaded. Check tickers and internet connection.")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["tic", "date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, "daily_raw.csv")
        combined.to_csv(path, index=False)
        print(f"Saved daily data → {path}")

    return combined


def download_intraday(
    tickers: list = DEFAULT_TICKERS,
    interval: str = "15m",
    period: str = "60d",
    save: bool = True,
) -> pd.DataFrame:
    """Download intraday OHLCV (5m / 15m).  yfinance caps history at ~60 days for intraday."""
    frames = []
    for tic in tickers:
        print(f"Downloading {interval} intraday data for {tic} ...")
        raw = yf.download(tic, period=period, interval=interval, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  WARNING: No data returned for {tic}")
            continue
        raw = _flatten_columns(raw)
        df = _clean_ohlcv(raw, tic)
        frames.append(df)

    if not frames:
        raise RuntimeError("No intraday data downloaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["tic", "date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f"intraday_{interval}_raw.csv")
        combined.to_csv(path, index=False)
        print(f"Saved intraday data → {path}")

    return combined


# ─── Train / Val / Test Split ─────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    date_col: str = "date",
) -> tuple:
    """
    Split a DataFrame chronologically.
    Returns (train_df, val_df, test_df).
    NO SHUFFLING — this is critical to avoid data leakage.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# ─── Load from disk ───────────────────────────────────────────────────────────

def load_daily(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "daily_raw.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {path}")
    return df


def load_intraday(interval: str = "15m", path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, f"intraday_{interval}_raw.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {path}")
    return df


if __name__ == "__main__":
    # Quick smoke-test
    daily = download_daily(tickers=["RELIANCE.NS", "TCS.NS"], period="2y")
    print(daily.head())
    train, val, test = time_split(daily)
