"""
features.py
===========
Feature engineering for NSE trading system.
Computes RSI, EMA, VWAP, ATR, MACD, Bollinger Bands, etc.
Normalises features and saves scalers for inference-time use.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MODELS_DIR = "models"

# ─── Technical Indicator Helpers ──────────────────────────────────────────────

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=window, min_periods=window).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP — correct per-day for intraday, full series for daily."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / (cum_vol + 1e-9)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    return upper, rolling_mean, lower


# ─── Main Feature Builder ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    """
    Add all technical features to a cleaned OHLCV DataFrame.
    Works per ticker to avoid cross-contamination.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["tic", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    result_frames = []

    for tic, group in df.groupby("tic"):
        g = group.copy().reset_index(drop=True)

        # ── Price-based ──────────────────────────────────────────────────────
        g["returns"] = g["close"].pct_change()
        g["log_returns"] = np.log(g["close"] / g["close"].shift(1))

        # ── EMAs ────────────────────────────────────────────────────────────
        g["ema_9"] = g["close"].ewm(span=9, min_periods=9).mean()
        g["ema_20"] = g["close"].ewm(span=20, min_periods=20).mean()
        g["ema_50"] = g["close"].ewm(span=50, min_periods=50).mean()
        g["ema_cross"] = (g["ema_9"] - g["ema_20"]) / (g["close"] + 1e-9)

        # ── RSI ──────────────────────────────────────────────────────────────
        g["rsi_14"] = compute_rsi(g["close"], 14)
        g["rsi_7"] = compute_rsi(g["close"], 7)

        # ── ATR ──────────────────────────────────────────────────────────────
        g["atr_14"] = compute_atr(g, 14)
        g["atr_pct"] = g["atr_14"] / (g["close"] + 1e-9)  # normalised volatility

        # ── VWAP ─────────────────────────────────────────────────────────────
        if intraday:
            # Reset VWAP each trading day
            g["trading_date"] = g["date"].dt.date
            vwap_parts = []
            for _date, day_grp in g.groupby("trading_date"):
                vwap_parts.append(compute_vwap(day_grp))
            g["vwap"] = pd.concat(vwap_parts)
        else:
            g["vwap"] = compute_vwap(g)

        g["price_vs_vwap"] = (g["close"] - g["vwap"]) / (g["vwap"] + 1e-9)

        # ── MACD ─────────────────────────────────────────────────────────────
        g["macd"], g["macd_signal"], g["macd_hist"] = compute_macd(g["close"])

        # ── Bollinger Bands ──────────────────────────────────────────────────
        g["bb_upper"], g["bb_mid"], g["bb_lower"] = compute_bollinger(g["close"])
        g["bb_pct"] = (g["close"] - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"] + 1e-9)
        g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / (g["bb_mid"] + 1e-9)

        # ── Volume features ──────────────────────────────────────────────────
        g["volume_ma"] = g["volume"].rolling(20).mean()
        g["volume_ratio"] = g["volume"] / (g["volume_ma"] + 1e-9)

        # ── Candlestick body / wick ──────────────────────────────────────────
        g["body"] = (g["close"] - g["open"]).abs() / (g["close"] + 1e-9)
        g["upper_wick"] = (g["high"] - g[["open", "close"]].max(axis=1)) / (g["close"] + 1e-9)
        g["lower_wick"] = (g[["open", "close"]].min(axis=1) - g["low"]) / (g["close"] + 1e-9)

        # ── Lagged closes ────────────────────────────────────────────────────
        for lag in [1, 2, 3, 5]:
            g[f"close_lag{lag}"] = g["close"].shift(lag)
            g[f"returns_lag{lag}"] = g["returns"].shift(lag)

        # ── Target labels ────────────────────────────────────────────────────
        # Next candle direction (classification)
        g["target_direction"] = (g["close"].shift(-1) > g["close"]).astype(int)
        # Next close (regression)
        g["target_next_close"] = g["close"].shift(-1)
        # Daily HIGH / LOW for same day (for high-low model)
        g["target_high"] = g["high"]
        g["target_low"] = g["low"]

        result_frames.append(g)

    out = pd.concat(result_frames, ignore_index=True)

    # Drop rows with NaN (from rolling windows + last row with no future target)
    out.dropna(inplace=True)
    out.reset_index(drop=True, inplace=True)

    print(f"Features built → {out.shape[0]} rows × {out.shape[1]} cols")
    return out


# ─── Feature Columns ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    "returns", "log_returns",
    "ema_9", "ema_20", "ema_50", "ema_cross",
    "rsi_14", "rsi_7",
    "atr_14", "atr_pct",
    "price_vs_vwap",
    "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width",
    "volume_ratio",
    "body", "upper_wick", "lower_wick",
    "close_lag1", "close_lag2", "close_lag3", "close_lag5",
    "returns_lag1", "returns_lag2", "returns_lag3", "returns_lag5",
]

REGRESSION_TARGETS = ["target_next_close", "target_high", "target_low"]
CLASSIFICATION_TARGET = "target_direction"


# ─── Normalisation ────────────────────────────────────────────────────────────

def fit_and_scale(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list = FEATURE_COLS,
    method: str = "standard",
    save_path: str = None,
) -> tuple:
    """
    Fit scaler on TRAIN only, transform all three splits.
    Returns (train_arr, val_arr, test_arr, scaler).
    """
    Scaler = StandardScaler if method == "standard" else MinMaxScaler
    scaler = Scaler()

    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val   = scaler.transform(val_df[feature_cols].values)
    X_test  = scaler.transform(test_df[feature_cols].values)

    if save_path:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"Scaler saved → {save_path}")

    return X_train, X_val, X_test, scaler


def load_scaler(path: str):
    return joblib.load(path)


# ─── Sliding Window Builder ───────────────────────────────────────────────────

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window: int = 30,
) -> tuple:
    """
    Build (samples, window, features) sequences for LSTM.
    X shape: (N, F)  →  out: (N-window, window, F)
    """
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window: i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


if __name__ == "__main__":
    from data_loader import download_daily, time_split
    daily = download_daily(["RELIANCE.NS"], period="2y")
    feat = build_features(daily)
    print(feat[FEATURE_COLS].describe())
