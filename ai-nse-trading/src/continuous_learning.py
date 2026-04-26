"""
continuous_learning.py
======================
Daily loop:
  1. Download latest data
  2. Build features
  3. Load existing model
  4. Predict on today's data
  5. Compare with actual (yesterday's actuals are now known)
  6. Log errors
  7. Incrementally retrain model on latest N days
  8. Save checkpoint to Google Drive
"""

import os
import datetime
import numpy as np
import pandas as pd
import torch
import joblib

from data_loader import download_daily, download_intraday, time_split
from features import build_features, FEATURE_COLS, fit_and_scale, make_sequences, load_scaler
from models import build_model, load_model, DEVICE
from train import incremental_train, save_to_drive
from evaluate import predict_classification, predict_regression, eval_classification, eval_regression, log_predictions

MODELS_DIR  = "models"
LOGS_DIR    = "logs"
DATA_DIR    = "data"
WINDOW      = 30    # sequence window for LSTM


def run_daily_loop(
    tickers: list,
    model_type: str = "lstm_direction",
    retrain_on_last_n_days: int = 90,
    drive_dir: str = "/content/drive/MyDrive/ai-nse-trading",
    use_intraday: bool = False,
):
    """
    Full daily loop. Safe to run from Colab every morning before market open.
    """
    today = str(datetime.date.today())
    print(f"\n{'='*60}")
    print(f"  DAILY LOOP — {today}")
    print(f"{'='*60}")

    # ── 1. Download latest data ───────────────────────────────────────────────
    if use_intraday:
        raw = download_intraday(tickers, interval="15m", period="60d")
        intraday = True
    else:
        raw = download_daily(tickers, period="2y")
        intraday = False

    # ── 2. Feature engineering ────────────────────────────────────────────────
    feat_df = build_features(raw, intraday=intraday)

    # ── 3. Prepare latest N days for fine-tuning ──────────────────────────────
    cutoff_date = pd.Timestamp.today() - pd.Timedelta(days=retrain_on_last_n_days)
    recent = feat_df[feat_df["date"] >= cutoff_date].copy()

    # Per-ticker loop
    for tic in tickers:
        print(f"\n[{tic}] Processing ...")
        tic_df = recent[recent["tic"] == tic].copy()
        if len(tic_df) < WINDOW + 10:
            print(f"  Not enough data for {tic}, skipping.")
            continue

        scaler_path = os.path.join(MODELS_DIR, f"scaler_{tic}.pkl")
        model_path  = os.path.join(MODELS_DIR, f"{model_type}_{tic}.pt")

        # ── Load scaler ───────────────────────────────────────────────────────
        if os.path.exists(scaler_path):
            scaler = load_scaler(scaler_path)
            X_scaled = scaler.transform(tic_df[FEATURE_COLS].values)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(tic_df[FEATURE_COLS].values)
            joblib.dump(scaler, scaler_path)

        y_dir = tic_df["target_direction"].values.astype(np.float32)

        # Build sequences
        X_seq, y_seq = make_sequences(X_scaled, y_dir, window=WINDOW)
        if len(X_seq) < 10:
            print(f"  Not enough sequences for {tic}")
            continue

        # ── Load / build model ────────────────────────────────────────────────
        model = build_model(model_type, input_size=len(FEATURE_COLS))
        if os.path.exists(model_path):
            model = load_model(model, model_path)
            print(f"  Loaded existing model ← {model_path}")
        else:
            print(f"  No existing model found, will train from scratch...")

        # ── Predict on latest data ────────────────────────────────────────────
        probs = predict_classification(model, X_seq)
        dates  = tic_df["date"].values[WINDOW:]
        actual = y_seq

        # ── Compare predicted vs actual (last known) ──────────────────────────
        log_predictions(dates, actual, probs, tic=tic, label="direction", run_date=today)
        metrics = eval_classification(actual, probs)

        # ── Incremental retrain ───────────────────────────────────────────────
        print(f"  Fine-tuning on last {len(X_seq)} sequences ...")
        incremental_train(model, X_seq, y_seq, task="classification", epochs=5, save_path=model_path)

        # ── Save to Drive ─────────────────────────────────────────────────────
        try:
            save_to_drive(model_path, drive_dir=os.path.join(drive_dir, "models"))
            save_to_drive(scaler_path, drive_dir=os.path.join(drive_dir, "models"))
        except Exception as e:
            print(f"  Drive save failed (not in Colab?): {e}")

    print(f"\nDaily loop complete for {today}.")


if __name__ == "__main__":
    run_daily_loop(
        tickers=["RELIANCE.NS", "TCS.NS"],
        model_type="lstm_direction",
        use_intraday=False,
    )
