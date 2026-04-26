"""
evaluate.py
===========
Model evaluation, metrics, and Predicted vs Actual logging.
Logs daily errors to CSV for the continuous learning loop.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error
)

from models import DEVICE

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_classification(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Returns probability of UP (0..1)."""
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        Xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE)
        logit = model(Xb)
        prob = torch.sigmoid(logit).cpu().numpy()
        probs.append(prob)
    return np.concatenate(probs)


@torch.no_grad()
def predict_regression(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Returns predicted values (regression)."""
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        Xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE)
        out = model(Xb).cpu().numpy()
        preds.append(out)
    return np.concatenate(preds)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def eval_classification(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print(f"\n  Directional Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {"accuracy": acc, "report": report}


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Close") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    errors = y_pred - y_true

    print(f"\n  [{label}] MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")
    print(f"           Mean error: {errors.mean():.4f} | Max abs error: {np.abs(errors).max():.4f}")

    return {
        "mae": mae, "rmse": rmse, "mape": mape,
        "mean_error": errors.mean(), "max_error": np.abs(errors).max()
    }


# ─── Predicted vs Actual Logging ──────────────────────────────────────────────

def log_predictions(
    dates: np.ndarray,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    tic: str,
    label: str = "close",
    run_date: str = None,
):
    """Append predicted vs actual to CSV log."""
    import datetime
    run_date = run_date or str(datetime.date.today())

    df = pd.DataFrame({
        "run_date": run_date,
        "date":     dates,
        "tic":      tic,
        "label":    label,
        "actual":   y_actual,
        "predicted": y_pred,
        "error":    y_pred - y_actual,
    })

    path = os.path.join(LOGS_DIR, "predictions.csv")
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)
    print(f"Logged {len(df)} predictions → {path}")

    # Also save separate actuals / errors for easy loading
    df[["run_date", "date", "tic", "label", "actual"]].to_csv(
        os.path.join(LOGS_DIR, "actuals.csv"), mode="a", index=False,
        header=not os.path.exists(os.path.join(LOGS_DIR, "actuals.csv"))
    )
    df[["run_date", "date", "tic", "label", "error"]].to_csv(
        os.path.join(LOGS_DIR, "errors.csv"), mode="a", index=False,
        header=not os.path.exists(os.path.join(LOGS_DIR, "errors.csv"))
    )

    return df


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(
    dates, y_actual, y_pred, title: str = "Predicted vs Actual",
    save_path: str = None
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    # ── Price panel ──────────────────────────────────────────────────────────
    ax1.plot(dates, y_actual, color="#00bfff", linewidth=1.8, label="Actual", zorder=3)
    ax1.plot(dates, y_pred,   color="#ff6b35", linewidth=1.5, linestyle="--", label="Predicted", zorder=2)
    ax1.fill_between(dates, y_actual, y_pred, alpha=0.1, color="#ff6b35")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Error panel ───────────────────────────────────────────────────────────
    errors = np.array(y_pred) - np.array(y_actual)
    ax2.bar(dates, errors, color=np.where(errors >= 0, "#2ecc71", "#e74c3c"), alpha=0.7, width=0.8)
    ax2.axhline(0, color="white", linewidth=0.8)
    ax2.set_ylabel("Error (Pred - Actual)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved → {save_path}")
    plt.show()


def plot_training_history(history: dict, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train Loss", color="#00bfff")
    ax.plot(history["val_loss"],   label="Val Loss",   color="#ff6b35")
    ax.set_title("Training History", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_price_with_indicators(df: pd.DataFrame, tic: str, save_path: str = None):
    """Price + VWAP + EMA + Volume panel."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [4, 1, 1]})
    ax1, ax2, ax3 = axes

    dates = pd.to_datetime(df["date"])

    ax1.plot(dates, df["close"],  color="#00bfff", linewidth=1.5, label="Close", zorder=3)
    if "vwap"  in df: ax1.plot(dates, df["vwap"],  color="#f39c12", linewidth=1.2, linestyle="--", label="VWAP")
    if "ema_9" in df: ax1.plot(dates, df["ema_9"], color="#2ecc71", linewidth=1.0, alpha=0.8, label="EMA9")
    if "ema_20" in df: ax1.plot(dates, df["ema_20"], color="#e74c3c", linewidth=1.0, alpha=0.8, label="EMA20")
    ax1.set_title(f"{tic} — Price + Indicators", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    if "rsi_14" in df:
        ax2.plot(dates, df["rsi_14"], color="#9b59b6", linewidth=1.2)
        ax2.axhline(70, color="#e74c3c", linewidth=0.8, linestyle="--")
        ax2.axhline(30, color="#2ecc71", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("RSI(14)")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

    ax3.bar(dates, df["volume"], color="#3498db", alpha=0.6)
    ax3.set_ylabel("Volume")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
