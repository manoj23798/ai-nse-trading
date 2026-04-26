"""
train.py
========
Training loops for all models.
Supports:
  - Full training from scratch
  - Incremental / fine-tuning on new data
  - Early stopping
  - Gradient clipping (prevents exploding gradients on time-series)
  - Google Drive checkpoint saving
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import DEVICE, save_model

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 50,
    "patience": 8,        # early stopping patience
    "grad_clip": 1.0,
    "weight_decay": 1e-4,
}


# ─── Dataset Helpers ──────────────────────────────────────────────────────────

def to_tensors(X, y, device=DEVICE):
    return (
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).to(device),
    )


def make_loader(X, y, batch_size=64, shuffle=False):
    Xt, yt = to_tensors(X, y)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ─── Generic Training Loop ────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "classification",   # 'classification' or 'regression'
    config: dict = None,
    save_path: str = None,
) -> dict:
    """
    Train model with early stopping and gradient clipping.

    Parameters
    ----------
    task : 'classification' → BCEWithLogitsLoss
           'regression'     → HuberLoss (robust to outliers)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ── Loss function ─────────────────────────────────────────────────────────
    if task == "classification":
        criterion = nn.BCEWithLogitsLoss()
    elif task == "regression_hl":   # high-low (2 outputs)
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.HuberLoss(delta=1.0)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    # Torch versions differ on whether `verbose` is accepted here.
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    train_loader = make_loader(X_train, y_train, batch_size=cfg["batch_size"], shuffle=False)
    val_loader   = make_loader(X_val,   y_val,   batch_size=cfg["batch_size"], shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"\n{'='*55}")
    print(f" Training: task={task}  epochs={cfg['epochs']}  device={DEVICE}")
    print(f"{'='*55}")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            if task == "regression_hl":
                loss = criterion(pred, yb)         # yb shape (batch, 2)
            else:
                loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                pred = model(Xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:03d}/{cfg['epochs']} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ── Early Stopping ───────────────────────────────────────────────────
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                save_model(model, save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch} (patience={cfg['patience']})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    print(f"\nBest val loss: {best_val_loss:.5f}")
    return history


# ─── Incremental Retraining ───────────────────────────────────────────────────

def incremental_train(
    model: nn.Module,
    X_new: np.ndarray,
    y_new: np.ndarray,
    task: str = "classification",
    epochs: int = 5,
    lr: float = 3e-4,
    save_path: str = None,
):
    """
    Fine-tune an existing model on the latest N rows of data.
    Used in the daily continuous learning loop.
    """
    print(f"[Incremental] Fine-tuning on {len(X_new)} new samples for {epochs} epochs...")
    config = {**DEFAULT_CONFIG, "epochs": epochs, "lr": lr, "patience": 99}

    # Use 90% as fake train, last 10% as val
    n = len(X_new)
    split = max(1, int(n * 0.9))
    X_tr, y_tr = X_new[:split], y_new[:split]
    X_vl, y_vl = X_new[split:], y_new[split:]

    return train_model(model, X_tr, y_tr, X_vl, y_vl, task=task, config=config, save_path=save_path)


# ─── Colab / Drive Helpers ────────────────────────────────────────────────────

def save_to_drive(local_path: str, drive_dir: str = "/content/drive/MyDrive/ai-nse-trading/models"):
    """Copy a checkpoint to Google Drive (only works in Colab)."""
    import shutil
    os.makedirs(drive_dir, exist_ok=True)
    dest = os.path.join(drive_dir, os.path.basename(local_path))
    shutil.copy2(local_path, dest)
    print(f"Checkpoint saved to Drive → {dest}")


def load_from_drive(filename: str, drive_dir: str = "/content/drive/MyDrive/ai-nse-trading/models",
                    local_dir: str = "models"):
    """Copy checkpoint from Google Drive to local models/ dir."""
    import shutil
    src = os.path.join(drive_dir, filename)
    os.makedirs(local_dir, exist_ok=True)
    dest = os.path.join(local_dir, filename)
    shutil.copy2(src, dest)
    print(f"Checkpoint loaded from Drive ← {src}")
    return dest
