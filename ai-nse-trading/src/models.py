"""
models.py
=========
PyTorch model definitions:
  1. LSTMDirectionModel   – classification (up/down)
  2. LSTMRegressionModel  – predicts next close price
  3. HighLowModel         – predicts daily HIGH and LOW
  4. TransformerModel     – optional, more powerful but heavier

All models are designed to run on CPU / free Colab GPU.
"""

import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[models.py] Using device: {DEVICE}")


# ─── 1. LSTM Direction Model (Classification) ─────────────────────────────────

class LSTMDirectionModel(nn.Module):
    """
    Predicts P(next candle is UP).
    Output: single sigmoid → probability in [0,1]
    Loss: BCEWithLogitsLoss
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # take last timestep
        last = self.norm(last)
        logit = self.head(last).squeeze(-1)
        return logit                  # raw logit (apply sigmoid externally)


# ─── 2. LSTM Regression Model (Next Close) ────────────────────────────────────

class LSTMRegressionModel(nn.Module):
    """
    Predicts next close price (normalised).
    Output: single float
    Loss: MSELoss / HuberLoss
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.norm(last)
        return self.head(last).squeeze(-1)


# ─── 3. High-Low Model ────────────────────────────────────────────────────────

class HighLowModel(nn.Module):
    """
    Predicts today's HIGH and LOW from features.
    Output: 2-dim vector [predicted_high, predicted_low]
    Loss: MSELoss
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),   # [high, low]
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.norm(last)
        return self.head(last)   # shape: (batch, 2)


# ─── 4. Transformer Model (Optional, More Powerful) ──────────────────────────

class TransformerDirectionModel(nn.Module):
    """
    Lightweight Transformer encoder for direction classification.
    More powerful than LSTM but slightly heavier.
    Still runs on free Colab GPU.
    """

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.2, max_seq: int = 60):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_seq)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]   # last token
        return self.head(x).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(model_type: str, input_size: int, **kwargs) -> nn.Module:
    """
    model_type: 'lstm_direction' | 'lstm_regression' | 'high_low' | 'transformer'
    """
    models = {
        "lstm_direction":  LSTMDirectionModel,
        "lstm_regression": LSTMRegressionModel,
        "high_low":        HighLowModel,
        "transformer":     TransformerDirectionModel,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(models.keys())}")
    return models[model_type](input_size=input_size, **kwargs).to(DEVICE)


def save_model(model: nn.Module, path: str):
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


def load_model(model: nn.Module, path: str) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"Model loaded ← {path}")
    return model
