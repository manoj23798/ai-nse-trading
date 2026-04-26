# 🇮🇳 AI NSE Trading Research System

> **DISCLAIMER**: This is a research/educational system. It does NOT guarantee profits.
> Stock trading carries significant financial risk. Consult a SEBI-registered advisor.

---

## What This System Does

An end-to-end ML pipeline for Indian stock market (NSE) research:

- Downloads NSE data via yfinance (RELIANCE, TCS, INFY, THANGAMAYL, etc.)
- Engineers 28+ technical features (RSI, EMA, VWAP, ATR, MACD, Bollinger)
- Trains LSTM / Transformer models for:
  - Next-candle direction (classification)
  - Next close price (regression)
  - Daily HIGH and LOW estimates
- Generates risk-managed trading signals
- Backtests with real transaction costs (brokerage + STT + slippage)
- Learns daily from prediction errors (incremental retraining)
- Saves all models and logs to Google Drive

---

## Quick Start (Google Colab)

1. Upload this repo to GitHub
2. Open `notebooks/run_colab.ipynb` in Google Colab
3. Enable GPU: Runtime → Change runtime type → T4 GPU (free)
4. Update `REPO_URL` in Step 1 to your GitHub URL
5. Run all cells top-to-bottom

**First run**: ~10–20 minutes (downloads data + trains models)
**Daily run**: ~5 minutes (loads saved models from Drive, fine-tunes)

---

## Project Structure

```
ai-nse-trading/
├── src/
│   ├── data_loader.py        # NSE data download + train/val/test split
│   ├── features.py           # 28+ technical indicators + feature matrix
│   ├── models.py             # LSTM + Transformer PyTorch models
│   ├── train.py              # Training loops + Google Drive checkpointing
│   ├── evaluate.py           # Metrics + Predicted vs Actual plots
│   ├── strategy.py           # Signal generation + risk management
│   ├── backtest.py           # Walk-forward backtesting + equity curve
│   └── continuous_learning.py # Daily update loop
├── notebooks/
│   └── run_colab.ipynb       # Master Colab notebook (run this)
├── data/                     # Downloaded raw data (auto-created)
├── models/                   # Saved model checkpoints (auto-created)
├── logs/                     # Predictions, errors, plots (auto-created)
├── requirements.txt
└── README.md
```

---

## Module Explanations

### `data_loader.py`
Downloads NSE (.NS) tickers via yfinance. Handles MultiIndex columns,
missing data, and ensures clean 1D float columns. Provides strict
**time-based train/val/test splits** (no shuffling = no data leakage).

### `features.py`
Computes 28+ features per-ticker:
- EMA(9,20,50), RSI(7,14), ATR(14), VWAP, MACD, Bollinger Bands
- Volume ratio, candlestick body/wick ratios, lagged returns
- Target labels: `target_direction`, `target_next_close`, `target_high`, `target_low`
- StandardScaler fitted on TRAIN only → prevents leakage into val/test

### `models.py`
Four PyTorch model architectures:
- `LSTMDirectionModel`  — classification (sigmoid output)
- `LSTMRegressionModel` — next close prediction
- `HighLowModel`        — predicts [HIGH, LOW] simultaneously
- `TransformerDirectionModel` — more powerful, optional

### `train.py`
- AdamW optimiser + ReduceLROnPlateau scheduler
- Early stopping (patience=10)
- Gradient clipping (prevents exploding gradients)
- Incremental fine-tuning for daily updates
- Google Drive checkpoint save/load helpers

### `evaluate.py`
- Directional accuracy, MAE, RMSE, MAPE
- Predicted vs Actual price plots with error panel
- Daily prediction logging → `logs/predictions.csv`, `actuals.csv`, `errors.csv`

### `strategy.py`
Strict signal rules:
- **BUY**: price > VWAP AND RSI > 55 AND model_prob_up > 0.60
- **SELL**: price < VWAP AND RSI < 45 AND model_prob_down > 0.60
- **HOLD**: everything else, or high volatility, or low confidence
- Position sizing: 1% capital risk per trade
- ATR-based stop-loss, 1:1.5 minimum risk:reward

### `backtest.py`
- Walk-forward validation (5 folds)
- Transaction costs: 0.03% brokerage + 0.1% STT + 0.02% slippage
- Stop-loss and take-profit enforcement within each bar
- Max daily loss cap (3%)
- Metrics: Sharpe ratio, max drawdown, win rate, profit factor

### `continuous_learning.py`
Daily loop:
1. Download latest N days of data
2. Load existing model checkpoint from Drive
3. Predict on latest data, log errors
4. Fine-tune model on recent data (5 epochs)
5. Save updated checkpoint back to Drive

---

## Risk Management Rules

| Rule | Value |
|------|-------|
| Risk per trade | ≤ 1% of capital |
| Max daily loss | ≤ 3% of capital |
| Min risk:reward | 1:1.5 |
| Stop-loss type | ATR × 1.5 |
| High volatility block | ATR% > 3% → no trade |
| Low confidence block | model_prob < 60% → no trade |
| Signal conflict block | Mixed indicators → HOLD |

---

## Google Drive Structure (auto-created)

```
MyDrive/ai-nse-trading/
├── models/
│   ├── lstm_direction_RELIANCE.NS.pt
│   ├── scaler_RELIANCE.NS.pkl
│   └── ... (one set per ticker)
├── logs/
│   ├── predictions.csv
│   ├── actuals.csv
│   └── errors.csv
└── backtest_summary.csv
```

---

## Free Resources Used

| Resource | Cost |
|----------|------|
| Google Colab (T4 GPU) | Free |
| Google Drive (15 GB) | Free |
| yfinance data | Free |
| PyTorch | Free / open-source |
| All libraries | Free / open-source |

**Total cost: ₹0**
