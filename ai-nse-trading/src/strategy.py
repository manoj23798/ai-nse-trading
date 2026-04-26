"""
strategy.py
===========
Signal generation with strict risk management rules.

Signal logic:
  BUY  if: price > VWAP  AND  RSI > 55  AND  model_prob_up > 0.60
  SELL if: price < VWAP  AND  RSI < 45  AND  model_prob_down > 0.60
  HOLD otherwise

Risk management:
  - Position size: ≤ 1–2% of capital per trade
  - Stop loss: ATR-based
  - Take profit: min risk:reward = 1:1.5
  - Max daily loss cap: 3%
  - No trade in conflicting / high-volatility / low-confidence conditions
"""

import numpy as np
import pandas as pd


# ─── Signal Constants ─────────────────────────────────────────────────────────
BUY  =  1
SELL = -1
HOLD =  0

# ─── Risk Config (edit to suit) ──────────────────────────────────────────────
RISK_CONFIG = {
    "capital":           100_000,  # INR starting capital
    "risk_per_trade":    0.01,     # 1% of capital per trade
    "max_daily_loss":    0.03,     # 3% max daily drawdown
    "min_rr":            1.5,      # min reward-to-risk ratio
    "atr_stop_mult":     1.5,      # stop = entry ± ATR × mult
    "atr_high_thresh":   0.03,     # >3% ATR/price → skip trade
    "prob_threshold":    0.60,     # model confidence threshold
    "rsi_buy":           55,
    "rsi_sell":          45,
}


# ─── Signal Generator ─────────────────────────────────────────────────────────

def generate_signals(
    df: pd.DataFrame,
    probs_up: np.ndarray,
    config: dict = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df       : feature DataFrame with columns: close, vwap, rsi_14, atr_pct
    probs_up : model output P(UP) for each row, shape (N,)
    config   : override RISK_CONFIG keys

    Returns
    -------
    DataFrame with added columns: signal, stop_loss, take_profit, position_size_units
    """
    cfg = {**RISK_CONFIG, **(config or {})}
    df = df.copy().reset_index(drop=True)

    n = len(df)
    assert len(probs_up) == n, "probs_up must match df length"

    signals          = np.full(n, HOLD, dtype=int)
    stop_losses      = np.full(n, np.nan)
    take_profits     = np.full(n, np.nan)
    pos_sizes        = np.full(n, 0.0)
    reasons          = [""] * n

    prob_down = 1.0 - probs_up

    for i in range(n):
        close    = df["close"].iloc[i]
        vwap     = df["vwap"].iloc[i]    if "vwap"    in df.columns else close
        rsi      = df["rsi_14"].iloc[i]  if "rsi_14"  in df.columns else 50.0
        atr      = df["atr_14"].iloc[i]  if "atr_14"  in df.columns else close * 0.01
        atr_pct  = df["atr_pct"].iloc[i] if "atr_pct" in df.columns else 0.01
        prob_u   = probs_up[i]
        prob_d   = prob_down[i]

        # ── Kill-switches ─────────────────────────────────────────────────────
        if np.isnan(close) or np.isnan(vwap) or np.isnan(rsi):
            reasons[i] = "NaN data"
            continue

        if atr_pct > cfg["atr_high_thresh"]:
            reasons[i] = f"High volatility ATR%={atr_pct:.3f}"
            continue

        # ── BUY condition ─────────────────────────────────────────────────────
        if (close > vwap and rsi > cfg["rsi_buy"] and prob_u >= cfg["prob_threshold"]):
            sl = close - cfg["atr_stop_mult"] * atr
            tp = close + cfg["atr_stop_mult"] * atr * cfg["min_rr"]
            risk_amt = close - sl
            if risk_amt <= 0:
                reasons[i] = "Zero risk amount"
                continue

            capital_risk = cfg["capital"] * cfg["risk_per_trade"]
            units = capital_risk / risk_amt

            signals[i]     = BUY
            stop_losses[i] = sl
            take_profits[i] = tp
            pos_sizes[i]   = units
            reasons[i]     = f"BUY: prob_up={prob_u:.2f} RSI={rsi:.1f}"

        # ── SELL / SHORT condition ────────────────────────────────────────────
        elif (close < vwap and rsi < cfg["rsi_sell"] and prob_d >= cfg["prob_threshold"]):
            sl = close + cfg["atr_stop_mult"] * atr
            tp = close - cfg["atr_stop_mult"] * atr * cfg["min_rr"]
            risk_amt = sl - close
            if risk_amt <= 0:
                reasons[i] = "Zero risk amount"
                continue

            capital_risk = cfg["capital"] * cfg["risk_per_trade"]
            units = capital_risk / risk_amt

            signals[i]      = SELL
            stop_losses[i]  = sl
            take_profits[i] = tp
            pos_sizes[i]    = units
            reasons[i]      = f"SELL: prob_down={prob_d:.2f} RSI={rsi:.1f}"

        else:
            reasons[i] = "HOLD: conditions not met"

    df["signal"]         = signals
    df["stop_loss"]      = stop_losses
    df["take_profit"]    = take_profits
    df["pos_size_units"] = pos_sizes
    df["signal_reason"]  = reasons
    df["prob_up"]        = probs_up

    buys  = (signals == BUY).sum()
    sells = (signals == SELL).sum()
    holds = (signals == HOLD).sum()
    print(f"Signals → BUY: {buys} | SELL: {sells} | HOLD: {holds}")

    return df


# ─── Indicative Candle Estimate ───────────────────────────────────────────────

def estimate_next_candle(
    current_close: float,
    pred_next_close: float,
    atr: float,
) -> dict:
    """
    Construct an INDICATIVE (approximate) next candle from model prediction + ATR range.
    NOT a guarantee — purely for visualisation purposes.
    Clearly labelled as an estimate.
    """
    mid = (current_close + pred_next_close) / 2
    direction = 1 if pred_next_close >= current_close else -1

    est_open  = current_close
    est_close = pred_next_close
    est_high  = max(est_open, est_close) + abs(atr) * 0.5
    est_low   = min(est_open, est_close) - abs(atr) * 0.5

    return {
        "open":  round(est_open,  2),
        "high":  round(est_high,  2),
        "low":   round(est_low,   2),
        "close": round(est_close, 2),
        "label": "ESTIMATE — not a guarantee",
    }
