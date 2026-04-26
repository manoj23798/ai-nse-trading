"""
backtest.py
===========
Walk-forward backtesting engine with:
  - Transaction costs (brokerage 0.03% + STT + slippage)
  - Position management (stop-loss / take-profit enforcement)
  - Equity curve tracking
  - Sharpe ratio, Max drawdown, Win rate calculation
  - NO lookahead bias — signals are generated strictly from past data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Transaction Cost Model ───────────────────────────────────────────────────
# NSE-realistic estimate (Zerodha-like)
BROKERAGE_PCT  = 0.0003   # 0.03% per leg
STT_PCT        = 0.001    # 0.1% on sell (equity delivery)
EXCHANGE_PCT   = 0.0000325
SLIPPAGE_PCT   = 0.0002   # 0.02% slippage estimate
TOTAL_COST_PCT = BROKERAGE_PCT * 2 + STT_PCT + EXCHANGE_PCT + SLIPPAGE_PCT


# ─── Backtest Engine ──────────────────────────────────────────────────────────

def run_backtest(
    signal_df: pd.DataFrame,
    capital: float = 100_000,
    max_daily_loss_frac: float = 0.03,
) -> dict:
    """
    Parameters
    ----------
    signal_df : output of strategy.generate_signals(), must have columns:
                date, close, signal, stop_loss, take_profit, pos_size_units

    Returns
    -------
    dict with equity_curve, trades, metrics
    """
    df = signal_df.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    equity        = capital
    peak_equity   = capital
    equity_curve  = [equity]
    dates_curve   = [df["date"].iloc[0]]

    trades        = []
    daily_loss    = 0.0
    current_date  = df["date"].iloc[0].date()

    in_position  = False
    position_dir = 0
    entry_price  = 0.0
    entry_units  = 0.0
    stop_loss    = 0.0
    take_profit  = 0.0
    entry_date   = None

    for i in range(len(df)):
        row      = df.iloc[i]
        price    = row["close"]
        signal   = row["signal"]
        today    = row["date"].date()

        # Reset daily loss tracker at new day
        if today != current_date:
            current_date = today
            daily_loss   = 0.0

        # ── Manage open position ──────────────────────────────────────────────
        if in_position:
            exit_triggered = False
            exit_price     = price
            exit_reason    = "signal_flip"

            # Stop-loss hit?
            if position_dir == 1 and price <= stop_loss:
                exit_price = stop_loss; exit_reason = "stop_loss"; exit_triggered = True
            elif position_dir == -1 and price >= stop_loss:
                exit_price = stop_loss; exit_reason = "stop_loss"; exit_triggered = True

            # Take-profit hit?
            if not exit_triggered:
                if position_dir == 1 and price >= take_profit:
                    exit_price = take_profit; exit_reason = "take_profit"; exit_triggered = True
                elif position_dir == -1 and price <= take_profit:
                    exit_price = take_profit; exit_reason = "take_profit"; exit_triggered = True

            # Opposite signal?
            if not exit_triggered and signal == -position_dir:
                exit_triggered = True

            if exit_triggered:
                gross_pnl = position_dir * (exit_price - entry_price) * entry_units
                cost      = TOTAL_COST_PCT * entry_price * entry_units
                net_pnl   = gross_pnl - cost

                equity    += net_pnl
                daily_loss = min(daily_loss + net_pnl, 0)   # accumulate losses only

                trades.append({
                    "entry_date":  entry_date,
                    "exit_date":   row["date"],
                    "direction":   "LONG" if position_dir == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "units":       entry_units,
                    "gross_pnl":   gross_pnl,
                    "cost":        cost,
                    "net_pnl":     net_pnl,
                    "exit_reason": exit_reason,
                })
                in_position = False

        # ── Daily loss cap check ──────────────────────────────────────────────
        if (-daily_loss / equity) >= max_daily_loss_frac:
            # Block new entries for today
            equity_curve.append(equity)
            dates_curve.append(row["date"])
            continue

        # ── Open new position ─────────────────────────────────────────────────
        if not in_position and signal in (1, -1):
            in_position  = True
            position_dir = signal
            entry_price  = price
            entry_units  = row.get("pos_size_units", 1.0)
            stop_loss    = row["stop_loss"] if not np.isnan(row["stop_loss"]) else (
                price - 0.02 * price if signal == 1 else price + 0.02 * price
            )
            take_profit  = row["take_profit"] if not np.isnan(row["take_profit"]) else (
                price + 0.03 * price if signal == 1 else price - 0.03 * price
            )
            entry_date = row["date"]

        peak_equity = max(peak_equity, equity)
        equity_curve.append(equity)
        dates_curve.append(row["date"])

    # Close any open position at the last price
    if in_position:
        last_price = df["close"].iloc[-1]
        gross_pnl  = position_dir * (last_price - entry_price) * entry_units
        cost       = TOTAL_COST_PCT * entry_price * entry_units
        net_pnl    = gross_pnl - cost
        equity    += net_pnl
        trades.append({
            "entry_date": entry_date, "exit_date": df["date"].iloc[-1],
            "direction": "LONG" if position_dir == 1 else "SHORT",
            "entry_price": entry_price, "exit_price": last_price,
            "units": entry_units, "gross_pnl": gross_pnl,
            "cost": cost, "net_pnl": net_pnl, "exit_reason": "end_of_data",
        })

    trades_df = pd.DataFrame(trades)
    metrics   = _compute_metrics(equity_curve, trades_df, capital)

    return {
        "equity_curve":  np.array(equity_curve),
        "dates_curve":   dates_curve,
        "trades":        trades_df,
        "metrics":       metrics,
        "final_equity":  equity,
    }


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _compute_metrics(equity_curve: list, trades_df: pd.DataFrame, initial_capital: float) -> dict:
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    # Sharpe (annualised, assume 252 trading days)
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0.0

    # Max drawdown
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / (peak + 1e-9)
    max_dd = dd.min()

    # Win rate
    if len(trades_df) > 0:
        win_rate = (trades_df["net_pnl"] > 0).mean()
        avg_win  = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"].mean() if (trades_df["net_pnl"] > 0).any() else 0
        avg_loss = trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"].mean() if (trades_df["net_pnl"] < 0).any() else 0
        profit_factor = abs(avg_win / (avg_loss + 1e-9))
    else:
        win_rate = profit_factor = avg_win = avg_loss = 0.0

    total_return = (equity[-1] - initial_capital) / initial_capital * 100

    m = {
        "total_return_pct":  round(total_return, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "win_rate":          round(win_rate, 4),
        "profit_factor":     round(profit_factor, 3),
        "num_trades":        len(trades_df),
        "avg_win_inr":       round(avg_win, 2),
        "avg_loss_inr":      round(avg_loss, 2),
    }

    print("\n" + "="*50)
    print("  BACKTEST RESULTS")
    print("="*50)
    for k, v in m.items():
        print(f"  {k:<25}: {v}")
    print("="*50)

    return m


# ─── Walk-Forward Validation ──────────────────────────────────────────────────

def walk_forward_backtest(
    signal_df: pd.DataFrame,
    n_splits: int = 5,
    capital: float = 100_000,
) -> list:
    """
    Split data into N time-ordered folds.
    Each fold: train on first K/n rows, test on next K/n rows.
    Returns list of result dicts.
    """
    n = len(signal_df)
    fold_size = n // n_splits
    results = []

    print(f"\n[Walk-Forward] {n_splits} folds × ~{fold_size} rows each")

    for k in range(n_splits):
        start = k * fold_size
        end   = start + fold_size if k < n_splits - 1 else n
        fold  = signal_df.iloc[start:end].copy()

        print(f"\n  Fold {k+1}/{n_splits}: rows {start}–{end}")
        result = run_backtest(fold, capital=capital)
        result["fold"] = k + 1
        results.append(result)

    avg_sharpe = np.mean([r["metrics"]["sharpe_ratio"]       for r in results])
    avg_return = np.mean([r["metrics"]["total_return_pct"]   for r in results])
    avg_dd     = np.mean([r["metrics"]["max_drawdown_pct"]   for r in results])

    print(f"\n[Walk-Forward Summary]")
    print(f"  Avg Sharpe: {avg_sharpe:.3f}")
    print(f"  Avg Return: {avg_return:.2f}%")
    print(f"  Avg MaxDD:  {avg_dd:.2f}%")

    return results


# ─── Equity Curve Plot ────────────────────────────────────────────────────────

def plot_equity_curve(result: dict, title: str = "Equity Curve", save_path: str = None):
    equity = result["equity_curve"]
    dates  = result["dates_curve"]
    trades = result["trades"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    # ── Equity curve ─────────────────────────────────────────────────────────
    ax1.plot(dates, equity, color="#00bfff", linewidth=2.0, label="Equity")
    ax1.fill_between(dates, equity[0], equity, alpha=0.15, color="#00bfff")

    # Drawdown shading
    peak = np.maximum.accumulate(equity)
    ax1.fill_between(dates, equity, peak, alpha=0.2, color="#e74c3c", label="Drawdown")

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (INR)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add metrics text box
    m = result["metrics"]
    info = (f"Return: {m['total_return_pct']}%  |  Sharpe: {m['sharpe_ratio']}  |  "
            f"MaxDD: {m['max_drawdown_pct']}%  |  WinRate: {m['win_rate']:.0%}  |  Trades: {m['num_trades']}")
    ax1.set_xlabel(info, fontsize=9)

    # ── Trade P&L distribution ────────────────────────────────────────────────
    if len(trades) > 0:
        colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in trades["net_pnl"]]
        ax2.bar(range(len(trades)), trades["net_pnl"], color=colors, alpha=0.8)
        ax2.axhline(0, color="white", linewidth=0.8)
        ax2.set_ylabel("Trade P&L (INR)")
        ax2.set_xlabel("Trade #")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved → {save_path}")
    plt.show()
