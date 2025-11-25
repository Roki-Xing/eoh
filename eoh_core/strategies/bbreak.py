"""
Reusable Bollinger Breakout backtest and associated dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BBreakParams:
    n: int
    k: float
    h: int


@dataclass(frozen=True)
class RiskParams:
    commission: float = 0.0
    slippage_bps: float = 0.0
    delay_days: int = 0
    max_position: float = 1.0
    max_daily_turnover: float = 1.0
    min_trades: int = 0
    min_exposure: float = 0.0


@dataclass
class BacktestMetrics:
    total_return: float
    sharpe: float
    max_drawdown: float
    trades: int
    exposure: float
    turnover: float
    cost_ratio: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "return": self.total_return,
            "sharpe": self.sharpe,
            "mdd": self.max_drawdown,
            "trades": float(self.trades),
            "exposure": self.exposure,
            "turnover": self.turnover,
            "cost_ratio": self.cost_ratio,
        }


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    positions: pd.DataFrame
    equity: pd.Series
    metrics: BacktestMetrics


def compute_bbands(close: pd.Series, n: int, k: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower


def _apply_delay(series: pd.Series, delay_days: int) -> pd.Series:
    if delay_days <= 0:
        return series
    return series.shift(delay_days)


def _exec_price(side: str, price: float, risk: RiskParams) -> float:
    slip = price * (risk.slippage_bps * 1e-4)
    if side == "BUY":
        return price + slip
    return price - slip


def _compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    risk: RiskParams,
) -> BacktestMetrics:
    daily_ret = equity.pct_change().fillna(0.0)
    if len(equity) > 1:
        total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    else:
        total_return = 0.0
    sharpe = 0.0
    std = daily_ret.std(ddof=0)
    if std > 1e-12:
        sharpe = (daily_ret.mean() / std) * np.sqrt(252.0)
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_drawdown = drawdown.min() if len(drawdown) else 0.0
    exposure = positions["position"].mean() if len(positions) else 0.0

    if len(positions) > 1:
        ts = pd.to_datetime(positions["timestamp"], errors="coerce")
        pos_change = positions["position"].diff().abs().fillna(0.0)
        turnover_series = pos_change.groupby(ts.dt.normalize()).sum()
        turnover = float(turnover_series.mean()) if not turnover_series.empty else 0.0
    else:
        turnover = 0.0

    trades_count = len(trades)
    cost_ratio = trades_count * (risk.commission + risk.slippage_bps * 1e-4)

    return BacktestMetrics(
        total_return=float(total_return),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
        trades=int(trades_count),
        exposure=float(exposure),
        turnover=float(turnover),
        cost_ratio=float(cost_ratio),
    )


def backtest_bbreak(frame: pd.DataFrame, params: BBreakParams, risk: RiskParams) -> BacktestResult:
    """
    Execute a long-only Bollinger breakout strategy and return evaluation artifacts.
    """
    px = frame["Close"].astype(float).copy()
    ma, up, lo = compute_bbands(px, params.n, params.k)

    buy_sig = px > up
    sell_sig = px < ma

    buy_sig = _apply_delay(buy_sig, risk.delay_days)
    sell_sig = _apply_delay(sell_sig, risk.delay_days)

    position = 0
    last_buy_idx = None
    equity_values = []
    positions = []
    trades = []
    prev_price = None

    for i, (timestamp, price) in enumerate(px.items()):
        if prev_price is None:
            eq = 1.0
        else:
            if position == 1:
                eq = equity_values[-1] * (price / prev_price)
            else:
                eq = equity_values[-1]

        if position == 0 and bool(buy_sig.iloc[i]) and not np.isnan(buy_sig.iloc[i]):
            exec_price = _exec_price("BUY", price, risk)
            eq *= (1.0 - risk.commission)
            position = 1
            last_buy_idx = i
            trades.append(
                {"timestamp": timestamp, "side": "BUY", "price": float(exec_price), "size": 1, "reason": "break_up"}
            )
        elif position == 1:
            can_exit = True
            if last_buy_idx is not None and (i - last_buy_idx) < params.h:
                can_exit = False
            if can_exit and bool(sell_sig.iloc[i]) and not np.isnan(sell_sig.iloc[i]):
                exec_price = _exec_price("SELL", price, risk)
                eq *= (1.0 - risk.commission)
                position = 0
                trades.append(
                    {"timestamp": timestamp, "side": "SELL", "price": float(exec_price), "size": 1, "reason": "below_ma"}
                )

        equity_values.append(eq)
        positions.append({"timestamp": timestamp, "position": position, "price": float(price), "equity": float(eq)})
        prev_price = price

    equity_series = pd.Series(equity_values, index=px.index, name="equity")
    positions_df = pd.DataFrame(positions)
    trades_df = pd.DataFrame(trades)

    metrics = _compute_metrics(equity_series, trades_df, positions_df, risk)
    return BacktestResult(trades=trades_df, positions=positions_df, equity=equity_series, metrics=metrics)


def enforce_constraints(result: BacktestResult, risk: RiskParams) -> bool:
    """
    Evaluate basic feasibility constraints against a backtest result.
    """
    if risk.min_trades and result.metrics.trades < risk.min_trades:
        return False
    if risk.min_exposure and result.metrics.exposure < risk.min_exposure:
        return False
    return True
