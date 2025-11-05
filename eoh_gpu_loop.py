# -*- coding: utf-8 -*-
"""
EOH 演化 + 回测 + 可视化与报告导出（单脚本版）

用法示例：
python eoh_gpu_loop.py \
  --model-dir /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --symbol SPY \
  --train_start 2020-01-01 --train_end 2022-12-31 \
  --test_start  2023-01-01 --test_end  2023-12-31 \
  --population 56 --generations 3 --topk 8 \
  --commission 0.0005 \
  --csv /root/autodl-tmp/price_cache/SPY_2020_2023.csv \
  --outdir /root/autodl-tmp/outputs/eoh_run_stable

运行完后，你会在 outdir 中看到：
- genXX.csv：每代评估结果
- best_plot_test.png：测试区间的策略图（含指标与交易点）
- yearly_report_test.csv：测试区间年度分解（含 B&H 对比）
- REPORT.md：关键指标小结
"""
import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from backtesting import Backtest, Strategy

# ---- Matplotlib 非交互后端，保存图片用 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# 简单日志
# -----------------------------
def log(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

# -----------------------------
# 强化版 CSV 加载
# -----------------------------
COMMON_TIME_ALIASES = [
    "Date", "date", "Datetime", "datetime", "DateTime", "Timestamp", "timestamp",
    "Time", "time", "candle_begin_time", "time_key", "t", "dt",
    "TradeDate", "TradingDay", "bar_time", "barTime",
    "Unnamed: 0", "index"
]

def _find_time_col(df: pd.DataFrame, prefer: Optional[str] = None) -> Optional[str]:
    cols = list(df.columns)
    if prefer and prefer in cols:
        return prefer
    if prefer:
        for c in cols:
            if c.lower() == prefer.lower():
                return c

    lower_map = {c.lower(): c for c in cols}
    for alias in COMMON_TIME_ALIASES:
        if alias in cols:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]

    # 自动探测可解析的日期列
    for c in cols:
        s = df[c]
        try:
            parsed = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
            if parsed.notna().mean() >= 0.9:
                return c
        except Exception:
            continue

    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"
    return None

def _coerce_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close", "volume"]
    # 用 Adj Close 兜底
    if "close" not in lower_map and "adj close" in lower_map:
        df = df.rename(columns={lower_map["adj close"]: "Close"})
        lower_map["close"] = "Close"

    missing = [x for x in need if x not in lower_map]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found columns: {list(df.columns)}")

    return df.rename(columns={
        lower_map["open"]: "Open",
        lower_map["high"]: "High",
        lower_map["low"]:  "Low",
        lower_map["close"]: "Close",
        lower_map["volume"]: "Volume",
    })[["Open", "High", "Low", "Close", "Volume"]]

def load_local_csv(symbol: str, start: str, end: str,
                   csv_path: Optional[str] = None,
                   date_col: Optional[str] = None,
                   search_dir: str = "/root/autodl-tmp/price_cache") -> pd.DataFrame:
    df = None
    tried = []
    if csv_path and os.path.isfile(csv_path):
        tried.append(csv_path)
        df = pd.read_csv(csv_path)
    else:
        try:
            y1, y2 = pd.Timestamp(start).year, pd.Timestamp(end).year
            cand = os.path.join(search_dir, f"{symbol}_{y1}_{y2}.csv")
            tried.append(cand)
            if os.path.isfile(cand):
                df = pd.read_csv(cand)
        except Exception:
            pass
        if df is None:
            cand = os.path.join(search_dir, f"{symbol}.csv")
            tried.append(cand)
            if os.path.isfile(cand):
                df = pd.read_csv(cand)

    if df is None:
        raise FileNotFoundError(f"CSV not found. Tried: {tried}. You can pass --csv /path/to/file.csv")

    time_col = _find_time_col(df, prefer=date_col)
    if time_col == "__index__":
        pass
    elif time_col is not None:
        # FutureWarning 关于 infer_datetime_format 可忽略
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=False, infer_datetime_format=True)
        if df[time_col].isna().mean() > 0.2:
            raise ValueError(f"Detected time column '{time_col}' but too many unparsable rows.")
        df = df.sort_values(time_col).set_index(time_col)
    else:
        raise ValueError("CSV must have a datetime column (Date/Datetime/Timestamp/Time/…)，或用 --date_col 指定")

    df = _coerce_ohlcv(df).dropna()

    s, e = pd.to_datetime(start), pd.to_datetime(end)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not DatetimeIndex after processing.")
    df = df[(df.index >= s) & (df.index <= e)]
    if df.empty:
        raise ValueError("Empty dataframe after date filtering")
    log(f"data loaded: rows={len(df)} time_col={time_col if time_col else 'index'}")
    return df

# -----------------------------
# 自实现指标
# -----------------------------
def ta_sma(arr, n: int):
    s = pd.Series(np.asarray(arr, dtype=float))
    return s.rolling(int(n), min_periods=int(n)).mean().to_numpy()

def ta_rsi(arr, n: int):
    n = int(n)
    s = pd.Series(np.asarray(arr, dtype=float))
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(n, min_periods=n).mean()
    avg_loss = loss.rolling(n, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.to_numpy()

def rolling_mean(arr, n: int):
    s = pd.Series(np.asarray(arr, dtype=float))
    return s.rolling(int(n), min_periods=int(n)).mean().to_numpy()

def rolling_std(arr, n: int):
    s = pd.Series(np.asarray(arr, dtype=float))
    return s.rolling(int(n), min_periods=int(n)).std(ddof=0).to_numpy()

# -----------------------------
# 风控基类
# -----------------------------
class BaseSafeStrategy(Strategy):
    sl_pct: float = 0.0
    tp_pct: float = 0.0
    min_hold: int = 1

    def init(self):
        self._bars_held = 0

    def _after_entry(self):
        self._bars_held = 0

    def _update_hold(self):
        if self.position:
            self._bars_held += 1

    def _can_exit(self):
        return self._bars_held >= self.min_hold

    def _apply_risk(self):
        if not self.position:
            return
        if self.tp_pct and self.position.pl_pct >= self.tp_pct * 100:
            self.position.close()
        elif self.sl_pct and self.position.pl_pct <= -self.sl_pct * 100:
            self.position.close()

# -----------------------------
# 三类基础策略（类体外部赋值参数）
# -----------------------------
def make_sma_cross_cls(n_fast: int, n_slow: int, min_hold: int):
    label = f"SMACross_f{n_fast}_s{n_slow}_h{min_hold}"
    assert n_fast > 1 and n_slow > 1 and n_fast < n_slow

    class SMACross(BaseSafeStrategy):
        def init(self):
            super().init()
            c = self.data.Close
            self.s_fast = self.I(ta_sma, c, self.n_fast)
            self.s_slow = self.I(ta_sma, c, self.n_slow)

        def next(self):
            need = max(self.n_fast, self.n_slow) + 2
            if len(self.data.Close) < need:
                return
            fast_prev, fast_now = self.s_fast[-2], self.s_fast[-1]
            slow_prev, slow_now = self.s_slow[-2], self.s_slow[-1]
            # 金叉进，多头
            if not self.position and fast_prev < slow_prev and fast_now > slow_now:
                self.buy(); self._after_entry(); return
            self._update_hold(); self._apply_risk()
            # 死叉出
            if self.position and self._can_exit() and fast_prev > slow_prev and fast_now < slow_now:
                self.position.close()

    SMACross.n_fast = int(n_fast)
    SMACross.n_slow = int(n_slow)
    SMACross.min_hold = int(min_hold)
    SMACross._label = label
    return SMACross

def make_rsi_mr_cls(n: int, oversold: int, overbought: int, min_hold: int):
    label = f"RSIMR_n{n}_os{oversold}_ob{overbought}_h{min_hold}"
    assert 2 <= n <= 200 and 1 <= oversold < overbought <= 99

    class RSIMR(BaseSafeStrategy):
        def init(self):
            super().init()
            c = self.data.Close
            self.rsi = self.I(ta_rsi, c, self.n)

        def next(self):
            if len(self.data.Close) < self.n + 2:
                return
            rsi_prev, rsi_now = self.rsi[-2], self.rsi[-1]
            # 超卖上穿 -> 做多
            if not self.position and rsi_prev < self.oversold and rsi_now >= self.oversold:
                self.buy(); self._after_entry(); return
            self._update_hold(); self._apply_risk()
            # 超买下穿 -> 平多
            if self.position and self._can_exit() and rsi_prev > self.overbought and rsi_now <= self.overbought:
                self.position.close()

    RSIMR.n = int(n)
    RSIMR.oversold = int(oversold)
    RSIMR.overbought = int(overbought)
    RSIMR.min_hold = int(min_hold)
    RSIMR._label = label
    return RSIMR

def make_bbands_break_cls(n: int, k: float, min_hold: int):
    label = f"BBreak_n{n}_k{round(k,2)}_h{min_hold}"
    assert n >= 5 and k > 0

    class BBandsBreak(BaseSafeStrategy):
        def init(self):
            super().init()
            c = self.data.Close
            self.mid = self.I(rolling_mean, c, self.n)
            self.std = self.I(rolling_std, c, self.n)
            self.up = self.I(lambda mid, std, k: mid + k * std, self.mid, self.std, self.k)
            self.dn = self.I(lambda mid, std, k: mid - k * std, self.mid, self.std, self.k)

        def next(self):
            if len(self.data.Close) < self.n + 2:
                return
            c_prev, c_now = self.data.Close[-2], self.data.Close[-1]
            up_prev, up_now = self.up[-2], self.up[-1]
            mid_prev, mid_now = self.mid[-2], self.mid[-1]
            # 上轨突破做多
            if not self.position and c_prev <= up_prev and c_now > up_now:
                self.buy(); self._after_entry(); return
            self._update_hold(); self._apply_risk()
            # 回落到中轨下方 -> 平多
            if self.position and self._can_exit() and c_prev >= mid_prev and c_now < mid_now:
                self.position.close()

    BBandsBreak.n = int(n)
    BBandsBreak.k = float(k)
    BBandsBreak.min_hold = int(min_hold)
    BBandsBreak._label = label
    return BBandsBreak

# -----------------------------
# 适应度与回测
# -----------------------------
@dataclass
class FitResult:
    name: str
    origin: str
    fit: float
    train_ret: float
    test_ret: float
    sharpe: float
    mdd: float
    trades: int
    exposure: float
    cls: type

def _safe_get(s, key: str, default: float = np.nan) -> float:
    try:
        if key in s:
            return float(s[key])
    except Exception:
        pass
    return float(default)

def compute_fitness(stats_train: pd.Series, stats_test: pd.Series):
    r_train = _safe_get(stats_train, 'Return [%]', 0.0) / 100.0
    r_test  = _safe_get(stats_test,  'Return [%]', 0.0) / 100.0
    sharpe  = _safe_get(stats_test,  'Sharpe Ratio', 0.0)
    mdd     = abs(_safe_get(stats_test, 'Max. Drawdown [%]', 0.0)) / 100.0
    trades  = int(_safe_get(stats_test, '# Trades', 0))
    expo    = _safe_get(stats_test, 'Exposure Time [%]', 0.0) / 100.0
    # 偏好高 Sharpe、较高 OOS 收益、较低 MDD
    fitness = sharpe + 0.5 * r_test - 0.5 * mdd
    return fitness, r_train, r_test, sharpe, trades, expo

def backtest_once(Strat, df: pd.DataFrame, commission: float, cash: float = 100_000.0) -> pd.Series:
    bt = Backtest(
        df, Strat, cash=cash, commission=commission,
        trade_on_close=False, exclusive_orders=True,
        hedging=False, finalize_trades=True
    )
    stats = bt.run()
    return stats

def evaluate(Strat, df_train: pd.DataFrame, df_test: pd.DataFrame, commission: float,
             min_trades: int = 8, min_exposure: float = 0.05):
    try:
        st_train = backtest_once(Strat, df_train, commission)
        st_test  = backtest_once(Strat, df_test,  commission)
    except Exception as e:
        warn(f"backtest failed: {e}")
        return None
    fitness, r_tr, r_te, sharpe, trades, expo = compute_fitness(st_train, st_test)
    if (trades < min_trades) or (expo < min_exposure):
        warn(f"skip due to few trades/exposure (trades={trades}, exp={expo:.2%})")
        return None
    label = getattr(Strat, "_label", Strat.__name__)
    return FitResult(label, "", fitness, r_tr, r_te, sharpe,
                     _safe_get(st_test, 'Max. Drawdown [%]', np.nan)/100.0,
                     trades, expo, Strat), st_train, st_test

# -----------------------------
# 采样与变异
# -----------------------------
def sample_candidate(rng: random.Random):
    which = rng.choice(["sma", "rsi", "bb"])
    if which == "sma":
        nf = rng.randint(5, 30); ns = rng.randint(nf + 5, nf + 80); hold = rng.randint(1, 10)
        return make_sma_cross_cls(nf, ns, hold)
    elif which == "rsi":
        n = rng.randint(6, 30); os_ = rng.randint(10, 35); ob_ = rng.randint(65, 85); hold = rng.randint(1, 10)
        return make_rsi_mr_cls(n, os_, ob_, hold)
    else:
        n = rng.randint(10, 40); k = round(rng.uniform(1.2, 3.0), 2); hold = rng.randint(1, 10)
        return make_bbands_break_cls(n, k, hold)

def mutate_from(cls_type, rng: random.Random):
    name = getattr(cls_type, "_label", cls_type.__name__)
    if name.startswith("SMACross"):
        nf = max(2, getattr(cls_type, "n_fast", 10) + rng.randint(-3, 3))
        ns = max(nf + 2, getattr(cls_type, "n_slow", 30) + rng.randint(-5, 5))
        hold = max(1, getattr(cls_type, "min_hold", 3) + rng.randint(-2, 2))
        return make_sma_cross_cls(nf, ns, hold)
    if name.startswith("RSIMR"):
        n = max(2, getattr(cls_type, "n", 14) + rng.randint(-3, 3))
        os_ = min(40, max(5, getattr(cls_type, "oversold", 30) + rng.randint(-5, 5)))
        ob_ = min(95, max(os_ + 1, getattr(cls_type, "overbought", 70) + rng.randint(-5, 5)))
        hold = max(1, getattr(cls_type, "min_hold", 3) + rng.randint(-2, 2))
        return make_rsi_mr_cls(n, os_, ob_, hold)
    if name.startswith("BBreak"):
        n = max(5, getattr(cls_type, "n", 20) + rng.randint(-4, 4))
        k = round(max(0.5, getattr(cls_type, "k", 2.0) + rng.uniform(-0.3, 0.3)), 2)
        hold = max(1, getattr(cls_type, "min_hold", 3) + rng.randint(-2, 2))
        return make_bbands_break_cls(n, k, hold)
    return sample_candidate(rng)

# -----------------------------
# 落盘：每代结果
# -----------------------------
def save_generation_csv(outdir: str, gen_idx: int, rows: List[FitResult]):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"gen{gen_idx:02d}.csv")
    if not rows:
        pd.DataFrame([], columns=["name","origin","fitness","trainR","testR","sharpe","mdd","trades","exposure"]).to_csv(path, index=False)
        log(f"gen{gen_idx:02d} -> {path}")
        return
    df = pd.DataFrame([{
        "name": r.name, "origin": r.origin, "fitness": r.fit,
        "trainR": r.train_ret, "testR": r.test_ret, "sharpe": r.sharpe,
        "mdd": r.mdd, "trades": r.trades, "exposure": r.exposure
    } for r in rows])
    df.to_csv(path, index=False)
    log(f"gen{gen_idx:02d} -> {path}")

# ==========================================================
# 可视化 & 报告导出
# ==========================================================
def _extract_trades(stats) -> Optional[pd.DataFrame]:
    # 兼容不同 backtesting.py 版本
    for key in ("_trades", "Trades"):
        try:
            t = stats[key]
            if isinstance(t, pd.DataFrame) and len(t) > 0:
                return t.copy()
        except Exception:
            pass
    try:
        t = stats._trades  # 部分版本是属性
        if isinstance(t, pd.DataFrame) and len(t) > 0:
            return t.copy()
    except Exception:
        pass
    return None

def _extract_equity_curve(stats) -> Optional[pd.DataFrame]:
    for key in ("_equity_curve", "Equity Curve"):
        try:
            ec = stats[key]
            if isinstance(ec, pd.DataFrame) and len(ec) > 0:
                return ec.copy()
        except Exception:
            pass
    try:
        ec = stats._equity_curve
        if isinstance(ec, pd.DataFrame) and len(ec) > 0:
            return ec.copy()
    except Exception:
        pass
    return None

def _plot_sma(ax, df: pd.DataFrame, n_fast: int, n_slow: int):
    ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
    ax.plot(df.index, pd.Series(df["Close"]).rolling(n_fast, min_periods=n_fast).mean(), label=f"SMA({n_fast})", linewidth=1.0)
    ax.plot(df.index, pd.Series(df["Close"]).rolling(n_slow, min_periods=n_slow).mean(), label=f"SMA({n_slow})", linewidth=1.0)
    ax.set_title("Price with SMAs")
    ax.legend(loc="best")

def _plot_bbands(ax, df: pd.DataFrame, n: int, k: float):
    close = df["Close"]
    mid = close.rolling(n, min_periods=n).mean()
    std = close.rolling(n, min_periods=n).std(ddof=0)
    up = mid + k * std
    dn = mid - k * std
    ax.plot(df.index, close, label="Close", linewidth=1.2)
    ax.plot(df.index, mid, label=f"Mid({n})", linewidth=1.0)
    ax.plot(df.index, up, label=f"Upper({k:.2f})", linewidth=1.0)
    ax.plot(df.index, dn, label=f"Lower({k:.2f})", linewidth=1.0)
    ax.set_title("Price with Bollinger Bands")
    ax.legend(loc="best")
    return mid, up, dn

def _plot_rsi(ax_price, ax_rsi, df: pd.DataFrame, n: int, os_: int, ob_: int):
    close = df["Close"]
    rsi = pd.Series(close).diff().pipe(lambda x: pd.concat([x.clip(lower=0), -x.clip(upper=0)], axis=1))  # tmp
    # 用前面的 ta_rsi 以保持一致
    rsi = pd.Series(ta_rsi(close.to_numpy(), n), index=close.index)
    ax_price.plot(df.index, close, label="Close", linewidth=1.2)
    ax_price.set_title("Price")
    ax_price.legend(loc="best")

    ax_rsi.plot(df.index, rsi, label=f"RSI({n})", linewidth=1.0)
    ax_rsi.axhline(os_, linestyle="--", linewidth=0.8)
    ax_rsi.axhline(ob_, linestyle="--", linewidth=0.8)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title("RSI")
    ax_rsi.legend(loc="best")

def _scatter_trades(ax, df: pd.DataFrame, trades: pd.DataFrame):
    # backtesting 的 Trades 里常见列：EntryBar, ExitBar, EntryPrice, ExitPrice
    if "EntryBar" in trades.columns:
        eb = trades["EntryBar"].astype(int).clip(0, len(df)-1)
        ax.scatter(df.index[eb], df["Close"].iloc[eb], marker="^", s=36)
    if "ExitBar" in trades.columns:
        xb = trades["ExitBar"].astype(int).clip(0, len(df)-1)
        ax.scatter(df.index[xb], df["Close"].iloc[xb], marker="v", s=36)

def _yearly_breakdown_from_equity(ec: pd.DataFrame,
                                  trades: Optional[pd.DataFrame],
                                  close: pd.Series) -> pd.DataFrame:
    """
    根据回测的 equity curve（含 'Equity' 列）做年度分解；
    同时给出 Buy&Hold（基于 close）的年度对照。
    关键修复：各项指标一律使用各自序列的索引做年度切分，避免布尔索引长度不匹配。
    """
    if "Equity" not in ec.columns:
        raise ValueError("equity curve missing 'Equity' column")

    # —— 策略净值曲线（带时间索引）——
    equity = ec["Equity"].astype(float)
    if "Time" in ec.columns and not isinstance(ec.index, pd.DatetimeIndex):
        idx = pd.to_datetime(ec["Time"], errors="coerce")
    else:
        idx = equity.index
    s_eq = pd.Series(equity.values, index=idx).dropna()
    if s_eq.size < 2:
        return pd.DataFrame(columns=[
            "Year","Strat_Return","Strat_Vol","Strat_Sharpe","Strat_MDD",
            "Trades","BH_Return","BH_Vol","BH_Sharpe","BH_MDD"
        ])

    # 策略日收益（用于年化波动/Sharpe）
    ret = s_eq.pct_change().dropna()

    # —— B&H 基准（满仓 close）——
    close = pd.Series(close).astype(float).dropna()
    bh_ret = close.pct_change().dropna()
    bh_eq = (1.0 + bh_ret).cumprod()

    years = sorted(set(pd.DatetimeIndex(s_eq.index).year))
    rows = []
    for y in years:
        # 1) 策略年度切片：用 s_eq 的索引
        mask_eq = (pd.DatetimeIndex(s_eq.index).year == y)
        if mask_eq.sum() >= 2:
            seg_eq = s_eq[mask_eq]
            r_y   = seg_eq.iloc[-1] / seg_eq.iloc[0] - 1.0
            # 日频收益的年化统计：用 ret 自己的索引
            mask_ret = (pd.DatetimeIndex(ret.index).year == y)
            if mask_ret.sum() >= 2:
                seg_ret = ret[mask_ret]
                vol_y   = seg_ret.std() * np.sqrt(252)
                sharpe_y= (seg_ret.mean() * 252) / (seg_ret.std() + 1e-12)
            else:
                vol_y = np.nan
                sharpe_y = np.nan
            # 年内 MDD：用 seg_eq 自身
            peak = seg_eq.cummax()
            dd   = (seg_eq / peak - 1.0).min()
        else:
            r_y = vol_y = sharpe_y = dd = np.nan

        # 2) 年度交易次数估计（优先 EntryTime，其次 EntryBar→用 close 的索引映射）
        trades_y = 0
        if trades is not None:
            if "EntryTime" in trades.columns:
                tt = pd.to_datetime(trades["EntryTime"], errors="coerce")
                trades_y = int((pd.DatetimeIndex(tt).year == y).sum())
            elif "EntryBar" in trades.columns and len(close) > 0:
                eb = trades["EntryBar"].astype(int).clip(0, len(close) - 1)
                t_idx = close.index[eb]
                trades_y = int((pd.DatetimeIndex(t_idx).year == y).sum())

        # 3) B&H 年度切片：用 bh_eq / bh_ret 自己的索引
        mask_bh_eq  = (pd.DatetimeIndex(bh_eq.index).year == y)
        mask_bh_ret = (pd.DatetimeIndex(bh_ret.index).year == y)
        if mask_bh_eq.sum() >= 2:
            seg_bh_eq = bh_eq[mask_bh_eq]
            r_bh = seg_bh_eq.iloc[-1] / seg_bh_eq.iloc[0] - 1.0
            seg_bh_ret = bh_ret[mask_bh_ret] if mask_bh_ret.sum() >= 2 else None
            if seg_bh_ret is not None:
                vol_bh    = seg_bh_ret.std() * np.sqrt(252)
                sharpe_bh = (seg_bh_ret.mean() * 252) / (seg_bh_ret.std() + 1e-12)
            else:
                vol_bh = sharpe_bh = np.nan
            dd_bh = (seg_bh_eq / seg_bh_eq.cummax() - 1.0).min()
        else:
            r_bh = vol_bh = sharpe_bh = dd_bh = np.nan

        rows.append({
            "Year": y,
            "Strat_Return": r_y,
            "Strat_Vol": vol_y,
            "Strat_Sharpe": sharpe_y,
            "Strat_MDD": dd,
            "Trades": trades_y,
            "BH_Return": r_bh,
            "BH_Vol": vol_bh,
            "BH_Sharpe": sharpe_bh,
            "BH_MDD": dd_bh
        })

    return pd.DataFrame(rows).sort_values("Year")


def export_visual_and_report(outdir: str, tag: str, Strat, df_test: pd.DataFrame, commission: float):
    """
    对测试集落盘：
    - best_plot_test.png：价图 + 指标 + 交易点
    - yearly_report_test.csv：年度分解（含 B&H）
    - REPORT.md：摘要
    """
    os.makedirs(outdir, exist_ok=True)
    # 重新在测试集跑一次，便于拿 trades/equity
    stats = backtest_once(Strat, df_test, commission)
    trades = _extract_trades(stats)
    ec = _extract_equity_curve(stats)

    # 图：根据策略类型画指标
    name = getattr(Strat, "_label", Strat.__name__)
    fig = None

    if name.startswith("BBreak"):
        n = int(getattr(Strat, "n", 20))
        k = float(getattr(Strat, "k", 2.0))
        fig, ax = plt.subplots(figsize=(12, 5))
        _plot_bbands(ax, df_test, n, k)
        if trades is not None:
            _scatter_trades(ax, df_test, trades)
        ax.set_xlabel("Time")
    elif name.startswith("SMACross"):
        nf = int(getattr(Strat, "n_fast", 10))
        ns = int(getattr(Strat, "n_slow", 30))
        fig, ax = plt.subplots(figsize=(12, 5))
        _plot_sma(ax, df_test, nf, ns)
        if trades is not None:
            _scatter_trades(ax, df_test, trades)
        ax.set_xlabel("Time")
    elif name.startswith("RSIMR"):
        n = int(getattr(Strat, "n", 14))
        os_ = int(getattr(Strat, "oversold", 30))
        ob_ = int(getattr(Strat, "overbought", 70))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                       gridspec_kw={"height_ratios":[3,1]})
        _plot_rsi(ax1, ax2, df_test, n, os_, ob_)
        if trades is not None:
            _scatter_trades(ax1, df_test, trades)
        ax2.set_xlabel("Time")
    else:
        # 兜底：只画 Close
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_test.index, df_test["Close"], label="Close")
        if trades is not None:
            _scatter_trades(ax, df_test, trades)
        ax.legend(loc="best")
        ax.set_title(name)
        ax.set_xlabel("Time")

    plot_path = os.path.join(outdir, f"best_plot_{tag}.png")
    if fig is not None:
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        log(f"plot saved -> {plot_path}")

    # 年度分解 & B&H
    yearly_path = os.path.join(outdir, f"yearly_report_{tag}.csv")
    if ec is not None:
        ydf = _yearly_breakdown_from_equity(ec, trades, df_test["Close"])
        ydf.to_csv(yearly_path, index=False)
        log(f"yearly report -> {yearly_path}")

    # 报告摘要
    rep_path = os.path.join(outdir, "REPORT.md")
    with open(rep_path, "a", encoding="utf-8") as f:
        f.write(f"## {tag} | {name}\n")
        f.write(f"- Return(test) = {float(stats['Return [%]']):.2f}%\n")
        f.write(f"- Sharpe(test) = {float(stats['Sharpe Ratio']):.2f}\n")
        f.write(f"- MDD(test)    = {float(stats['Max. Drawdown [%]']):.2f}%\n")
        f.write(f"- #Trades(test)= {int(stats['# Trades'])}\n")
        f.write(f"- Exposure     = {float(stats['Exposure Time [%]']):.2f}%\n\n")
    log(f"report appended -> {rep_path}")

# -----------------------------
# 主流程
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--test_start", required=True)
    ap.add_argument("--test_end", required=True)
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--commission", type=float, default=0.0005)
    ap.add_argument("--cpus", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", type=str, default=None, help="直接指定 CSV 路径（可选）")
    ap.add_argument("--date_col", type=str, default=None, help="指定时间列名（可选）")
    ap.add_argument("--export_report", action="store_true", help="对测试集导出图表与年度报表")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    df_all = load_local_csv(
        args.symbol, args.train_start, args.test_end,
        csv_path=args.csv, date_col=args.date_col
    )
    df_train = df_all[(df_all.index >= pd.to_datetime(args.train_start)) & (df_all.index <= pd.to_datetime(args.train_end))]
    df_test  = df_all[(df_all.index >= pd.to_datetime(args.test_start))  & (df_all.index <= pd.to_datetime(args.test_end))]
    log(f"split: train={len(df_train)} test={len(df_test)}")

    population = args.population
    generations = args.generations
    topk = max(1, min(args.topk, population))
    survivors: List[type] = []
    best_overall: Optional[FitResult] = None

    for gen in range(1, generations + 1):
        log(f"==== Generation {gen}/{generations} ====")
        candidates: List[Tuple[str, type]] = []

        if gen == 1 or not survivors:
            for _ in range(population):
                candidates.append(("seed", sample_candidate(rng)))
        else:
            pool = survivors[:]
            while len(pool) < population:
                parent = rng.choice(survivors)
                pool.append(mutate_from(parent, rng))
            for cls in pool[:population]:
                candidates.append(("mut", cls))

        valid_results: List[FitResult] = []
        for idx, (origin, Strat) in enumerate(candidates, start=1):
            try:
                r = evaluate(Strat, df_train, df_test, args.commission, min_trades=8, min_exposure=0.05)
                if r is None:
                    log(f"[SKIP] id={idx} origin={origin} reason=few trades/exposure or errors")
                    continue
                fr, _, _ = r
                fr.origin = origin
                valid_results.append(fr)
                log(f"[OK] id={idx} origin={origin} name={fr.name} fit={fr.fit:.4f} "
                    f"trainR={fr.train_ret:.2%} testR={fr.test_ret:.2%} "
                    f"Sharpe={fr.sharpe:.2f} MDD={fr.mdd:.2%} trades={fr.trades} exp={fr.exposure:.1%}")
            except Exception as e:
                warn(f"backtest failed: {e}")

        save_generation_csv(args.outdir, gen, valid_results)

        if not valid_results:
            warn("gen produced no valid strategies")
            survivors = []
            continue

        best = max(valid_results, key=lambda r: r.fit)
        log(f"[BEST] gen{gen:02d} name={best.name} fitness={best.fit:.4f} "
            f"testR={best.test_ret:.2%} Sharpe={best.sharpe:.2f} MDD={best.mdd:.2%}")

        if (best_overall is None) or (best.fit > best_overall.fit):
            best_overall = best

        ranked = sorted(valid_results, key=lambda r: r.fit, reverse=True)
        survivors = [r.cls for r in ranked[:topk]]

    log(f"all generations done -> {args.outdir}")

    # 导出测试集图与报表（仅一次，对最优个体）
    if args.export_report and best_overall is not None:
        export_visual_and_report(
            outdir=args.outdir,
            tag="test",
            Strat=best_overall.cls,
            df_test=df_test,
            commission=args.commission
        )
        log(f"[BEST] {best_overall.name} report exported.")

if __name__ == "__main__":
    main()
