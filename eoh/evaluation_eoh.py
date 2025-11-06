# /root/autodl-tmp/quant/evaluation_eoh.py
# 多策略（BBreak/SMACross/RSIMR）评估器，兼容 eoh_evolve_main.py 的官方 EoH 调用。
# 依赖：numpy, pandas

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

# =============== 工具函数 ===============

def _sma(a: pd.Series, n: int) -> pd.Series:
    return a.rolling(int(n), min_periods=int(n)).mean()

def _rsi_wilder(close: pd.Series, n: int) -> pd.Series:
    # Wilder RSI 实现
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder 的 alpha = 1/n 的 EMA
    alpha = 1.0 / float(n)
    avg_gain = up.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = down.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)
    return rsi

def _max_drawdown_from_equity(eq: np.ndarray) -> float:
    # eq 为累计净值（>=0）
    roll_max = np.maximum.accumulate(eq)
    dd = eq / roll_max - 1.0
    return float(np.min(dd))

def _sharpe_daily(eq: np.ndarray, eps=1e-12) -> float:
    # 基于净值序列计算日收益后年化 Sharpe（252）
    ret = np.diff(eq) / (eq[:-1] + eps)
    if ret.size < 2:
        return 0.0
    mu = np.mean(ret)
    sd = np.std(ret, ddof=1) + eps
    return float(np.sqrt(252.0) * mu / sd)

def _exposure_from_pos(pos: np.ndarray) -> float:
    return float(np.mean(pos)) if pos.size else 0.0

# =============== 三类策略的信号与回测 ===============

@dataclass
class BBreakParams:
    n: int
    k: float
    h: int

@dataclass
class SMACrossParams:
    f: int  # fast
    s: int  # slow
    h: int

@dataclass
class RSIMRParams:
    n: int
    os: int
    ob: int
    h: int

def _backtest_bbreak(close: pd.Series, p: BBreakParams, commission: float) -> Dict[str, Any]:
    n, k, hold = int(p.n), float(p.k), int(p.h)

    ma = _sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    upper = ma + k * std
    # lower = ma - k * std  # 可用于可视化

    eq = np.zeros(len(close), dtype=float); eq[0] = 1.0
    pos = np.zeros(len(close), dtype=int)
    trades: List[Tuple[pd.Timestamp, str, float, int, str]] = []
    hold_ctr = 0
    in_pos = False

    for i in range(1, len(close)):
        c = close.iloc[i]
        m = ma.iloc[i]
        u = upper.iloc[i]
        # 进场：收盘上破上轨
        if (not in_pos) and (u == u) and (c > u):
            in_pos = True
            hold_ctr = 0
            pos[i] = 1
            eq[i] = eq[i-1] * (1.0 - commission)
            trades.append((close.index[i], "BUY", float(c), 1, "break_up"))
        elif in_pos:
            hold_ctr += 1
            pos[i] = 1
            # 出场：收盘跌破均线 或 达到最小持有
            if ((m == m) and (c < m)) or (hold_ctr >= hold):
                in_pos = False
                pos[i] = 0
                eq[i] = eq[i-1] * (1.0 - commission)
                trades.append((close.index[i], "SELL", float(c), 1, "min_hold" if (hold_ctr >= hold and (m != m or c >= m)) else "below_ma"))

        if pos[i] == 0 and eq[i] == 0:
            eq[i] = eq[i-1]
        # 净值推进
        if pos[i] == 1:
            eq[i] = eq[i] if eq[i] != 0 else eq[i-1]
            eq[i] *= float(c / close.iloc[i-1])
        else:
            eq[i] = eq[i] if eq[i] != 0 else eq[i-1]

    return {
        "pos": pos,
        "equity": eq,
        "trades": trades,
        "overlay": {"ma": ma.values, "upper": upper.values}
    }

def _backtest_smacross(close: pd.Series, p: SMACrossParams, commission: float) -> Dict[str, Any]:
    f, s, hold = int(p.f), int(p.s), int(p.h)
    f = max(2, f); s = max(f+1, s)  # 保证 s > f

    fast = _sma(close, f)
    slow = _sma(close, s)

    eq = np.zeros(len(close), dtype=float); eq[0] = 1.0
    pos = np.zeros(len(close), dtype=int)
    trades: List[Tuple[pd.Timestamp, str, float, int, str]] = []
    hold_ctr = 0
    in_pos = False

    # 上穿/下穿：使用前一日位置判断
    for i in range(1, len(close)):
        c = close.iloc[i]
        f0, s0 = fast.iloc[i-1], slow.iloc[i-1]
        f1, s1 = fast.iloc[i], slow.iloc[i]

        cross_up = (f0 <= s0) and (f1 > s1)
        cross_dn = (f0 >= s0) and (f1 < s1)

        if (not in_pos) and cross_up and (f1 == f1) and (s1 == s1):
            in_pos = True; hold_ctr = 0
            pos[i] = 1
            eq[i] = eq[i-1] * (1.0 - commission)
            trades.append((close.index[i], "BUY", float(c), 1, "cross_up"))
        elif in_pos:
            hold_ctr += 1
            pos[i] = 1
            if cross_dn or (hold_ctr >= hold):
                in_pos = False; pos[i] = 0
                eq[i] = eq[i-1] * (1.0 - commission)
                trades.append((close.index[i], "SELL", float(c), 1, "cross_down" if cross_dn else "min_hold"))

        if pos[i] == 0 and eq[i] == 0:
            eq[i] = eq[i-1]
        if pos[i] == 1:
            eq[i] = eq[i] if eq[i] != 0 else eq[i-1]
            eq[i] *= float(c / close.iloc[i-1])
        else:
            eq[i] = eq[i] if eq[i] != 0 else eq[i-1]

    return {
        "pos": pos,
        "equity": eq,
        "trades": trades,
        "overlay": {"fast": fast.values, "slow": slow.values}
    }

def _backtest_rsimr(close: pd.Series, p: RSIMRParams, commission: float) -> Dict[str, Any]:
    n, os, ob, hold = int(p.n), int(p.os), int(p.ob), int(p.h)
    os = max(1, min(os, 49)); ob = max(51, min(ob, 99))
    rsi = _rsi_wilder(close, n)

    eq = np.zeros(len(close), dtype=float); eq[0] = 1.0
    pos = np.zeros(len(close), dtype=int)
    trades: List[Tuple[pd.Timestamp, str, float, int, str]] = []
    hold_ctr = 0
    in_pos = False

    for i in range(1, len(close)):
        c = close.iloc[i]
        rr = rsi.iloc[i]
        if (not in_pos) and (rr < os):
            in_pos = True; hold_ctr = 0
            pos[i] = 1
            eq[i] = eq[i-1] * (1.0 - commission)
            trades.append((close.index[i], "BUY", float(c), 1, "rsi_oversold"))
        elif in_pos:
            hold_ctr += 1
            pos[i] = 1
            if (rr > ob) or (hold_ctr >= hold):
                in_pos = False; pos[i] = 0
                eq[i] = eq[i-1] * (1.0 - commission)
                trades.append((close.index[i], "SELL", float(c), 1, "rsi_overbought" if rr > ob else "min_hold"))

        if pos[i] == 0 and eq[i] == 0:
            eq[i] = eq[i-1]
        if pos[i] == 1:
            eq[i] = eq[i] if eq[i] != 0 else eq[i-1]
            eq[i] *= float(c / close.iloc[i-1])
        else:
            eq[i] = eq[i] if eq[i] != 0 else eq[i-1]

    return {
        "pos": pos,
        "equity": eq,
        "trades": trades,
        "overlay": {"rsi": rsi.values, "os": float(os), "ob": float(ob)}
    }

# =============== 解析候选 & 计算指标 ===============

_F_BBREAK = re.compile(r"^BBreak_n(?P<n>\d+)_k(?P<k>\d+\.?\d*)_h(?P<h>\d+)$")
_F_SMA    = re.compile(r"^SMACross_f(?P<f>\d+)_s(?P<s>\d+)_h(?P<h>\d+)$")
_F_RSIMR  = re.compile(r"^RSIMR_n(?P<n>\d+)_os(?P<os>\d+)_ob(?P<ob>\d+)_h(?P<h>\d+)$")

def parse_program_text(name: str) -> Tuple[str, Dict[str, float]]:
    """
    解析程序名 -> (family, params_dict)
    支持：
      - BBreak_n{n}_k{k}_h{h}
      - SMACross_f{f}_s{s}_h{h}
      - RSIMR_n{n}_os{os}_ob{ob}_h{h}
    """
    m = _F_BBREAK.match(name)
    if m:
        d = {k: float(v) for k, v in m.groupdict().items()}
        d["n"] = int(d["n"]); d["h"] = int(d["h"]); d["k"] = float(d["k"])
        return "BBreak", d
    m = _F_SMA.match(name)
    if m:
        d = {k: float(v) for k, v in m.groupdict().items()}
        d["f"] = int(d["f"]); d["s"] = int(d["s"]); d["h"] = int(d["h"])
        return "SMACross", d
    m = _F_RSIMR.match(name)
    if m:
        d = {k: float(v) for k, v in m.groupdict().items()}
        d["n"] = int(d["n"]); d["os"] = int(d["os"]); d["ob"] = int(d["ob"]); d["h"] = int(d["h"])
        return "RSIMR", d
    raise ValueError(f"无法解析候选名：{name}")

def _metrics_from_equity(eq: np.ndarray) -> Dict[str, float]:
    mdd = _max_drawdown_from_equity(eq)
    total_ret = float(eq[-1] - 1.0)
    sharpe = _sharpe_daily(eq)
    return {"ret": total_ret, "sharpe": sharpe, "mdd": mdd}

# =============== 官方 EoH 评估封装 ===============

class MultiStrategyEvaluation:
    """
    EoH 官方调用期望的接口：
      - __init__(df_train, df_test, commission, constraints...)
      - evaluate_program(program_text: str) -> dict(fitness=..., metrics_train=..., metrics_test=..., name=program_text)
    """
    def __init__(self,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 commission: float = 0.0005,
                 min_trades: int = 8,
                 min_expo: float = 0.45,
                 max_expo: float = 0.95):
        self.close_train = df_train["Close"].astype(float)
        self.close_test  = df_test["Close"].astype(float)
        self.commission = float(commission)
        self.min_trades = int(min_trades)
        self.min_expo = float(min_expo)
        self.max_expo = float(max_expo)

    def _run_one(self, family: str, params: Dict[str, float], side: str) -> Dict[str, Any]:
        close = self.close_train if side == "train" else self.close_test
        if family == "BBreak":
            res = _backtest_bbreak(close, BBreakParams(n=int(params["n"]), k=float(params["k"]), h=int(params["h"])), self.commission)
        elif family == "SMACross":
            f, s = int(params["f"]), int(params["s"])
            if s <= f:
                s = f + 1
            res = _backtest_smacross(close, SMACrossParams(f=f, s=s, h=int(params["h"])), self.commission)
        elif family == "RSIMR":
            n, os, ob, h = int(params["n"]), int(params["os"]), int(params["ob"]), int(params["h"])
            if ob <= os: ob = os + 1
            res = _backtest_rsimr(close, RSIMRParams(n=n, os=os, ob=ob, h=h), self.commission)
        else:
            raise ValueError(f"未知 family: {family}")
        return res

    def _check_constraints(self, res: Dict[str, Any]) -> Optional[str]:
        pos = res["pos"]
        expo = _exposure_from_pos(pos)
        trades = len(res["trades"])
        if trades < self.min_trades:
            return f"few_trades({trades}<{self.min_trades})"
        if not (self.min_expo <= expo <= self.max_expo):
            return f"expo_out({expo:.2f})"
        return None

    def evaluate_program(self, program_text: str) -> Dict[str, Any]:
        # 解析
        family, params = parse_program_text(program_text)

        # 训练侧
        train_res = self._run_one(family, params, "train")
        msg = self._check_constraints(train_res)
        if msg is not None:
            return {
                "name": program_text,
                "fitness": -1e9,
                "reason": msg,
                "metrics_train": {},
                "metrics_test": {}
            }

        # 测试侧
        test_res = self._run_one(family, params, "test")

        # 指标
        m_train = _metrics_from_equity(train_res["equity"])
        m_test  = _metrics_from_equity(test_res["equity"])
        expo = _exposure_from_pos(train_res["pos"])

        # 适应度：Sharpe 优先 + 轻度收益奖励 + MDD 惩罚（可按需调整）
        fitness = (
            1.00 * m_train["sharpe"] +
            0.25 * m_train["ret"] -
            0.50 * abs(m_train["mdd"]) +
            0.10 * (expo - 0.6)   # 过低/过高的暴露给予轻微惩罚
        )

        return {
            "name": program_text,
            "fitness": float(fitness),
            "metrics_train": {
                "ret": m_train["ret"], "sharpe": m_train["sharpe"], "mdd": m_train["mdd"],
                "trades": len(train_res["trades"]), "expo": expo
            },
            "metrics_test": {
                "ret": m_test["ret"], "sharpe": m_test["sharpe"], "mdd": m_test["mdd"],
                "trades": len(test_res["trades"]), "expo": _exposure_from_pos(test_res["pos"])
            }
        }
