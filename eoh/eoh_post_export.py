#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export best strategy artifacts from the latest generation CSV.
"""
import argparse
import json
import os
import platform
import re
import sys
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eoh_core import (
    BBreakParams,
    RiskParams,
    backtest_bbreak,
    compute_bbands,
    robust_read_csv,
    slice_by_date,
)
from eoh_core.utils import ensure_dir, log

def bh_equity_from_close(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    return (close / close.iloc[0])

def yearly_report(equity: pd.Series, trades: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    """Return yearly stats with B&H columns merged outside"""
    # align equity to px index (避免布尔掩码长度不一致)
    equity = equity.reindex(px.index).ffill().fillna(1.0)
    years = sorted(set(pd.to_datetime(px.index).year))
    rows = []
    for y in years:
        mask = (pd.to_datetime(px.index).year == y)
        s_eq = equity[mask]
        if len(s_eq) < 2:
            continue
        r_y = s_eq.iloc[-1]/s_eq.iloc[0]-1.0
        ret = s_eq.pct_change().dropna()
        vol = ret.std(ddof=0)*np.sqrt(252.0) if len(ret)>2 else 0.0
        sharpe = (ret.mean()/ret.std(ddof=0)*np.sqrt(252.0)) if len(ret)>2 and ret.std(ddof=0)>1e-12 else 0.0
        roll_max = s_eq.cummax()
        mdd = (s_eq/roll_max - 1.0).min() if len(s_eq)>0 else 0.0
        # trades in year
        if not trades.empty:
            tmask = (pd.to_datetime(trades["timestamp"]).dt.year == y)
            tcount = int(trades[tmask].shape[0])
        else:
            tcount = 0
        # exposure
        exposure = None
        # approximate exposure by positions
        # (caller会在外面另算更精准的，这里先留空或由外部填充)
        rows.append({
            "year": int(y), "strat_return": float(r_y), "strat_vol": float(vol),
            "strat_sharpe": float(sharpe), "strat_mdd": float(mdd),
            "strat_trades": tcount
        })
    return pd.DataFrame(rows)

# -------------------------
# Parse best from gen csv
# -------------------------
def parse_best_from_gencsv(outdir: str) -> Tuple[str, Dict[str,Any], str]:
    # pick the highest genXX
    gens = []
    for f in os.listdir(outdir):
        m = re.match(r"gen(\d+)\.csv$", f)
        if m:
            gens.append(int(m.group(1)))
    if not gens:
        raise FileNotFoundError("未找到 genxx.csv")
    g = max(gens)
    path = os.path.join(outdir, f"gen{g:02d}.csv")
    log(f"[INFO] gen{g:02d}.csv -> {path}")
    df = pd.read_csv(path)
    if "fitness" not in df.columns:
        raise ValueError("gen csv 缺少 fitness 列")
    best = df.iloc[df["fitness"].argmax()]
    name = str(best["name"])
    # supports only BBreak
    m = re.match(r"BBreak_n(\d+)_k([0-9.]+)_h(\d+)", name)
    if not m:
        raise ValueError(f"无法从 name 解析参数：{name}")
    n = int(m.group(1)); k = float(m.group(2)); h = int(m.group(3))
    params = {"n": n, "k": k, "h": h}
    return "BBreak", params, name

# -------------------------
# Plotting
# -------------------------
def plot_price_bbands_trades(df: pd.DataFrame, n: int, k: float,
                             trades: pd.DataFrame, positions: pd.DataFrame,
                             out_png: str, title: str,
                             add_bh: bool=True):
    px = df["Close"].astype(float)
    ma, up, lo = compute_bbands(px, n, k)
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0])
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    # price + bands
    ax1.plot(px.index, px.values, label="Price")
    ax1.plot(ma.index, ma.values, label=f"MA{n}")
    ax1.plot(up.index, up.values, label=f"Upper k={k}")
    ax1.plot(lo.index, lo.values, label=f"Lower k={k}")
    # trades markers
    if not trades.empty:
        buys  = trades[trades["side"]=="BUY"]
        sells = trades[trades["side"]=="SELL"]
        ax1.scatter(buys["timestamp"], px.reindex(pd.to_datetime(buys["timestamp"])).values, marker="^", s=40, label="BUY")
        ax1.scatter(sells["timestamp"], px.reindex(pd.to_datetime(sells["timestamp"])).values, marker="v", s=40, label="SELL")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    # equity panel
    eq = positions.set_index("timestamp")["equity"].astype(float)
    eq = eq.reindex(px.index).ffill().fillna(1.0)
    ax2.plot(eq.index, eq.values, label="Strategy equity")
    if add_bh:
        bh = bh_equity_from_close(px)
        ax2.plot(bh.index, bh.values, label="Buy&Hold")
    ax2.legend(loc="upper left")
    plt.savefig(out_png, dpi=150)
    log(f"[INFO] plot saved -> {out_png}")

def plot_side_by_side(df_train, df_test, n,k, trades_train, pos_train, trades_test, pos_test, out_png):
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])

    def _one(ax1, ax2, df, tt, pp, title):
        px = df["Close"].astype(float)
        ma, up, lo = compute_bbands(px, n, k)
        ax1.plot(px.index, px.values, label="Price")
        ax1.plot(ma.index, ma.values, label=f"MA{n}")
        ax1.plot(up.index, up.values, label=f"Upper k={k}")
        ax1.plot(lo.index, lo.values, label=f"Lower k={k}")
        if not tt.empty:
            buys = tt[tt["side"]=="BUY"]
            sells = tt[tt["side"]=="SELL"]
            ax1.scatter(buys["timestamp"], px.reindex(pd.to_datetime(buys["timestamp"])).values, marker="^", s=40, label="BUY")
            ax1.scatter(sells["timestamp"], px.reindex(pd.to_datetime(sells["timestamp"])).values, marker="v", s=40, label="SELL")
        ax1.set_title(title)
        ax1.legend(loc="upper left")
        eq = pp.set_index("timestamp")["equity"].astype(float).reindex(px.index).ffill().fillna(1.0)
        ax2.plot(eq.index, eq.values, label="Strategy equity")
        bh = bh_equity_from_close(px)
        ax2.plot(bh.index, bh.values, label="Buy&Hold")
        ax2.legend(loc="upper left")

    ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[1,0])
    _one(ax1, ax2, df_train, trades_train, pos_train, "TRAIN")

    ax3 = fig.add_subplot(gs[0,1]); ax4 = fig.add_subplot(gs[1,1])
    _one(ax3, ax4, df_test, trades_test, pos_test, "TEST")

    plt.savefig(out_png, dpi=150)
    log(f"[INFO] side-by-side plot saved -> {out_png}")

def plot_html(df: pd.DataFrame, n: int, k: float,
              trades: pd.DataFrame, positions: pd.DataFrame, out_html: str, title: str):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        log(f"[WARN] Plotly not available ({e}), skip HTML for {title}")
        return
    px = df["Close"].astype(float)
    ma, up, lo = compute_bbands(px, n, k)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7,0.3], vertical_spacing=0.03,
                        subplot_titles=(title,"Equity vs B&H"))
    fig.add_trace(go.Scatter(x=px.index, y=px.values, name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ma.index, y=ma.values, name=f"MA{n}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=up.index, y=up.values, name=f"Upper k={k}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=lo.index, y=lo.values, name=f"Lower k={k}"), row=1, col=1)
    if not trades.empty:
        buys = trades[trades["side"]=="BUY"]
        sells= trades[trades["side"]=="SELL"]
        fig.add_trace(go.Scatter(x=buys["timestamp"], y=px.reindex(pd.to_datetime(buys["timestamp"])).values,
                                 mode="markers", marker_symbol="triangle-up", name="BUY"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=sells["timestamp"], y=px.reindex(pd.to_datetime(sells["timestamp"])).values,
                                 mode="markers", marker_symbol="triangle-down", name="SELL"),
                      row=1, col=1)
    eq = positions.set_index("timestamp")["equity"].astype(float).reindex(px.index).ffill().fillna(1.0)
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy equity"), row=2, col=1)
    bh = bh_equity_from_close(px)
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy&Hold"), row=2, col=1)
    fig.update_layout(height=700, width=1200)
    fig.write_html(out_html, include_plotlyjs="cdn")
    log(f"[INFO] html saved -> {out_html}")

# -------------------------
# Export helpers
# -------------------------
def export_table(df: pd.DataFrame, path: str, expect_cols: List[str]):
    if not df.empty:
        # enforce column order if existing
        cols = [c for c in expect_cols if c in df.columns]
        out = df.copy()
        if cols:
            out = out[cols]
        out.to_csv(path, index=False)
        log(f"[INFO] saved -> {path} (rows={out.shape[0]})")
    else:
        # write empty file with header
        pd.DataFrame(columns=expect_cols).to_csv(path, index=False)
        log(f"[INFO] saved -> {path} (rows=0)")

def export_best_strategy_py(outdir: str, family: str, params: Dict[str,Any]):
    path = os.path.join(outdir, "best_strategy.py")
    code = f'''# Auto-generated best strategy
# Family: {family}
# Params: {json.dumps(params)}
import pandas as pd
import numpy as np

def compute_bbands(close: pd.Series, n: int, k: float):
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    up = ma + k*sd
    lo = ma - k*sd
    return ma, up, lo

def run_bbreak(df: pd.DataFrame, n={int(params["n"])}, k={float(params["k"])}, hold={int(params["h"])}, commission=0.0005):
    px = df["Close"].astype(float)
    ma, up, lo = compute_bbands(px, n, k)
    buy_sig  = (px > up)
    sell_sig = (px < ma)
    pos=0; last_buy=None; eq=1.0; prev=None
    equity=[]; trades=[]; pos_rows=[]
    for i,(dt, price) in enumerate(px.items()):
        if prev is None:
            eq = 1.0
        else:
            if pos==1: eq *= (price/prev)
        # signals
        if pos==0 and buy_sig.iloc[i]:
            eq *= (1.0-commission); pos=1; last_buy=i
            trades.append({{"timestamp":dt,"side":"BUY","price":float(price),"size":1,"reason":"break_up"}})
        elif pos==1:
            can_exit = True if (last_buy is None) else ((i-last_buy)>=hold)
            if can_exit and sell_sig.iloc[i]:
                eq *= (1.0-commission); pos=0
                trades.append({{"timestamp":dt,"side":"SELL","price":float(price),"size":1,"reason":"below_ma"}})
        equity.append(eq); pos_rows.append({{"timestamp":dt,"position":pos,"price":float(price),"equity":float(eq)}})
        prev = price
    return pd.DataFrame(trades), pd.DataFrame(pos_rows), pd.Series(equity, index=px.index, name="equity")
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    log(f"[INFO] best_strategy.py saved -> {path}")

def snapshot_run_params(outdir: str, args: argparse.Namespace, family: str, params: Dict[str,Any], best_name: str):
    snap = {
        "symbol": args.symbol, "csv": args.csv,
        "train_start": args.train_start, "train_end": args.train_end,
        "test_start": args.test_start, "test_end": args.test_end,
        "commission": args.commission,
        "population": args.population, "generations": args.generations, "topk": args.topk,
        "seed": args.seed, "model_dir": getattr(args, "model_dir", ""),
        "family": family, "best_name": best_name, "params": params,
        # 风控参数快照（如果 evolve 阶段传过来，可在这里一并保存；默认给出）
        "slippage_bps": getattr(args, "slippage_bps", 0.0),
        "delay_days": getattr(args, "delay_days", 0),
        "max_position": getattr(args, "max_position", 1.0),
        "max_daily_turnover": getattr(args, "max_daily_turnover", 1.0),
        "min_trades": getattr(args, "min_trades", 0),
        "min_exposure": getattr(args, "min_exposure", 0.0),
        "alpha": getattr(args, "alpha", 1.0),
        "beta": getattr(args, "beta", 0.5),
        "gamma": getattr(args, "gamma", 1.0),
        "delta": getattr(args, "delta", 0.0),
        "zeta": getattr(args, "zeta", 0.0),
    }
    p = os.path.join(outdir, "run_params.json")
    with open(p,"w",encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    log(f"[INFO] run_params.json saved -> {p}")

def snapshot_env(outdir: str):
    p = os.path.join(outdir, "env_summary.txt")
    lines = []
    lines.append(f"Python: {sys.version}")
    lines.append(f"Platform: {platform.platform()}")
    # key packages
    def _ver(modname):
        try:
            mod = __import__(modname)
            v = getattr(mod, "__version__", "unknown")
            return f"{modname}: {v}"
        except Exception as e:
            return f"{modname}: not installed"
    for m in ["numpy","pandas","matplotlib","plotly","torch","transformers"]:
        lines.append(_ver(m))
    # GPU info via torch if available
    try:
        import torch
        lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA device count: {torch.cuda.device_count()}")
            lines.append(f"Current device: {torch.cuda.current_device()}")
            lines.append(f"Device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        pass
    with open(p,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"[INFO] env_summary.txt saved -> {p}")

def append_report(
    outdir: str,
    side_name: str,
    family: str,
    params: Dict[str, Any],
    ypath: str,
    best_name: str,
    args: argparse.Namespace,
    *,
    metrics: Dict[str, float] | None = None,
) -> None:
    report = os.path.join(outdir, "REPORT.md")
    # load yearly report sample lines
    df = pd.read_csv(ypath)
    lines = []
    lines.append(f"## {side_name.upper()} — {best_name}")
    lines.append(f"- Family: **{family}**, Params: `{json.dumps(params)}`")
    if metrics:
        lines.append(
            "- Metrics: return={return:.2%}, sharpe={sharpe:.2f}, mdd={mdd:.2%}, "
            "turnover={turnover:.2%}, exposure={exposure:.2%}, trades={trades:.0f}, cost={cost_ratio:.4f}".format(
                **metrics
            )
        )
    if not df.empty:
        # show last row quickly
        tail = df.iloc[-1].to_dict()
        lines.append(f"- Last year: stratR={tail.get('strat_return',0):.2%}, bhR={tail.get('bh_return',0):.2%}, "
                     f"stratSharpe={tail.get('strat_sharpe',0):.2f}, bhSharpe={tail.get('bh_sharpe',0):.2f}")
    lines.append("")
    if not os.path.isfile(report):
        # write header with param snapshot
        head = [
            "# REPORT",
            "",
            "### Run params snapshot",
            "```json",
            json.dumps({
                "symbol": args.symbol, "csv": args.csv,
                "train": [args.train_start, args.train_end],
                "test": [args.test_start, args.test_end],
                "commission": args.commission,
                "population": args.population, "generations": args.generations, "topk": args.topk,
                "seed": args.seed, "model_dir": getattr(args,"model_dir","")
            }, ensure_ascii=False, indent=2),
            "```",
            ""
        ]
        with open(report, "w", encoding="utf-8") as f:
            f.write("\n".join(head + lines) + "\n")
    else:
        with open(report, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    log(f"[INFO] report appended -> {report}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", dest="model_dir", default="", help="optional for snapshot only")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--test_start", required=True)
    ap.add_argument("--test_end", required=True)
    ap.add_argument("--population", type=int, default=56)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--commission", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--html", action="store_true", help="export plotly html in addition to png")
    # 允许把 evolve 的风险参数一起快照进 run_params.json（不强制传）
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--delay_days", type=int, default=0)
    ap.add_argument("--max_position", type=float, default=1.0)
    ap.add_argument("--max_daily_turnover", type=float, default=1.0)
    ap.add_argument("--min_trades", type=int, default=0)
    ap.add_argument("--min_exposure", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--zeta", type=float, default=0.0)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    family, params, best_name = parse_best_from_gencsv(args.outdir)
    log(f"[INFO] best from gen -> {best_name} (family={family}, params={params})")

    # data
    df_all = robust_read_csv(args.csv)
    df_train = slice_by_date(df_all, args.train_start, args.train_end)
    df_test  = slice_by_date(df_all, args.test_start, args.test_end)
    log(f"[INFO] data loaded: train={len(df_train)} rows, test={len(df_test)} rows")

    n, k, h = int(params["n"]), float(params["k"]), int(params["h"])

    # ---- RUN train
    log("--------------------------------------------------")
    log(f"[DEBUG] Backtest BBreak n={n}, k={k}, hold={h}, commission={args.commission}")
    train_result = backtest_bbreak(df_train, strategy_params, risk)
    tr_trades = train_result.trades
    tr_pos = train_result.positions
    tr_eq = train_result.equity
    train_metrics = train_result.metrics.as_dict()
    log(
        "[TRAIN] return={return:.2%} sharpe={sharpe:.2f} mdd={mdd:.2%} trades={trades} cost={cost_ratio:.4f}".format(
            **train_metrics
        )
    )
    export_table(tr_trades, os.path.join(args.outdir, "best_trades_train.csv"),
                 ["timestamp","side","price","size","reason"])
    export_table(tr_pos, os.path.join(args.outdir, "best_positions_train.csv"),
                 ["timestamp","position","price","equity"])
    eq_train = pd.DataFrame({
        "timestamp": df_train.index,
        "equity": tr_eq.reindex(df_train.index).ffill().fillna(1.0).values,
        "bh_equity": bh_equity_from_close(df_train["Close"]).values
    })
    export_table(eq_train, os.path.join(args.outdir, "best_equity_train.csv"),
                 ["timestamp","equity","bh_equity"])
    y_train = yearly_report(tr_eq, tr_trades, df_train["Close"])
    bh = bh_equity_from_close(df_train["Close"])
    bh_rows=[]
    for y in sorted(set(df_train.index.year)):
        mask = (df_train.index.year==y)
        seq = bh[mask]; ret = seq.pct_change().dropna()
        r_y = seq.iloc[-1]/seq.iloc[0]-1.0 if len(seq)>1 else 0.0
        vol = ret.std(ddof=0)*np.sqrt(252.0) if len(ret)>2 else 0.0
        sharpe = (ret.mean()/ret.std(ddof=0)*np.sqrt(252.0)) if len(ret)>2 and ret.std(ddof=0)>1e-12 else 0.0
        roll_max = seq.cummax(); mdd = (seq/roll_max - 1.0).min() if len(seq)>0 else 0.0
        bh_rows.append({"year": int(y), "bh_return": float(r_y), "bh_vol": float(vol),
                        "bh_sharpe": float(sharpe), "bh_mdd": float(mdd)})
    y_train = y_train.merge(pd.DataFrame(bh_rows), on="year", how="left")
    ypath = os.path.join(args.outdir, "yearly_report_train.csv")
    y_train.to_csv(ypath, index=False)
    log(f"[INFO] yearly report -> {ypath}")
    plot_price_bbands_trades(df_train, n,k, tr_trades, tr_pos,
                             os.path.join(args.outdir, "best_plot_train.png"),
                             f"[TRAIN] {best_name}")
    if args.html:
        plot_html(df_train, n,k, tr_trades, tr_pos,
                  os.path.join(args.outdir, "best_plot_train.html"),
                  f"[TRAIN] {best_name}")

    # ---- RUN test
    test_result = backtest_bbreak(df_test, strategy_params, risk)
    tr_trades_t = test_result.trades
    tr_pos_t = test_result.positions
    tr_eq_t = test_result.equity
    test_metrics = test_result.metrics.as_dict()
    log(
        "[TEST] return={return:.2%} sharpe={sharpe:.2f} mdd={mdd:.2%} trades={trades} cost={cost_ratio:.4f}".format(
            **test_metrics
        )
    )
    export_table(tr_trades_t, os.path.join(args.outdir, "best_trades_test.csv"),
                 ["timestamp","side","price","size","reason"])
    export_table(tr_pos_t, os.path.join(args.outdir, "best_positions_test.csv"),
                 ["timestamp","position","price","equity"])
    eq_test = pd.DataFrame({
        "timestamp": df_test.index,
        "equity": tr_eq_t.reindex(df_test.index).ffill().fillna(1.0).values,
        "bh_equity": bh_equity_from_close(df_test["Close"]).values
    })
    export_table(eq_test, os.path.join(args.outdir, "best_equity_test.csv"),
                 ["timestamp","equity","bh_equity"])
    y_test = yearly_report(tr_eq_t, tr_trades_t, df_test["Close"])
    bh = bh_equity_from_close(df_test["Close"])
    bh_rows=[]
    for y in sorted(set(df_test.index.year)):
        mask = (df_test.index.year==y)
        seq = bh[mask]; ret = seq.pct_change().dropna()
        r_y = seq.iloc[-1]/seq.iloc[0]-1.0 if len(seq)>1 else 0.0
        vol = ret.std(ddof=0)*np.sqrt(252.0) if len(ret)>2 else 0.0
        sharpe = (ret.mean()/ret.std(ddof=0)*np.sqrt(252.0)) if len(ret)>2 and ret.std(ddof=0)>1e-12 else 0.0
        roll_max = seq.cummax(); mdd = (seq/roll_max - 1.0).min() if len(seq)>0 else 0.0
        bh_rows.append({"year": int(y), "bh_return": float(r_y), "bh_vol": float(vol),
                        "bh_sharpe": float(sharpe), "bh_mdd": float(mdd)})
    y_test = y_test.merge(pd.DataFrame(bh_rows), on="year", how="left")
    ypath_t = os.path.join(args.outdir, "yearly_report_test.csv")
    y_test.to_csv(ypath_t, index=False)
    log(f"[INFO] yearly report -> {ypath_t}")
    plot_price_bbands_trades(df_test, n,k, tr_trades_t, tr_pos_t,
                             os.path.join(args.outdir, "best_plot_test.png"),
                             f"[TEST] {best_name}")
    if args.html:
        plot_html(df_test, n,k, tr_trades_t, tr_pos_t,
                  os.path.join(args.outdir, "best_plot_test.html"),
                  f"[TEST] {best_name}")

    # side-by-side PNG
    plot_side_by_side(df_train, df_test, n,k,
                      tr_trades, tr_pos, tr_trades_t, tr_pos_t,
                      os.path.join(args.outdir, "best_plot_side_by_side.png"))

    # snapshots
    snapshot_run_params(args.outdir, args, family, params, best_name)
    snapshot_env(args.outdir)
    export_best_strategy_py(args.outdir, family, params)

    # report append
    append_report(
        args.outdir,
        "train",
        family,
        params,
        os.path.join(args.outdir, "yearly_report_train.csv"),
        best_name,
        args,
        metrics=train_metrics,
    )
    append_report(
        args.outdir,
        "test",
        family,
        params,
        os.path.join(args.outdir, "yearly_report_test.csv"),
        best_name,
        args,
        metrics=test_metrics,
    )
    log(f"[INFO] All done for best '{best_name}'.")

if __name__ == "__main__":
    main()
