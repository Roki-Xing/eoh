#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Walk-Forward 汇总脚本
读取多个 OUTDIR（每个 OUTDIR 来自 eoh_post_export.py 的导出），汇总 test 集表现并出图。

需要的文件（每个 OUTDIR 最好具备，缺失也会容错）：
- best_equity_test.csv       (timestamp, equity)
- best_positions_test.csv    (timestamp, position, price, equity)  # 用于计算 Buy&Hold
- best_trades_test.csv       (timestamp, side, price, size, reason) # 仅用于计数
- yearly_report_test.csv     # 若包含 bh_* 列可直接使用，否则用 price 计算 BH

输出：
- walkforward_summary.csv
- walkforward_equity_compare.png  （每个目录一个小面板：策略 vs B&H）
- walkforward_equity_overlay.png  （所有策略净值叠加）

用法示例：
python /root/autodl-tmp/quant/eoh_walkforward_aggregate.py \
  --outdirs /root/autodl-tmp/outputs/eoh_run_20to21 \
           /root/autodl-tmp/outputs/eoh_run_20to22 \
           /root/autodl-tmp/outputs/eoh_run_stable \
  --result-dir /root/autodl-tmp/outputs/walkforward
"""

import argparse
import os
import sys
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ANNUAL_DAYS = 252.0


def _read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # 尝试标准化时间列
    for c in ["timestamp", "Timestamp", "date", "Date", "time", "Time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).copy()
            df = df.sort_values(c)
            df = df.rename(columns={c: "timestamp"})
            df = df.reset_index(drop=True)
            break
    return df


def _mdd_from_equity(eq: pd.Series) -> float:
    """最大回撤，返回负数（如 -0.12 表示 -12%）"""
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    return float(dd.min()) if len(dd) else np.nan


def _sharpe_from_equity(eq: pd.Series) -> float:
    """基于日频净值的年化 Sharpe（rf=0）"""
    if len(eq) < 3:
        return np.nan
    ret = eq.pct_change().dropna()
    if len(ret) < 2 or ret.std() == 0:
        return np.nan
    return float(np.sqrt(ANNUAL_DAYS) * ret.mean() / ret.std())


def _ann_return_from_equity(eq: pd.Series) -> float:
    """年化收益（CAGR）：(last/first)^(252/N) - 1"""
    if len(eq) < 2:
        return np.nan
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n = float(len(eq))
    cagr = (float(eq.iloc[-1] / eq.iloc[0])) ** (ANNUAL_DAYS / n) - 1.0
    return float(cagr)


def _total_return_from_equity(eq: pd.Series) -> float:
    """总收益：last/first - 1"""
    if len(eq) < 2:
        return np.nan
    return float(eq.iloc[-1] / eq.iloc[0] - 1.0)


def _exposure_from_positions(pos_df: Optional[pd.DataFrame]) -> float:
    """暴露 = 平均持仓非零比例"""
    if pos_df is None or "position" not in pos_df.columns:
        return np.nan
    p = (pos_df["position"].values != 0).astype(float)
    return float(np.mean(p)) if len(p) else np.nan


def _bh_equity_from_positions(pos_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """用 positions 的 price 列构造 Buy&Hold 净值：price / price[0]"""
    if pos_df is None or "price" not in pos_df.columns:
        return None
    px = pos_df[["timestamp", "price"]].dropna().copy()
    if px.empty:
        return None
    px = px.sort_values("timestamp")
    eq = px["price"] / px["price"].iloc[0]
    eq.index = px["timestamp"].values
    eq.name = "bh_equity"
    return eq


def _align_equities(eq: pd.Series, bh: Optional[pd.Series]) -> Tuple[pd.Series, Optional[pd.Series]]:
    """对齐策略与 BH 净值（按交集对齐）"""
    if bh is None:
        return eq, None
    df = pd.DataFrame({"eq": eq})
    df["bh"] = bh
    df = df.dropna()
    return df["eq"], df["bh"]


def analyze_one_outdir(outdir: str) -> Dict:
    """
    返回一个 dict，包含：
    - label: 目录名
    - period: 'YYYY-MM-DD -> YYYY-MM-DD'
    - metrics for strategy: total_return, ann_return, sharpe, mdd, trades, exposure
    - metrics for bh: bh_total_return, bh_ann_return, bh_sharpe, bh_mdd
    - equity series: eq (pd.Series), bh (pd.Series or None)
    """
    label = os.path.basename(os.path.abspath(outdir))
    res = {
        "label": label,
        "outdir": outdir,
        "period": "",
        "total_return": np.nan,
        "ann_return": np.nan,
        "sharpe": np.nan,
        "mdd": np.nan,
        "trades": np.nan,
        "exposure": np.nan,
        "bh_total_return": np.nan,
        "bh_ann_return": np.nan,
        "bh_sharpe": np.nan,
        "bh_mdd": np.nan,
        "eq_series": None,
        "bh_series": None,
    }

    # 读取 test 侧导出
    p_eq = os.path.join(outdir, "best_equity_test.csv")
    p_pos = os.path.join(outdir, "best_positions_test.csv")
    p_trd = os.path.join(outdir, "best_trades_test.csv")
    p_yrl = os.path.join(outdir, "yearly_report_test.csv")

    df_eq = _read_csv_safe(p_eq)
    df_pos = _read_csv_safe(p_pos)
    df_trd = _read_csv_safe(p_trd)
    df_yrl = _read_csv_safe(p_yrl)

    if df_eq is None or "timestamp" not in df_eq.columns or "equity" not in df_eq.columns:
        print(f"[WARN] skip '{outdir}': missing best_equity_test.csv or columns.", file=sys.stderr)
        return res

    # 策略净值
    eq = df_eq.sort_values("timestamp").copy()
    s_eq = eq["equity"].astype(float)
    s_eq.index = eq["timestamp"].values
    s_eq.name = "equity"

    # BH 净值优先从 positions 的 price 计算
    s_bh = _bh_equity_from_positions(df_pos)

    # 若 yearly_report_test.csv 带有 bh 年度数据，不能直接汇总成总净值，因此仍优先用 price 复原 bh 净值
    # （若未来在导出阶段另存 bh_equity_test.csv，可在此读取替代）

    # 对齐净值
    s_eq, s_bh = _align_equities(s_eq, s_bh)

    # 统计周期
    if len(s_eq) >= 2:
        res["period"] = f"{pd.Timestamp(s_eq.index[0]).date()} -> {pd.Timestamp(s_eq.index[-1]).date()}"

    # 策略指标
    res["total_return"] = _total_return_from_equity(s_eq)
    res["ann_return"]   = _ann_return_from_equity(s_eq)
    res["sharpe"]       = _sharpe_from_equity(s_eq)
    res["mdd"]          = _mdd_from_equity(s_eq)

    # 交易次数
    if df_trd is not None and not df_trd.empty:
        res["trades"] = float(len(df_trd))
    else:
        res["trades"] = np.nan

    # 暴露
    res["exposure"] = _exposure_from_positions(df_pos)

    # BH 指标（若 s_bh 存在）
    if s_bh is not None and len(s_bh) >= 2:
        res["bh_total_return"] = _total_return_from_equity(s_bh)
        res["bh_ann_return"]   = _ann_return_from_equity(s_bh)
        res["bh_sharpe"]       = _sharpe_from_equity(s_bh)
        res["bh_mdd"]          = _mdd_from_equity(s_bh)

    res["eq_series"] = s_eq
    res["bh_series"] = s_bh
    return res


def save_summary(results: List[Dict], outdir: str) -> str:
    rows = []
    for r in results:
        rows.append({
            "label": r["label"],
            "period": r["period"],
            "total_return": r["total_return"],
            "ann_return": r["ann_return"],
            "sharpe": r["sharpe"],
            "mdd": r["mdd"],
            "trades": r["trades"],
            "exposure": r["exposure"],
            "bh_total_return": r["bh_total_return"],
            "bh_ann_return": r["bh_ann_return"],
            "bh_sharpe": r["bh_sharpe"],
            "bh_mdd": r["bh_mdd"],
            "outdir": r["outdir"],
        })
    df = pd.DataFrame(rows)
    # 百分比列友好格式（导出 raw 数值，另存一份 pretty）
    p_csv = os.path.join(outdir, "walkforward_summary.csv")
    df.to_csv(p_csv, index=False, float_format="%.6f")

    # 再导出一份带百分比字符串的友好版
    df2 = df.copy()
    pct_cols = ["total_return", "ann_return", "mdd", "exposure", "bh_total_return", "bh_ann_return", "bh_mdd"]
    for c in pct_cols:
        if c in df2.columns:
            df2[c] = (df2[c] * 100.0).map(lambda x: f"{x:.2f}%")
    for c in ["sharpe", "bh_sharpe"]:
        if c in df2.columns:
            df2[c] = df2[c].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    p_csv_pretty = os.path.join(outdir, "walkforward_summary_pretty.csv")
    df2.to_csv(p_csv_pretty, index=False)
    print(f"[INFO] summary saved -> {p_csv}")
    print(f"[INFO] pretty  saved -> {p_csv_pretty}")
    return p_csv


def plot_compare(results: List[Dict], outdir: str) -> Tuple[str, str]:
    # 每个目录一个子图：策略 vs BH
    n = len(results)
    h = max(3.0 * n, 3.0)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, h), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        s_eq: pd.Series = r["eq_series"]
        s_bh: Optional[pd.Series] = r["bh_series"]
        if s_eq is None or len(s_eq) == 0:
            ax.set_title(f"{r['label']} (no equity)")
            continue

        # 归一处理方便对比
        eq_norm = s_eq / s_eq.iloc[0]
        ax.plot(eq_norm.index, eq_norm.values, label=f"{r['label']} (Strategy)")

        if s_bh is not None and len(s_bh) > 0:
            bh_norm = s_bh / s_bh.iloc[0]
            ax.plot(bh_norm.index, bh_norm.values, linestyle="--", label="Buy&Hold")

        ax.set_title(f"{r['label']} | {r['period']}")
        ax.set_ylabel("Normalized Equity")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Date")
    p_panel = os.path.join(outdir, "walkforward_equity_compare.png")
    plt.tight_layout()
    plt.savefig(p_panel, dpi=150)
    plt.close(fig)
    print(f"[INFO] panel plot saved -> {p_panel}")

    # 叠加图（仅在所有测试起止相同/或你仍想直观看趋势时）
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    aligned = 0
    for r in results:
        s_eq = r["eq_series"]
        if s_eq is None or len(s_eq) == 0:
            continue
        eq_norm = s_eq / s_eq.iloc[0]
        ax2.plot(eq_norm.index, eq_norm.values, label=r["label"])
        aligned += 1
    ax2.set_title("All Strategies (Overlay, normalized)")
    ax2.set_ylabel("Normalized Equity")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    p_overlay = os.path.join(outdir, "walkforward_equity_overlay.png")
    plt.tight_layout()
    plt.savefig(p_overlay, dpi=150)
    plt.close(fig2)
    print(f"[INFO] overlay plot saved -> {p_overlay}")

    return p_panel, p_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdirs", nargs="+", required=True, help="多个 eoh_post_export 的输出目录")
    ap.add_argument("--result-dir", required=False, default=None, help="汇总输出目录（默认为第一个 outdir 的同级 walkforward）")
    args = ap.parse_args()

    outdirs = [os.path.abspath(p) for p in args.outdirs]
    for p in outdirs:
        if not os.path.isdir(p):
            print(f"[WARN] not a directory: {p}", file=sys.stderr)

    # 决定结果目录
    if args.result_dir:
        result_dir = os.path.abspath(args.result_dir)
    else:
        parent = os.path.dirname(outdirs[0])
        result_dir = os.path.join(parent, "walkforward")
    os.makedirs(result_dir, exist_ok=True)
    print(f"[INFO] result dir -> {result_dir}")

    # 逐目录分析
    results = []
    for od in outdirs:
        r = analyze_one_outdir(od)
        # 仅保留有 equity 的
        if isinstance(r.get("eq_series", None), pd.Series) and len(r["eq_series"]) > 1:
            results.append(r)
        else:
            print(f"[WARN] skip in summary due to missing equity: {od}", file=sys.stderr)

    if not results:
        print("[ERROR] no valid results. Ensure you have run eoh_post_export.py for each outdir.", file=sys.stderr)
        sys.exit(1)

    save_summary(results, result_dir)
    plot_compare(results, result_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
