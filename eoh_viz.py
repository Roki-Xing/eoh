#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EOH Visualization & Report
- 读入价格/交易/持仓/净值 CSV
- 重建 pos（如缺失）
- 计算总体与按年指标：年化收益/年化波动/Sharpe/MDD/交易数/曝光
- 画图：价格+布林带+买卖点；策略 vs 买入持有；pos 阶梯线
- 导出：REPORT.md、metrics.csv（总体+按年）、plot.png、positions.csv（如重建）

用法示例：
python eoh_viz.py \
  --price_csv /root/autodl-tmp/price_cache/SPY_2020_2023.csv \
  --trades_csv /root/autodl-tmp/outputs/eoh_run_stable/trades_test.csv \
  --outdir /root/autodl-tmp/outputs/eoh_run_stable \
  --tag test \
  --start 2023-01-01 --end 2023-12-31 \
  --bb_n 12 --bb_k 1.2 \
  --commission 0.0005
"""

import os
import math
import argparse
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# 基础工具
# ---------------------------

def log(msg: str):
    print(f"[INFO] {msg}")


def read_price_csv(path: str) -> pd.DataFrame:
    """
    读取价格 CSV，返回包含 datetime 索引与列：open/high/low/close/volume（尽量兼容常见命名）
    必须包含：时间列（自动识别）、收盘价列（Close/close/Adj Close 任一）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"price_csv not found: {path}")

    df = pd.read_csv(path)
    # 猜时间列
    time_candidates = ["timestamp", "time", "date", "Date", "Datetime", "datetime", "Time"]
    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # 若没有明显时间列，且无索引，报错
        raise ValueError("CSV 必须包含时间列（如 timestamp/date/Datetime 等）")

    # 解析时间列
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    df = df.dropna(subset=[time_col]).copy()
    df = df.sort_values(time_col)
    df = df.set_index(time_col)

    # 标准化列名
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ["open", "o"]:
            rename_map[col] = "open"
        elif lc in ["high", "h"]:
            rename_map[col] = "high"
        elif lc in ["low", "l"]:
            rename_map[col] = "low"
        elif lc in ["close", "c"]:
            rename_map[col] = "close"
        elif lc in ["adj close", "adj_close", "adjusted close", "adjc"]:
            # 如果没有 close，用调整收盘价作为 close
            if "close" not in [x.lower() for x in df.columns]:
                rename_map[col] = "close"
            else:
                rename_map[col] = "adj_close"
        elif lc in ["volume", "vol", "v"]:
            rename_map[col] = "volume"

    if rename_map:
        df = df.rename(columns=rename_map)

    # 至少要有 close
    if "close" not in df.columns:
        # 再尝试直接找几种常见列
        for c in ["Close", "CLOSE"]:
            if c in df.columns:
                df = df.rename(columns={c: "close"})
                break
    if "close" not in df.columns:
        raise ValueError("价格 CSV 中未找到收盘价列（close/Close/Adj Close），请检查。")

    # 其余列补齐（若缺）
    for need in ["open", "high", "low", "volume"]:
        if need not in df.columns:
            df[need] = np.nan

    return df


def read_trades_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not os.path.exists(path):
        warnings.warn(f"trades_csv not found: {path}")
        return None
    df = pd.read_csv(path)
    return df


def read_positions_csv(path: Optional[str], index_like: Optional[pd.Index]) -> Optional[pd.Series]:
    if not path:
        return None
    if not os.path.exists(path):
        warnings.warn(f"positions_csv not found: {path}")
        return None
    df = pd.read_csv(path)
    # 兼容两种格式：1) Time, Pos；2) 只有 Pos 且长度=索引长度
    if "Pos" in df.columns:
        if "Time" in df.columns:
            t = pd.to_datetime(df["Time"], errors="coerce")
            s = pd.Series(df["Pos"].astype(int).values, index=t)
            s = s.sort_index()
            # 若给了 index_like，按最近对齐到该索引
            if index_like is not None:
                s = s.reindex(index_like, method="nearest", tolerance=None)
                s = s.fillna(method="ffill").fillna(0).astype(int)
            return s
        else:
            # 无时间列，则要求 index_like 存在且长度一致
            if index_like is None or len(index_like) != len(df):
                warnings.warn("positions.csv 没有时间列且长度与价格索引不符，将忽略该文件。")
                return None
            return pd.Series(df["Pos"].astype(int).values, index=index_like)
    else:
        warnings.warn("positions.csv 未包含 'Pos' 列，将忽略。")
        return None


def reconstruct_pos_series(index_like: pd.Index, trades: Optional[pd.DataFrame]) -> pd.Series:
    """
    根据 trades 表重建逐根 pos：0/1。优先用 EntryBar/ExitBar，否则用 EntryTime/ExitTime。
    """
    pos = pd.Series(0, index=index_like, dtype=int)
    if trades is None or len(trades) == 0:
        return pos

    cols = set([c.lower() for c in trades.columns])

    def col(name):  # 不区分大小写取列
        for c in trades.columns:
            if c.lower() == name.lower():
                return trades[c]
        return None

    if "entrybar" in cols:
        eb = col("EntryBar").astype(float)
        xb = col("ExitBar").astype(float) if "exitbar" in cols else pd.Series([np.nan]*len(trades))
        for i in range(len(trades)):
            s = int(np.nan_to_num(eb.iloc[i], nan=-1.0))
            if s < 0: 
                continue
            s = max(0, min(s, len(index_like)-1))
            e = xb.iloc[i]
            if pd.isna(e):
                e = len(index_like) - 1
            e = int(max(s, min(int(e), len(index_like)-1)))
            pos.iloc[s:e+1] = 1
        return pos

    # 用时间戳
    et = col("EntryTime")
    xt = col("ExitTime")
    if et is None:
        # 可能叫 Entry 或 entry_time 等
        et = col("Entry") or col("entry_time")
    if xt is None:
        xt = col("Exit") or col("exit_time")

    for i in range(len(trades)):
        st = pd.to_datetime(et.iloc[i], errors="coerce") if et is not None else pd.NaT
        en = pd.to_datetime(xt.iloc[i], errors="coerce") if xt is not None else pd.NaT
        if pd.isna(st):
            continue
        s = index_like.get_indexer([st], method="nearest")[0]
        if pd.isna(en):
            e = len(index_like) - 1
        else:
            e = index_like.get_indexer([en], method="nearest")[0]
            e = max(s, e)
        pos.iloc[s:e+1] = 1

    return pos


def reconstruct_equity_from_pos(close: pd.Series,
                                pos: pd.Series,
                                commission: float = 0.0) -> Tuple[pd.Series, pd.Series]:
    """
    根据 pos × close 计算日收益与净值。换仓日扣除手续费（每次 0->1 或 1->0 扣 commission）。
    返回：(daily_ret, equity_curve)
    """
    close = close.astype(float)
    ret_price = close.pct_change().fillna(0.0)
    # 策略收益：昨日持仓 * 当日价格收益
    pos_shift = pos.shift(1).fillna(0.0)
    strat_ret = pos_shift * ret_price

    # 交易日扣佣金：pos 变化的那天（含进/出）
    delta = pos.diff().fillna(pos.iloc[0]).astype(float)
    cost = commission * np.abs(delta)  # 进一次扣，出一次也扣
    strat_ret = strat_ret - cost

    equity = (1.0 + strat_ret).cumprod()
    return strat_ret, equity


def compute_mdd(equity: pd.Series) -> float:
    highwater = equity.cummax()
    dd = equity / highwater - 1.0
    return dd.min() if len(dd) else 0.0


def annual_metrics_from_returns(ret: pd.Series,
                                trades: Optional[pd.DataFrame],
                                pos: Optional[pd.Series],
                                freq: int = 252) -> pd.DataFrame:
    """
    按自然年汇总指标：AnnRet/AnnVol/Sharpe/MDD/Trades/Exposure
    """
    if ret.empty:
        return pd.DataFrame()

    df = pd.DataFrame({"ret": ret})
    df["year"] = df.index.year
    rows = []
    for y, g in df.groupby("year"):
        rr = g["ret"].dropna()
        if len(rr) == 0:
            continue
        eq = (1 + rr).cumprod()
        T = max(len(rr) / freq, 1e-9)
        ann_ret = eq.iloc[-1] ** (1 / T) - 1.0
        ann_vol = rr.std(ddof=0) * math.sqrt(freq)
        sharpe = (ann_ret / ann_vol) if ann_vol > 1e-12 else np.nan
        mdd = compute_mdd(eq)

        # 该年交易数/持仓占比
        if trades is not None and len(trades) > 0:
            # 以 EntryTime 属于该年计数（若无 EntryTime，则用 EntryBar 无法按年，只能 NaN）
            ent_col = None
            for cand in ["EntryTime", "Entry", "entry_time"]:
                if cand in trades.columns:
                    ent_col = cand
                    break
            if ent_col:
                t_in_year = pd.to_datetime(trades[ent_col], errors="coerce")
                trades_n = int(((t_in_year.dt.year == y).fillna(False)).sum())
            else:
                trades_n = np.nan
        else:
            # 通过 pos 的 0->1 次数估算（该年）
            if pos is not None and len(pos) > 0:
                pos_y = pos[pos.index.year == y].astype(float)
                trades_n = int(((pos_y.diff() > 0.5).fillna(pos_y.iloc[0] > 0.5)).sum())
            else:
                trades_n = np.nan

        exposure = float(pos[pos.index.year == y].mean()) if pos is not None and len(pos) else np.nan

        rows.append({
            "Year": int(y),
            "AnnRet": ann_ret,
            "AnnVol": ann_vol,
            "Sharpe": sharpe,
            "MDD": mdd,
            "Trades": trades_n,
            "Exposure": exposure,
        })
    return pd.DataFrame(rows).sort_values("Year")


def overall_metrics_from_returns(ret: pd.Series,
                                 trades: Optional[pd.DataFrame],
                                 pos: Optional[pd.Series],
                                 freq: int = 252) -> dict:
    if ret.empty:
        return {"AnnRet": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MDD": np.nan, "Trades": 0, "Exposure": 0.0}

    eq = (1 + ret).cumprod()
    T = max(len(ret) / freq, 1e-9)
    ann_ret = eq.iloc[-1] ** (1 / T) - 1.0
    ann_vol = ret.std(ddof=0) * math.sqrt(freq)
    sharpe = (ann_ret / ann_vol) if ann_vol > 1e-12 else np.nan
    mdd = compute_mdd(eq)

    if trades is not None and len(trades) > 0:
        trades_n = int(len(trades))
    else:
        if pos is not None and len(pos) > 0:
            trades_n = int(((pos.diff() > 0.5).fillna(pos.iloc[0] > 0.5)).sum())
        else:
            trades_n = 0

    exposure = float(pos.mean()) if pos is not None and len(pos) > 0 else 0.0

    return {
        "AnnRet": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MDD": mdd,
        "Trades": trades_n,
        "Exposure": exposure,
    }


def compute_bbands(close: pd.Series, n: int, k: float) -> pd.DataFrame:
    ma = close.rolling(n, min_periods=1).mean()
    sd = close.rolling(n, min_periods=1).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return pd.DataFrame({"ma": ma, "upper": upper, "lower": lower})


def extract_markers_from_trades(df: pd.DataFrame, trades: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    从 trades 提取买卖点（时间与价格），用于 scatter 标注。优先使用 EntryTime/ExitTime。
    若没有时间，尝试使用 EntryBar/ExitBar 索引对齐到 df。
    """
    buy_t, buy_p = [], []
    sell_t, sell_p = [], []

    cols = [c.lower() for c in trades.columns]
    def col(name):
        for c in trades.columns:
            if c.lower() == name.lower():
                return trades[c]
        return None

    et = col("EntryTime") or col("Entry") or col("entry_time")
    xt = col("ExitTime") or col("Exit") or col("exit_time")
    ep = col("EntryPrice") or col("entry_price")
    xp = col("ExitPrice") or col("exit_price")

    if et is not None:
        t = pd.to_datetime(et, errors="coerce")
        for i in range(len(trades)):
            ts = t.iloc[i]
            if pd.isna(ts): 
                continue
            idx = df.index.get_indexer([ts], method="nearest")[0]
            buy_t.append(df.index[idx])
            buy_p.append(float(ep.iloc[i]) if ep is not None and not pd.isna(ep.iloc[i]) else float(df["close"].iloc[idx]))
        if xt is not None:
            t2 = pd.to_datetime(xt, errors="coerce")
            for i in range(len(trades)):
                ts = t2.iloc[i]
                if pd.isna(ts):
                    continue
                idx = df.index.get_indexer([ts], method="nearest")[0]
                sell_t.append(df.index[idx])
                sell_p.append(float(xp.iloc[i]) if xp is not None and not pd.isna(xp.iloc[i]) else float(df["close"].iloc[idx]))
    elif "entrybar" in cols:
        eb = col("EntryBar").astype(float)
        xb = col("ExitBar").astype(float) if "exitbar" in cols else pd.Series([np.nan]*len(trades))
        for i in range(len(trades)):
            s = int(np.nan_to_num(eb.iloc[i], nan=-1.0))
            if 0 <= s < len(df):
                buy_t.append(df.index[s])
                buy_p.append(float(df["close"].iloc[s]))
            e = xb.iloc[i]
            if not pd.isna(e):
                e = int(e)
                if 0 <= e < len(df):
                    sell_t.append(df.index[e])
                    sell_p.append(float(df["close"].iloc[e]))
    else:
        # 无法识别，返回空
        pass

    return pd.Series(buy_p, index=pd.Index(buy_t)), pd.Series(sell_p, index=pd.Index(sell_t))


# ---------------------------
# 画图与导出
# ---------------------------

def plot_all(df: pd.DataFrame,
             pos: pd.Series,
             equity: pd.Series,
             bh_equity: pd.Series,
             buy_mark: Optional[pd.Series],
             sell_mark: Optional[pd.Series],
             bb: Optional[pd.DataFrame],
             title_tag: str,
             out_png: str):
    """
    三联图：
      1) 价格 + 布林带 + 买/卖点
      2) 净值（策略 vs Buy&Hold）
      3) pos 阶梯线
    """
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.5, 0.6], hspace=0.25)

    # --- (1) 价格图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df["close"].values, label="Close", linewidth=1.0)
    if bb is not None:
        ax1.plot(df.index, bb["ma"].values, label="BB MA", linewidth=0.9)
        ax1.plot(df.index, bb["upper"].values, label="BB Upper", linewidth=0.9, linestyle="--")
        ax1.plot(df.index, bb["lower"].values, label="BB Lower", linewidth=0.9, linestyle="--")

    if buy_mark is not None and len(buy_mark) > 0:
        ax1.scatter(buy_mark.index, buy_mark.values, marker="^", s=40, alpha=0.8, label="Buy")
    if sell_mark is not None and len(sell_mark) > 0:
        ax1.scatter(sell_mark.index, sell_mark.values, marker="v", s=40, alpha=0.8, label="Sell")

    ax1.set_title(f"{title_tag} - Price / BBands / Trades")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="best")

    # 右轴 pos（在价格图上叠一条 0/1 阶梯线，便于看在场/离场）
    ax1p = ax1.twinx()
    ax1p.step(df.index, pos.values, where="post", linewidth=0.9, alpha=0.35)
    ax1p.set_ylim(-0.1, 1.1)
    ax1p.set_yticks([0, 1])
    ax1p.set_yticklabels(["flat", "pos"])
    ax1p.grid(False)

    # --- (2) 净值对比
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(equity.index, equity.values, label="Strategy", linewidth=1.2)
    ax2.plot(bh_equity.index, bh_equity.values, label="Buy&Hold", linewidth=1.2, linestyle="--")
    ax2.set_title("Equity Curve")
    ax2.set_ylabel("Equity (× initial)")
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="best")

    # --- (3) pos 单独子图
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.step(df.index, pos.values, where="post", linewidth=1.2)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["flat", "pos"])
    ax3.set_title("Position (0/1)")
    ax3.grid(True, linestyle=":")

    for ax in [ax1, ax2, ax3]:
        for label in ax.get_xticklabels():
            label.set_rotation(0)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    log(f"plot -> {out_png}")


def save_report_and_csv(outdir: str,
                        tag: str,
                        overall: dict,
                        yearly: pd.DataFrame):
    # 总体指标 CSV
    overall_path = os.path.join(outdir, f"{tag}_metrics_overall.csv")
    pd.DataFrame([overall]).to_csv(overall_path, index=False)
    log(f"metrics_overall -> {overall_path}")

    # 按年指标 CSV
    yearly_path = os.path.join(outdir, f"{tag}_metrics_yearly.csv")
    if yearly is not None and len(yearly) > 0:
        yearly.to_csv(yearly_path, index=False)
        log(f"metrics_yearly -> {yearly_path}")

    # 报告 Markdown
    rep_path = os.path.join(outdir, f"{tag}_REPORT.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"# Report: {tag}\n\n")
        f.write("## Overall Metrics\n\n")
        f.write(f"- Annualized Return: {overall['AnnRet']*100:.2f}%\n")
        f.write(f"- Annualized Vol   : {overall['AnnVol']*100:.2f}%\n")
        f.write(f"- Sharpe           : {overall['Sharpe']:.3f}\n")
        f.write(f"- Max Drawdown     : {overall['MDD']*100:.2f}%\n")
        f.write(f"- Trades           : {overall['Trades']}\n")
        f.write(f"- Exposure         : {overall['Exposure']*100:.2f}%\n")

        f.write("\n## Yearly Breakdown\n\n")
        if yearly is not None and len(yearly) > 0:
            f.write(yearly.to_markdown(index=False))
            f.write("\n")
        else:
            f.write("_No yearly data._\n")
    log(f"report -> {rep_path}")


# ---------------------------
# 主流程
# ---------------------------

def main():
    ap = argparse.ArgumentParser("EOH Visualization & Report")
    ap.add_argument("--price_csv", required=True, help="价格 CSV（含时间与close）")
    ap.add_argument("--trades_csv", default=None, help="交易 CSV（可选）")
    ap.add_argument("--positions_csv", default=None, help="持仓 CSV（可选，若无则依据 trades 重建）")
    ap.add_argument("--equity_csv", default=None, help="策略净值 CSV（可选，若无则依据 pos & close 重建）")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--tag", default="test", help="输出文件名标签")
    ap.add_argument("--start", default=None, help="起始日期（可选，如 2023-01-01）")
    ap.add_argument("--end", default=None, help="结束日期（可选，如 2023-12-31）")
    ap.add_argument("--bb_n", type=int, default=20, help="布林带窗口")
    ap.add_argument("--bb_k", type=float, default=2.0, help="布林带倍数")
    ap.add_argument("--commission", type=float, default=0.0, help="单侧手续费（比例），用于重建净值")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 价格数据
    df_all = read_price_csv(args.price_csv)

    # 按需截取时间窗口
    if args.start:
        s = pd.to_datetime(args.start, errors="coerce")
        if not pd.isna(s):
            df_all = df_all[df_all.index >= s]
    if args.end:
        e = pd.to_datetime(args.end, errors="coerce")
        if not pd.isna(e):
            df_all = df_all[df_all.index <= e]
    if len(df_all) == 0:
        raise ValueError("选择的时间窗口内没有价格数据。")

    # 交易与持仓
    trades = read_trades_csv(args.trades_csv)
    pos = read_positions_csv(args.positions_csv, index_like=df_all.index)
    if pos is None:
        pos = reconstruct_pos_series(df_all.index, trades)
        # 落盘以便检查
        pos_path = os.path.join(args.outdir, f"{args.tag}_positions.csv")
        pd.DataFrame({"Time": df_all.index, "Pos": pos.values}).to_csv(pos_path, index=False)
        log(f"positions (reconstructed) -> {pos_path}")

    # 净值：策略与基准
    if args.equity_csv and os.path.exists(args.equity_csv):
        ec_df = pd.read_csv(args.equity_csv)
        # 尝试识别 equity 列与时间列
        time_candidates = ["Time", "time", "timestamp", "Datetime", "date"]
        eq_candidates = ["equity", "Equity", "strategy", "Strategy"]
        tcol = None
        for c in time_candidates:
            if c in ec_df.columns:
                tcol = c; break
        if tcol:
            ec_df[tcol] = pd.to_datetime(ec_df[tcol], errors="coerce")
            ec_df = ec_df.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)
            # 最近对齐到价格索引
            ec_df = ec_df.reindex(df_all.index, method="nearest")
        # 找净值列
        ecol = None
        for c in eq_candidates:
            if c in ec_df.columns:
                ecol = c; break
        if ecol is None:
            # 否则找第一列
            ecol = ec_df.columns[0]
        equity = ec_df[ecol].astype(float)
        # 从净值反推日收益
        ret = equity.pct_change().fillna(0.0)
    else:
        # 根据 pos 与 close 重建
        ret, equity = reconstruct_equity_from_pos(df_all["close"], pos, commission=args.commission)

    # 基准：满仓买入持有（不含佣金）
    bh_ret = df_all["close"].pct_change().fillna(0.0)
    bh_equity = (1 + bh_ret).cumprod()

    # 布林带（仅用于画图）
    bb = compute_bbands(df_all["close"], n=args.bb_n, k=args.bb_k) if args.bb_n > 0 else None

    # 买卖点（如有 trades）
    buy_mark, sell_mark = (None, None)
    if trades is not None and len(trades) > 0:
        buy_mark, sell_mark = extract_markers_from_trades(df_all, trades)

    # 指标
    overall = overall_metrics_from_returns(ret, trades, pos, freq=252)
    yearly = annual_metrics_from_returns(ret, trades, pos, freq=252)

    # 画图
    out_png = os.path.join(args.outdir, f"{args.tag}_plot.png")
    plot_all(df_all, pos, equity, bh_equity, buy_mark, sell_mark, bb, args.tag, out_png)

    # 导出报告与 CSV
    save_report_and_csv(args.outdir, args.tag, overall, yearly)

    log("done.")


if __name__ == "__main__":
    main()
