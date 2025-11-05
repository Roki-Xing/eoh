#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-symbol batch runner:
- 对一组 symbols 循环：
  1) 调用 eoh_evolve_main.py 做进化（可传风控/适应度参数）
  2) 调用 eoh_post_export.py 导出 CSV/PNG/可选HTML、报告与快照
- 产出：
  - 每个标的单独目录：<outdir-base>/<SYMBOL>/（含 genXX.csv、best_*、yearly_report_*、REPORT.md 等）
  - 总汇总：<outdir-base>/symbols_leaderboard.csv / symbols_leaderboard_pretty.csv
  - 净值叠加图：<outdir-base>/symbols_equity_overlay_test.png

依赖：
- 需要同目录已有 eoh_evolve_main.py / eoh_post_export.py
- 本脚本仅负责 orchestration 与汇总
"""
import os, re, sys, json, argparse, subprocess, glob
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def log(msg: str):
    print(msg, flush=True)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def parse_symbols(sym_arg: str=None, file_path: str=None) -> List[str]:
    syms = []
    if sym_arg:
        for x in sym_arg.replace(";", ",").split(","):
            x = x.strip().upper()
            if x:
                syms.append(x)
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                x = line.strip().upper()
                if x:
                    syms.append(x)
    # unique
    syms = list(dict.fromkeys(syms))
    if not syms:
        raise ValueError("未提供任何 symbol（--symbols 或 --symbols-file）")
    return syms

def fmt_csv_path(tmpl: str, sym: str) -> str:
    # 支持 {sym}, {SYM}, {sym_lower}, {symbol}
    return (tmpl
            .replace("{sym}", sym)
            .replace("{SYM}", sym)
            .replace("{symbol}", sym)
            .replace("{sym_lower}", sym.lower()))

def last_gen_csv(outdir: str) -> str:
    gens = []
    for f in os.listdir(outdir):
        m = re.match(r"gen(\d+)\.csv$", f)
        if m:
            gens.append(int(m.group(1)))
    if not gens:
        return ""
    return os.path.join(outdir, f"gen{max(gens):02d}.csv")

def pick_best_row(gen_csv: str) -> Dict[str, Any]:
    df = pd.read_csv(gen_csv)
    if "fitness" not in df.columns:
        raise ValueError("gen csv 缺少 fitness 列")
    r = df.iloc[df["fitness"].argmax()].to_dict()
    return r

def plot_equity_overlay(equity_dict: Dict[str, pd.DataFrame], out_png: str, title: str="Strategy equity (TEST)"):
    plt.figure(figsize=(12,7))
    for sym, df in equity_dict.items():
        if df is None or df.empty: 
            continue
        if "timestamp" not in df.columns or "equity" not in df.columns:
            continue
        ts = pd.to_datetime(df["timestamp"])
        eq = df["equity"].astype(float)
        # 归一
        base = eq.iloc[0] if len(eq)>0 else 1.0
        if base == 0: base = 1.0
        plt.plot(ts, eq/base, label=sym)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    log(f"[INFO] overlay plot saved -> {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="LLM 模型目录（传给 evolve 脚本；即便关闭 --use-llm 也保留以统一快照）")
    ap.add_argument("--symbols", default="", help="逗号分隔，如 SPY,QQQ,AAPL")
    ap.add_argument("--symbols-file", default="", help="文本文件，每行一个symbol")
    ap.add_argument("--csv-template", required=True,
                    help="CSV 模板路径，支持 {sym}/{SYM}/{sym_lower}/{symbol} 占位符，如 /root/.../price_cache/{sym}_2020_2023.csv")
    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--test_start", required=True)
    ap.add_argument("--test_end", required=True)
    ap.add_argument("--population", type=int, default=56)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--commission", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir-base", required=True, help="所有标的的输出根目录，每个标的一个子目录")
    ap.add_argument("--use-llm", action="store_true", help="转发给 evolve 脚本")
    # 风控与适应度（转发给 evolve；同时快照给 post_export）
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--delay-days", type=int, default=0)
    ap.add_argument("--max-position", type=float, default=1.0)
    ap.add_argument("--max-daily-turnover", type=float, default=1.0)
    ap.add_argument("--min-trades", type=int, default=0)
    ap.add_argument("--min-exposure", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--zeta", type=float, default=0.0)
    # 可视化
    ap.add_argument("--html", action="store_true", help="转发给 post_export，生成交互式HTML图")
    args = ap.parse_args()

    ensure_dir(args.outdir_base)

    symbols = parse_symbols(args.symbols, args.symbols_file)
    leaderboard_rows = []
    overlay_series: Dict[str, pd.DataFrame] = {}

    this_dir = os.path.dirname(os.path.abspath(__file__))
    evolve_py = os.path.join(this_dir, "eoh_evolve_main.py")
    export_py = os.path.join(this_dir, "eoh_post_export.py")

    if not os.path.isfile(evolve_py):
        raise FileNotFoundError(f"未找到 {evolve_py}")
    if not os.path.isfile(export_py):
        raise FileNotFoundError(f"未找到 {export_py}")

    for sym in symbols:
        log("="*70)
        log(f"[RUN] symbol = {sym}")
        csv_path = fmt_csv_path(args.csv_template, sym)
        if not os.path.isfile(csv_path):
            log(f"[WARN] CSV 不存在: {csv_path}，跳过 {sym}")
            leaderboard_rows.append({
                "symbol": sym, "status": "csv_missing", "outdir": "", "best_name": "",
                "fitness": np.nan, "testReturn": np.nan, "testSharpe": np.nan, "testMDD": np.nan,
                "testTrades": np.nan, "testExposure": np.nan
            })
            continue

        outdir = os.path.join(args.outdir_base, sym)
        ensure_dir(outdir)

        # 1) evolve
        evolve_cmd = [
            sys.executable, evolve_py,
            "--model-dir", args.model_dir,
            "--symbol", sym,
            "--csv", csv_path,
            "--train_start", args.train_start, "--train_end", args.train_end,
            "--test_start", args.test_start, "--test_end", args.test_end,
            "--population", str(args.population),
            "--generations", str(args.generations),
            "--topk", str(args.topk),
            "--commission", str(args.commission),
            "--seed", str(args.seed),
            "--outdir", outdir,
            "--slippage-bps", str(args.slippage_bps),
            "--delay-days", str(args.delay_days),
            "--max-position", str(args.max_position),
            "--max-daily-turnover", str(args.max_daily_turnover),
            "--min-trades", str(args.min_trades),
            "--min-exposure", str(args.min_exposure),
            "--alpha", str(args.alpha),
            "--beta", str(args.beta),
            "--gamma", str(args.gamma),
            "--delta", str(args.delta),
            "--zeta", str(args.zeta),
        ]
        if args.use_llm:
            evolve_cmd.append("--use-llm")

        log("[CMD] " + " ".join(evolve_cmd))
        ev_log_path = os.path.join(outdir, "evolve_run.log")
        with open(ev_log_path, "w", encoding="utf-8") as f:
            proc = subprocess.run(evolve_cmd, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            log(f"[ERR] evolve 失败，见日志：{ev_log_path}")
            leaderboard_rows.append({
                "symbol": sym, "status": "evolve_failed", "outdir": outdir, "best_name": "",
                "fitness": np.nan, "testReturn": np.nan, "testSharpe": np.nan, "testMDD": np.nan,
                "testTrades": np.nan, "testExposure": np.nan
            })
            continue

        # 2) export
        export_cmd = [
            sys.executable, export_py,
            "--model-dir", args.model_dir,        # 仅快照
            "--symbol", sym,
            "--csv", csv_path,
            "--train_start", args.train_start, "--train_end", args.train_end,
            "--test_start", args.test_start, "--test_end", args.test_end,
            "--commission", str(args.commission),
            "--population", str(args.population),
            "--generations", str(args.generations),
            "--topk", str(args.topk),
            "--seed", str(args.seed),
            "--outdir", outdir,
            "--slippage_bps", str(args.slippage_bps),
            "--delay_days", str(args.delay_days),
            "--max_position", str(args.max_position),
            "--max_daily_turnover", str(args.max_daily_turnover),
            "--min_trades", str(args.min_trades),
            "--min_exposure", str(args.min_exposure),
            "--alpha", str(args.alpha),
            "--beta", str(args.beta),
            "--gamma", str(args.gamma),
            "--delta", str(args.delta),
            "--zeta", str(args.zeta),
        ]
        if args.html:
            export_cmd.append("--html")

        log("[CMD] " + " ".join(export_cmd))
        ex_log_path = os.path.join(outdir, "export_run.log")
        with open(ex_log_path, "w", encoding="utf-8") as f:
            proc = subprocess.run(export_cmd, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            log(f"[ERR] export 失败，见日志：{ex_log_path}")
            leaderboard_rows.append({
                "symbol": sym, "status": "export_failed", "outdir": outdir, "best_name": "",
                "fitness": np.nan, "testReturn": np.nan, "testSharpe": np.nan, "testMDD": np.nan,
                "testTrades": np.nan, "testExposure": np.nan
            })
            continue

        # 3) 汇总该标的指标（读最后一代 genXX.csv 的best 行）
        gen_csv = last_gen_csv(outdir)
        if not gen_csv:
            log(f"[WARN] 未找到 genXX.csv: {outdir}")
            leaderboard_rows.append({
                "symbol": sym, "status": "no_gen_csv", "outdir": outdir, "best_name": "",
                "fitness": np.nan, "testReturn": np.nan, "testSharpe": np.nan, "testMDD": np.nan,
                "testTrades": np.nan, "testExposure": np.nan
            })
            continue
        try:
            best = pick_best_row(gen_csv)
            leaderboard_rows.append({
                "symbol": sym,
                "status": "ok",
                "outdir": outdir,
                "best_name": best.get("name",""),
                "fitness": float(best.get("fitness", np.nan)),
                "trainReturn": float(best.get("trainReturn", np.nan)),
                "trainSharpe": float(best.get("trainSharpe", np.nan)),
                "trainMDD": float(best.get("trainMDD", np.nan)),
                "testReturn": float(best.get("testReturn", np.nan)),
                "testSharpe": float(best.get("testSharpe", np.nan)),
                "testMDD": float(best.get("testMDD", np.nan)),
                "testTrades": int(best.get("testTrades", 0)) if not pd.isna(best.get("testTrades", np.nan)) else np.nan,
                "testExposure": float(best.get("testExposure", np.nan)),
            })
        except Exception as e:
            log(f"[WARN] 解析 {gen_csv} 失败：{e}")
            leaderboard_rows.append({
                "symbol": sym, "status": "parse_gen_failed", "outdir": outdir, "best_name": "",
                "fitness": np.nan, "testReturn": np.nan, "testSharpe": np.nan, "testMDD": np.nan,
                "testTrades": np.nan, "testExposure": np.nan
            })

        # 4) 收集测试净值，用于叠加图
        eq_path = os.path.join(outdir, "best_equity_test.csv")
        if os.path.isfile(eq_path):
            try:
                eq_df = pd.read_csv(eq_path)
                overlay_series[sym] = eq_df
            except Exception as e:
                log(f"[WARN] 读取 {eq_path} 失败：{e}")

    # === 汇总输出 ===
    lb = pd.DataFrame(leaderboard_rows)
    lb_path = os.path.join(args.outdir_base, "symbols_leaderboard.csv")
    lb.to_csv(lb_path, index=False)
    log(f"[INFO] leaderboard saved -> {lb_path}")
    # pretty（按 testSharpe/fitness 排序）
    order_cols = [c for c in ["testSharpe","fitness","testReturn","testMDD","testTrades","testExposure"] if c in lb.columns]
    if order_cols:
        lb_pretty = lb.sort_values(["testSharpe","fitness"], ascending=[False, False], na_position="last")
    else:
        lb_pretty = lb
    lbp = os.path.join(args.outdir_base, "symbols_leaderboard_pretty.csv")
    lb_pretty.to_csv(lbp, index=False)
    log(f"[INFO] leaderboard pretty saved -> {lbp}")

    # 叠加净值图（TEST）
    overlay_png = os.path.join(args.outdir_base, "symbols_equity_overlay_test.png")
    plot_equity_overlay(overlay_series, overlay_png, title="Strategy Equity Overlay (TEST)")

    log("[INFO] Done.")

if __name__ == "__main__":
    main()
