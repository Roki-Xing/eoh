# /root/autodl-tmp/quant/eoh_evolve_main.py
# 官方 EoH 驱动，多策略候选（BBreak/SMACross/RSIMR）
# 依赖：transformers/你已有的 EoH 框架、evaluation_eoh.MultiStrategyEvaluation

from __future__ import annotations
import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from evaluation_eoh import MultiStrategyEvaluation

# ==== 你环境中已可用的 LLM 调用/官方 EoH 入口（保持你现有的导入方式）====
# 这里假定你已有一个轻量 EoH 管理器（上一版能成功运行的那套）
# 如果你用的是你前面那版的 EoH wrapper，直接替换其中的 prompt 文案即可。

LLM_FAMILIES_GUIDE = """\
You are evolving trading strategy candidates. Only output ONE line per candidate, with NO explanation.
Allowed families and formats:
1) Bollinger breakout (BBreak):  BBreak_n{n}_k{k}_h{h}
   - n: integer [8, 40]
   - k: float   [0.8, 2.8]
   - h: integer [1, 10]
   Logic: long when Close > MA(n)+k*Std(n), exit when Close < MA(n) or min hold h days.

2) SMA Cross (SMACross):  SMACross_f{f}_s{s}_h{h}
   - f: integer [5, 30], s: integer [f+1, 120]
   - h: integer [1, 10]
   Logic: long when SMA(f) crosses above SMA(s); exit when cross down or min hold h days.

3) RSI mean reversion (RSIMR):  RSIMR_n{n}_os{os}_ob{ob}_h{h}
   - n: integer [5, 30], os: [10,40], ob: [60,90], and ob>os
   - h: integer [1, 10]
   Logic: long when RSI(n) < os; exit when RSI(n) > ob or min hold h days.

Only return a single line like:
BBreak_n12_k1.18_h6
or
SMACross_f10_s60_h5
or
RSIMR_n14_os30_ob70_h6
"""

def load_local_csv(symbol: str, csv_path: str,
                   train_start: str, train_end: str,
                   test_start: str, test_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    # 兼容常见列名
    time_col = None
    for c in ["timestamp", "date", "Date", "Datetime", "datetime"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("CSV 必须包含时间列，例如 'Date' 或 'timestamp'。")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    df = df.rename(columns={time_col: "timestamp"})
    if "Close" not in df.columns:
        # 兼容 close/Adj Close
        if "close" in df.columns: df["Close"] = df["close"]
        elif "Adj Close" in df.columns: df["Close"] = df["Adj Close"]
        else: raise ValueError("CSV 需包含 Close/close/Adj Close 之一")
    df = df[["timestamp", "Close"]].copy()

    df_train = df[(df["timestamp"] >= pd.Timestamp(train_start)) & (df["timestamp"] <= pd.Timestamp(train_end))].copy()
    df_test  = df[(df["timestamp"] >= pd.Timestamp(test_start))  & (df["timestamp"] <= pd.Timestamp(test_end))].copy()
    if df_train.empty or df_test.empty:
        raise ValueError("切分后为空，请检查时间范围。")
    df_train = df_train.set_index("timestamp")
    df_test  = df_test.set_index("timestamp")
    return df_train, df_test

def seed_random_candidates(k: int) -> list[str]:
    """ 生成多策略的随机候选，提升多样性 """
    out = []
    for _ in range(k):
        fam = random.choices(["BBreak", "SMACross", "RSIMR"], weights=[0.4, 0.35, 0.25], k=1)[0]
        if fam == "BBreak":
            n = random.randint(8, 40)
            kf = round(random.uniform(0.8, 2.8), 2)
            h = random.randint(1, 10)
            out.append(f"BBreak_n{n}_k{kf}_h{h}")
        elif fam == "SMACross":
            f = random.randint(5, 30)
            s = random.randint(f+1, 120)
            h = random.randint(1, 10)
            out.append(f"SMACross_f{f}_s{s}_h{h}")
        else:
            n = random.randint(5, 30)
            os_ = random.randint(10, 40)
            ob_ = random.randint(60, 90)
            if ob_ <= os_:
                ob_ = os_ + 1
            h = random.randint(1, 10)
            out.append(f"RSIMR_n{n}_os{os_}_ob{ob_}_h{h}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--symbol", default="SPY")
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
    ap.add_argument("--use-llm", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    df_train, df_test = load_local_csv(
        args.symbol, args.csv,
        args.train_start, args.train_end,
        args.test_start, args.test_end
    )

    evaluator = MultiStrategyEvaluation(df_train, df_test, commission=args.commission)

    # ====== 初始化候选：随机 + （可选）LLM ======
    cand = seed_random_candidates(args.population // 2)
    if args.use-llm:
        # 这里使用你原脚本里与 LLM 交互的函数，替换提示词为 LLM_FAMILIES_GUIDE
        # 伪代码：llm_generate(N, LLM_FAMILIES_GUIDE) -> List[str]
        llm_more = seed_random_candidates(args.population - len(cand))  # 如果不接 LLM，也用随机兜底
        cand.extend(llm_more)
    else:
        cand.extend(seed_random_candidates(args.population - len(cand)))

    # ====== EoH 主循环（简化形态，与上一版一致）======
    best = None
    for gen in range(1, args.generations + 1):
        records = []
        for name in cand:
            res = evaluator.evaluate_program(name)
            rec = {
                "name": res["name"],
                "fitness": res["fitness"],
                "train_ret": res["metrics_train"].get("ret", np.nan),
                "train_sharpe": res["metrics_train"].get("sharpe", np.nan),
                "train_mdd": res["metrics_train"].get("mdd", np.nan),
                "train_trades": res["metrics_train"].get("trades", np.nan),
                "train_expo": res["metrics_train"].get("expo", np.nan),
                "test_ret": res["metrics_test"].get("ret", np.nan),
                "test_sharpe": res["metrics_test"].get("sharpe", np.nan),
                "test_mdd": res["metrics_test"].get("mdd", np.nan),
                "test_trades": res["metrics_test"].get("trades", np.nan),
                "test_expo": res["metrics_test"].get("expo", np.nan),
            }
            records.append(rec)
            if (best is None) or (rec["fitness"] > best["fitness"]):
                best = rec
        # 存盘
        df_gen = pd.DataFrame.from_records(records)
        df_gen = df_gen.sort_values("fitness", ascending=False)
        df_gen.to_csv(os.path.join(args.outdir, f"gen{gen:02d}.csv"), index=False)
        print(f"[INFO] gen{gen:02d} -> {os.path.join(args.outdir, f'gen{gen:02d}.csv')}")
        # 变异/扩展：从 TopK 采样做随机扰动（简单版）
        top = df_gen.head(args.topk)["name"].tolist()
        next_cand = []
        for t in top:
            # 家族内轻微扰动
            if t.startswith("BBreak_"):
                m = re.match(r"BBreak_n(\d+)_k(\d+\.?\d*)_h(\d+)", t)
                n, kf, h = int(m.group(1)), float(m.group(2)), int(m.group(3))
                for _ in range(args.population // (args.topk*2) + 1):
                    n2 = max(5, n + random.randint(-3, 3))
                    k2 = round(max(0.5, kf + random.uniform(-0.2, 0.2)), 2)
                    h2 = max(1, h + random.randint(-2, 2))
                    next_cand.append(f"BBreak_n{n2}_k{k2}_h{h2}")
            elif t.startswith("SMACross_"):
                m = re.match(r"SMACross_f(\d+)_s(\d+)_h(\d+)", t)
                f, s, h = int(m.group(1)), int(m.group(2)), int(m.group(3))
                for _ in range(args.population // (args.topk*2) + 1):
                    f2 = max(3, f + random.randint(-3, 3))
                    s2 = max(f2+1, s + random.randint(-6, 6))
                    h2 = max(1, h + random.randint(-2, 2))
                    next_cand.append(f"SMACross_f{f2}_s{s2}_h{h2}")
            elif t.startswith("RSIMR_"):
                m = re.match(r"RSIMR_n(\d+)_os(\d+)_ob(\d+)_h(\d+)", t)
                n, os_, ob_, h = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                for _ in range(args.population // (args.topk*2) + 1):
                    n2 = max(3, n + random.randint(-3, 3))
                    os2 = min(45, max(5, os_ + random.randint(-3, 3)))
                    ob2 = max(os2+1, min(95, ob_ + random.randint(-3, 3)))
                    h2 = max(1, h + random.randint(-2, 2))
                    next_cand.append(f"RSIMR_n{n2}_os{os2}_ob{ob2}_h{h2}")
        # 填满
        while len(next_cand) < args.population:
            next_cand.extend(seed_random_candidates(args.population - len(next_cand)))
        cand = next_cand[:args.population]

    if best is not None:
        print(f"[BEST] name={best['name']} fitness={best['fitness']:.4f} "
              f"trainR={best['train_ret']*100:.2f}% testR={best['test_ret']*100:.2f}% "
              f"trainSharpe={best['train_sharpe']:.2f} testSharpe={best['test_sharpe']:.2f}")
        with open(os.path.join(args.outdir, "best.json"), "w") as f:
            json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
