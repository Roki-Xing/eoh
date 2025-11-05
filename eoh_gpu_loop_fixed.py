
import os, io, json, textwrap, argparse, random
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests

# yfinance 可选
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False

# 只需要 Strategy（指标我们自带）
from backtesting import Backtest, Strategy

# HF Transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from eoh_core.prompts import PromptLibrary, PromptStyle
from eoh_core.llm import LocalHFClient, extract_code_blocks
from eoh_core.utils import ensure_dir, log

# -------------------------
# args
# -------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="HF local model dir")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--test_start", required=True)
    ap.add_argument("--test_end", required=True)
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--commission", type=float, default=0.0005)
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--prompt-style",
        default="normal",
        help="Comma separated prompt style(s) to sample from (default: normal).",
    )
    ap.add_argument(
        "--prompt-dir",
        default="prompts",
        help="Directory containing optional <style>_system.txt and <style>_user.txt overrides.",
    )
    return ap.parse_args()
def save_text(path: str, txt: str):
    with open(path, "w", encoding="utf-8") as f: f.write(txt)

def df_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    def pick(name):
        ln = name.lower()
        if ln in cols: return cols[ln]
        for c in df.columns:
            if c.lower() == ln: return c
        return None
    mapping = {}
    for lo, Hi in [("open","Open"),("high","High"),("low","Low"),("close","Close"),("volume","Volume")]:
        c = pick(lo)
        if c is not None: mapping[c] = Hi
    if mapping: df = df.rename(columns=mapping)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    elif "timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["timestamp"]); df = df.set_index("Date").sort_index()
    else:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
    for need in ["Open","High","Low","Close","Volume"]:
        if need not in df.columns:
            if need == "Volume": df[need] = 0
            else: raise ValueError(f"Missing column: {need}")
    return df[["Open","High","Low","Close","Volume"]]

def slice_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))].copy()

# -------------------------
# Local CSV priority
# -------------------------
PRICE_CACHE_ROOT = "/root/autodl-tmp/price_cache"

def load_local_csv(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    path = os.path.join(PRICE_CACHE_ROOT, f"{symbol}_2020_2023.csv")
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns and "Date" not in df.columns:
            df["Date"] = pd.to_datetime(df["timestamp"])
        elif "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        elif "date" in df.columns:
            df["Date"] = pd.to_datetime(df["date"])
        else:
            raise ValueError("No Date/timestamp column in local CSV")
        df = df_ohlcv(df)
        df = slice_df(df, start, end)
        if len(df)==0: return None
        log(f"[INFO] local CSV hit: {path} rows={len(df)}")
        return df
    except Exception as e:
        log(f"[WARN] local CSV load failed: {e}")
        return None

# -------------------------
# Alpha Vantage (CSV)
# -------------------------
def load_alpha(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    key = os.environ.get("ALPHA_VANTAGE_API_KEY","").strip()
    if not key:
        log("[WARN] ALPHA key missing")
        return None
    url = "https://www.alphavantage.co/query"
    params = dict(function="TIME_SERIES_DAILY", symbol=symbol, apikey=key, datatype="csv", outputsize="full")
    try:
        r = requests.get(url, params=params, timeout=30)
        ct = r.headers.get("Content-Type","")
        if (ct and "text/csv" in ct) or (r.text and r.text.strip().startswith("timestamp")):
            df = pd.read_csv(io.StringIO(r.text)).rename(columns={"timestamp":"Date"})
            df = df_ohlcv(df)
            df = slice_df(df, start, end)
            if len(df)==0: return None
            log(f"[INFO] alpha CSV rows={len(df)}")
            return df
        else:
            log("[WARN] Alpha responded non-CSV (likely rate-limited/premium)")
            return None
    except Exception as e:
        log(f"[WARN] alpha error: {e}")
        return None

# -------------------------
# yfinance
# -------------------------
def load_yf(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    if not _HAS_YF: return None
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, interval="1d")
        if isinstance(df, pd.DataFrame) and len(df):
            df = df.rename(columns={"Adj Close":"Close"})
            df = df_ohlcv(df)
            log(f"[INFO] yfinance rows={len(df)}")
            return df
        return None
    except Exception as e:
        log(f"[WARN] yfinance error: {e}")
        return None

# -------------------------
# Synthetic
# -------------------------
def make_synth(symbol: str, start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="B")
    px = 100.0
    opn=[]; hi=[]; lo=[]; cls=[]; vol=[]
    rng = np.random.default_rng(42)
    for _ in range(len(idx)):
        ret = rng.normal(0, 0.01)
        px = max(1.0, px * (1+ret))
        c = px
        o = c*(1+rng.normal(0,0.002))
        h = max(o,c)*(1+abs(rng.normal(0,0.003)))
        l = min(o,c)*(1-abs(rng.normal(0,0.003)))
        v = int(abs(rng.normal(1e6, 2e5)))
        opn.append(o); hi.append(h); lo.append(l); cls.append(c); vol.append(v)
    df = pd.DataFrame({"Open":opn,"High":hi,"Low":lo,"Close":cls,"Volume":vol}, index=idx)
    df.index.name="Date"
    log(f"[INFO] synthetic rows={len(df)}")
    return df

# -------------------------
# Unified loader
# -------------------------
def load_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    for fn in (load_local_csv, load_alpha, load_yf):
        df = fn(symbol, start, end)
        if df is not None and len(df): return df
    log("[WARN] Falling back to synthetic")
    return make_synth(symbol, start, end)

# -------------------------
# Our minimal indicators (SMA/RSI/crossover)
# -------------------------
def SMA(series, n=10):
    s = pd.Series(series, copy=False)
    return s.rolling(int(n)).mean().to_numpy()

def RSI(series, n=14):
    s = pd.Series(series, copy=False).astype(float)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.rolling(int(n)).mean()
    roll_down = down.rolling(int(n)).mean()
    rs = roll_up / (roll_down.replace(0, np.nan) + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).to_numpy()

def crossover(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2 or len(b) < 2: return False
    return (a[-2] < b[-2]) and (a[-1] > b[-1])

# -------------------------
# Strategy sandbox
# -------------------------
ALLOWED_GLOBALS = {
    "__builtins__": {
        "abs": abs, "min": min, "max": max, "range": range, "len": len,
        "sum": sum, "any": any, "all": all, "round": round
    },
    "np": np, "numpy": np,
    "pd": pd, "pandas": pd,
    # expose Strategy API and our indicators
    "Strategy": Strategy,
    "SMA": SMA, "RSI": RSI, "crossover": crossover,
}

def safe_exec_strategy(code: str) -> Optional[Callable]:
    try:
        loc: Dict[str,Any] = {}
        exec(compile(code, "<llm_code>", "exec"), ALLOWED_GLOBALS, loc)
        Strat = loc.get("Strat", None)
        if Strat is None: return None
        if not hasattr(Strat, "init") or not hasattr(Strat, "next"): return None
        return Strat
    except Exception as e:
        log(f"[WARN] exec failed: {e}")
        return None

# -------------------------
# Evaluation
# -------------------------
def run_bt(df: pd.DataFrame, Strat: Strategy, commission: float) -> Tuple[pd.Series, Backtest]:
    bt = Backtest(df, Strat, cash=100_000, commission=commission, exclusive_orders=True)
    stats = bt.run()
    return stats, bt

def fitness_from_stats(st: pd.Series) -> float:
    r = float(st.get("Return [%]", 0.0))
    dd = float(st.get("Max. Drawdown [%]", 1.0))
    sharpe = float(st.get("Sharpe Ratio", 0.0))
    return r - 0.5*max(0.0, dd) + 10.0*max(0.0, sharpe)

# -------------------------
# Main
# -------------------------
def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = ensure_dir(args.outdir)

    prompt_dir = Path(args.prompt_dir).expanduser() if args.prompt_dir else None
    prompt_dir_display = "built-in defaults"
    if prompt_dir:
        if prompt_dir.exists():
            prompt_dir_display = str(prompt_dir)
        else:
            log(f"[WARN] prompt dir {prompt_dir} not found, using built-in templates")
            prompt_dir = None

    prompt_library = PromptLibrary(base_dir=prompt_dir)
    style_tokens = [token.strip().lower() for token in args.prompt_style.split(",") if token.strip()]
    if not style_tokens:
        style_tokens = ["normal"]
    unique_style_tokens: List[str] = []
    for token in style_tokens:
        if token not in unique_style_tokens:
            unique_style_tokens.append(token)

    resolved_styles: List[PromptStyle] = []
    for token in unique_style_tokens:
        try:
            resolved_styles.append(PromptStyle(token))
        except ValueError:
            log(f"[WARN] unknown prompt style '{token}', falling back to 'normal'")
            resolved_styles.append(PromptStyle.NORMAL)

    templates = {style: prompt_library.get(style) for style in resolved_styles}
    default_template = templates[resolved_styles[0]]

    prompt_specs: List[Dict[str, Any]] = []
    for idx in range(args.population):
        style = resolved_styles[idx % len(resolved_styles)]
        template = templates[style]
        prompt_specs.append(
            {
                "index": idx + 1,
                "style": style,
                "template": template,
                "prompt": template.user.format(symbol=args.symbol),
            }
        )

    span_start = min(args.train_start, args.test_start)
    span_end = max(args.train_end, args.test_end)
    df_all = load_prices(args.symbol, span_start, span_end)
    df_train = slice_df(df_all, args.train_start, args.train_end)
    df_test = slice_df(df_all, args.test_start, args.test_end)
    log(f"[INFO] split: train={len(df_train)} test={len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    log(f"[INFO] model loaded on {model.device}")
    client = LocalHFClient(tokenizer=tokenizer, model=model, system_prompt=default_template.system)

    gen_dir = ensure_dir(outdir / "gen01_codes")
    rows: List[Dict[str, Any]] = []

    for spec in prompt_specs:
        idx = spec["index"]
        template = spec["template"]
        prompt_text = spec["prompt"]
        log(f"[GEN] {idx}/{len(prompt_specs)} style={template.name}")
        raw = client.generate(
            prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            system_prompt=template.system,
        )
        code = extract_code_blocks(raw)
        if not code:
            log("[WARN] no code extracted")
            continue
        Strat = safe_exec_strategy(code)
        if Strat is None:
            log("[WARN] no valid Strat class")
            continue

        try:
            st_train, _ = run_bt(df_train, Strat, args.commission)
            st_test, _ = run_bt(df_test, Strat, args.commission)
            fit = fitness_from_stats(st_train)
            code_path = gen_dir / f"strat_{idx:03d}.py"
            raw_path = gen_dir / f"raw_{idx:03d}.txt"
            save_text(code_path, code)
            save_text(raw_path, raw)
            row = {
                "id": idx,
                "prompt_style": template.name,
                "prompt_user": prompt_text,
                "fitness": fit,
                "train_Return_%": float(st_train.get("Return [%]", float("nan"))),
                "train_Sharpe": float(st_train.get("Sharpe Ratio", float("nan"))),
                "train_MaxDD_%": float(st_train.get("Max. Drawdown [%]", float("nan"))),
                "test_Return_%": float(st_test.get("Return [%]", float("nan"))),
                "test_Sharpe": float(st_test.get("Sharpe Ratio", float("nan"))),
                "test_MaxDD_%": float(st_test.get("Max. Drawdown [%]", float("nan"))),
                "code_path": str(code_path),
                "raw_path": str(raw_path),
            }
            rows.append(row)
            log(
                "[OK] id={id} style={style} fit={fit:.2f} trainR={train_Return_%:.2f} testR={test_Return_%:.2f}".format(
                    id=idx,
                    style=template.name,
                    fit=fit,
                    train_Return_=row["train_Return_%"],
                    test_Return_=row["test_Return_%"],
                )
            )
        except Exception as exc:
            log(f"[WARN] backtest failed: {exc}")
            continue

    if not rows:
        log("[INFO] generation 1 done, valid=0")
        return

    df_res = pd.DataFrame(rows).sort_values("fitness", ascending=False)
    result_csv = outdir / "gen01.csv"
    df_res.to_csv(result_csv, index=False)
    best = df_res.iloc[0].to_dict()

    best_code = Path(best["code_path"]).read_text(encoding="utf-8")
    save_text(outdir / "best_strategy.py", best_code)
    save_text(outdir / "best_metrics.json", json.dumps(best, indent=2, ensure_ascii=False))
    readme = textwrap.dedent(
        f"""
        symbol: {args.symbol}
        train: {args.train_start} ~ {args.train_end}
        test : {args.test_start} ~ {args.test_end}
        population: {args.population}
        temperature: {args.temperature}
        commission: {args.commission}
        model_dir: {args.model_dir}
        prompt_styles: {", ".join(style.value for style in resolved_styles)}
        prompt_dir: {prompt_dir_display}
        """
    ).strip()
    save_text(outdir / "README.txt", readme + "\n")
    log(f"[INFO] generation 1 done, valid={len(rows)}, best id={int(best['id'])}")
    log(f"[INFO] results -> {outdir}")


if __name__ == "__main__":
    main()
