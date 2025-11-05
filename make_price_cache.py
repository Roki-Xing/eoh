#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更稳健的价格拉取脚本：
1) 首选 yfinance（定制 Session、防并发、可选代理）
2) 失败则回退 Stooq（qqq.us / aapl.us）
3) 统一输出列：timestamp, Open, High, Low, Close, Volume
"""

import argparse, os, sys, json, time
from typing import Optional, Dict
import pandas as pd

def _unify_df(df: pd.DataFrame) -> pd.DataFrame:
    # 统一列 & 时间戳
    rename_map = {
        "date": "timestamp",
        "Date": "timestamp",
        "Datetime": "timestamp",
        "Open": "Open", "High": "High", "Low": "Low", "Close": "Close",
        "close": "Close",
        "open": "Open", "high": "High", "low": "Low",
        "Volume": "Volume", "volume": "Volume",
        "Adj Close": "AdjClose", "AdjClose": "AdjClose"
    }
    df = df.rename(columns=rename_map)
    # 有些源给的是小写列名
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns and col.lower() in df.columns:
            df[col] = df[col.lower()]
    # 时间列兜底
    if "timestamp" not in df.columns:
        if "Date" in df.columns:
            df["timestamp"] = df["Date"]
        elif df.index.name in ("Date","Datetime") or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index":"timestamp"})
        else:
            raise ValueError("无法识别时间列")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["timestamp","Close"]).sort_values("timestamp")
    # 只保留需要列
    keep = ["timestamp","Open","High","Low","Close","Volume"]
    for k in keep:
        if k not in df.columns:
            # 体面地补列（成交量缺失时置 0）
            df[k] = 0.0 if k != "Volume" else 0
    return df[keep]

def _download_yf(sym: str, start: str, end: str, proxy: Optional[str]) -> pd.DataFrame:
    import yfinance as yf
    import requests

    # 自定义 Session（避免 999/403）
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; EOH-Downloader/1.0; +local)"
    })
    if proxy:
        sess.proxies.update({"http": proxy, "https": proxy})

    # 尝试两种方式：download & history（某些版本对 TZ 处理不同）
    for attempt in range(2):
        try:
            if attempt == 0:
                df = yf.download(
                    sym, start=start, end=end, interval="1d",
                    auto_adjust=False, progress=False, threads=False, session=sess
                )
            else:
                df = yf.Ticker(sym, session=sess).history(
                    start=start, end=end, interval="1d", auto_adjust=False
                )
            if df is not None and not df.empty:
                return _unify_df(df)
        except Exception as e:
            # 常见限流/网络抖动，稍等再试另外路径
            time.sleep(0.8)
    raise RuntimeError("yfinance 拉取失败")

def _download_stooq(sym: str) -> pd.DataFrame:
    # stooq 符号一般是 {sym}.us，A股/港股/其他交易所需改后缀
    import requests, io
    for ticker in (f"{sym.lower()}.us", sym.lower()):
        url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
        r = requests.get(url, timeout=10)
        if r.ok and r.text.strip() and r.text.lower().startswith("date,"):
            df = pd.read_csv(io.StringIO(r.text))
            return _unify_df(df)
    raise RuntimeError("stooq 回退也失败")

def save_one(sym: str, start: str, end: str, outdir: str, proxy: Optional[str]) -> str:
    os.makedirs(outdir, exist_ok=True)
    y0, y1 = start[:4], end[:4]
    out = os.path.join(outdir, f"{sym}_{y0}_{y1}.csv")

    # 先 yfinance
    try:
        print(f"[DL] (yfinance) {sym} {start} ~ {end}")
        df = _download_yf(sym, start, end, proxy)
        df.to_csv(out, index=False)
        print(f"[OK] {sym} saved -> {out} (rows={len(df)})")
        return out
    except Exception as e_yf:
        print(f"[WARN] yfinance 失败：{e_yf}")

    # 回退 stooq
    try:
        print(f"[DL] (stooq) {sym}")
        df = _download_stooq(sym)
        # stooq 是全历史，这里截断到区间
        mask = (df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end))
        df = df.loc[mask]
        if df.empty:
            raise RuntimeError("stooq 返回为空或区间切分为空")
        df.to_csv(out, index=False)
        print(f"[OK] {sym} (stooq) saved -> {out} (rows={len(df)})")
        return out
    except Exception as e_sq:
        print(f"[ERR] stooq 也失败：{e_sq}")
        raise RuntimeError(f"{sym} 两个数据源均失败（可能当前环境外网受限或被限流）")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="逗号分隔：SPY,QQQ,AAPL")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--proxy", default="", help="可选：HTTP(S) 代理，如 http://user:pass@host:port")
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.to_datetime(args.start).strftime("%Y-%m-%d")
    end   = pd.to_datetime(args.end).strftime("%Y-%m-%d")

    results: Dict[str, dict] = {}
    for s in syms:
        try:
            p = save_one(s, start, end, args.outdir, args.proxy or None)
            results[s] = {"ok": True, "path": p}
        except Exception as e:
            results[s] = {"ok": False, "error": str(e)}
    print("[SUMMARY]", json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
