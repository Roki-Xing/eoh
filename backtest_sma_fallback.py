import os, time, argparse, requests, pandas as pd, numpy as np, matplotlib.pyplot as plt
import yfinance as yf
from backtesting import Backtest, Strategy

# ---------- Data fetchers ----------
def yf_session():
    s = requests.Session()
    s.headers.update({"User-Agent":"Mozilla/5.0"})
    return s

def fetch_yf(symbol, start, end, tries=5):
    last_err=None
    for k in range(tries):
        try:
            df = yf.download(symbol, start=start, end=end, auto_adjust=False,
                             progress=False, threads=False, session=yf_session())
            if isinstance(df.columns, pd.MultiIndex):
                if symbol in df.columns.levels[0]:
                    df = df.xs(symbol, axis=1, level=0)
                else:
                    df.columns = df.columns.get_level_values(-1)
            if not df.empty:
                return df.dropna()
        except Exception as e:
            last_err = e
        time.sleep(2*(k+1))
    raise last_err or RuntimeError("yfinance empty")

def fetch_alpha_vantage(symbol, start, end):
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY not set")
    url = "https://www.alphavantage.co/query"
    params = {
        "function":"TIME_SERIES_DAILY",
        "symbol":symbol,
        "outputsize":"full",
        "datatype":"csv",
        "apikey":api_key,
    }
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df.rename(columns={"timestamp":"Date","open":"Open","high":"High","low":"Low",
                       "close":"Close","volume":"Volume"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]); df.set_index("Date", inplace=True)
    df = df.sort_index()
    df = df.loc[(df.index>=pd.to_datetime(start))&(df.index<pd.to_datetime(end))]
    need = ["Open","High","Low","Close","Volume"]
    if not all(c in df.columns for c in need) or df.empty:
        raise RuntimeError("alpha_vantage empty")
    return df[need].dropna()

def fetch_stooq(symbol):
    code = symbol.lower()
    url = f"https://stooq.com/q/d/l/?s={code}&i=d"
    df = pd.read_csv(url)
    df.rename(columns=str.title, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]); df.set_index("Date", inplace=True)
    need = ["Open","High","Low","Close","Volume"]
    if not all(c in df.columns for c in need) or df.empty:
        raise RuntimeError("stooq empty")
    return df[need].dropna()

def fetch_with_fallback(symbol, start, end):
    try:
        df = fetch_yf(symbol, start, end); return df, "yfinance"
    except Exception:
        pass
    try:
        df = fetch_alpha_vantage(symbol, start, end); return df, "alphavantage"
    except Exception:
        pass
    try:
        df = fetch_stooq(symbol)
        df = df.loc[(df.index>=pd.to_datetime(start))&(df.index<pd.to_datetime(end))]
        if df.empty: raise RuntimeError("stooq empty after slicing")
        return df, "stooq"
    except Exception:
        pass
    idx = pd.date_range(start, end, freq="B")
    ret = np.random.normal(0.0005, 0.01, size=len(idx))
    px = 100 * (1+ret).cumprod()
    df = pd.DataFrame({"Open":px, "High":px*1.002, "Low":px*0.998, "Close":px, "Volume":1_000_000}, index=idx)
    return df, "synthetic"

# ---------- Strategy ----------
def add_sma(df, n): return df["Close"].rolling(int(n)).mean()

class SMACross(Strategy):
    n1 = 10
    n2 = 50
    def init(self):
        close = pd.Series(self.data.Close, name="Close")
        sma1 = add_sma(close.to_frame(name="Close"), self.n1)
        sma2 = add_sma(close.to_frame(name="Close"), self.n2)
        self.sma1 = self.I(lambda: sma1.values)
        self.sma2 = self.I(lambda: sma2.values)
    def next(self):
        if self.sma1[-1] > self.sma2[-1] and not self.position:
            self.buy()
        elif self.sma1[-1] < self.sma2[-1] and self.position.is_long:
            self.position.close()

# ---------- Plot ----------
def plot_price_sma(df, out_png, title):
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["Close"], label="Close")
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    plt.plot(df.index, df["SMA_10"], label="SMA_10")
    plt.plot(df.index, df["SMA_50"], label="SMA_50")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_png); plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end",   default="2023-01-01")
    ap.add_argument("--outdir", default="/root/autodl-tmp/outputs")
    ap.add_argument("--commission", type=float, default=0.001)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df, src = fetch_with_fallback(args.symbol, args.start, args.end)
    if df.empty: raise RuntimeError("No data obtained from any source.")
    raw_csv = os.path.join(args.outdir, f"{args.symbol}_{args.start}_{args.end}_{src}.csv")
    df.to_csv(raw_csv)

    # 关键修复点：把 finalize_trades=True 放在 Backtest(...) 构造器里
    bt = Backtest(df, SMACross, cash=100_000, commission=args.commission, finalize_trades=True)
    stats = bt.run()

    stats_csv = os.path.join(args.outdir, f"{args.symbol}_stats_{src}.csv")
    stats.to_csv(stats_csv)

    out_png = os.path.join(args.outdir, f"{args.symbol}_sma_{src}.png")
    plot_price_sma(df.copy(), out_png, f"{args.symbol} SMA10/50 [{src}]")
    bt.plot(filename=os.path.join(args.outdir, f"{args.symbol}_bt_{src}.html"), open_browser=False)

    print("\n=== DONE ===")
    print("Source:", src)
    print("CSV:", raw_csv)
    print("Stats CSV:", stats_csv)
    print("Chart PNG:", out_png)
    print("Backtest HTML:", os.path.join(args.outdir, f"{args.symbol}_bt_{src}.html"))
    print("\nKey stats (finalized):")
    print(stats[["Return [%]","Sharpe Ratio","Max. Drawdown [%]"]])
if __name__ == "__main__":
    main()
