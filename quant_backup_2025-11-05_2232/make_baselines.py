import os, io, json, warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd, numpy as np

# ---------- 数据加载：AlphaVantage -> yfinance -> synthetic ----------
def _read_alpha_csv(text):
    df = pd.read_csv(io.StringIO(text))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    return df[["Open","High","Low","Close","Volume"]]

def load_alpha(symbol, start, end):
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise RuntimeError("no alpha key")
    import requests

    # 先试 ADJUSTED，不行就退回 DAILY（避免 premium 提示）
    for fn in ("TIME_SERIES_DAILY_ADJUSTED", "TIME_SERIES_DAILY"):
        url=(f"https://www.alphavantage.co/query?function={fn}"
             f"&symbol={symbol}&apikey={key}&datatype=csv&outputsize=full")
        r = requests.get(url, timeout=60); r.raise_for_status()
        txt = r.text.strip()
        if txt.startswith("{"):     # 被限流/提示 premium
            continue
        df = _read_alpha_csv(txt)
        df = df.loc[start:end]
        if len(df):
            return df
    raise RuntimeError("alpha premium or rate limit")

def load_yf(symbol, start, end):
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yf empty or rate limited")
    return df[["Open","High","Low","Close","Volume"]]

def load_syn(start, end):
    idx = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(7)
    ret = rng.normal(0.0003, 0.01, len(idx))
    px  = 100*np.cumprod(1+ret)
    df  = pd.DataFrame({"Close": px}, index=idx)
    df["Open"]=df["High"]=df["Low"]=df["Close"]; df["Volume"]=1_000_000
    return df[["Open","High","Low","Close","Volume"]]

def get_data(symbol, start, end):
    for loader in (load_alpha, load_yf):
        try:
            df = loader(symbol, start, end)
            src = loader.__name__.replace("load_", "")
            return df, src
        except Exception as e:
            print(f"[WARN] {loader.__name__} failed: {e}")
    return load_syn(start, end), "synthetic"

# ---------- 回测 ----------
from backtesting import Backtest, Strategy

class BuyHold(Strategy):
    def init(self): pass
    def next(self):
        if not self.position:
            self.buy()

class SMACross(Strategy):
    n1=10; n2=50
    def init(self):
        # 用一个简单的向量化 SMA 函数，避免 pandas 依赖
        def sma(x, n):
            x = np.asarray(x, dtype=float)
            s = pd.Series(x).rolling(int(n)).mean().to_numpy()
            return s
        self.s1 = self.I(sma, self.data.Close, self.n1)
        self.s2 = self.I(sma, self.data.Close, self.n2)
    def next(self):
        if self.s1[-1] > self.s2[-1] and not self.position:
            self.buy()
        elif self.s1[-1] < self.s2[-1] and self.position:
            self.position.close()

class RSIRev(Strategy):
    look=14; lo=30; hi=70
    def init(self):
        # 向量化 RSI（简单均线版），传给 self.I(close, look)
        def rsi_vec(close, n):
            close = np.asarray(close, dtype=float)
            r = np.diff(close, prepend=close[0])
            up = np.where(r>0,  r, 0.0)
            dn = np.where(r<0, -r, 0.0)
            up_ma = pd.Series(up).rolling(int(n)).mean().to_numpy()
            dn_ma = pd.Series(dn).rolling(int(n)).mean().to_numpy()
            rs = up_ma / (dn_ma + 1e-12)
            return 100 - 100/(1 + rs)
        self.rsi = self.I(rsi_vec, self.data.Close, self.look)
    def next(self):
        if self.rsi[-1] < self.lo and not self.position:
            self.buy()
        if self.rsi[-1] > self.hi and self.position:
            self.position.close()

def run_bt(df, strat):
    bt = Backtest(df, strat, cash=100_000, commission=0.0005, finalize_trades=True)
    st = bt.run()
    return {
        "Return[%]": float(st["Return [%]"]),
        "Sharpe": float(st.get("Sharpe Ratio", float("nan"))),
        "MaxDD[%]": float(st.get("Max. Drawdown [%]", float("nan"))),
        "Trades": int(st.get("# Trades", 0)),
    }, bt

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--train_start", default="2020-01-01")
    ap.add_argument("--train_end",   default="2022-12-31")
    ap.add_argument("--test_start",  default="2023-01-01")
    ap.add_argument("--test_end",    default="2023-12-31")
    ap.add_argument("--outdir", default="/root/autodl-tmp/outputs/baselines")
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for split,(s,e) in {
        "train":(args.train_start,args.train_end),
        "test": (args.test_start, args.test_end),
    }.items():
        df,src = get_data(args.symbol, s, e)
        print(f"[INFO] split={split}, source={src}, rows={len(df)}")
        df.to_csv(f"{args.outdir}/{args.symbol}_{split}_{src}.csv")
        rows=[]
        for name,strat in [("BuyHold",BuyHold),("SMA10-50",SMACross),("RSI14",RSIRev)]:
            stats,_ = run_bt(df,strat)
            rows.append({"split":split,"source":src,"strategy":name, **stats})
        pd.DataFrame(rows).to_csv(f"{args.outdir}/{args.symbol}_baselines_{split}.csv", index=False)
        print(f"[OK] {split} baselines -> {args.outdir}")
