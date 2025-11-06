# Auto-generated best strategy
# Family: BBreak
# Params: {"n": 12, "k": 0.86, "h": 11}
import pandas as pd
import numpy as np

def compute_bbands(close: pd.Series, n: int, k: float):
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    up = ma + k*sd
    lo = ma - k*sd
    return ma, up, lo

def run_bbreak(df: pd.DataFrame, n=12, k=0.86, hold=11, commission=0.0005):
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
            trades.append({"timestamp":dt,"side":"BUY","price":float(price),"size":1,"reason":"break_up"})
        elif pos==1:
            can_exit = True if (last_buy is None) else ((i-last_buy)>=hold)
            if can_exit and sell_sig.iloc[i]:
                eq *= (1.0-commission); pos=0
                trades.append({"timestamp":dt,"side":"SELL","price":float(price),"size":1,"reason":"below_ma"})
        equity.append(eq); pos_rows.append({"timestamp":dt,"position":pos,"price":float(price),"equity":float(eq)})
        prev = price
    return pd.DataFrame(trades), pd.DataFrame(pos_rows), pd.Series(equity, index=px.index, name="equity")
