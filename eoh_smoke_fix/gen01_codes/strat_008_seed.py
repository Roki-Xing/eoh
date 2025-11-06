import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

def SMA(s, n):
    n = int(n)
    return pd.Series(s).rolling(n).mean()

def RSI(s, n=14):
    s = pd.Series(s)
    d = s.diff()
    up   = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = -d.clip(upper=0).ewm(alpha=1/n, adjust=False).mean()
    rs = up/(down+1e-12)
    return 100 - (100/(1+rs))

class Strat(Strategy):
    def init(self):
        self.fast = self.I(SMA, self.data.Close.shift(1), 50)
        self.slow = self.I(SMA, self.data.Close.shift(1), 14)
        self.rsi  = self.I(RSI, self.data.Close.shift(1), 70)

    def next(self):
        # 简单规则：均线金叉且 RSI>50 做多，死叉或 RSI<45 平仓
        if crossover(self.fast, self.slow) and self.rsi[-1] > 50:
            if not self.position:
                self.buy(size=1)
        elif crossover(self.slow, self.fast) or self.rsi[-1] < 45:
            if self.position:
                self.position.close()
