import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

def SMA(s, n):
    n = int(n)
    s = pd.Series(s).shift(1)
    return s.rolling(n).mean()

def RSI(s, n=14):
    s = pd.Series(s).shift(1)
    d = s.diff()
    up   = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = -d.clip(upper=0).ewm(alpha=1/n, adjust=False).mean()
    rs = up/(down+1e-12)
    return 100 - (100/(1+rs))

class Strat(Strategy):
    def init(self):
        self.fast = self.I(SMA, self.data.Close, 50)
        self.slow = self.I(SMA, self.data.Close, 200)
        self.rsi  = self.I(RSI, self.data.Close, 14)

    def next(self):
        if crossover(self.fast, self.slow) and self.rsi[-1] > 50:
            if not self.position:
                self.buy(size=1)
        elif crossover(self.slow, self.fast) or self.rsi[-1] < 45:
            if self.position:
                self.position.close()
