import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

def SMA(series, n=10):
    s = pd.Series(series, copy=False)
    return s.rolling(int(n)).mean().to_numpy()

def RSI(series, n=14):
    s = pd.Series(series, copy=False).astype(float)
    d = s.diff()
    up = d.clip(lower=0.0).rolling(int(n)).mean()
    dn = (-d).clip(lower=0.0).rolling(int(n)).mean()
    rs = up / (dn.replace(0, np.nan) + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).to_numpy()

class Strat(Strategy):
    n1=10; n2=50
    def init(self):
        self.s1 = self.I(SMA, self.data.Close, self.n1)
        self.s2 = self.I(SMA, self.data.Close, self.n2)
    def next(self):
        if crossover(self.s1, self.s2):
            self.buy()
        elif crossover(self.s2, self.s1):
            self.position.close()
