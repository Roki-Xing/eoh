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
    def __init__(self, sma1=50, sma2=200, rsi_period=14):
        self.sma1 = sma1
        self.sma2 = sma2
        self.rsi_period = rsi_period

    def calculate_signal(self, data):
        close = data['close']
        sma1 = close.rolling(window=self.sma1).mean()
        sma2 = close.rolling(window=self.sma2).mean()
        rsi = RSI(close, self.rsi_period)

        signal = 0
        if sma1[-1] > sma2[-1] and sma1[-2] <= sma2[-2]:
            signal = 1
        elif sma1[-1] < sma2[-1] and sma1[-2] >= sma2[-2]:
            signal = -1
        elif rsi[-1] > 70:
            signal = -1
        elif rsi[-1] < 30:
            signal = 1

        return signal

    def apply_signal(self, data, signal):
        if signal == 1:
            self.buy(data)
        elif signal == -1:
            self.sell(data)

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

