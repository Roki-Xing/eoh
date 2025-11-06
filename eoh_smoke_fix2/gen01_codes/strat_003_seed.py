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
    def __init__(self, short_window=40, long_window=100, rsi_window=14):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_window = rsi_window

    def calculate_sma(self, data, window):
        return data['close'].rolling(window=window).mean()

    def calculate_rsi(self, data, window):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def apply_rule(self, data):
        sma_short = self.calculate_sma(data, self.short_window)
        sma_long = self.calculate_sma(data, self.long_window)
        rsi = self.calculate_rsi(data, self.rsi_window)

        if sma_short[-2] < sma_long[-2] and sma_short[-1] > sma_long[-1]:
            return 'buy'
        elif sma_short[-2] > sma_long[-2] and sma_short[-1] < sma_long[-1]:
            return 'sell'
        else:
            return None

    def backtest_strategy(self, data):
        signals = []

def init(self):
        self.fast = self.I(SMA, self.data.Close, 40)
        self.slow = self.I(SMA, self.data.Close, 100)
        self.rsi  = self.I(RSI, self.data.Close, 14)


def next(self):
        if crossover(self.fast, self.slow) and self.rsi[-1] > 50:
            if not self.position:
                self.buy(size=1)
        elif crossover(self.slow, self.fast) or self.rsi[-1] < 45:
            if self.position:
                self.position.close()

