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
    def __init__(self):
        self.size = 1
        self.stop_loss = None
        self.take_profit = None

    def calculate_sma(self, data, window):
        return data.rolling(window=window).mean()

    def calculate_rsi(self, data, window):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).fillna(0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def apply_risk_rules(self, data):
        self.size = min(self.size, 1)
        if self.stop_loss is not None:
            if data['Close'] <= self.stop_loss:
                self.size = 0
        if self.take_profit is not None:
            if data['Close'] >= self.take_profit:
                self.size = 0

    def generate_signals(self, data):
        sma20 = self.calculate_sma(data['Close'], 20)
        sma50 = self.calculate_sma(data['Close'], 50)
        rsi14 = self.calculate_rsi(data['Close'], 14)

        long_condition = (data['Close'] > sma20) & (sma20 > sma50)

def init(self):
        self.fast = self.I(SMA, self.data.Close.shift(1), 100)
        self.slow = self.I(SMA, self.data.Close.shift(1), 100)
        self.rsi  = self.I(RSI, self.data.Close.shift(1), 20)


def next(self):
        if crossover(self.fast, self.slow) and self.rsi[-1] > 50:
            if not self.position:
                self.buy(size=1)
        elif crossover(self.slow, self.fast) or self.rsi[-1] < 45:
            if self.position:
                self.position.close()

