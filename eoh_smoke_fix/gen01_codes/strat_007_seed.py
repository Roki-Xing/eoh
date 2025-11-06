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
    def __init__(self, short_window=40, long_window=100, rsi_window=14, take_profit=5, stop_loss=2):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_window = rsi_window
        self.take_profit = take_profit
        self.stop_loss = stop_loss

    def calculate_sma(self, data, window):
        return data['Close'].rolling(window=window).mean()

    def calculate_rsi(self, data, window):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def apply_strategy(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        short_sma = self.calculate_sma(data, self.short_window)
        long_sma = self.calculate_sma(data, self.long_window)
        rsi = self.calculate_rsi(data, self.rsi_window)

        signals['short_sma'] = short_sma
        signals['long_sma'] = long_sma
        signals['rsi'] = rsi

        signals['signal'][self.short_window:]

def init(self):
        self.fast = self.I(SMA, self.data.Close.shift(1), 40)
        self.slow = self.I(SMA, self.data.Close.shift(1), 100)
        self.rsi  = self.I(RSI, self.data.Close.shift(1), 14)


def next(self):
        if crossover(self.fast, self.slow) and self.rsi[-1] > 50:
            if not self.position:
                self.buy(size=1)
        elif crossover(self.slow, self.fast) or self.rsi[-1] < 45:
            if self.position:
                self.position.close()

