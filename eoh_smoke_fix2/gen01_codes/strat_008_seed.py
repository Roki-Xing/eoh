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
    def __init__(self, short_window=50, long_window=200, rsi_window=14):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_window = rsi_window

    def calculate_sma(self, data):
        return data['Close'].rolling(window=self.short_window).mean(), data['Close'].rolling(window=self.long_window).mean()

    def calculate_rsi(self, data):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def apply_rule(self, data):
        sma_short, sma_long = self.calculate_sma(data)
        rsi = self.calculate_rsi(data)

        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['sma_cross'] = np.where(sma_short > sma_long, 1.0, 0.0)
        signals['rsi'] = rsi
        signals['position'] = signals['signal'].diff()

        signals.loc[signals['sma_cross'] == 1.0, 'signal'] = 1.0

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

