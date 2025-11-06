class Strat(Strategy):
    n=14; over=70; under=30
    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.n)
    def next(self):
        if self.rsi[-1] < self.under and not self.position:
            self.buy()
        if self.rsi[-1] > self.over and self.position:
            self.position.close()
