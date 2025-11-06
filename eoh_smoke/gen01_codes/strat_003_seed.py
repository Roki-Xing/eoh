class Strat(Strategy):
    n=20; k=2.0
    def init(self):
        c=self.data.Close
        m=self.I(SMA, c, self.n)
        s=pd.Series(c).rolling(int(self.n)).std().to_numpy()
        self.up = m + self.k*s
        self.lo = m - self.k*s
    def next(self):
        c=self.data.Close[-1]
        if c > self.up[-1]:
            self.buy()
        elif c < self.lo[-1] and self.position:
            self.position.close()
