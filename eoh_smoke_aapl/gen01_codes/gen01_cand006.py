def strategy(df):
    sma_short = ta.trend.sma_indicator(df['close'], window=50)
    sma_long = ta.trend.sma_indicator(df['close'], window=200)
    rsi = ta.momentum.rsi(df['close'], window=14)
    
    signals = pd.Series(0, index=df.index)
    signals[sma_short > sma_long] = 1
    signals[sma_short < sma_long] = -1
    signals[rsi >= 70] = -1
    signals[rsi <= 30] = 1
    
    return signals