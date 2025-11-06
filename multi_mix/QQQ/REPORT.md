# REPORT

### Run params snapshot
```json
{
  "symbol": "QQQ",
  "csv": "/root/autodl-tmp/price_cache/QQQ_2020_2023.csv",
  "train": [
    "2020-01-01",
    "2022-12-31"
  ],
  "test": [
    "2023-01-01",
    "2023-12-29"
  ],
  "commission": 0.0005,
  "population": 56,
  "generations": 5,
  "topk": 8,
  "seed": 42,
  "model_dir": "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
}
```

## TRAIN — BBreak_n13_k1.45_h6
- Family: **BBreak**, Params: `{"n": 13, "k": 1.45, "h": 6}`
- Last year: stratR=-11.91%, bhR=-33.22%, stratSharpe=-0.86, bhSharpe=-1.10

## TEST — BBreak_n13_k1.45_h6
- Family: **BBreak**, Params: `{"n": 13, "k": 1.45, "h": 6}`
- Last year: stratR=29.74%, bhR=55.83%, stratSharpe=2.22, bhSharpe=2.61

