# REPORT

### Run params snapshot
```json
{
  "symbol": "AAPL",
  "csv": "/root/autodl-tmp/price_cache/AAPL_2020_2023.csv",
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

## TRAIN — BBreak_n12_k0.86_h11
- Family: **BBreak**, Params: `{"n": 12, "k": 0.86, "h": 11}`
- Last year: stratR=-25.29%, bhR=-28.20%, stratSharpe=-1.08, bhSharpe=-0.76

## TEST — BBreak_n12_k0.86_h11
- Family: **BBreak**, Params: `{"n": 12, "k": 0.86, "h": 11}`
- Last year: stratR=31.81%, bhR=54.80%, stratSharpe=1.94, bhSharpe=2.32

