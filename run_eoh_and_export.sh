#!/usr/bin/env bash
set -e

MODEL_DIR="/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
CSV="/root/autodl-tmp/price_cache/SPY_2020_2023.csv"
SYMBOL="SPY"
TRAIN_START="2020-01-01"
TRAIN_END="2022-12-31"
TEST_START="2023-01-01"
TEST_END="2023-12-31"
OUTDIR="/root/autodl-tmp/outputs/eoh_run_stable"
COMMISSION="0.0005"

# 1) 先跑主脚本（会生成 gen*.csv）
python /root/autodl-tmp/quant/eoh_gpu_loop.py \
  --model-dir "$MODEL_DIR" \
  --symbol "$SYMBOL" \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --test_start  "$TEST_START"  --test_end  "$TEST_END" \
  --population 56 --generations 3 --topk 8 \
  --commission "$COMMISSION" \
  --outdir "$OUTDIR"

# 2) 再跑导出脚本（读取 gen*.csv 的最优解 → 回测导出）
python /root/autodl-tmp/quant/eoh_post_export.py \
  --symbol "$SYMBOL" \
  --csv "$CSV" \
  --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
  --test_start  "$TEST_START"  --test_end  "$TEST_END" \
  --commission "$COMMISSION" \
  --outdir "$OUTDIR"
