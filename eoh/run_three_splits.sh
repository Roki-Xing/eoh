#!/usr/bin/env bash
set -euo pipefail

# ====== 基本配置 ======
SYMBOL="SPY"
MODEL_DIR="/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
CSV="/root/autodl-tmp/price_cache/SPY_2020_2023.csv"   # 统一使用超集CSV
OUTBASE="/root/autodl-tmp/outputs"
COMMISSION="0.0005"

# 演化搜索参数（可按需调整）
POP=56
GENS=3
TOPK=8

# 主脚本/后处理脚本路径
MAIN="/root/autodl-tmp/quant/eoh_gpu_loop.py"
POST="/root/autodl-tmp/quant/eoh_post_export.py"

# ====== 检查依赖 ======
if [[ ! -f "$CSV" ]]; then
  echo "[ERROR] CSV not found: $CSV"
  echo "请确认文件路径，或修改此脚本 CSV 变量为你的实际CSV路径。"
  exit 1
fi

if [[ ! -f "$MAIN" ]]; then
  echo "[ERROR] main script not found: $MAIN"
  exit 1
fi

if [[ ! -f "$POST" ]]; then
  echo "[ERROR] post-export script not found: $POST"
  exit 1
fi

# ====== 通用运行函数 ======
run_split () {
  local OUTDIR="$1"
  local TRAIN_START="$2"
  local TRAIN_END="$3"
  local TEST_START="$4"
  local TEST_END="$5"

  mkdir -p "$OUTDIR"

  echo "--------------------------------------------------"
  echo "[INFO] Running split:"
  echo "       OUTDIR      = $OUTDIR"
  echo "       TRAIN       = $TRAIN_START -> $TRAIN_END"
  echo "       TEST        = $TEST_START  -> $TEST_END"
  echo "--------------------------------------------------"

  # 1) 演化搜索主流程
  python "$MAIN" \
    --model-dir "$MODEL_DIR" \
    --symbol "$SYMBOL" \
    --csv "$CSV" \
    --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
    --test_start  "$TEST_START"  --test_end  "$TEST_END" \
    --population $POP --generations $GENS --topk $TOPK \
    --commission "$COMMISSION" \
    --outdir "$OUTDIR"

  # 2) 导出最优策略（交易、持仓、净值、年报、图）
  python "$POST" \
    --symbol "$SYMBOL" \
    --csv "$CSV" \
    --train_start "$TRAIN_START" --train_end "$TRAIN_END" \
    --test_start  "$TEST_START"  --test_end  "$TEST_END" \
    --commission "$COMMISSION" \
    --outdir "$OUTDIR"

  echo "[INFO] Done -> $OUTDIR"
  echo
}

# ====== 三组切分 ======
# A) 2020–2021 训练 → 2022 测试
run_split "$OUTBASE/eoh_20to21_train_22_test" \
  "2020-01-01" "2021-12-31" \
  "2022-01-01" "2022-12-31"

# B) 2020–2022 训练 → 2023 测试（更接近生产）
run_split "$OUTBASE/eoh_20to22_train_23_test" \
  "2020-01-01" "2022-12-31" \
  "2023-01-01" "2023-12-31"

# C) 2022 训练 → 2023 测试（单年训练的泛化）
run_split "$OUTBASE/eoh_22_train_23_test" \
  "2022-01-01" "2022-12-31" \
  "2023-01-01" "2023-12-31"

echo "[INFO] All splits completed."
