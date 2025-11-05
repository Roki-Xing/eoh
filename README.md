# eoh

Modular research environment for the Evolution of Heuristics (EOH) workflow.

## Core library

The `eoh_core` package consolidates utilities that were previously duplicated across scripts:

- `eoh_core.data` – robust CSV ingestion (`robust_read_csv`), date slicing, and synthetic price helpers.
- `eoh_core.strategies` – Bollinger breakout (BBreak) strategy models plus reusable backtesting logic and metrics.
- `eoh_core.evaluation` – fitness scoring helpers with configurable weightings.
- `eoh_core.prompts` – prompt templates, style selection, and on-disk overrides.
- `eoh_core.llm` – lightweight abstractions for local Hugging Face models and code block extraction.
- `eoh_core.utils` – logging, directory helpers, and small collection utilities.

Import the package directly (`import eoh_core as eoh`) to access these components in notebooks or additional scripts.

## Key scripts

### `eoh_evolve_main.py`

- Uses the shared BBreak backtester to run evolutionary searches over `(n, k, h)` parameters.
- New CLI options:
  - `--test-share` controls the train/test weighting in the blended fitness (default `0.8`).
  - `--multi-objective` 与 `--objectives` 可直接输出帕累托前沿（默认优化 `testReturn+/testSharpe+/testMDD-`）。
  - `--weight-presets` 支持一次性比较多组 α/β/γ/δ/ζ 权重，并生成 `weight_presets_summary.csv`。
  - Existing risk knobs map directly onto `eoh_core.strategies.RiskParams`.
- Outputs land in the requested `--outdir`, with generation CSVs that now include turnover, cost ratio, and per-split fitness columns.

Example:

```bash
python3 eoh_evolve_main.py \
  --model-dir /models/llm \
  --csv data/SPY.csv \
  --train_start 2016-01-01 --train_end 2020-12-31 \
  --test_start 2021-01-01 --test_end 2023-12-31 \
  --outdir runs/exp01 \
  --population 64 --generations 5 \
  --test-share 0.9
```

### `eoh_post_export.py`

- Replays the best BBreak configuration on train/test sets using the shared engine.
- Exports tables, equity curves, and yearly reports with buy & hold comparisons; metric summaries (return, Sharpe, turnover, cost ratio) are appended to `REPORT.md`.
- The script now accepts the same risk knobs used during evolution so artefacts are reproducible.

### `scripts/run_experiments.py`

- Reads YAML configs（参见 `config/experiments/sample.yaml`）批量执行演化与导出流程，可配合 `--dry-run` 先审查命令。
- 自动为每个实验生成独立输出目录，方便对比不同参数组合或 Prompt 设置。

### `eoh_gpu_loop_fixed.py`

- Generates candidate `Strat` implementations through a local Hugging Face model.
- Prompt management is configurable:
  - `--prompt-style normal,conservative,aggressive` cycles styles (default `normal`).
  - `--prompt-dir` points to a directory containing `<style>_system.txt` and `<style>_user.txt` overrides; otherwise defaults from `prompts/` are used.
- Each run stores the raw LLM output, prompt metadata, and backtest metrics for traceability.

## Prompt templates

Editable prompt files live in `prompts/`:

- `normal_system.txt` / `normal_user.txt`
- `conservative_system.txt` / `conservative_user.txt`
- `aggressive_system.txt` / `aggressive_user.txt`

Add new styles by creating matching files and referencing them via `--prompt-style`. The `PromptLibrary` automatically picks up overrides.

## 数据与配置

- `.env.example` 描述了 AlphaVantage、OpenAI 等外部依赖的环境变量，可复制为 `.env` 并填入实际值。
- `config/experiments/sample.yaml` 给出了批量实验的模板；可在其中配置不同的时间窗口、种群规模或模型目录。
- 如需自定义数据下载/缓存目录，可通过环境变量（如 `DATA_CACHE_DIR`）与 `eoh_core.data` 模块配合使用。

## Suggested workflow

1. Prepare price data (or allow `load_prices` to fetch/craft synthetic inputs).
2. Run the evolutionary loop (`eoh_evolve_main.py`) with the desired risk/fitness settings.
3. Export artefacts (`eoh_post_export.py`) for reports and plots.
4. Optionally explore LLM-generated strategies using `eoh_gpu_loop_fixed.py`, varying prompt styles for diversity.

Prefer the shared library for future scripts or notebooks instead of copying helpers across files.

## Testing & CI

- 轻量级单元测试位于 `tests/`，覆盖基础回测与 CSV 读取流程，使用 `pytest` 运行：

  ```bash
  python3 -m pip install -r requirements.txt
  pytest
  ```

- `.github/workflows/ci.yml` 在每次提交和 PR 上自动执行测试，确保重构后的核心逻辑保持稳定。
