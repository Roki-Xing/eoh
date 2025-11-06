#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modularised evolution loop for the Bollinger Breakout (BBreak) strategy family.

The script now delegates data loading, backtesting, fitness scoring, and logging
to the shared `eoh_core` package. This reduces duplication with the export and
GPU scripts and makes it easier to plug the engine into other orchestration
workflows or test harnesses.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from eoh_core import (
    BBreakParams,
    FitnessWeights,
    RiskParams,
    backtest_bbreak,
    blended_score,
    enforce_constraints,
    metric_score,
    robust_read_csv,
    slice_by_date,
)
from eoh_core.strategies import BacktestMetrics
from eoh_core.utils import ensure_dir, log


def parse_objectives(value: str) -> List[Tuple[str, bool]]:
    """
    Parse objective specification string, e.g. "testReturn+,testSharpe+,testMDD-".
    The trailing '+' means maximise, '-' means minimise.
    """
    objectives: List[Tuple[str, bool]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token[-1] in {"+", "-"}:
            column = token[:-1]
            maximise = token[-1] == "+"
        else:
            column = token
            maximise = True
        objectives.append((column, maximise))
    return objectives


def parse_weight_presets(value: str) -> Dict[str, FitnessWeights]:
    """
    Parse presets like "return:alpha=0.5,beta=1.5;balanced:alpha=1,beta=0.7,gamma=1.2".
    Missing coefficients fall back to FitnessWeights defaults.
    """
    presets: Dict[str, FitnessWeights] = {}
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid weight preset '{chunk}'")
        name, spec = chunk.split(":", 1)
        coeffs: Dict[str, float] = {}
        for pair in spec.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Invalid coefficient '{pair}' in preset '{name}'")
            key, raw = pair.split("=", 1)
            coeffs[key.strip()] = float(raw.strip())
        presets[name.strip()] = FitnessWeights(**coeffs)
    return presets


DEFAULT_OBJECTIVES: List[Tuple[str, bool]] = [
    ("testReturn", True),
    ("testSharpe", True),
    ("testMDD", False),
]


def make_name(params: BBreakParams) -> str:
    return f"BBreak_n{params.n}_k{params.k}_h{params.h}"


def sample_params(rng: random.Random) -> BBreakParams:
    return BBreakParams(
        n=rng.randint(5, 40),
        k=round(rng.uniform(0.5, 1.8), 2),
        h=rng.randint(2, 10),
    )


def mutate_params(base: BBreakParams, rng: random.Random) -> BBreakParams:
    n = min(80, max(3, int(round(base.n + rng.gauss(0, 5)))))
    k = min(3.0, round(max(0.4, base.k + rng.gauss(0, 0.2)), 2))
    h = min(20, max(1, int(round(base.h + rng.gauss(0, 2)))))
    return BBreakParams(n=n, k=k, h=h)


@dataclass
class CandidateSummary:
    params: BBreakParams
    train_metrics: dict
    test_metrics: dict
    train_score: float
    test_score: float
    fitness: float
    feasible: bool

    def to_row(self) -> dict:
        return {
            "name": make_name(self.params),
            "family": "BBreak",
            "params": json.dumps(asdict(self.params)),
            "trainReturn": self.train_metrics["return"],
            "trainSharpe": self.train_metrics["sharpe"],
            "trainMDD": self.train_metrics["mdd"],
            "trainTrades": self.train_metrics["trades"],
            "trainExposure": self.train_metrics["exposure"],
            "trainTurnover": self.train_metrics["turnover"],
            "trainCostRatio": self.train_metrics["cost_ratio"],
            "testReturn": self.test_metrics["return"],
            "testSharpe": self.test_metrics["sharpe"],
            "testMDD": self.test_metrics["mdd"],
            "testTrades": self.test_metrics["trades"],
            "testExposure": self.test_metrics["exposure"],
            "testTurnover": self.test_metrics["turnover"],
            "testCostRatio": self.test_metrics["cost_ratio"],
            "trainFitness": self.train_score,
            "testFitness": self.test_score,
            "fitness": self.fitness,
            "feasible": int(self.feasible),
        }


@dataclass
class EvolutionConfig:
    population: int
    generations: int
    topk: int
    weights: FitnessWeights
    risk: RiskParams
    test_share: float = 0.8
    multi_objective: bool = False
    objectives: Optional[Sequence[Tuple[str, bool]]] = None
    weight_presets: Optional[Dict[str, FitnessWeights]] = None


class EvolutionEngine:
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        outdir: str,
        config: EvolutionConfig,
        rng: random.Random,
    ) -> None:
        if df_train.empty or df_test.empty:
            raise ValueError("Train/test 时间段无数据")
        self.df_train = df_train
        self.df_test = df_test
        self.outdir = ensure_dir(outdir)
        self.config = config
        self.rng = rng
        self.objectives = list(self.config.objectives or DEFAULT_OBJECTIVES)
        self.history_rows: List[dict] = []

    def _initial_population(self) -> List[BBreakParams]:
        return [sample_params(self.rng) for _ in range(self.config.population)]

    def _load_elites(self, generation: int) -> List[BBreakParams]:
        prev_path = self.outdir / f"gen{generation:02d}.csv"
        if not prev_path.exists():
            return []
        prev = pd.read_csv(prev_path)
        if prev.empty or "params" not in prev.columns:
            return []
        prev = prev.sort_values("fitness", ascending=False).head(min(self.config.topk, len(prev)))
        elites: List[BBreakParams] = []
        for _, row in prev.iterrows():
            try:
                params = json.loads(row["params"])
                elites.append(
                    BBreakParams(
                        n=int(params["n"]),
                        k=float(params["k"]),
                        h=int(params["h"]),
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                log(f"[WARN] failed to parse params from previous generation: {exc}")
        return elites

    def _mutate_population(self, seeds: Sequence[BBreakParams]) -> List[BBreakParams]:
        if not seeds:
            return self._initial_population()
        new_pop: List[BBreakParams] = []
        while len(new_pop) < self.config.population:
            base = self.rng.choice(seeds)
            new_pop.append(mutate_params(base, self.rng))
        return new_pop

    def _summarise(self, params: BBreakParams) -> CandidateSummary:
        train_result = backtest_bbreak(self.df_train, params, self.config.risk)
        test_result = backtest_bbreak(self.df_test, params, self.config.risk)

        feasible = enforce_constraints(test_result, self.config.risk)
        train_score = metric_score(train_result.metrics, self.config.weights)
        test_score = metric_score(test_result.metrics, self.config.weights)
        if feasible:
            fitness = blended_score(
                train_result.metrics,
                test_result.metrics,
                self.config.weights,
                self.config.test_share,
            )
        else:
            fitness = -1e9

        return CandidateSummary(
            params=params,
            train_metrics=train_result.metrics.as_dict(),
            test_metrics=test_result.metrics.as_dict(),
            train_score=float(train_score),
            test_score=float(test_score),
            fitness=float(fitness),
            feasible=feasible,
        )

    def run_generation(self, generation: int, population: Iterable[BBreakParams]) -> pd.DataFrame:
        rows = []
        for idx, individual in enumerate(population, 1):
            summary = self._summarise(individual)
            row_dict = summary.to_row()
            row_dict["generation"] = generation
            rows.append(row_dict)
            log(
                f"[GEN {generation:02d}] {idx:03d}/{self.config.population} "
                f"{row_dict['name']} fitness={summary.fitness:.4f} feasible={summary.feasible}"
            )
        frame = pd.DataFrame(rows)
        path = self.outdir / f"gen{generation:02d}.csv"
        frame.to_csv(path, index=False)
        log(f"[INFO] gen{generation:02d} -> {path}")
        if not frame.empty:
            self.history_rows.extend(frame.to_dict("records"))
        return frame

    def _normalise_objective(self, frame: pd.DataFrame, column: str, maximise: bool) -> pd.Series:
        values = frame[column].astype(float)
        if maximise:
            return values
        if "mdd" in column.lower() or "drawdown" in column.lower():
            return -values.abs()
        return -values

    def _pareto_front(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not self.objectives:
            return frame.copy()
        scores = [
            self._normalise_objective(frame, column, maximise).to_numpy()
            for column, maximise in self.objectives
        ]
        matrix = np.vstack(scores).T
        mask = np.ones(len(frame), dtype=bool)
        for i in range(len(frame)):
            if not mask[i]:
                continue
            for j in range(len(frame)):
                if i == j or not mask[j]:
                    continue
                if np.all(matrix[j] >= matrix[i]) and np.any(matrix[j] > matrix[i]):
                    mask[i] = False
                    break
        return frame.loc[mask].copy()

    def _write_multiobjective_outputs(self, aggregated: pd.DataFrame) -> None:
        if aggregated.empty:
            return
        aggregated.to_csv(self.outdir / "all_candidates.csv", index=False)
        if self.config.multi_objective:
            front = self._pareto_front(aggregated)
            front.to_csv(self.outdir / "pareto_front.csv", index=False)
            log(f"[INFO] pareto front size={len(front)} -> {self.outdir / 'pareto_front.csv'}")

    def _write_weight_presets(self, aggregated: pd.DataFrame) -> None:
        if not self.config.weight_presets or aggregated.empty:
            return
        rows = []
        for name, weights in self.config.weight_presets.items():
            best_score = None
            best_row = None
            for record in aggregated.to_dict("records"):
                metrics = BacktestMetrics(
                    total_return=float(record.get("testReturn", 0.0)),
                    sharpe=float(record.get("testSharpe", 0.0)),
                    max_drawdown=float(record.get("testMDD", 0.0)),
                    trades=int(record.get("testTrades", 0)),
                    exposure=float(record.get("testExposure", 0.0)),
                    turnover=float(record.get("testTurnover", 0.0)),
                    cost_ratio=float(record.get("testCostRatio", 0.0)),
                )
                score = metric_score(metrics, weights)
                if best_score is None or score > best_score:
                    best_score = score
                    best_row = record
            if best_row is not None:
                rows.append(
                    {
                        "preset": name,
                        "score": best_score,
                        "candidate": best_row["name"],
                        "params": best_row["params"],
                        "testReturn": best_row.get("testReturn"),
                        "testSharpe": best_row.get("testSharpe"),
                        "testMDD": best_row.get("testMDD"),
                    }
                )
        if rows:
            pd.DataFrame(rows).to_csv(self.outdir / "weight_presets_summary.csv", index=False)

    def run(self) -> pd.Series | None:
        population = self._initial_population()
        best_row: pd.Series | None = None

        for gen in range(1, self.config.generations + 1):
            if gen > 1:
                seeds = self._load_elites(gen - 1)
                population = self._mutate_population(seeds if seeds else population)

            frame = self.run_generation(gen, population)
            if not frame.empty:
                gen_best = frame.loc[frame["fitness"].idxmax()]
                if best_row is None or gen_best["fitness"] > best_row["fitness"]:
                    best_row = gen_best
        aggregated = pd.DataFrame(self.history_rows)
        if not aggregated.empty:
            self._write_multiobjective_outputs(aggregated)
            self._write_weight_presets(aggregated)
        return best_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--train_start", required=True)
    parser.add_argument("--train_end", required=True)
    parser.add_argument("--test_start", required=True)
    parser.add_argument("--test_end", required=True)
    parser.add_argument("--population", type=int, default=56)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--use-llm", dest="use_llm", action="store_true", help="标记位（兼容旧参数）")
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--delay-days", type=int, default=0)
    parser.add_argument("--max-position", type=float, default=1.0)
    parser.add_argument("--max-daily-turnover", type=float, default=1.0)
    parser.add_argument("--min-trades", type=int, default=0)
    parser.add_argument("--min-exposure", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--zeta", type=float, default=0.0)
    parser.add_argument("--test-share", type=float, default=0.8, help="测试集在整体 fitness 中的权重")
    parser.add_argument(
        "--multi-objective",
        action="store_true",
        help="启用多目标选择，生成帕累托前沿（基于 --objectives 配置）。",
    )
    parser.add_argument(
        "--objectives",
        default="testReturn+,testSharpe+,testMDD-",
        help="逗号分隔的目标列，结尾 + 为最大化，- 为最小化，例如 testReturn+,testSharpe+,testMDD-。",
    )
    parser.add_argument(
        "--weight-presets",
        default="",
        help="可选的权重预设，格式 name:alpha=1,beta=0.5;... 用于运行后比较不同权重下的最优解。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df_all = robust_read_csv(args.csv)
    df_train = slice_by_date(df_all, args.train_start, args.train_end)
    df_test = slice_by_date(df_all, args.test_start, args.test_end)

    risk = RiskParams(
        commission=args.commission,
        slippage_bps=args.slippage_bps,
        delay_days=args.delay_days,
        max_position=args.max_position,
        max_daily_turnover=args.max_daily_turnover,
        min_trades=args.min_trades,
        min_exposure=args.min_exposure,
    )
    weights = FitnessWeights(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        zeta=args.zeta,
    )
    objectives = parse_objectives(args.objectives) if args.objectives else DEFAULT_OBJECTIVES
    if not objectives:
        objectives = DEFAULT_OBJECTIVES
    weight_presets = parse_weight_presets(args.weight_presets) if args.weight_presets.strip() else None
    config = EvolutionConfig(
        population=args.population,
        generations=args.generations,
        topk=args.topk,
        weights=weights,
        risk=risk,
        test_share=args.test_share,
        multi_objective=args.multi_objective,
        objectives=objectives,
        weight_presets=weight_presets,
    )

    engine = EvolutionEngine(df_train, df_test, args.outdir, config, random.Random(args.seed))
    best_row = engine.run()

    if best_row is not None:
        log(
            "[BEST] name={name} fitness={fitness:.4f} "
            "trainR={trainReturn:.2%} testR={testReturn:.2%} "
            "trainSharpe={trainSharpe:.2f} testSharpe={testSharpe:.2f}".format(**best_row.to_dict())
        )
    log(f"[INFO] all generations done -> {engine.outdir}")


if __name__ == "__main__":
    main()
