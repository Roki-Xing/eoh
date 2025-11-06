"""
Fitness utilities shared by the evolution scripts.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..strategies.bbreak import BacktestMetrics


@dataclass(frozen=True)
class FitnessWeights:
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 1.0
    delta: float = 0.0
    zeta: float = 0.0


def metric_score(metrics: BacktestMetrics, weights: FitnessWeights) -> float:
    """
    Compute the scalar fitness score from a set of metrics.

    The score matches the legacy implementation and scales returns/drawdowns to percentage points.
    """
    return (
        weights.alpha * metrics.sharpe
        + weights.beta * (metrics.total_return * 100.0)
        - weights.gamma * abs(metrics.max_drawdown) * 100.0
        - weights.delta * (metrics.turnover * 100.0)
        - weights.zeta * (metrics.cost_ratio * 100.0)
    )


def blended_score(
    train_metrics: BacktestMetrics,
    test_metrics: BacktestMetrics,
    weights: FitnessWeights,
    test_share: float = 0.8,
) -> float:
    """
    Blend train/test scores with the provided share (default 80% test, 20% train).
    """
    test_score = metric_score(test_metrics, weights)
    train_score = metric_score(train_metrics, weights)
    return test_share * test_score + (1.0 - test_share) * train_score
