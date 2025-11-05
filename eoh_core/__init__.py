"""
EOH core helpers extracted from the monolithic scripts.
"""

from . import utils
from .data import PriceDataLoader, robust_read_csv, slice_by_date, synthetic_price_series
from .evaluation import FitnessWeights, blended_score, metric_score
from .llm import LLMClient, LocalHFClient, extract_code_blocks
from .prompts import PromptLibrary, PromptStyle, PromptTemplate
from .strategies import (
    BBreakParams,
    RiskParams,
    BacktestMetrics,
    BacktestResult,
    backtest_bbreak,
    compute_bbands,
    enforce_constraints,
)

__all__ = [
    "utils",
    "PriceDataLoader",
    "robust_read_csv",
    "slice_by_date",
    "synthetic_price_series",
    "FitnessWeights",
    "blended_score",
    "metric_score",
    "LLMClient",
    "LocalHFClient",
    "extract_code_blocks",
    "PromptLibrary",
    "PromptStyle",
    "PromptTemplate",
    "BBreakParams",
    "RiskParams",
    "BacktestMetrics",
    "BacktestResult",
    "backtest_bbreak",
    "compute_bbands",
    "enforce_constraints",
]
