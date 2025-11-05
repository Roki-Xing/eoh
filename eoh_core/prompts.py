"""
Prompt management utilities for LLM-driven strategy generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Mapping, Optional


class PromptStyle(str, Enum):
    NORMAL = "normal"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class PromptTemplate:
    name: str
    system: str
    user: str


DEFAULT_PROMPTS: Dict[PromptStyle, PromptTemplate] = {
    PromptStyle.NORMAL: PromptTemplate(
        name="normal",
        system=(
            "You are an expert quantitative strategist.\n"
            "Output ONLY a Python code block that defines a Backtesting.py Strategy named `Strat`.\n"
            "Rules:\n"
            "- Use only numpy, pandas, Strategy API.\n"
            "- SMA, RSI, and crossover are AVAILABLE in the environment; you can call them directly.\n"
            "- Register indicators via self.I(...).\n"
            "- Provide class parameters as class variables (e.g., n1=10).\n"
            "- Trade in next() using crossover(...) or simple comparisons.\n"
            "- No external imports / I/O / plotting / print.\n"
            "- Keep code short and runnable."
        ),
        user=(
            "Design a simple rule-based trading strategy for {symbol} daily OHLCV data.\n"
            "Constraints:\n"
            "- Define `class Strat(Strategy):` with parameters and init/next methods.\n"
            "- You may use SMA, RSI, and crossover (available) via self.I(...)\n"
            "- Commission is handled externally; you don't need to apply it.\n"
            "Return ONLY code wrapped in a Python code block like:\n"
            "```python\n"
            "# your code here\n"
            "```"
        ),
    ),
    PromptStyle.CONSERVATIVE: PromptTemplate(
        name="conservative",
        system=(
            "You are an expert quantitative strategist tasked with producing LOW RISK long-only trading ideas.\n"
            "Follow the platform constraints and output a single Python code block defining `Strat`.\n"
            "Rules:\n"
            "- Focus on reducing drawdowns and trade frequency.\n"
            "- Use only numpy, pandas, Strategy API.\n"
            "- SMA, RSI, and crossover are available.\n"
            "- Register indicators via self.I(...).\n"
            "- Avoid overly aggressive leverage or rapid flipping of positions.\n"
            "- No external imports / I/O / plotting / print."
        ),
        user=(
            "Build a conservative daily trading strategy for {symbol} that prioritises stability and low drawdown.\n"
            "Use class `Strat(Strategy)` and rely on SMA/RSI/crossover helpers.\n"
            "Favour filters that avoid whipsaw trades, and require multiple confirmations before entering.\n"
            "Return ONLY code in a Python fenced block."
        ),
    ),
    PromptStyle.AGGRESSIVE: PromptTemplate(
        name="aggressive",
        system=(
            "You are an expert quantitative strategist working on HIGH BETA, return-seeking ideas.\n"
            "Output a Python code block defining `Strat` that obeys the platform constraints.\n"
            "Rules:\n"
            "- Use only numpy, pandas, Strategy API.\n"
            "- SMA, RSI, and crossover are available.\n"
            "- Favour momentum and breakout style logic to capture strong trends.\n"
            "- Register indicators via self.I(...).\n"
            "- Keep the code executable without additional imports."
        ),
        user=(
            "Design an aggressive daily trading strategy for {symbol} that actively seeks high returns.\n"
            "Use indicators like fast/slow SMAs, RSI momentum filters, or other combinations to enter trends early.\n"
            "Express the logic in class `Strat(Strategy)` and return ONLY a Python code block."
        ),
    ),
}


class PromptLibrary:
    """
    Manage prompt templates with optional overrides from the filesystem.
    """

    def __init__(self, base_dir: Optional[Path] = None, defaults: Optional[Mapping[PromptStyle, PromptTemplate]] = None):
        self.base_dir = Path(base_dir) if base_dir else None
        self.defaults = dict(defaults or DEFAULT_PROMPTS)

    def _load_override(self, style: PromptStyle) -> Optional[PromptTemplate]:
        if not self.base_dir:
            return None
        system_path = self.base_dir / f"{style.value}_system.txt"
        user_path = self.base_dir / f"{style.value}_user.txt"
        if system_path.exists() and user_path.exists():
            system = system_path.read_text(encoding="utf-8").strip()
            user = user_path.read_text(encoding="utf-8").strip()
            return PromptTemplate(name=style.value, system=system, user=user)
        return None

    def get(self, style: str | PromptStyle) -> PromptTemplate:
        style_enum = PromptStyle(style) if not isinstance(style, PromptStyle) else style
        override = self._load_override(style_enum)
        if override:
            return override
        if style_enum not in self.defaults:
            available = ", ".join(s.value for s in PromptStyle)
            raise KeyError(f"Prompt style '{style_enum.value}' not found. Available: {available}")
        return self.defaults[style_enum]

    def available_styles(self) -> list[str]:
        return sorted({s.value for s in PromptStyle})
