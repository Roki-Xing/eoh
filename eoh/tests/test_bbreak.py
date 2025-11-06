import io
import textwrap

import pandas as pd
import pytest

from eoh_core.data import robust_read_csv, slice_by_date
from eoh_core.strategies import BBreakParams, RiskParams, backtest_bbreak


def _make_trending_prices() -> pd.DataFrame:
    """Construct a small deterministic price series for tests."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    closes = pd.Series(100 + idx.dayofyear % 5 * 2, index=idx).astype(float)
    frame = pd.DataFrame({"Close": closes, "Open": closes, "High": closes + 1, "Low": closes - 1})
    frame.index.name = "Date"
    return frame


def test_backtest_bbreak_basic_trade_cycles():
    df = _make_trending_prices()
    params = BBreakParams(n=3, k=0.5, h=1)
    risk = RiskParams(commission=0.0005, slippage_bps=5)

    result = backtest_bbreak(df, params, risk)

    # Expect at least one completed trade cycle on deterministic data
    assert not result.trades.empty
    assert set(result.trades["side"].unique()) <= {"BUY", "SELL"}
    assert result.metrics.trades == len(result.trades)
    assert result.metrics.total_return is not None
    assert 0.0 <= result.metrics.exposure <= 1.0


def test_robust_read_csv_handles_generic_headers(tmp_path):
    csv_content = textwrap.dedent(
        """\
        Timestamp,Open,High,Low,Adj Close,Volume
        2024-01-01,100,101,99,100.5,1200
        2024-01-02,101,102,100,101.5,1300
        2024-01-03,102,103,101,102.5,1400
        """
    )
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    df = robust_read_csv(str(csv_path))
    sliced = slice_by_date(df, "2024-01-01", "2024-01-02")

    assert list(df.columns) == ["Open", "High", "Low", "Adj Close", "Volume", "Close"]
    assert len(sliced) == 2
    assert sliced.index[0].isoformat().startswith("2024-01-01")
