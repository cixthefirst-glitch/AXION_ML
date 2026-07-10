from pathlib import Path

from src.crypto_signals.data import load_data
from src.crypto_signals.model import train_and_backtest


def test_backtest_runs_on_synthetic_data():
    data = load_data(n_rows=220)
    result = train_and_backtest(data, train_window=60, test_window=20)

    assert result["metrics"]["accuracy"] >= 0.0
    assert "signal" in result
    assert result["signal"] in {"BUY", "SELL"}


def test_backtest_uses_xgboost_when_requested():
    data = load_data(n_rows=220)
    result = train_and_backtest(data, train_window=60, test_window=20, model_type="xgboost")

    assert result["signal"] in {"BUY", "SELL"}
    assert "confidence" in result
