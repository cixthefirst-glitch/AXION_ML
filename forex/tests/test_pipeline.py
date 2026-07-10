from pathlib import Path

from src.crypto_signals.data import load_data
from src.crypto_signals.features import get_feature_columns, prepare_features
from src.crypto_signals.model import train_and_backtest


def test_backtest_runs_on_synthetic_data():
    data = load_data(n_rows=220)
    result = train_and_backtest(data, train_window=60, test_window=20)

    assert result["metrics"]["accuracy"] >= 0.0
    assert "signal" in result
    assert result["signal"] in {"BUY", "SELL", "HOLD"}


def test_backtest_uses_xgboost_when_requested():
    data = load_data(n_rows=220)
    result = train_and_backtest(data, train_window=60, test_window=20, model_type="xgboost")

    assert result["signal"] in {"BUY", "SELL", "HOLD"}
    assert "confidence" in result


def test_smc_features_are_added_when_enabled():
    data = load_data(n_rows=220)
    features = prepare_features(data, include_smc=True)

    assert "smc_bos" in features.columns
    assert "smc_choch" in features.columns
    assert "smc_order_block_distance" in features.columns
    assert "smc_swing_highs" in features.columns
    assert "smc_swing_lows" in features.columns


def test_get_feature_columns_includes_smc_when_enabled():
    feature_columns = get_feature_columns(include_smc=True)
    assert "smc_market_structure" in feature_columns
    assert "smc_trend_direction" in feature_columns
    base_columns = get_feature_columns()
    assert "smc_market_structure" not in base_columns
