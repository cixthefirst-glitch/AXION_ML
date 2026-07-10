import numpy as np
import pandas as pd

from src.crypto_signals.signals import build_signal, format_telegram_message


def test_build_signal_computes_risk_reward():
    features = pd.Series(
        {
            "close": 100.0,
            "atr_14": 2.0,
            "trend_bias": 1,
            "volume": 100000.0,
            "volatility_21": 0.001,
            "volume_spike": 0,
        }
    )

    signal = build_signal(
        "BTCUSDT",
        "15m",
        features,
        label=1,
        probabilities=np.array([0.05, 0.9, 0.05]),
        threshold=0.8,
    )

    assert signal is not None
    assert signal.risk_reward == 1.67
    assert "Risk : Reward: 1.67" in format_telegram_message([signal])
