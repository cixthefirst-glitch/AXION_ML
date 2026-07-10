from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class BacktestResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    win_rate: float
    profit_factor: float
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    trades: int


def calculate_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    return float(np.min(drawdowns))


def calculate_sortino(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    downside_std = np.std(downside) if downside.size else 0.0
    if downside_std == 0.0:
        return float("inf") if np.mean(returns) > 0 else 0.0
    return float(np.mean(returns) / downside_std * np.sqrt(252))


def build_backtest_metrics(returns: List[float], labels: List[int], predictions: List[int]) -> BacktestResult:
    returns_array = np.array(returns, dtype=float)
    equity = np.cumprod(1 + returns_array)

    profit = returns_array[returns_array > 0].sum()
    loss = -returns_array[returns_array < 0].sum()
    profit_factor = float(profit / loss) if loss != 0 else float("inf")
    total_return = float(equity[-1] - 1.0)
    sharpe = float(np.mean(returns_array) / (np.std(returns_array) + 1e-9) * np.sqrt(252))
    sortino = calculate_sortino(returns_array)
    max_dd = calculate_drawdown(equity)
    win_rate = float(np.mean(np.array(returns_array) > 0))

    return BacktestResult(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, average="macro", zero_division=0)),
        recall=float(recall_score(labels, predictions, average="macro", zero_division=0)),
        f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        trades=len(returns),
    )


def _normalize_label(value: object) -> str:
    if isinstance(value, str):
        return value
    label_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    return label_map.get(int(value), str(value))


def backtest_model(model, features: pd.DataFrame, labels: pd.Series, returns: pd.Series) -> BacktestResult:
    predictions = model.predict(features)
    normalized_labels = [_normalize_label(value) for value in labels.tolist()]
    normalized_predictions = [_normalize_label(value) for value in predictions]
    return build_backtest_metrics(returns.tolist(), normalized_labels, normalized_predictions)
