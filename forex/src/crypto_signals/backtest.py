from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


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
    cagr: float
    avg_trade_return: float
    roc_auc: float
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


def calculate_cagr(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    total_return = np.prod(1 + returns) - 1
    periods = len(returns)
    if periods == 0:
        return 0.0
    return float((1 + total_return) ** (252.0 / periods) - 1)


def build_backtest_metrics(
    returns: List[float],
    labels: List[int],
    predictions: List[int],
    probabilities: Optional[np.ndarray] = None,
) -> BacktestResult:
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
    cagr = calculate_cagr(returns_array)
    avg_trade_return = float(np.mean(returns_array)) if returns_array.size else 0.0

    roc_auc = 0.0
    if probabilities is not None and probabilities.size > 0:
        try:
            unique_labels = sorted(set(labels))
            if len(unique_labels) > 1:
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                encoded_labels = [label_map[label] for label in labels]
                roc_auc = float(roc_auc_score(encoded_labels, probabilities, multi_class="ovo"))
        except Exception:
            roc_auc = 0.0

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
        cagr=cagr,
        avg_trade_return=avg_trade_return,
        roc_auc=roc_auc,
        trades=len(returns),
    )


def _normalize_label(value: object) -> str:
    if isinstance(value, str):
        return value
    label_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    return label_map.get(int(value), str(value))


def backtest_model(model, features: pd.DataFrame, labels: pd.Series, returns: pd.Series, probabilities: Optional[np.ndarray] = None) -> BacktestResult:
    predictions = model.predict(features)
    normalized_labels = [_normalize_label(value) for value in labels.tolist()]
    normalized_predictions = [_normalize_label(value) for value in predictions]
    return build_backtest_metrics(returns.tolist(), normalized_labels, normalized_predictions, probabilities=probabilities)


def rolling_window_backtest(
    model_factory: Callable[[], object],
    features: pd.DataFrame,
    labels: pd.Series,
    returns: pd.Series,
    train_window: int,
    test_window: int,
    step: int = 1,
) -> List[BacktestResult]:
    results: List[BacktestResult] = []
    max_start = len(features) - train_window - test_window
    for start in range(0, max_start + 1, step):
        train_slice = slice(start, start + train_window)
        test_slice = slice(start + train_window, start + train_window + test_window)
        model = model_factory()
        model.fit(features.iloc[train_slice], labels.iloc[train_slice])
        predictions = model.predict(features.iloc[test_slice])
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features.iloc[test_slice])
        results.append(
            build_backtest_metrics(
                returns.iloc[test_slice].tolist(),
                labels.iloc[test_slice].tolist(),
                predictions,
                probabilities=probabilities,
            )
        )
    return results


def compare_feature_sets(
    model_factory: Callable[[], object],
    named_features: Dict[str, pd.DataFrame],
    labels: pd.Series,
    returns: pd.Series,
    train_window: int,
    test_window: int,
    step: int = 1,
) -> Dict[str, List[BacktestResult]]:
    comparisons: Dict[str, List[BacktestResult]] = {}
    for name, feature_set in named_features.items():
        comparisons[name] = rolling_window_backtest(
            model_factory=model_factory,
            features=feature_set,
            labels=labels,
            returns=returns,
            train_window=train_window,
            test_window=test_window,
            step=step,
        )
    return comparisons
