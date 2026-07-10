from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Signal:
    symbol: str
    timeframe: str
    direction: str
    probability: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    trend_bias: int
    volume: float
    volatility: float
    grade: str
    risk_reward: float
    rank: int
    timestamp: datetime
    note: str


def build_signal(
    symbol: str,
    timeframe: str,
    features: pd.Series,
    label: int,
    probabilities: np.ndarray,
    threshold: float,
    stop_loss_multiplier: float = 1.2,
    take_profit_1_multiplier: float = 2.0,
    take_profit_2_multiplier: float = 4.0,
) -> Optional[Signal]:
    class_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    direction = class_map.get(label, "HOLD")
    probability = float(np.max(probabilities))
    if direction == "HOLD" or probability < threshold:
        return None

    close = float(features["close"])
    atr = float(features["atr_14"])
    trend_bias = int(features["trend_bias"])
    volume = float(features["volume"])
    volatility = float(features["volatility_21"])
    volume_spike = bool(features["volume_spike"])

    if direction == "BUY":
        entry = close
        stop_loss = close - atr * stop_loss_multiplier
        take_profit_1 = close + atr * take_profit_1_multiplier
        take_profit_2 = close + atr * take_profit_2_multiplier
        note = "Strong uptrend" if trend_bias > 0 else "Momentum setup"
    else:
        entry = close
        stop_loss = close + atr * stop_loss_multiplier
        take_profit_1 = close - atr * take_profit_1_multiplier
        take_profit_2 = close - atr * take_profit_2_multiplier
        note = "Strong downtrend" if trend_bias < 0 else "Momentum setup"

    risk_reward = abs((take_profit_1 - entry) / (entry - stop_loss)) if stop_loss != entry else 0.0
    grade = "A" if probability >= 0.95 and risk_reward >= 2.0 else "B" if probability >= 0.92 else "C"

    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        probability=probability,
        entry=round(entry, 6),
        stop_loss=round(stop_loss, 6),
        take_profit_1=round(take_profit_1, 6),
        take_profit_2=round(take_profit_2, 6),
        trend_bias=trend_bias,
        volume=round(volume, 2),
        volatility=round(volatility, 6),
        grade=grade,
        risk_reward=round(risk_reward, 2),
        rank=0,
        timestamp=datetime.utcnow(),
        note=note + (" | Volume spike" if volume_spike else ""),
    )


def filter_signed_signals(signals: List[Signal], top_n: int) -> List[Signal]:
    ranked = sorted(signals, key=lambda item: item.probability, reverse=True)
    for index, signal in enumerate(ranked, start=1):
        signal.rank = index
    return ranked[:top_n]


def format_telegram_message(signals: List[Signal]) -> str:
    if not signals:
        return "No high-conviction setups detected at this time."

    lines = ["🚨 *Market Signal Summary*", ""]
    for signal in signals:
        lines.append(f"*{signal.symbol}* — {signal.direction} — {signal.probability:.0%}")
        lines.append(f"Timeframe: {signal.timeframe}")
        lines.append(f"Entry: {signal.entry}")
        lines.append(f"Stop Loss: {signal.stop_loss}")
        lines.append(f"Take Profit 1: {signal.take_profit_1}")
        lines.append(f"Take Profit 2: {signal.take_profit_2}")
        lines.append(f"Risk : Reward: {signal.risk_reward:.2f} | Grade: {signal.grade}")
        lines.append(f"Trend bias: {signal.trend_bias}, Volatility: {signal.volatility:.6f}")
        lines.append(f"Volume: {signal.volume}")
        lines.append(f"Note: {signal.note}")
        lines.append("---")
    return "\n".join(lines)
