from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Signal:
    """Trading signal with comprehensive information for execution and tracking."""

    symbol: str
    pair: str
    timeframe: str
    direction: str
    probability: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: Optional[float] = None
    risk_reward: float = 0.0
    confidence_score: float = 0.0
    trend_bias: int = 0
    volume: float = 0.0
    volatility: float = 0.0
    grade: str = ""
    rank: int = 0
    timestamp: datetime = None
    note: str = ""
    explanation: str = ""
    pattern_detected: str = ""
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    score_components: Optional[dict] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if not self.pair:
            self.pair = self.symbol
        if self.confidence_score == 0.0:
            self.confidence_score = self.probability * 100


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
    take_profit_3_multiplier: Optional[float] = 6.0,
) -> Optional[Signal]:
    class_map = {0: "HOLD", 1: "LONG", 2: "SHORT"}
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

    if direction == "LONG":
        entry = close
        stop_loss = close - atr * stop_loss_multiplier
        take_profit_1 = close + atr * take_profit_1_multiplier
        take_profit_2 = close + atr * take_profit_2_multiplier
        take_profit_3 = close + atr * take_profit_3_multiplier if take_profit_3_multiplier else None
        explanation = "Strong uptrend detected" if trend_bias > 0 else "Bullish momentum setup"
    else:
        entry = close
        stop_loss = close + atr * stop_loss_multiplier
        take_profit_1 = close - atr * take_profit_1_multiplier
        take_profit_2 = close - atr * take_profit_2_multiplier
        take_profit_3 = close - atr * take_profit_3_multiplier if take_profit_3_multiplier else None
        explanation = "Strong downtrend detected" if trend_bias < 0 else "Bearish momentum setup"

    risk_reward = (
        abs((take_profit_1 - entry) / (entry - stop_loss))
        if stop_loss != entry
        else 0.0
    )
    grade = (
        "A" if probability >= 0.85 and risk_reward >= 2.0
        else "B" if probability >= 0.75
        else "C"
    )

    note = f"{'Strong' if abs(trend_bias) > 2 else 'Moderate'} {'uptrend' if trend_bias > 0 else 'downtrend'}"
    if volume_spike:
        note += " | Volume spike detected"

    score_components = {
        "model_confidence": probability,
        "trend_strength": abs(trend_bias) / 3,
        "volatility": min(volatility / 0.01, 1.0),
        "volume_factor": 1.0 if volume_spike else 0.7,
    }

    return Signal(
        symbol=symbol,
        pair=symbol,
        timeframe=timeframe,
        direction=direction,
        probability=probability,
        entry=round(entry, 8),
        stop_loss=round(stop_loss, 8),
        take_profit_1=round(take_profit_1, 8),
        take_profit_2=round(take_profit_2, 8),
        take_profit_3=round(take_profit_3, 8) if take_profit_3 else None,
        risk_reward=round(risk_reward, 2),
        confidence_score=round(probability * 100, 1),
        trend_bias=trend_bias,
        volume=round(volume, 2),
        volatility=round(volatility, 8),
        grade=grade,
        rank=0,
        timestamp=datetime.utcnow(),
        note=note,
        explanation=explanation,
        score_components=score_components,
    )


def filter_signed_signals(signals: List[Signal], top_n: int) -> List[Signal]:
    ranked = sorted(
        signals, key=lambda item: item.probability * item.risk_reward, reverse=True
    )
    for index, signal in enumerate(ranked, start=1):
        signal.rank = index
    return ranked[:top_n]


def format_telegram_message(signals: List[Signal], include_emoji: bool = True) -> str:
    """Format signals for Telegram using HTML parse mode."""
    if not signals:
        return "✅ No high-conviction setups detected at this time."

    lines = [
        "🚨 <b>MARKET SIGNALS ALERT</b>",
        "",
        f"⏰ Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "⚠️ <i>Risk Warning: Trading involves significant risk. Please trade responsibly.</i>",
        "",
    ]

    for signal in signals:
        emoji = "📈" if signal.direction == "LONG" else "📉"
        side_emoji = "🟢" if signal.direction == "LONG" else "🔴"
        lines.append(f"{side_emoji} <b>{signal.symbol}</b> — {emoji} <b>{signal.direction}</b>")
        lines.append(f"Grade: <b>{signal.grade}</b> | Confidence: <b>{signal.confidence_score:.0f}%</b>")
        lines.append("")
        lines.append(f"📍 Entry:     <code>{signal.entry:,.8f}</code>")
        lines.append(f"🛑 Stop Loss: <code>{signal.stop_loss:,.8f}</code>")
        lines.append(f"🎯 TP1:       <code>{signal.take_profit_1:,.8f}</code>")
        lines.append(f"🎯 TP2:       <code>{signal.take_profit_2:,.8f}</code>")
        if signal.take_profit_3:
            lines.append(f"🎯 TP3:       <code>{signal.take_profit_3:,.8f}</code>")
        lines.append(f"📊 R:R Ratio: {signal.risk_reward:.2f}")
        lines.append(f"⏱ Timeframe: {signal.timeframe}")
        lines.append(f"💬 {signal.explanation}")
        lines.append("─────────────────────")
        lines.append("")

    return "\n".join(lines)