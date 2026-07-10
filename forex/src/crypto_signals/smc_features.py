from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SMC_FEATURE_COLUMNS = [
    "smc_trend_direction",
    "smc_market_structure",
    "smc_bos",
    "smc_choch",
    "smc_order_block_distance",
    "smc_order_block_type",
    "smc_fvg_presence",
    "smc_liquidity_sweep",
    "smc_equal_high",
    "smc_equal_low",
    "smc_premium_zone",
    "smc_discount_zone",
    "smc_swing_highs",
    "smc_swing_lows",
]

DEFAULT_SMC_FEATURES = SMC_FEATURE_COLUMNS.copy()


@dataclass
class SMCFeatureExtractor:
    enabled_features: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.enabled_features is None:
            self.enabled_features = DEFAULT_SMC_FEATURES.copy()
        self.enabled = [feature for feature in self.enabled_features if feature in SMC_FEATURE_COLUMNS]

    def extract(self, frame: pd.DataFrame) -> pd.DataFrame:
        features: Dict[str, pd.Series] = {}

        if "smc_trend_direction" in self.enabled:
            features["smc_trend_direction"] = self.detect_trend_direction(frame)

        if "smc_market_structure" in self.enabled:
            features["smc_market_structure"] = self.detect_market_structure(frame)

        if "smc_bos" in self.enabled:
            features["smc_bos"] = self.detect_break_of_structure(frame)

        if "smc_choch" in self.enabled:
            features["smc_choch"] = self.detect_change_of_character(frame)

        if "smc_order_block_distance" in self.enabled or "smc_order_block_type" in self.enabled:
            order_block = self.detect_order_blocks(frame)
            if "smc_order_block_distance" in self.enabled:
                features["smc_order_block_distance"] = order_block["distance"]
            if "smc_order_block_type" in self.enabled:
                features["smc_order_block_type"] = order_block["type"]

        if "smc_fvg_presence" in self.enabled:
            features["smc_fvg_presence"] = self.detect_fair_value_gap(frame)

        if "smc_liquidity_sweep" in self.enabled:
            features["smc_liquidity_sweep"] = self.detect_liquidity_sweep(frame)

        if "smc_equal_high" in self.enabled or "smc_equal_low" in self.enabled:
            equals = self.detect_equal_highs_lows(frame)
            if "smc_equal_high" in self.enabled:
                features["smc_equal_high"] = equals["equal_high"]
            if "smc_equal_low" in self.enabled:
                features["smc_equal_low"] = equals["equal_low"]

        if "smc_premium_zone" in self.enabled:
            features["smc_premium_zone"] = self.detect_premium_zone(frame)

        if "smc_discount_zone" in self.enabled:
            features["smc_discount_zone"] = self.detect_discount_zone(frame)

        if "smc_swing_highs" in self.enabled or "smc_swing_lows" in self.enabled:
            swings = self.detect_swing_extrema(frame)
            if "smc_swing_highs" in self.enabled:
                features["smc_swing_highs"] = swings["swing_highs"]
            if "smc_swing_lows" in self.enabled:
                features["smc_swing_lows"] = swings["swing_lows"]

        extracted = pd.DataFrame(features, index=frame.index)
        return extracted.fillna(0)

    def detect_trend_direction(self, frame: pd.DataFrame) -> pd.Series:
        if "ema_50" not in frame or "ema_200" not in frame:
            ema_50 = frame["Close"].ewm(span=50, adjust=False).mean()
            ema_200 = frame["Close"].ewm(span=200, adjust=False).mean()
        else:
            ema_50 = frame["ema_50"]
            ema_200 = frame["ema_200"]
        direction = np.sign(ema_50 - ema_200)
        return direction.fillna(0).astype(int)

    def detect_market_structure(self, frame: pd.DataFrame) -> pd.Series:
        high_pivot = frame["High"].rolling(window=20, min_periods=1).max().shift(1)
        low_pivot = frame["Low"].rolling(window=20, min_periods=1).min().shift(1)
        bull = frame["Close"] > high_pivot
        bear = frame["Close"] < low_pivot
        structure = np.where(bull, 1, np.where(bear, -1, 0))
        return pd.Series(structure, index=frame.index, dtype=int)

    def detect_break_of_structure(self, frame: pd.DataFrame) -> pd.Series:
        high_band = frame["High"].rolling(window=20, min_periods=1).max().shift(1)
        low_band = frame["Low"].rolling(window=20, min_periods=1).min().shift(1)
        bos = np.where(frame["Close"] > high_band, 1, np.where(frame["Close"] < low_band, -1, 0))
        return pd.Series(bos, index=frame.index, dtype=int)

    def detect_change_of_character(self, frame: pd.DataFrame) -> pd.Series:
        trend = self.detect_trend_direction(frame)
        change = trend.diff().fillna(0)
        choch = np.where(change > 0, 1, np.where(change < 0, -1, 0))
        return pd.Series(choch, index=frame.index, dtype=int)

    def detect_order_blocks(self, frame: pd.DataFrame) -> Dict[str, pd.Series]:
        is_bull_block = (
            (frame["Close"].shift(1) > frame["Open"].shift(1))
            & (frame["Close"].shift(2) < frame["Open"].shift(2))
        )
        is_bear_block = (
            (frame["Close"].shift(1) < frame["Open"].shift(1))
            & (frame["Close"].shift(2) > frame["Open"].shift(2))
        )
        bull_level = frame["Low"].shift(1).where(is_bull_block)
        bear_level = frame["High"].shift(1).where(is_bear_block)
        bull_level = bull_level.ffill().fillna(method="bfill")
        bear_level = bear_level.ffill().fillna(method="bfill")
        distance = np.minimum(
            np.abs(frame["Close"] - bull_level),
            np.abs(frame["Close"] - bear_level),
        )
        atr = frame["atr_14"] if "atr_14" in frame else self._compute_atr(frame)
        distance_normalized = distance / (atr + 1e-9)
        block_type = np.where(is_bull_block, 1, np.where(is_bear_block, -1, 0))
        return {
            "distance": pd.Series(distance_normalized.fillna(0), index=frame.index),
            "type": pd.Series(block_type, index=frame.index, dtype=int),
        }

    def detect_fair_value_gap(self, frame: pd.DataFrame) -> pd.Series:
        gap_bull = (frame["Low"] > frame["High"].shift(1)) & (frame["Open"].shift(1) > frame["Close"].shift(2))
        gap_bear = (frame["High"] < frame["Low"].shift(1)) & (frame["Open"].shift(1) < frame["Close"].shift(2))
        gap = gap_bull | gap_bear
        return gap.astype(int).fillna(0)

    def detect_liquidity_sweep(self, frame: pd.DataFrame) -> pd.Series:
        lower_break = frame["Low"] < frame["Low"].rolling(window=10, min_periods=1).min().shift(1)
        upper_break = frame["High"] > frame["High"].rolling(window=10, min_periods=1).max().shift(1)
        sweep = (lower_break | upper_break) & (frame["Close"] != frame["Open"])
        return sweep.astype(int).fillna(0)

    def detect_equal_highs_lows(self, frame: pd.DataFrame) -> Dict[str, pd.Series]:
        high_diff = (frame["High"] - frame["High"].shift(1)).abs() / frame["High"].shift(1).replace(0, np.nan)
        low_diff = (frame["Low"] - frame["Low"].shift(1)).abs() / frame["Low"].shift(1).replace(0, np.nan)
        return {
            "equal_high": (high_diff < 0.002).astype(int).fillna(0),
            "equal_low": (low_diff < 0.002).astype(int).fillna(0),
        }

    def detect_premium_zone(self, frame: pd.DataFrame) -> pd.Series:
        premium = (frame["Close"] > frame["vwap"] * 1.02) & (frame["Close"] > frame["ema_50"])
        return premium.astype(int).fillna(0)

    def detect_discount_zone(self, frame: pd.DataFrame) -> pd.Series:
        discount = (frame["Close"] < frame["vwap"] * 0.98) & (frame["Close"] < frame["ema_50"])
        return discount.astype(int).fillna(0)

    def detect_swing_extrema(self, frame: pd.DataFrame) -> Dict[str, pd.Series]:
        swing_high = (
            (frame["High"] > frame["High"].shift(1))
            & (frame["High"] > frame["High"].shift(-1))
        )
        swing_low = (
            (frame["Low"] < frame["Low"].shift(1))
            & (frame["Low"] < frame["Low"].shift(-1))
        )
        swing_highs = swing_high.rolling(window=10, min_periods=1).sum().fillna(0).astype(int)
        swing_lows = swing_low.rolling(window=10, min_periods=1).sum().fillna(0).astype(int)
        return {
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
        }

    def _compute_atr(self, frame: pd.DataFrame, window: int = 14) -> pd.Series:
        prev_close = frame["Close"].shift(1)
        ranges = pd.concat(
            [frame["High"] - frame["Low"], (frame["High"] - prev_close).abs(), (frame["Low"] - prev_close).abs()],
            axis=1,
        )
        return ranges.max(axis=1).rolling(window=window, min_periods=1).mean().fillna(0)
