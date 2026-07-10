from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.crypto_signals.api import ExchangeClient
from src.crypto_signals.features import prepare_features
from src.crypto_signals.model import SignalModel
from src.crypto_signals.signals import Signal, build_signal, filter_signed_signals


@dataclass
class ScannerConfig:
    timeframe: str = "15m"
    scan_limit: int = 300
    lookback_bars: int = 500
    min_history: int = 120
    signal_threshold: float = 0.90
    top_signals: int = 8
    max_workers: int = 8
    min_volume: float = 100_000
    min_atr: float = 0.0005
    min_risk_reward: float = 2.0
    stop_loss_multiplier: float = 1.2
    take_profit_1_multiplier: float = 2.0
    take_profit_2_multiplier: float = 4.0
    rank_by: str = "probability"


class SignalScanner:
    def __init__(self, client: ExchangeClient, model: SignalModel, config: ScannerConfig, logger: Any):
        self.client = client
        self.model = model
        self.config = config
        self.logger = logger
        self.feature_columns = model.feature_columns

    def fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            self.logger.debug("Downloading %s", symbol)
            df = self.client.get_klines(symbol, self.config.timeframe, self.config.lookback_bars)
            if df.shape[0] < self.config.min_history:
                self.logger.debug("Skipping %s because history is too short (%s bars)", symbol, df.shape[0])
                return None
            return df
        except Exception as exc:
            self.logger.warning("Failed to fetch data for %s: %s", symbol, exc)
            return None

    def scan(self, symbols: List[str]) -> List[Signal]:
        self.logger.info("Scanning %s symbols with timeframe %s", len(symbols), self.config.timeframe)
        tasks = {}
        results: List[Signal] = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            for symbol in symbols[: self.config.scan_limit]:
                tasks[executor.submit(self.fetch_symbol_data, symbol)] = symbol

            for future in as_completed(tasks):
                symbol = tasks[future]
                df = future.result()
                if df is None:
                    continue

                features = prepare_features(df, include_smc=self.model.include_smc, smc_features=self.model.smc_features)
                latest = features.tail(1)
                if latest.empty:
                    continue

                current = latest.iloc[0]
                if current["volume"] < self.config.min_volume or current["atr_14"] < self.config.min_atr:
                    continue

                proba = self.model.predict_proba(latest[self.feature_columns])[0]
                label = self.model.predict(latest[self.feature_columns])[0]

                if label == "BUY" and current["close"] < current["ema_50"]:
                    continue
                if label == "SELL" and current["close"] > current["ema_50"]:
                    continue

                signal = build_signal(
                    symbol=symbol,
                    timeframe=self.config.timeframe,
                    features=current,
                    label=label,
                    probabilities=proba,
                    threshold=self.config.signal_threshold,
                    stop_loss_multiplier=self.config.stop_loss_multiplier,
                    take_profit_1_multiplier=self.config.take_profit_1_multiplier,
                    take_profit_2_multiplier=self.config.take_profit_2_multiplier,
                )
                if signal is not None and signal.risk_reward >= self.config.min_risk_reward:
                    results.append(signal)

        signals = filter_signed_signals(results, self.config.top_signals)
        self.logger.info("Found %s actionable signals", len(signals))
        return signals
