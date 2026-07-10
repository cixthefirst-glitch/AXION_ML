from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def load_env_file(env_path: Optional[str] = None) -> None:
    if load_dotenv is None:
        return

    if env_path:
        load_dotenv(env_path)
    else:
        default_path = Path(".env")
        if default_path.exists():
            load_dotenv(default_path)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def get_env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    return float(value) if value is not None else default


@dataclass
class Config:
    api_key: Optional[str]
    api_secret: Optional[str]
    api_base_url: Optional[str]
    telegram_token: Optional[str]
    telegram_chat_id: Optional[str]
    timeframe: str
    scan_limit: int
    training_pairs: int
    lookback_bars: int
    signal_threshold: float
    top_signals: int
    min_volume: float
    min_atr: float
    risk_reward: float
    min_risk_reward: float
    stop_loss_multiplier: float
    take_profit_1_multiplier: float
    take_profit_2_multiplier: float
    model_path: str
    retrain: bool
    log_level: str
    model_type: str

    @classmethod
    def from_cli(cls) -> "Config":
        parser = argparse.ArgumentParser(description="Crypto signal generator configuration")
        parser.add_argument("--env-file", default=None, help="Optional .env path with API and Telegram keys")
        parser.add_argument("--api-key", default=None, help="Exchange API key")
        parser.add_argument("--api-secret", default=None, help="Exchange API secret")
        parser.add_argument("--api-base-url", default=None, help="Exchange API base URL")
        parser.add_argument("--telegram-token", default=None, help="Telegram bot token")
        parser.add_argument("--telegram-chat-id", default=None, help="Telegram chat ID")
        parser.add_argument("--timeframe", default="15m", help="Kline interval for scanning")
        parser.add_argument("--scan-limit", type=int, default=300, help="Number of symbols to scan")
        parser.add_argument("--training-pairs", type=int, default=80, help="Number of symbols to collect for training")
        parser.add_argument("--lookback-bars", type=int, default=500, help="History bars to collect per symbol")
        parser.add_argument("--signal-threshold", type=float, default=0.92, help="Probability threshold for signals")
        parser.add_argument("--top-signals", type=int, default=8, help="Maximum number of signals to send")
        parser.add_argument("--min-volume", type=float, default=100_000.0, help="Minimum average volume for a valid signal")
        parser.add_argument("--min-atr", type=float, default=0.0005, help="Minimum ATR for a valid signal")
        parser.add_argument("--risk-reward", type=float, default=2.0, help="Target risk:reward ratio for training labels and profit targets")
        parser.add_argument("--min-risk-reward", type=float, default=2.0, help="Minimum risk:reward ratio for actionable signals")
        parser.add_argument("--stop-loss-multiplier", type=float, default=1.2, help="ATR multiplier used to calculate stop loss")
        parser.add_argument("--take-profit-1-multiplier", type=float, default=2.0, help="ATR multiplier used to calculate first take profit")
        parser.add_argument("--take-profit-2-multiplier", type=float, default=4.0, help="ATR multiplier used to calculate second take profit")
        parser.add_argument("--model-path", default="models/crypto_signal_v1.joblib", help="Path to save or load trained model")
        parser.add_argument("--retrain", action="store_true", help="Force retraining of the model")
        parser.add_argument("--model-type", default="xgboost", choices=["xgboost", "random_forest"], help="Model family to use")
        parser.add_argument("--log-level", default="INFO", help="Logging level")
        args = parser.parse_args()

        load_env_file(args.env_file)

        api_key = args.api_key or get_env("API_KEY") or get_env("EXCHANGE_API_KEY")
        api_secret = args.api_secret or get_env("API_SECRET") or get_env("EXCHANGE_API_SECRET")
        api_base_url = args.api_base_url or get_env("API_BASE_URL")
        telegram_token = args.telegram_token or get_env("TELEGRAM_TOKEN") or get_env("TG_BOT_TOKEN")
        telegram_chat_id = args.telegram_chat_id or get_env("TELEGRAM_CHAT_ID")

        return cls(
            api_key=api_key,
            api_secret=api_secret,
            api_base_url=api_base_url,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            timeframe=args.timeframe,
            scan_limit=args.scan_limit,
            training_pairs=args.training_pairs,
            lookback_bars=args.lookback_bars,
            signal_threshold=args.signal_threshold,
            top_signals=args.top_signals,
            min_volume=args.min_volume,
            min_atr=args.min_atr,
            risk_reward=get_env_float("RISK_REWARD", args.risk_reward),
            min_risk_reward=get_env_float("MIN_RISK_REWARD", args.min_risk_reward),
            stop_loss_multiplier=get_env_float("STOP_LOSS_MULTIPLIER", args.stop_loss_multiplier),
            take_profit_1_multiplier=get_env_float("TAKE_PROFIT_1_MULTIPLIER", args.take_profit_1_multiplier),
            take_profit_2_multiplier=get_env_float("TAKE_PROFIT_2_MULTIPLIER", args.take_profit_2_multiplier),
            model_path=args.model_path,
            retrain=args.retrain,
            log_level=args.log_level,
            model_type=args.model_type,
        )
