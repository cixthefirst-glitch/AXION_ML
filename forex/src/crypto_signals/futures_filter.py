"""Module for filtering and validating perpetual futures trading pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.crypto_signals.coin_info import CoinInfo
from src.crypto_signals.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class FuturesFilter:
    """Configuration for filtering futures pairs."""

    min_volume_usdt: float = 1_000_000  # Minimum 24h volume in USDT
    min_funding_rate: Optional[float] = None  # Minimum funding rate (can be negative)
    max_funding_rate: Optional[float] = 0.01  # Maximum funding rate
    require_funding_rate: bool = False  # Require funding rate to be present
    min_open_interest: float = 0  # Minimum open interest
    min_price_change: float = -100  # Minimum price change percentage
    max_price_change: float = 100  # Maximum price change percentage
    exclude_symbols: List[str] = None  # Symbols to exclude
    include_symbols: Optional[List[str]] = None  # If set, only include these

    def __post_init__(self):
        if self.exclude_symbols is None:
            self.exclude_symbols = []


class FuturesPairFilter:
    """Filter for perpetual futures trading pairs."""

    def __init__(self, config: FuturesFilter):
        """Initialize the filter.
        
        Args:
            config: FuturesFilter configuration object
        """
        self.config = config
        self.logger = setup_logger(__name__)

    def is_valid_futures_pair(self, coin_info: CoinInfo) -> bool:
        """Check if a trading pair is a valid futures pair for trading.
        
        Args:
            coin_info: CoinInfo object with pair information
            
        Returns:
            True if pair passes all validation criteria, False otherwise.
        """
        # Check if in exclusion list
        if coin_info.symbol in self.config.exclude_symbols:
            self.logger.debug(f"Pair {coin_info.symbol} is in exclusion list")
            return False

        # Check if in inclusion list (if set)
        if (
            self.config.include_symbols
            and coin_info.symbol not in self.config.include_symbols
        ):
            self.logger.debug(f"Pair {coin_info.symbol} not in inclusion list")
            return False

        # Check trading status
        if coin_info.trading_status != "ACTIVE":
            self.logger.debug(
                f"Pair {coin_info.symbol} not active (status: {coin_info.trading_status})"
            )
            return False

        # Check volume
        if coin_info.volume_24h < self.config.min_volume_usdt:
            self.logger.debug(
                f"Pair {coin_info.symbol} volume too low: {coin_info.volume_24h} < {self.config.min_volume_usdt}"
            )
            return False

        # Check funding rate (if required and available)
        if coin_info.funding_rate is not None:
            if self.config.min_funding_rate is not None:
                if coin_info.funding_rate < self.config.min_funding_rate:
                    self.logger.debug(
                        f"Pair {coin_info.symbol} funding rate too low: {coin_info.funding_rate} < {self.config.min_funding_rate}"
                    )
                    return False

            if self.config.max_funding_rate is not None:
                if coin_info.funding_rate > self.config.max_funding_rate:
                    self.logger.debug(
                        f"Pair {coin_info.symbol} funding rate too high: {coin_info.funding_rate} > {self.config.max_funding_rate}"
                    )
                    return False
        elif self.config.require_funding_rate:
            self.logger.debug(
                f"Pair {coin_info.symbol} funding rate not available but required"
            )
            return False

        # Check open interest (if available)
        if coin_info.open_interest is not None:
            if coin_info.open_interest < self.config.min_open_interest:
                self.logger.debug(
                    f"Pair {coin_info.symbol} open interest too low: {coin_info.open_interest} < {self.config.min_open_interest}"
                )
                return False

        # Check price change
        if coin_info.price_change_24h < self.config.min_price_change:
            self.logger.debug(
                f"Pair {coin_info.symbol} price change too low: {coin_info.price_change_24h} < {self.config.min_price_change}"
            )
            return False

        if coin_info.price_change_24h > self.config.max_price_change:
            self.logger.debug(
                f"Pair {coin_info.symbol} price change too high: {coin_info.price_change_24h} > {self.config.max_price_change}"
            )
            return False

        return True

    def filter_pairs(self, coin_infos: List[CoinInfo]) -> List[CoinInfo]:
        """Filter a list of coin infos to valid futures pairs.
        
        Args:
            coin_infos: List of CoinInfo objects
            
        Returns:
            Filtered list of valid CoinInfo objects
        """
        valid_pairs = [
            coin_info
            for coin_info in coin_infos
            if self.is_valid_futures_pair(coin_info)
        ]

        self.logger.info(
            f"Filtered {len(coin_infos)} pairs to {len(valid_pairs)} valid futures pairs"
        )
        return valid_pairs

    def filter_symbols(self, symbols: List[str], coin_manager) -> List[str]:
        """Filter a list of symbols by fetching their info and validating.
        
        Args:
            symbols: List of trading symbols
            coin_manager: CoinInfoManager instance to fetch coin info
            
        Returns:
            List of valid trading symbols
        """
        valid_symbols = []

        for symbol in symbols:
            coin_info = coin_manager.get_coin_info(symbol)
            if coin_info and self.is_valid_futures_pair(coin_info):
                valid_symbols.append(symbol)

        self.logger.info(
            f"Filtered {len(symbols)} symbols to {len(valid_symbols)} valid pairs"
        )
        return valid_symbols
