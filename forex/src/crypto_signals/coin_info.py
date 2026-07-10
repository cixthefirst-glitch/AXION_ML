"""Module for gathering coin market information from MEXC and CoinGecko."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import requests

from src.crypto_signals.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CoinInfo:
    """Standardized coin information from multiple sources."""

    symbol: str
    name: str
    pair: str
    # MEXC data
    current_price: float
    volume_24h: float
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    price_change_24h: float = 0.0
    trading_status: str = "ACTIVE"
    # CoinGecko data
    market_cap: Optional[float] = None
    market_cap_rank: Optional[int] = None
    coingecko_volume_24h: Optional[float] = None
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    max_supply: Optional[float] = None
    developer_activity: Optional[float] = None
    community_score: Optional[float] = None
    market_sentiment: Optional[str] = None
    trending: bool = False
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


class MEXCClient:
    """Client for MEXC Futures API interactions."""

    BASE_URL = "https://contract.mexc.com"
    SPOT_BASE_URL = "https://api.mexc.com"

    def __init__(self, api_key: str, api_secret: str, timeout: int = 10):
        """Initialize MEXC client."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"X-MEXC-APIKEY": api_key})

    def get_futures_symbols(self) -> List[str]:
        """Fetch all USDT perpetual futures trading pairs from MEXC.
        
        Returns:
            List of active USDT perpetual futures symbols.
        """
        try:
            url = f"{self.BASE_URL}/open/api/v2/productList"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                logger.error("MEXC API error: %s", data.get("message", "Unknown error"))
                return []

            symbols = []
            for product in data.get("data", []):
                # Only get USDT perpetual futures
                if (
                    product.get("symbol", "").endswith("_USDT")
                    and product.get("productType") == 1  # 1 = perpetual futures
                    and product.get("state") == 0  # 0 = trading enabled
                ):
                    symbols.append(product.get("symbol"))

            return symbols
        except Exception as e:
            logger.error("Failed to fetch MEXC symbols: %s", str(e))
            return []

    def get_futures_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker information for a MEXC futures symbol.
        
        Args:
            symbol: MEXC futures symbol (e.g., "BTC_USDT")
            
        Returns:
            Dictionary with ticker data or None if failed.
        """
        try:
            url = f"{self.BASE_URL}/open/api/v2/ticker"
            response = self.session.get(url, params={"symbol": symbol}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                return None

            ticker = data.get("data", {})
            return {
                "symbol": symbol,
                "current_price": float(ticker.get("lastPrice", 0)),
                "volume_24h": float(ticker.get("volume24h", 0)),
                "price_change_24h": float(ticker.get("change24h", 0)),
                "funding_rate": float(ticker.get("fundingRate", 0)) if ticker.get("fundingRate") else None,
                "open_interest": float(ticker.get("openInterest", 0)) if ticker.get("openInterest") else None,
            }
        except Exception as e:
            logger.error("Failed to fetch MEXC ticker for %s: %s", symbol, str(e))
            return None

    def get_account_balance(self) -> Optional[float]:
        """Get account balance in USDT."""
        try:
            url = f"{self.BASE_URL}/open/api/v2/account/assets"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                return None

            # Find USDT balance
            for asset in data.get("data", {}).get("userAssets", []):
                if asset.get("currency") == "USDT":
                    return float(asset.get("accountEquity", 0))

            return None
        except Exception as e:
            logger.error("Failed to fetch account balance: %s", str(e))
            return None


class CoinGeckoClient:
    """Client for CoinGecko API interactions."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        """Initialize CoinGecko client."""
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    def get_coin_info(self, coin_id: str) -> Optional[Dict]:
        """Fetch coin information from CoinGecko.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin")
            
        Returns:
            Dictionary with coin data or None if failed.
        """
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}"
            params = {
                "localization": False,
                "tickers": False,
                "market_data": True,
                "community_data": True,
                "developer_data": True,
            }
            if self.api_key:
                params["x_cg_pro_api_key"] = self.api_key

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            return {
                "name": data.get("name"),
                "symbol": data.get("symbol", "").upper(),
                "market_cap": data.get("market_data", {}).get("market_cap", {}).get("usd"),
                "market_cap_rank": data.get("market_cap_rank"),
                "volume_24h": data.get("market_data", {}).get("total_volume", {}).get("usd"),
                "circulating_supply": data.get("market_data", {}).get("circulating_supply"),
                "total_supply": data.get("market_data", {}).get("total_supply"),
                "max_supply": data.get("market_data", {}).get("max_supply"),
                "developer_activity": data.get("developer_data", {}).get("commit_count_4_weeks"),
                "community_score": data.get("community_data", {}).get("community_score"),
                "sentiment_votes_up_percentage": data.get("sentiment_votes_up_percentage"),
                "trending": False,  # Will check separately
            }
        except Exception as e:
            logger.error("Failed to fetch CoinGecko info for %s: %s", coin_id, str(e))
            return None

    def get_trending_coins(self) -> List[str]:
        """Get list of trending coin IDs on CoinGecko.
        
        Returns:
            List of trending coin IDs.
        """
        try:
            url = f"{self.BASE_URL}/search/trending"
            params = {}
            if self.api_key:
                params["x_cg_pro_api_key"] = self.api_key

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            trending = []
            for item in data.get("coins", [])[:10]:  # Get top 10
                coin_id = item.get("item", {}).get("id")
                if coin_id:
                    trending.append(coin_id)

            return trending
        except Exception as e:
            logger.error("Failed to fetch trending coins: %s", str(e))
            return []


class CoinInfoManager:
    """Manager for combining information from multiple sources."""

    def __init__(
        self,
        mexc_key: str,
        mexc_secret: str,
        coingecko_key: Optional[str] = None,
        cache_duration_seconds: int = 300,
    ):
        """Initialize coin info manager.
        
        Args:
            mexc_key: MEXC API key
            mexc_secret: MEXC API secret
            coingecko_key: Optional CoinGecko API key (for premium features)
            cache_duration_seconds: How long to cache results
        """
        self.mexc = MEXCClient(mexc_key, mexc_secret)
        self.coingecko = CoinGeckoClient(coingecko_key)
        self.cache_duration = cache_duration_seconds
        self._coin_cache: Dict[str, CoinInfo] = {}
        self._cache_timestamp: Dict[str, datetime] = {}

    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        """Convert trading symbol to CoinGecko coin ID.
        
        Args:
            symbol: Trading symbol (e.g., "BTC_USDT" -> "bitcoin")
            
        Returns:
            CoinGecko coin ID.
        """
        # Simple mapping for common coins
        mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "BNB": "binancecoin",
            "SOL": "solana",
            "ADA": "cardano",
            "XRP": "ripple",
            "DOGE": "dogecoin",
            "MATIC": "matic-network",
            "LINK": "chainlink",
            "LTC": "litecoin",
            "BCH": "bitcoin-cash",
            "AVAX": "avalanche-2",
            "DOT": "polkadot",
            "UNI": "uniswap",
            "ATOM": "cosmos",
        }

        coin_symbol = symbol.split("_")[0]  # Extract base symbol
        return mapping.get(coin_symbol, coin_symbol.lower())

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cache is still valid for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if cache is valid, False otherwise.
        """
        if symbol not in self._cache_timestamp:
            return False

        elapsed = (datetime.utcnow() - self._cache_timestamp[symbol]).total_seconds()
        return elapsed < self.cache_duration

    def get_coin_info(self, symbol: str, use_cache: bool = True) -> Optional[CoinInfo]:
        """Get combined coin information from all sources.
        
        Args:
            symbol: MEXC trading symbol (e.g., "BTC_USDT")
            use_cache: Whether to use cached data if available
            
        Returns:
            CoinInfo object or None if failed.
        """
        # Check cache
        if use_cache and self._is_cache_valid(symbol):
            return self._coin_cache.get(symbol)

        # Fetch from MEXC
        mexc_ticker = self.mexc.get_futures_ticker(symbol)
        if not mexc_ticker:
            logger.warning("Could not fetch MEXC data for %s", symbol)
            return None

        # Fetch from CoinGecko (best effort, don't fail if unavailable)
        coingecko_id = self._symbol_to_coingecko_id(symbol)
        coingecko_data = self.coingecko.get_coin_info(coingecko_id) or {}

        # Merge data
        coin_info = CoinInfo(
            symbol=symbol,
            name=coingecko_data.get("name", symbol.split("_")[0]),
            pair=symbol,
            current_price=mexc_ticker.get("current_price", 0),
            volume_24h=mexc_ticker.get("volume_24h", 0),
            funding_rate=mexc_ticker.get("funding_rate"),
            open_interest=mexc_ticker.get("open_interest"),
            price_change_24h=mexc_ticker.get("price_change_24h", 0),
            market_cap=coingecko_data.get("market_cap"),
            market_cap_rank=coingecko_data.get("market_cap_rank"),
            coingecko_volume_24h=coingecko_data.get("volume_24h"),
            circulating_supply=coingecko_data.get("circulating_supply"),
            total_supply=coingecko_data.get("total_supply"),
            max_supply=coingecko_data.get("max_supply"),
            developer_activity=coingecko_data.get("developer_activity"),
            community_score=coingecko_data.get("community_score"),
            market_sentiment=coingecko_data.get("sentiment_votes_up_percentage"),
        )

        # Cache the result
        self._coin_cache[symbol] = coin_info
        self._cache_timestamp[symbol] = datetime.utcnow()

        return coin_info

    def get_trending_coins_info(self) -> List[CoinInfo]:
        """Get information for trending coins.
        
        Returns:
            List of CoinInfo objects for trending coins.
        """
        trending_ids = self.coingecko.get_trending_coins()
        results = []

        # For each trending coin, try to find MEXC equivalent
        for coin_id in trending_ids:
            coingecko_data = self.coingecko.get_coin_info(coin_id) or {}
            symbol = coingecko_data.get("symbol", "").upper()

            if symbol:
                # Try to fetch MEXC data
                mexc_symbol = f"{symbol}_USDT"
                mexc_ticker = self.mexc.get_futures_ticker(mexc_symbol)

                if mexc_ticker:
                    coin_info = CoinInfo(
                        symbol=mexc_symbol,
                        name=coingecko_data.get("name", symbol),
                        pair=mexc_symbol,
                        current_price=mexc_ticker.get("current_price", 0),
                        volume_24h=mexc_ticker.get("volume_24h", 0),
                        funding_rate=mexc_ticker.get("funding_rate"),
                        open_interest=mexc_ticker.get("open_interest"),
                        price_change_24h=mexc_ticker.get("price_change_24h", 0),
                        market_cap=coingecko_data.get("market_cap"),
                        market_cap_rank=coingecko_data.get("market_cap_rank"),
                        coingecko_volume_24h=coingecko_data.get("volume_24h"),
                        trending=True,
                    )
                    results.append(coin_info)

        return results

    def get_all_futures_coins(self) -> List[CoinInfo]:
        """Get information for all available MEXC futures pairs.
        
        Returns:
            List of CoinInfo objects for all trading pairs.
        """
        symbols = self.mexc.get_futures_symbols()
        results = []

        for symbol in symbols:
            coin_info = self.get_coin_info(symbol, use_cache=True)
            if coin_info:
                results.append(coin_info)

        return results
