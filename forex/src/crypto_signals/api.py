from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

DEFAULT_API_BASE_URL = "https://api.binance.com"


class ExchangeClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        auth_headers: Optional[Dict[str, str]] = None,
        timeout: int = 15,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url or DEFAULT_API_BASE_URL
        self.timeout = timeout
        self.session = requests.Session()
        if auth_headers:
            self.session.headers.update(auth_headers)
        elif self.api_key:
            self.session.headers["X-MBX-APIKEY"] = self.api_key

    def _request(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_top_symbols(self, limit: int = 300) -> List[str]:
        tickers = self._request("/api/v3/ticker/24hr", {})
        filtered = [
            ticker
            for ticker in tickers
            if ticker.get("symbol", "").endswith("USDT") and float(ticker.get("quoteVolume", 0)) > 10_000
        ]
        filtered.sort(key=lambda item: float(item.get("quoteVolume", 0)), reverse=True)
        return [item["symbol"] for item in filtered[:limit]]

    def get_exchange_symbols(self) -> List[str]:
        info = self._request("/api/v3/exchangeInfo", {})
        symbols = [
            symbol["symbol"]
            for symbol in info.get("symbols", [])
            if symbol.get("status") == "TRADING"
            and symbol.get("quoteAsset") == "USDT"
            and symbol.get("isSpotTradingAllowed", True)
        ]
        return symbols

    def get_klines(self, symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
        payload = self._request("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        frame = pd.DataFrame(
            payload,
            columns=[
                "open_time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        frame["Date"] = pd.to_datetime(frame["open_time"], unit="ms")
        frame = frame[["Date", "Open", "High", "Low", "Close", "Volume"]]
        frame = frame.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})
        return frame

    def safe_sleep(self, pause_seconds: float = 0.2) -> None:
        time.sleep(pause_seconds)
