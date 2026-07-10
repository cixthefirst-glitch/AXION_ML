from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.crypto_signals.api import ExchangeClient


def load_data(
    path: Optional[str] = None,
    api_symbol: Optional[str] = None,
    api_interval: str = "1h",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_base_url: Optional[str] = None,
    n_rows: int = 400,
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file, exchange API, or synthetic sample dataset."""
    if path:
        frame = pd.read_csv(path, parse_dates=["Date"])
        frame = frame.sort_values("Date").reset_index(drop=True)
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        return frame

    if api_symbol:
        return load_data_from_api(
            api_symbol,
            api_interval,
            n_rows,
            api_key=api_key,
            api_secret=api_secret,
            api_base_url=api_base_url,
        )

    return make_synthetic_crypto_data(n_rows=n_rows)


def load_data_from_api(
    symbol: str,
    interval: str = "1h",
    limit: int = 400,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> pd.DataFrame:
    client = ExchangeClient(api_key=api_key, api_secret=api_secret, base_url=api_base_url)
    return client.get_klines(symbol, interval=interval, limit=limit)


def load_history_for_symbols(
    symbols: List[str],
    interval: str = "15m",
    limit: int = 500,
    max_workers: int = 8,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    client = ExchangeClient(api_key=api_key, api_secret=api_secret, base_url=api_base_url)
    results: Dict[str, pd.DataFrame] = {}

    def _fetch(symbol: str) -> Optional[pd.DataFrame]:
        try:
            df = client.get_klines(symbol, interval=interval, limit=limit)
            if df.shape[0] < 100:
                return None
            return df
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            frame = future.result()
            if frame is not None:
                results[symbol] = frame
    return results


def make_synthetic_crypto_data(n_rows: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.linspace(0.0002, 0.0015, n_rows)
    noise = rng.normal(0, 0.012, size=n_rows)
    returns = drift + noise
    prices = np.exp(np.cumsum(returns))

    open_prices = prices * (1 + rng.normal(0, 0.003, size=n_rows))
    close_prices = prices * (1 + rng.normal(0, 0.002, size=n_rows))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(rng.normal(0, 0.004, size=n_rows)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(rng.normal(0, 0.004, size=n_rows)))
    volume = rng.integers(1_200, 7_000, size=n_rows)

    frame = pd.DataFrame(
        {
            "Date": pd.date_range(start="2020-01-01", periods=n_rows, freq="D"),
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volume,
        }
    )
    return frame
