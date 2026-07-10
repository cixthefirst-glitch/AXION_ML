# Crypto Signal Lab

This starter project builds a machine learning pipeline for crypto price prediction and backtesting.

## What it does
- Loads crypto OHLCV data from a CSV file or generates a synthetic sample dataset when no file is supplied.
- Engineers technical indicators such as returns, moving averages, RSI, and volatility.
- Trains a tree-based classifier to predict whether the next bar will be up or down.
- Runs a walk-forward backtest that retrains the model over time.
- Prints a simple signal for the latest point in the series.

## Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the scanner once with synthetic fallback data:
   ```bash
   python -m src.crypto_signals.main
   ```
3. Fetch live OHLCV data from an exchange API:
   ```bash
   python -m src.crypto_signals.main --api-symbol BTCUSDT --api-interval 15m
   ```
4. Load exchange API credentials from CLI or `.env`:
   ```bash
   python -m src.crypto_signals.main --api-symbol BTCUSDT --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET
   ```
   or save credentials in a `.env` file and use:
   ```bash
   python -m src.crypto_signals.main --env-file .env
   ```
5. Send high-probability signals to Telegram:
   ```bash
   python -m src.crypto_signals.main --api-symbol BTCUSDT --api-interval 15m --telegram-token YOUR_BOT_TOKEN --telegram-chat-id YOUR_CHAT_ID
   ```
6. Retrain the production model:
   ```bash
   python -m src.crypto_signals.main --retrain
   ```

Supported environment variables:
- `API_KEY` or `EXCHANGE_API_KEY`
- `API_SECRET` or `EXCHANGE_API_SECRET`
- `TELEGRAM_TOKEN` or `TG_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `API_SYMBOL`
- `API_INTERVAL`
- `MODEL_PATH`

The CSV should contain at least these columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
