"""Telegram bot integration for sending signals and notifications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

from src.crypto_signals.logger import setup_logger, get_notification_logger

logger = setup_logger(__name__)
notification_logger = get_notification_logger()


@dataclass
class TelegramConfig:
    bot_token: str
    channel_id: str
    user_id: str


class TelegramNotifier:
    API_BASE_URL = "https://api.telegram.org"

    def __init__(self, config: TelegramConfig):
        self.config = config
        self.session = requests.Session()

    def _send_message(self, chat_id: str, text: str) -> bool:
        """Send a message using HTML parse mode (reliable, no escaping issues)."""
        try:
            url = f"{self.API_BASE_URL}/bot{self.config.bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            response = self.session.post(url, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data.get("ok", False):
                return True
            else:
                logger.error("Telegram error: %s", data.get("description", "Unknown error"))
                return False
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False

    def send_signal_to_channel(self, signal_text: str) -> bool:
        """Send trading signal to the public channel."""
        try:
            result = self._send_message(self.config.channel_id, signal_text)
            if result:
                notification_logger.info("Signal sent to channel: %s", self.config.channel_id)
            else:
                notification_logger.error("Failed to send signal to channel")
            return result
        except Exception as e:
            logger.error("Error sending signal to channel: %s", e)
            return False

    def send_notification_to_user(self, notification_type: str, message: str) -> bool:
        """Send a notification to the admin's private chat."""
        if not self.config.user_id:
            return False
        try:
            emoji_map = {
                "TRADE_OPENED":       "🟢",
                "TRADE_CLOSED":       "🔴",
                "TP_1_REACHED":       "🎯",
                "TP_2_REACHED":       "🎯",
                "TP_3_REACHED":       "🎯",
                "SL_HIT":             "❌",
                "MARKET_SIGNAL":      "🚨",
                "BACKTEST_COMPLETED": "✅",
                "ERROR":              "⚠️",
                "API_ERROR":          "🚨",
                "DAILY_SUMMARY":      "📊",
            }
            emoji = emoji_map.get(notification_type, "📬")
            full_message = (
                f"{emoji} <b>{notification_type}</b>\n\n"
                f"{message}\n\n"
                f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            result = self._send_message(self.config.user_id, full_message)
            if result:
                notification_logger.info("Notification sent to user: %s", notification_type)
            return result
        except Exception as e:
            logger.error("Error sending notification to user: %s", e)
            return False

    def send_error_notification(self, error_type: str, error_message: str) -> bool:
        message = f"Error Type: {error_type}\nMessage: {error_message}"
        notification_type = "API_ERROR" if "API" in error_type else "ERROR"
        return self.send_notification_to_user(notification_type, message)

    def send_daily_summary(self, total_trades: int, profitable_trades: int,
                           total_pnl: float, win_rate: float) -> bool:
        message = (
            f"Total Trades: {total_trades}\n"
            f"Profitable: {profitable_trades}\n"
            f"Total P&L: {total_pnl:+,.2f} USDT\n"
            f"Win Rate: {win_rate:.1f}%"
        )
        return self.send_notification_to_user("DAILY_SUMMARY", message)


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> bool:
    """Legacy helper for sending basic HTML Telegram messages."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                   "disable_web_page_preview": True}
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json().get("ok", False)
    except Exception as e:
        logger.error("Failed to send message: %s", e)
        return False