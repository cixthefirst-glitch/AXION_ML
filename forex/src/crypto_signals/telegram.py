"""Telegram bot integration for sending signals and notifications."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import requests

from src.crypto_signals.logger import setup_logger, get_notification_logger

logger = setup_logger(__name__)
notification_logger = get_notification_logger()


_TELEGRAM_MARKDOWN_ESCAPE = re.compile(r"([_\*\[\]\(\)~`>#+\-=|{}.!])")


def escape_markdown(text: str) -> str:
    """Escape special markdown characters for Telegram MarkdownV2.
    
    Args:
        text: Text to escape
        
    Returns:
        Escaped text safe for MarkdownV2
    """
    return _TELEGRAM_MARKDOWN_ESCAPE.sub(r"\\\1", text)


@dataclass
class TelegramConfig:
    """Telegram configuration."""

    bot_token: str
    channel_id: str
    user_id: str  # Your personal Telegram user ID for DM


class TelegramNotifier:
    """Handle Telegram notifications for signals and events."""

    API_BASE_URL = "https://api.telegram.org"

    def __init__(self, config: TelegramConfig):
        """Initialize Telegram notifier.
        
        Args:
            config: TelegramConfig with bot token and chat IDs
        """
        self.config = config
        self.session = requests.Session()

    def _send_message(self, chat_id: str, text: str, parse_mode: str = "MarkdownV2") -> bool:
        """Send a message to a Telegram chat.
        
        Args:
            chat_id: Telegram chat ID (can be negative for channels)
            text: Message text
            parse_mode: Parsing mode (MarkdownV2 or HTML)
            
        Returns:
            True if successful
        """
        try:
            url = f"{self.API_BASE_URL}/bot{self.config.bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            response = self.session.post(url, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get("ok", False):
                return True
            else:
                logger.error(f"Telegram error: {data.get('description', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {str(e)}")
            return False

    def send_signal_to_channel(self, signal_text: str) -> bool:
        """Send a trading signal to the public channel.
        
        Args:
            signal_text: Formatted signal message
            
        Returns:
            True if successful
        """
        try:
            escaped_text = escape_markdown(signal_text)
            result = self._send_message(self.config.channel_id, escaped_text)

            if result:
                notification_logger.info(f"Signal sent to channel: {self.config.channel_id}")
            else:
                notification_logger.error("Failed to send signal to channel")

            return result
        except Exception as e:
            logger.error(f"Error sending signal to channel: {str(e)}")
            return False

    def send_notification_to_user(self, notification_type: str, message: str) -> bool:
        """Send a notification to the user's private chat.
        
        Args:
            notification_type: Type of notification (TRADE_OPENED, TP_1_REACHED, etc.)
            message: Notification message
            
        Returns:
            True if successful
        """
        try:
            # Format with emoji and type
            emoji_map = {
                "TRADE_OPENED": "🟢",
                "TRADE_CLOSED": "🔴",
                "TP_1_REACHED": "🎯",
                "TP_2_REACHED": "🎯",
                "TP_3_REACHED": "🎯",
                "SL_HIT": "❌",
                "TRAILING_STOP": "📍",
                "BACKTEST_COMPLETED": "✅",
                "LIVE_TEST_COMPLETED": "✅",
                "ERROR": "⚠️",
                "API_ERROR": "🚨",
                "RESTART": "🔄",
                "DAILY_SUMMARY": "📊",
            }

            emoji = emoji_map.get(notification_type, "📬")
            full_message = f"{emoji} *{notification_type}*\n\n{message}\n\n⏰ {datetime.utcnow().strftime('%Y--%m--%d %H:%M:%S UTC')}"

            escaped_text = escape_markdown(full_message)
            result = self._send_message(self.config.user_id, escaped_text)

            if result:
                notification_logger.info(
                    f"Notification sent to user: {notification_type}"
                )
            else:
                notification_logger.error(
                    f"Failed to send {notification_type} notification"
                )

            return result
        except Exception as e:
            logger.error(f"Error sending notification to user: {str(e)}")
            return False

    def send_trade_opened_notification(
        self,
        coin: str,
        pair: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        direction: str,
        signal_confidence: float,
    ) -> bool:
        """Send trade opened notification.
        
        Args:
            coin: Coin symbol
            pair: Trading pair
            entry_price: Entry price
            stop_loss: Stop loss level
            take_profit_1: Take profit 1 level
            direction: LONG or SHORT
            signal_confidence: Signal confidence score
            
        Returns:
            True if successful
        """
        message = f"""Pair: {pair}
Direction: {direction}
Entry: {entry_price:,.8f}
Stop Loss: {stop_loss:,.8f}
Take Profit: {take_profit_1:,.8f}
Signal Confidence: {signal_confidence:.1f}%"""

        return self.send_notification_to_user("TRADE_OPENED", message)

    def send_trade_closed_notification(
        self,
        pair: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        duration_minutes: int,
    ) -> bool:
        """Send trade closed notification.
        
        Args:
            pair: Trading pair
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/Loss in USDT
            pnl_percent: Profit/Loss percentage
            duration_minutes: Trade duration in minutes
            
        Returns:
            True if successful
        """
        message = f"""Pair: {pair}
Entry: {entry_price:,.8f}
Exit: {exit_price:,.8f}
P&L: {pnl:+,.2f} USDT ({pnl_percent:+.2f}%)
Duration: {duration_minutes} minutes"""

        return self.send_notification_to_user("TRADE_CLOSED", message)

    def send_tp_reached_notification(
        self,
        pair: str,
        tp_level: int,
        price: float,
        quantity: float,
        pnl: float,
    ) -> bool:
        """Send take profit reached notification.
        
        Args:
            pair: Trading pair
            tp_level: TP level (1, 2, or 3)
            price: TP price
            quantity: Quantity closed
            pnl: Realized P&L
            
        Returns:
            True if successful
        """
        message = f"""Pair: {pair}
TP Level: {tp_level}
Price: {price:,.8f}
Quantity Closed: {quantity:,.8f}
Realized P&L: {pnl:+,.2f} USDT"""

        notification_type = f"TP_{tp_level}_REACHED"
        return self.send_notification_to_user(notification_type, message)

    def send_stop_loss_notification(
        self,
        pair: str,
        entry_price: float,
        sl_price: float,
        loss: float,
    ) -> bool:
        """Send stop loss hit notification.
        
        Args:
            pair: Trading pair
            entry_price: Entry price
            sl_price: Stop loss price
            loss: Loss amount in USDT
            
        Returns:
            True if successful
        """
        message = f"""Pair: {pair}
Entry: {entry_price:,.8f}
Stop Loss Hit: {sl_price:,.8f}
Loss: {loss:,.2f} USDT"""

        return self.send_notification_to_user("SL_HIT", message)

    def send_error_notification(self, error_type: str, error_message: str) -> bool:
        """Send error notification.
        
        Args:
            error_type: Type of error
            error_message: Error message
            
        Returns:
            True if successful
        """
        message = f"""Error Type: {error_type}
Message: {error_message}"""

        notification_type = "API_ERROR" if "API" in error_type else "ERROR"
        return self.send_notification_to_user(notification_type, message)

    def send_daily_summary(
        self,
        total_trades: int,
        profitable_trades: int,
        total_pnl: float,
        win_rate: float,
    ) -> bool:
        """Send daily trading summary.
        
        Args:
            total_trades: Total trades executed
            profitable_trades: Number of profitable trades
            total_pnl: Total P&L in USDT
            win_rate: Win rate percentage
            
        Returns:
            True if successful
        """
        message = f"""Total Trades: {total_trades}
Profitable: {profitable_trades}
Total P&L: {total_pnl:+,.2f} USDT
Win Rate: {win_rate:.1f}%"""

        return self.send_notification_to_user("DAILY_SUMMARY", message)


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> bool:
    """Legacy function for sending basic Telegram messages.
    
    Args:
        bot_token: Telegram bot token
        chat_id: Target chat ID
        text: Message text
        
    Returns:
        True if successful
    """
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        escaped_text = escape_markdown(text)
        payload = {
            "chat_id": chat_id,
            "text": escaped_text,
            "parse_mode": "MarkdownV2",
        }
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json().get("ok", False)
    except Exception as e:
        logger.error(f"Failed to send message: {str(e)}")
        return False

