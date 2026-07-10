from __future__ import annotations

import re
from typing import Dict

import requests


_TELEGRAM_MARKDOWN_ESCAPE = re.compile(r"([_\*\[\]\(\)~`>#+\-=|{}.!])")


def escape_markdown(text: str) -> str:
    return _TELEGRAM_MARKDOWN_ESCAPE.sub(r"\\\1", text)


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> bool:
    """Send a text message to a Telegram chat using a bot token."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload: Dict[str, str] = {
        "chat_id": chat_id,
        "text": escape_markdown(text),
        "parse_mode": "MarkdownV2",
    }
    response = requests.post(url, json=payload, timeout=15)
    response.raise_for_status()
    return response.json().get("ok", False)
