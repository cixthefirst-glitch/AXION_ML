"""Send pipeline reports to Telegram after training/backtesting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.crypto_signals.telegram import TelegramConfig, TelegramNotifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send pipeline reports to Telegram")
    parser.add_argument("--telegram-token", required=True, help="Telegram bot token")
    parser.add_argument("--telegram-channel-id", required=True, help="Telegram channel/chat id")
    parser.add_argument("--telegram-user-id", default=None, help="Telegram user id for DM notifications")
    parser.add_argument("--download-report", default=None, help="Path to download_summary.json")
    parser.add_argument("--backtest-report", default=None, help="Path to backtest report JSON")
    return parser.parse_args()


def format_report_message(
    download_report: Optional[str] = None,
    backtest_report: Optional[str] = None,
) -> str:
    """Format reports into a readable Telegram message."""
    lines = ["🤖 AXION_ML Pipeline Completed\n"]

    if download_report and Path(download_report).exists():
        try:
            with open(download_report, "r") as f:
                data = json.load(f)
            lines.append("📊 Data Download Summary:")
            lines.append(f"  • Symbols downloaded: {data.get('symbols_downloaded', 0)}")
            lines.append(f"  • Symbols failed: {data.get('symbols_failed', 0)}")
            lines.append(f"  • Total records: {data.get('total_records', 0)}")
            lines.append("")
        except Exception as e:
            lines.append(f"⚠️  Could not read download report: {e}\n")

    if backtest_report and Path(backtest_report).exists():
        try:
            with open(backtest_report, "r") as f:
                data = json.load(f)
            lines.append("📈 Backtest Results:")
            lines.append(f"  • Total trades: {data.get('total_trades', 0)}")
            lines.append(f"  • Win rate: {data.get('win_rate', 0):.1f}%")
            lines.append(f"  • Total return: {data.get('total_return', 0):.2f}%")
            lines.append(f"  • Max drawdown: {data.get('max_drawdown', 0):.2f}%")
            lines.append(f"  • Sharpe ratio: {data.get('sharpe_ratio', 0):.2f}")
            lines.append("")
        except Exception as e:
            lines.append(f"⚠️  Could not read backtest report: {e}\n")

    lines.append("✅ Model trained and ready for trading")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    try:
        tg_config = TelegramConfig(
            bot_token=args.telegram_token,
            channel_id=str(args.telegram_channel_id),
            user_id=str(args.telegram_user_id or ""),
        )
        notifier = TelegramNotifier(tg_config)
    except Exception as e:
        print(f"Failed to initialize Telegram notifier: {e}")
        return 1

    message = format_report_message(
        download_report=args.download_report,
        backtest_report=args.backtest_report,
    )

    try:
        ok = notifier.send_signal_to_channel(message)
        print(f"Telegram notification sent: {ok}")
        return 0 if ok else 1
    except Exception as e:
        print(f"Failed to send Telegram notification: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
