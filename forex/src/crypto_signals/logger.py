"""Logging configuration with rotating file handlers for different log categories."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def setup_logger(
    name: str = "crypto_signal",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_type: str = "general",
) -> logging.Logger:
    """Setup logger with both console and rotating file handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional custom log file path
        log_type: Type of log (general, signals, trades, errors, api, positions, notifications)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    if log_file is None:
        log_file = f"{log_type}.log"

    log_path = LOGS_DIR / log_file
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10,  # Keep 10 backup files
    )
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def get_signal_logger() -> logging.Logger:
    """Get logger for signal generation."""
    return setup_logger("signals", logging.DEBUG, log_type="signals")


def get_trade_logger() -> logging.Logger:
    """Get logger for trade execution."""
    return setup_logger("trades", logging.DEBUG, log_type="trades")


def get_error_logger() -> logging.Logger:
    """Get logger for errors."""
    return setup_logger("errors", logging.DEBUG, log_type="errors")


def get_api_logger() -> logging.Logger:
    """Get logger for API requests."""
    return setup_logger("api", logging.DEBUG, log_type="api")


def get_position_logger() -> logging.Logger:
    """Get logger for position updates."""
    return setup_logger("positions", logging.DEBUG, log_type="positions")


def get_notification_logger() -> logging.Logger:
    """Get logger for notifications."""
    return setup_logger("notifications", logging.DEBUG, log_type="notifications")

