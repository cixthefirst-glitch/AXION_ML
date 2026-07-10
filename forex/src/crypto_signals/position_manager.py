"""Module for managing open positions with advanced features like trailing stops and break-even."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from src.crypto_signals.logger import setup_logger
from src.crypto_signals.trading_engine import Trade

logger = setup_logger(__name__)


class PositionStatus(Enum):
    """Status of a position."""

    OPEN = "OPEN"
    BREAK_EVEN = "BREAK_EVEN"
    PARTIAL_TP = "PARTIAL_TP"
    TRAILING_STOP = "TRAILING_STOP"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


@dataclass
class PositionManagerConfig:
    """Configuration for position management."""

    enable_break_even: bool = True
    break_even_offset_percent: float = 0.2  # Move SL to entry + 0.2%
    enable_trailing_stop: bool = True
    trailing_stop_percent: float = 2.0  # Activate at 2% profit
    trailing_stop_distance_percent: float = 1.0  # Trail by 1%
    enable_partial_tp: bool = True
    partial_tp_1_percent: float = 50.0  # Close 50% at TP1
    partial_tp_2_percent: float = 30.0  # Close 30% at TP2
    check_interval_seconds: int = 60
    timeout_minutes: int = 1440  # 24 hours


class PositionMonitor:
    """Monitor and manage open positions."""

    def __init__(self, trading_engine, config: PositionManagerConfig = None):
        """Initialize position monitor.
        
        Args:
            trading_engine: MEXCTradingEngine instance
            config: PositionManagerConfig
        """
        self.engine = trading_engine
        self.config = config or PositionManagerConfig()
        self.logger = setup_logger(__name__)
        self.position_status: Dict[str, PositionStatus] = {}
        self.tp_levels_closed: Dict[str, List[bool]] = {}
        self.trailing_stop_activated: Dict[str, bool] = {}
        self.last_check: Dict[str, datetime] = {}

    def check_break_even(self, trade: Trade) -> bool:
        """Check and activate break-even stop loss.
        
        Args:
            trade: Trade to check
            
        Returns:
            True if break-even was activated
        """
        if not self.config.enable_break_even:
            return False

        if trade.break_even_activated:
            return False

        # Calculate break-even level based on current profit
        if trade.direction == "LONG":
            current_profit_percent = (
                (trade.current_price - trade.entry_price) / trade.entry_price
            ) * 100
        else:
            current_profit_percent = (
                (trade.entry_price - trade.current_price) / trade.entry_price
            ) * 100

        # Activate at configured profit threshold
        if current_profit_percent >= 1.0:  # At least 1% profit
            new_stop_loss = (
                trade.entry_price
                * (1 + self.config.break_even_offset_percent / 100)
                if trade.direction == "LONG"
                else trade.entry_price
                * (1 - self.config.break_even_offset_percent / 100)
            )

            trade.stop_loss = new_stop_loss
            trade.break_even_activated = True
            trade.notes.append(
                f"Break-even activated at {current_profit_percent:.2f}% profit"
            )
            logger.info(f"Break-even activated for {trade.pair}")

            return True

        return False

    def check_trailing_stop(self, trade: Trade) -> bool:
        """Check and activate trailing stop.
        
        Args:
            trade: Trade to check
            
        Returns:
            True if trailing stop should be updated
        """
        if not self.config.enable_trailing_stop:
            return False

        # Calculate current profit
        if trade.direction == "LONG":
            current_profit_percent = (
                (trade.current_price - trade.entry_price) / trade.entry_price
            ) * 100
        else:
            current_profit_percent = (
                (trade.entry_price - trade.current_price) / trade.entry_price
            ) * 100

        # Check if we should activate trailing stop
        if current_profit_percent >= self.config.trailing_stop_percent:
            if not trade.trailing_stop_percent:
                trade.trailing_stop_percent = self.config.trailing_stop_distance_percent
                trade.notes.append(
                    f"Trailing stop activated at {current_profit_percent:.2f}% profit"
                )
                logger.info(f"Trailing stop activated for {trade.pair}")
                return True

            # Update trailing stop level
            trail_distance = (
                trade.current_price
                * (self.config.trailing_stop_distance_percent / 100)
            )
            new_stop_loss = (
                trade.current_price - trail_distance
                if trade.direction == "LONG"
                else trade.current_price + trail_distance
            )

            # Only move stop loss upward (for LONG) or downward (for SHORT)
            if trade.direction == "LONG" and new_stop_loss > trade.stop_loss:
                trade.stop_loss = new_stop_loss
                trade.notes.append(f"Trailing stop updated to {new_stop_loss}")
                return True
            elif trade.direction == "SHORT" and new_stop_loss < trade.stop_loss:
                trade.stop_loss = new_stop_loss
                trade.notes.append(f"Trailing stop updated to {new_stop_loss}")
                return True

        return False

    def check_partial_take_profit(self, trade: Trade) -> List[Dict]:
        """Check if partial take profits should be closed.
        
        Args:
            trade: Trade to check
            
        Returns:
            List of close orders to execute
        """
        if not self.config.enable_partial_tp:
            return []

        if trade.trade_id not in self.tp_levels_closed:
            self.tp_levels_closed[trade.trade_id] = [False, False]

        close_orders = []

        # Check TP1
        if not self.tp_levels_closed[trade.trade_id][0]:
            if (trade.direction == "LONG" and trade.current_price >= trade.take_profit_1) or (
                trade.direction == "SHORT" and trade.current_price <= trade.take_profit_1
            ):
                quantity = trade.quantity * (self.config.partial_tp_1_percent / 100)
                close_orders.append(
                    {
                        "trade_id": trade.trade_id,
                        "pair": trade.pair,
                        "quantity": quantity,
                        "tp_level": 1,
                    }
                )
                self.tp_levels_closed[trade.trade_id][0] = True
                trade.closed_quantity += quantity
                trade.notes.append(f"Partial TP1 ({self.config.partial_tp_1_percent}%) closed")
                logger.info(f"Partial TP1 closed for {trade.pair}")

        # Check TP2
        if trade.take_profit_2 and not self.tp_levels_closed[trade.trade_id][1]:
            if (trade.direction == "LONG" and trade.current_price >= trade.take_profit_2) or (
                trade.direction == "SHORT" and trade.current_price <= trade.take_profit_2
            ):
                quantity = trade.quantity * (self.config.partial_tp_2_percent / 100)
                close_orders.append(
                    {
                        "trade_id": trade.trade_id,
                        "pair": trade.pair,
                        "quantity": quantity,
                        "tp_level": 2,
                    }
                )
                self.tp_levels_closed[trade.trade_id][1] = True
                trade.closed_quantity += quantity
                trade.notes.append(f"Partial TP2 ({self.config.partial_tp_2_percent}%) closed")
                logger.info(f"Partial TP2 closed for {trade.pair}")

        return close_orders

    def check_position_timeout(self, trade: Trade) -> bool:
        """Check if a position has exceeded the timeout duration.
        
        Args:
            trade: Trade to check
            
        Returns:
            True if position should be closed due to timeout
        """
        elapsed = datetime.utcnow() - trade.entry_time
        timeout = timedelta(minutes=self.config.timeout_minutes)

        if elapsed > timeout:
            trade.notes.append(f"Position timeout after {self.config.timeout_minutes} minutes")
            logger.warning(f"Position timeout for {trade.pair}")
            return True

        return False

    def update_all_positions(self, trades: Dict[str, Trade]) -> Dict[str, list]:
        """Update all open positions.
        
        Args:
            trades: Dictionary of trade_id -> Trade
            
        Returns:
            Dictionary with updates to execute (close_orders, break_even_updates, etc.)
        """
        updates = {
            "partial_tp_closes": [],
            "break_even_updates": [],
            "trailing_stop_updates": [],
            "timeout_closes": [],
        }

        for trade_id, trade in trades.items():
            # Update P&L based on current price (trading engine should provide this)

            # Check break-even
            if self.check_break_even(trade):
                updates["break_even_updates"].append(trade)

            # Check trailing stop
            if self.check_trailing_stop(trade):
                updates["trailing_stop_updates"].append(trade)

            # Check partial take profits
            tp_closes = self.check_partial_take_profit(trade)
            if tp_closes:
                updates["partial_tp_closes"].extend(tp_closes)

            # Check timeout
            if self.check_position_timeout(trade):
                updates["timeout_closes"].append(trade)

        return updates

    def close_trade(self, trade: Trade, reason: str = "Manual") -> bool:
        """Close a trade.
        
        Args:
            trade: Trade to close
            reason: Reason for closing
            
        Returns:
            True if close was initiated
        """
        trade.status = "CLOSED"
        trade.exit_time = datetime.utcnow()
        trade.notes.append(f"Closed: {reason}")

        logger.info(f"Trade closed: {trade.pair} - {reason}")
        return True

    def get_portfolio_summary(self, trades: Dict[str, Trade]) -> Dict:
        """Get summary of all open positions.
        
        Args:
            trades: Dictionary of trade_id -> Trade
            
        Returns:
            Summary dictionary with totals and stats
        """
        if not trades:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "total_pnl_percent": 0.0,
                "long_trades": 0,
                "short_trades": 0,
                "profitable_trades": 0,
            }

        total_pnl = 0.0
        total_exposure = 0.0
        long_count = 0
        short_count = 0
        profitable = 0

        for trade in trades.values():
            total_pnl += trade.unrealized_pnl
            total_exposure += trade.entry_price * trade.quantity

            if trade.direction == "LONG":
                long_count += 1
            else:
                short_count += 1

            if trade.unrealized_pnl > 0:
                profitable += 1

        avg_pnl_percent = (total_pnl / total_exposure * 100) if total_exposure > 0 else 0.0

        return {
            "total_trades": len(trades),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_percent": round(avg_pnl_percent, 2),
            "long_trades": long_count,
            "short_trades": short_count,
            "profitable_trades": profitable,
            "losing_trades": len(trades) - profitable,
        }
