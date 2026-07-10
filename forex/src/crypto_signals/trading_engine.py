"""Module for executing trades on MEXC Futures with advanced order management."""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import requests

from src.crypto_signals.logger import setup_logger

logger = setup_logger(__name__)


class OrderType(Enum):
    """Order types supported."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(Enum):
    """Order side (direction)."""

    LONG = "LONG"
    SHORT = "SHORT"


class PositionMode(Enum):
    """Position mode for futures."""

    ISOLATED = "ISOLATED"
    CROSS = "CROSS"


@dataclass
class Trade:
    """Represents an open trade."""

    trade_id: str
    pair: str
    direction: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    entry_order_id: str
    quantity: float
    leverage: int
    position_mode: str
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    status: str = "OPEN"
    trailing_stop_percent: Optional[float] = None
    break_even_activated: bool = False
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_order_id: Optional[str] = None
    closed_quantity: float = 0.0
    closed_pnl: float = 0.0
    closed_pnl_percent: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class TradeConfig:
    """Configuration for trade execution."""

    leverage: int = 5
    position_mode: PositionMode = PositionMode.ISOLATED
    risk_percent: float = 1.0  # Risk % of account per trade
    max_open_trades: int = 5
    use_market_orders: bool = True
    timeout_minutes: int = 1440  # 24 hours


class MEXCTradingEngine:
    """Trading engine for MEXC Futures."""

    BASE_URL = "https://contract.mexc.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        config: TradeConfig = None,
        timeout: int = 10,
    ):
        """Initialize trading engine.
        
        Args:
            api_key: MEXC API key
            api_secret: MEXC API secret
            config: TradeConfig object with trading parameters
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.config = config or TradeConfig()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"X-MEXC-APIKEY": api_key})
        self.open_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []

    def _sign_request(self, method: str, path: str, params: Dict) -> Dict[str, str]:
        """Generate request signature for authenticated endpoints.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path
            params: Request parameters
            
        Returns:
            Headers with signature
        """
        timestamp = str(int(time.time() * 1000))
        query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self.api_secret.encode(),
            f"{query_string}{timestamp}".encode(),
            hashlib.sha256,
        ).hexdigest()

        headers = self.session.headers.copy()
        headers.update(
            {
                "X-MEXC-Timestamp": timestamp,
                "X-MEXC-Signature": signature,
            }
        )
        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        authenticated: bool = False,
    ) -> Optional[Dict]:
        """Make an API request.
        
        Args:
            method: HTTP method
            path: API path
            params: Request parameters
            authenticated: Whether to sign the request
            
        Returns:
            Response data or None if failed
        """
        try:
            url = f"{self.BASE_URL}{path}"
            params = params or {}

            if authenticated:
                headers = self._sign_request(method, path, params)
            else:
                headers = self.session.headers

            if method.upper() == "GET":
                response = self.session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
            elif method.upper() == "POST":
                response = self.session.post(
                    url, data=params, headers=headers, timeout=self.timeout
                )
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None

            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                logger.error(f"API error: {data.get('message', 'Unknown error')}")
                return None

            return data.get("data")

        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def get_account_info(self) -> Optional[Dict]:
        """Get account information including balance.
        
        Returns:
            Account info or None if failed
        """
        return self._request("GET", "/open/api/v2/account/assets", {}, authenticated=True)

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC_USDT")
            
        Returns:
            Position data or None if no position
        """
        positions = self._request(
            "GET",
            "/open/api/v2/positionList",
            {"symbol": symbol},
            authenticated=True,
        )

        if positions and isinstance(positions, list) and len(positions) > 0:
            return positions[0]
        return None

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int = None,
    ) -> Optional[str]:
        """Place a market order.
        
        Args:
            symbol: Trading symbol
            side: Order side (LONG or SHORT)
            quantity: Order quantity
            leverage: Leverage to use
            
        Returns:
            Order ID or None if failed
        """
        leverage = leverage or self.config.leverage

        params = {
            "symbol": symbol,
            "side": side,
            "positionType": 1,  # 1 = perpetual
            "type": OrderType.MARKET.value,
            "quantity": quantity,
            "leverage": leverage,
        }

        result = self._request(
            "POST", "/open/api/v2/orders/open", params, authenticated=True
        )

        if result and result.get("orderId"):
            logger.info(
                f"Market order placed: {symbol} {side} {quantity} @ leverage {leverage}"
            )
            return result.get("orderId")

        return None

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        leverage: int = None,
    ) -> Optional[str]:
        """Place a limit order.
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price
            leverage: Leverage to use
            
        Returns:
            Order ID or None if failed
        """
        leverage = leverage or self.config.leverage

        params = {
            "symbol": symbol,
            "side": side,
            "positionType": 1,
            "type": OrderType.LIMIT.value,
            "quantity": quantity,
            "price": price,
            "leverage": leverage,
        }

        result = self._request(
            "POST", "/open/api/v2/orders/open", params, authenticated=True
        )

        if result and result.get("orderId"):
            logger.info(
                f"Limit order placed: {symbol} {side} {quantity} @ {price} (leverage {leverage})"
            )
            return result.get("orderId")

        return None

    def set_stop_loss(self, symbol: str, price: float) -> bool:
        """Set stop loss for a position.
        
        Args:
            symbol: Trading symbol
            price: Stop loss price
            
        Returns:
            True if successful
        """
        params = {
            "symbol": symbol,
            "stopLossPrice": price,
        }

        result = self._request(
            "POST", "/open/api/v2/orders/stopLoss", params, authenticated=True
        )

        if result:
            logger.info(f"Stop loss set for {symbol} @ {price}")
            return True

        return False

    def set_take_profit(self, symbol: str, price: float) -> bool:
        """Set take profit for a position.
        
        Args:
            symbol: Trading symbol
            price: Take profit price
            
        Returns:
            True if successful
        """
        params = {
            "symbol": symbol,
            "takeProfitPrice": price,
        }

        result = self._request(
            "POST", "/open/api/v2/orders/takeProfit", params, authenticated=True
        )

        if result:
            logger.info(f"Take profit set for {symbol} @ {price}")
            return True

        return False

    def close_position(self, symbol: str, quantity: float) -> Optional[str]:
        """Close a position by placing a market order in opposite direction.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to close
            
        Returns:
            Order ID or None if failed
        """
        position = self.get_position(symbol)
        if not position:
            logger.warning(f"No position found for {symbol}")
            return None

        current_side = position.get("side", "LONG")
        close_side = "SHORT" if current_side == "LONG" else "LONG"

        order_id = self.place_market_order(
            symbol, close_side, quantity, leverage=self.config.leverage
        )

        if order_id:
            logger.info(f"Close order placed for {symbol}: {quantity} {close_side}")

        return order_id

    def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Get status of an order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to check
            
        Returns:
            Order status data or None
        """
        params = {"symbol": symbol, "orderId": order_id}

        result = self._request(
            "GET", "/open/api/v2/orders/get", params, authenticated=True
        )

        return result

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        params = {"symbol": symbol, "orderId": order_id}

        result = self._request(
            "POST", "/open/api/v2/orders/cancel", params, authenticated=True
        )

        if result:
            logger.info(f"Order {order_id} cancelled for {symbol}")
            return True

        return False

    def can_open_trade(self) -> bool:
        """Check if we can open a new trade based on max open trades limit.
        
        Returns:
            True if we can open a new trade
        """
        return len(self.open_trades) < self.config.max_open_trades

    def execute_signal_trade(
        self,
        signal,
        account_balance: float,
    ) -> Optional[str]:
        """Execute a trade based on a signal.
        
        Args:
            signal: Signal object with trading parameters
            account_balance: Current account balance in USDT
            
        Returns:
            Trade ID or None if execution failed
        """
        if not self.can_open_trade():
            logger.warning(
                f"Cannot open trade: max open trades ({self.config.max_open_trades}) reached"
            )
            return None

        # Calculate position size based on risk
        risk_amount = account_balance * (self.config.risk_percent / 100)
        risk_distance = abs(signal.entry - signal.stop_loss)

        if risk_distance == 0:
            logger.error("Invalid stop loss - cannot calculate position size")
            return None

        quantity = risk_amount / risk_distance

        # Place entry order
        order_id = self.place_market_order(
            signal.symbol,
            signal.direction,
            quantity,
            self.config.leverage,
        )

        if not order_id:
            logger.error(f"Failed to place entry order for {signal.symbol}")
            return None

        # Set stop loss
        self.set_stop_loss(signal.symbol, signal.stop_loss)

        # Set take profit
        self.set_take_profit(signal.symbol, signal.take_profit_1)

        # Create trade record
        trade_id = f"{signal.symbol}_{int(time.time())}"
        trade = Trade(
            trade_id=trade_id,
            pair=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry,
            entry_time=datetime.utcnow(),
            entry_order_id=order_id,
            quantity=quantity,
            leverage=self.config.leverage,
            position_mode=self.config.position_mode.value,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            take_profit_3=getattr(signal, "take_profit_3", None),
        )

        self.open_trades[trade_id] = trade
        logger.info(f"Trade opened: {trade_id}")

        return trade_id

    def update_trade_pnl(self, trade_id: str, current_price: float) -> None:
        """Update P&L for a trade.
        
        Args:
            trade_id: Trade ID
            current_price: Current market price
        """
        if trade_id not in self.open_trades:
            return

        trade = self.open_trades[trade_id]
        trade.current_price = current_price

        if trade.direction == "LONG":
            pnl = (current_price - trade.entry_price) * trade.quantity
            pnl_percent = ((current_price - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl = (trade.entry_price - current_price) * trade.quantity
            pnl_percent = (
                (trade.entry_price - current_price) / trade.entry_price
            ) * 100

        trade.unrealized_pnl = pnl
        trade.unrealized_pnl_percent = pnl_percent
