"""Base strategy class with lifecycle hooks."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_percent: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None
    order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.timestamp.timestamp()}"


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    def update_price(self, price: float):
        """Update current price and trailing stop."""
        self.current_price = price

        # Update trailing stop if set
        if self.trailing_stop_pct is not None:
            new_stop = price * (1 - self.trailing_stop_pct)
            if self.stop_loss is None or new_stop > self.stop_loss:
                self.stop_loss = new_stop


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.

    Lifecycle hooks:
    - on_start: Called once at strategy initialization
    - on_data: Called for each new data point
    - on_bar_close: Called at the end of each bar/candle
    - on_order_fill: Called when an order is filled
    - on_order_reject: Called when an order is rejected
    - on_position_close: Called when a position is closed
    - on_stop: Called once at strategy termination
    """

    def __init__(self, name: Optional[str] = None, **params):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            **params: Strategy parameters
        """
        self.name = name or self.__class__.__name__
        self.params = params

        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash = 0.0
        self.initial_capital = 0.0

        # Order tracking
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []

        # Performance tracking
        self.equity_curve: List[float] = []
        self.timestamps: List[pd.Timestamp] = []
        self.trades: List[Dict[str, Any]] = []

        # Internal state
        self._current_timestamp = None
        self._current_data = None
        self._is_running = False

        logger.info(f"Initialized strategy: {self.name}")

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    @property
    def positions_value(self) -> float:
        """Total value of open positions."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def leverage(self) -> float:
        """Current leverage ratio."""
        if self.portfolio_value == 0:
            return 0.0
        return self.positions_value / self.portfolio_value

    # Lifecycle hooks

    def on_start(self, initial_capital: float):
        """
        Called once at strategy start.

        Args:
            initial_capital: Starting capital
        """
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self._is_running = True
        logger.info(f"Strategy started with capital: ${initial_capital:,.2f}")

    @abstractmethod
    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """
        Called for each new data point.

        Args:
            timestamp: Current timestamp
            data: Market data up to current timestamp
        """
        pass

    def on_bar_close(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """
        Called at the end of each bar/candle.

        Args:
            timestamp: Bar close timestamp
            data: Market data
        """
        pass

    def on_order_fill(self, order: Order, fill_price: float, fill_timestamp: pd.Timestamp):
        """
        Called when an order is filled.

        Args:
            order: Filled order
            fill_price: Execution price
            fill_timestamp: Fill timestamp
        """
        logger.info(
            f"Order filled: {order.side.value} {order.quantity} {order.symbol} @ ${fill_price:.2f}"
        )

        # Update position
        if order.side == OrderSide.BUY:
            if order.symbol in self.positions:
                # Add to existing position
                pos = self.positions[order.symbol]
                total_quantity = pos.quantity + order.quantity
                pos.entry_price = (
                    (pos.entry_price * pos.quantity + fill_price * order.quantity) / total_quantity
                )
                pos.quantity = total_quantity
            else:
                # Create new position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=fill_price,
                    entry_timestamp=fill_timestamp,
                    current_price=fill_price,
                )
        else:  # SELL
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pos.quantity -= order.quantity

                # Close position if quantity is zero or negative
                if pos.quantity <= 0:
                    self.on_position_close(pos, fill_price, fill_timestamp)
                    del self.positions[order.symbol]

        # Record trade
        self.trades.append({
            'timestamp': fill_timestamp,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': fill_price,
            'order_type': order.order_type.value,
        })

    def on_order_reject(self, order: Order, reason: str):
        """
        Called when an order is rejected.

        Args:
            order: Rejected order
            reason: Rejection reason
        """
        logger.warning(f"Order rejected: {order.symbol} - {reason}")

    def on_position_close(self, position: Position, close_price: float, timestamp: pd.Timestamp):
        """
        Called when a position is closed.

        Args:
            position: Closed position
            close_price: Closing price
            timestamp: Close timestamp
        """
        pnl = (close_price - position.entry_price) * position.quantity
        pnl_pct = (close_price - position.entry_price) / position.entry_price

        logger.info(
            f"Position closed: {position.symbol} - P&L: ${pnl:.2f} ({pnl_pct*100:.2f}%)"
        )

    def on_stop(self):
        """Called once at strategy termination."""
        self._is_running = False
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        logger.info(f"Strategy stopped. Total return: {total_return*100:.2f}%")

    # Trading methods

    def buy(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        size: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Create a buy order.

        Args:
            symbol: Symbol to buy
            quantity: Number of shares/contracts
            size: Position size as fraction of portfolio (alternative to quantity)
            order_type: Type of order
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            Created order
        """
        if quantity is None and size is not None:
            # Calculate quantity from position size
            if self._current_data is not None:
                # Get current price - handle both single and multi-symbol data
                try:
                    # Check if multi-symbol (multi-level columns)
                    if hasattr(self._current_data.columns, 'levels') and len(self._current_data.columns.levels) > 1:
                        # Multi-symbol: columns are ('SYMBOL', 'close')
                        if symbol in self._current_data.columns.get_level_values(0):
                            current_price = self._current_data[symbol]['close'].iloc[-1]
                        else:
                            raise ValueError(f"Symbol {symbol} not found in data")
                    else:
                        # Single-symbol: columns are ['close', 'open', etc.]
                        current_price = self._current_data['close'].iloc[-1]

                    quantity = (self.portfolio_value * size) / current_price
                except (KeyError, IndexError, AttributeError) as e:
                    raise ValueError(f"Cannot calculate quantity: {e}")
            else:
                raise ValueError("Cannot calculate quantity without current price")

        if quantity is None:
            raise ValueError("Must specify either quantity or size")

        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            timestamp=self._current_timestamp,
            **kwargs
        )

        self.pending_orders.append(order)
        return order

    def sell(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        size: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Create a sell order.

        Args:
            symbol: Symbol to sell
            quantity: Number of shares/contracts (None = sell all)
            size: Position size as fraction of current position
            order_type: Type of order
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)

        Returns:
            Created order
        """
        if quantity is None and size is not None:
            # Calculate quantity from position size
            if symbol in self.positions:
                quantity = self.positions[symbol].quantity * size
            else:
                raise ValueError(f"No position in {symbol}")

        if quantity is None:
            # Sell entire position
            if symbol in self.positions:
                quantity = self.positions[symbol].quantity
            else:
                raise ValueError(f"No position in {symbol}")

        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            timestamp=self._current_timestamp,
            **kwargs
        )

        self.pending_orders.append(order)
        return order

    def close_position(self, symbol: str) -> Optional[Order]:
        """Close position in symbol."""
        if symbol in self.positions:
            return self.sell(symbol)
        return None

    def close_all_positions(self) -> List[Order]:
        """Close all open positions."""
        orders = []
        for symbol in list(self.positions.keys()):
            order = self.close_position(symbol)
            if order:
                orders.append(order)
        return orders

    def set_stop_loss(self, symbol: str, stop_price: float):
        """Set stop loss for position."""
        if symbol in self.positions:
            self.positions[symbol].stop_loss = stop_price

    def set_take_profit(self, symbol: str, take_profit_price: float):
        """Set take profit for position."""
        if symbol in self.positions:
            self.positions[symbol].take_profit = take_profit_price

    def set_trailing_stop(self, symbol: str, trailing_pct: float):
        """Set trailing stop for position."""
        if symbol in self.positions:
            self.positions[symbol].trailing_stop_pct = trailing_pct

    # Utility methods

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if position exists."""
        return symbol in self.positions

    def update_equity_curve(self):
        """Update equity curve tracking."""
        self.equity_curve.append(self.portfolio_value)
        if self._current_timestamp:
            self.timestamps.append(self._current_timestamp)

    def get_symbol(self, data: Optional[pd.DataFrame] = None) -> str:
        """
        Get the trading symbol from data.

        For single-symbol strategies, returns 'ASSET' or the symbol name if available.
        For multi-symbol strategies, returns the first symbol or requires explicit symbol parameter.

        Args:
            data: DataFrame to extract symbol from (uses self._current_data if None)

        Returns:
            Symbol string
        """
        if data is None:
            data = self._current_data

        if data is None:
            return 'ASSET'

        # Check if multi-symbol (multi-level columns)
        if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
            # Multi-symbol: return first symbol
            symbols = data.columns.get_level_values(0).unique()
            return symbols[0] if len(symbols) > 0 else 'ASSET'
        else:
            # Single-symbol: try to get from index name or use generic
            if hasattr(data.index, 'name') and data.index.name:
                return str(data.index.name)
            # Check if there's a symbol column
            if 'symbol' in data.columns:
                return str(data['symbol'].iloc[0])
            return 'ASSET'

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary."""
        if len(self.equity_curve) < 2:
            return {}

        equity = pd.Series(self.equity_curve, index=self.timestamps)
        returns = equity.pct_change().dropna()

        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        num_trades = len(self.trades)

        return {
            'total_return': total_return,
            'final_value': equity.iloc[-1],
            'num_trades': num_trades,
            'avg_return': returns.mean(),
            'volatility': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        }
