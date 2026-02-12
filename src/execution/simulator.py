"""Execution simulator for backtesting."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from src.strategy.base import BaseStrategy, Order, OrderType, OrderSide
from src.execution.cost_models import CostModel, FixedCostModel, TransactionCost
from src.execution.order_book import OrderBook, OrderBookSimulator, LimitOrderTracker
from config import config


@dataclass
class BacktestResult:
    """Results from backtest execution."""
    equity_curve: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    metrics: Dict[str, Any]
    strategy_name: str


class ExecutionSimulator:
    """
    Simulate strategy execution with realistic market mechanics.

    Features:
    - Multiple order types (market, limit, stop)
    - Realistic slippage and transaction costs
    - Order book simulation
    - Partial fills
    - Position tracking
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        cost_model: Optional[CostModel] = None,
        use_order_book: bool = True,
        partial_fills: bool = True,
        max_position_size: float = 0.25,
        max_leverage: float = 1.0,
    ):
        """
        Initialize execution simulator.

        Args:
            initial_capital: Starting capital
            cost_model: Transaction cost model
            use_order_book: Whether to simulate order book
            partial_fills: Allow partial order fills
            max_position_size: Maximum position size as fraction of portfolio
            max_leverage: Maximum portfolio leverage
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or FixedCostModel()
        self.use_order_book = use_order_book
        self.partial_fills = partial_fills
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage

        # Order book simulator
        self.order_book_simulator = OrderBookSimulator()
        self.limit_order_tracker = LimitOrderTracker()

        # State tracking
        self.current_timestamp = None
        self.current_data = None

        # Results
        self.equity_history: List[Dict] = []
        self.position_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.order_history: List[Dict] = []

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            strategy: Trading strategy
            data: Market data (OHLCV)
            start_date: Start date (None = use all data)
            end_date: End date (None = use all data)

        Returns:
            BacktestResult
        """
        logger.info(f"Starting backtest for {strategy.name}")

        # Filter data by date range
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]

        # Initialize strategy
        strategy.on_start(self.initial_capital)

        # Main backtest loop
        for i, (timestamp, row) in enumerate(data.iterrows()):
            self.current_timestamp = timestamp
            self.current_data = data.iloc[:i+1]

            # Update position prices
            self._update_positions(strategy, row)

            # Check for limit order fills
            if self.use_order_book:
                self._check_limit_orders(strategy, row)

            # Check stop losses and take profits
            self._check_stops(strategy, row)

            # Call strategy
            strategy._current_timestamp = timestamp
            strategy._current_data = self.current_data
            strategy.on_data(timestamp, self.current_data)

            # Execute pending orders
            self._execute_orders(strategy, row)

            # Call bar close hook
            strategy.on_bar_close(timestamp, self.current_data)

            # Update equity curve
            strategy.update_equity_curve()

            # Record state
            self._record_state(strategy, timestamp)

        # Finalize strategy
        strategy.on_stop()

        # Generate results
        result = self._generate_results(strategy, data)

        logger.info(f"Backtest completed: {len(self.trade_history)} trades executed")
        return result

    def _update_positions(self, strategy: BaseStrategy, row: pd.Series):
        """Update current prices for all positions."""
        if 'close' not in row:
            return

        for symbol, position in strategy.positions.items():
            # For multi-symbol data
            if hasattr(row.index, 'levels') and len(row.index.levels) > 1:
                if symbol in row.index.get_level_values(0):
                    price = row[symbol]['close']
                    position.update_price(price)
            else:
                # Single symbol
                position.update_price(row['close'])

    def _check_limit_orders(self, strategy: BaseStrategy, row: pd.Series):
        """Check and fill limit orders."""
        for position in strategy.positions.values():
            symbol = position.symbol

            # Generate order book
            order_book = self.order_book_simulator.generate_from_ohlcv(
                symbol, row, self.current_timestamp
            )

            # Check fills
            fills = self.limit_order_tracker.check_fills(
                symbol, order_book, self.current_timestamp
            )

            for fill in fills:
                # Process fill
                logger.debug(f"Limit order filled: {fill}")

    def _check_stops(self, strategy: BaseStrategy, row: pd.Series):
        """Check stop losses and take profits."""
        for symbol, position in list(strategy.positions.items()):
            current_price = position.current_price

            # Stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                strategy.close_position(symbol)

            # Take profit
            if position.take_profit and current_price >= position.take_profit:
                logger.info(f"Take profit triggered for {symbol} at {current_price}")
                strategy.close_position(symbol)

    def _execute_orders(self, strategy: BaseStrategy, row: pd.Series):
        """Execute pending orders."""
        executed_orders = []

        for order in strategy.pending_orders:
            # Check position size limits
            if not self._check_limits(strategy, order):
                strategy.on_order_reject(
                    order,
                    "Position size or leverage limit exceeded"
                )
                executed_orders.append(order)
                continue

            # Execute based on order type
            if order.order_type == OrderType.MARKET:
                self._execute_market_order(strategy, order, row)
                executed_orders.append(order)

            elif order.order_type == OrderType.LIMIT:
                # Add to limit order tracker
                self.limit_order_tracker.add_order(
                    order.order_id,
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.limit_price,
                    self.current_timestamp,
                )
                executed_orders.append(order)

            elif order.order_type == OrderType.STOP_LOSS:
                # Set stop loss on position
                if order.symbol in strategy.positions:
                    strategy.positions[order.symbol].stop_loss = order.stop_price
                executed_orders.append(order)

            elif order.order_type == OrderType.TAKE_PROFIT:
                # Set take profit on position
                if order.symbol in strategy.positions:
                    strategy.positions[order.symbol].take_profit = order.stop_price
                executed_orders.append(order)

        # Remove executed orders
        strategy.pending_orders = [
            o for o in strategy.pending_orders if o not in executed_orders
        ]

    def _execute_market_order(
        self,
        strategy: BaseStrategy,
        order: Order,
        row: pd.Series
    ):
        """Execute market order."""
        symbol = order.symbol

        # Get execution price
        if self.use_order_book:
            # Use order book for execution
            order_book = self.order_book_simulator.generate_from_ohlcv(
                symbol, row, self.current_timestamp
            )

            if order.side == OrderSide.BUY:
                fill_price, filled_qty = order_book.execute_market_order(
                    "buy", order.quantity
                )
            else:
                fill_price, filled_qty = order_book.execute_market_order(
                    "sell", order.quantity
                )

            # Check for partial fill
            if self.partial_fills and filled_qty < order.quantity:
                logger.warning(
                    f"Partial fill: {filled_qty}/{order.quantity} for {symbol}"
                )
                order.quantity = filled_qty

        else:
            # Simple execution at close price
            fill_price = row['close']
            filled_qty = order.quantity

        # Calculate transaction costs
        volume = row.get('volume', 1000000)
        volatility = self._estimate_volatility(row)

        costs = self.cost_model.calculate_cost(
            quantity=filled_qty,
            price=fill_price,
            side=order.side.value,
            volatility=volatility,
            volume=volume,
        )

        # Adjust fill price for costs
        if order.side == OrderSide.BUY:
            effective_price = fill_price + (costs.total / filled_qty)
        else:
            effective_price = fill_price - (costs.total / filled_qty)

        # Update cash
        if order.side == OrderSide.BUY:
            strategy.cash -= (filled_qty * effective_price)
        else:
            strategy.cash += (filled_qty * effective_price)

        # Record trade
        self.trade_history.append({
            'timestamp': self.current_timestamp,
            'symbol': symbol,
            'side': order.side.value,
            'quantity': filled_qty,
            'price': fill_price,
            'effective_price': effective_price,
            'commission': costs.commission,
            'slippage': costs.slippage,
            'market_impact': costs.market_impact,
            'total_cost': costs.total,
        })

        # Call strategy hook
        strategy.on_order_fill(order, effective_price, self.current_timestamp)

        # Record order
        self.order_history.append({
            'timestamp': self.current_timestamp,
            'order_id': order.order_id,
            'symbol': symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'filled_quantity': filled_qty,
            'price': fill_price,
        })

    def _check_limits(self, strategy: BaseStrategy, order: Order) -> bool:
        """Check position size and leverage limits."""
        # Calculate order value
        if hasattr(self.current_data, 'columns') and hasattr(self.current_data.columns, 'levels'):
            if order.symbol in self.current_data.columns.get_level_values(0):
                price = self.current_data[order.symbol]['close'].iloc[-1]
            else:
                return False
        else:
            price = self.current_data['close'].iloc[-1]

        order_value = order.quantity * price

        # Check position size limit
        max_position_value = strategy.portfolio_value * self.max_position_size
        if order.side == OrderSide.BUY and order_value > max_position_value:
            return False

        # Check leverage limit
        if order.side == OrderSide.BUY:
            new_positions_value = strategy.positions_value + order_value
            new_leverage = new_positions_value / strategy.portfolio_value
            if new_leverage > self.max_leverage:
                return False

        return True

    def _estimate_volatility(self, row: pd.Series) -> float:
        """Estimate volatility from OHLCV data."""
        if 'high' in row and 'low' in row and 'close' in row and row['close'] > 0:
            # Parkinson volatility estimator
            return (row['high'] - row['low']) / row['close']
        return 0.02  # Default 2%

    def _record_state(self, strategy: BaseStrategy, timestamp: pd.Timestamp):
        """Record portfolio state."""
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': strategy.portfolio_value,
            'cash': strategy.cash,
            'positions_value': strategy.positions_value,
            'num_positions': len(strategy.positions),
        })

        # Record positions
        for symbol, position in strategy.positions.items():
            self.position_history.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
            })

    def _generate_results(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame
    ) -> BacktestResult:
        """Generate backtest results."""
        # Convert to DataFrames
        equity_df = pd.DataFrame(self.equity_history).set_index('timestamp')
        positions_df = pd.DataFrame(self.position_history)
        trades_df = pd.DataFrame(self.trade_history)
        orders_df = pd.DataFrame(self.order_history)

        # Calculate metrics
        from src.analytics.performance import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()

        if len(equity_df) > 0:
            metrics = analyzer.calculate_metrics(equity_df['equity'], trades_df)
        else:
            metrics = {}

        return BacktestResult(
            equity_curve=equity_df['equity'] if len(equity_df) > 0 else pd.Series(),
            positions=positions_df,
            trades=trades_df,
            orders=orders_df,
            metrics=metrics,
            strategy_name=strategy.name,
        )
