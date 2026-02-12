"""Order book simulation."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from collections import defaultdict
import heapq


@dataclass
class OrderBookLevel:
    """Single price level in order book."""
    price: float
    quantity: float


@dataclass
class OrderBook:
    """Order book with bid/ask levels."""
    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)  # Sorted descending by price
    asks: List[OrderBookLevel] = field(default_factory=list)  # Sorted ascending by price
    timestamp: Optional[pd.Timestamp] = None

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price (average of best bid and ask)."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points."""
        if self.spread is not None and self.mid_price is not None and self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return None

    def get_total_volume(self, side: str, levels: int = 5) -> float:
        """Get total volume for side up to N levels."""
        if side == "bid":
            return sum(level.quantity for level in self.bids[:levels])
        else:
            return sum(level.quantity for level in self.asks[:levels])

    def execute_market_order(self, side: str, quantity: float) -> Tuple[float, float]:
        """
        Simulate market order execution.

        Args:
            side: "buy" or "sell"
            quantity: Order quantity

        Returns:
            (average_price, filled_quantity)
        """
        if side == "buy":
            # Walk up the ask side
            levels = self.asks
        else:
            # Walk down the bid side
            levels = self.bids

        remaining = quantity
        total_cost = 0.0
        filled = 0.0

        for level in levels:
            if remaining <= 0:
                break

            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            filled += fill_qty
            remaining -= fill_qty

        if filled > 0:
            avg_price = total_cost / filled
        else:
            avg_price = 0.0

        return avg_price, filled

    def execute_limit_order(
        self,
        side: str,
        quantity: float,
        limit_price: float
    ) -> Tuple[float, float]:
        """
        Simulate limit order execution.

        Args:
            side: "buy" or "sell"
            quantity: Order quantity
            limit_price: Limit price

        Returns:
            (average_price, filled_quantity)
        """
        if side == "buy":
            # Can only fill at limit price or better (lower)
            levels = [l for l in self.asks if l.price <= limit_price]
        else:
            # Can only fill at limit price or better (higher)
            levels = [l for l in self.bids if l.price >= limit_price]

        remaining = quantity
        total_cost = 0.0
        filled = 0.0

        for level in levels:
            if remaining <= 0:
                break

            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            filled += fill_qty
            remaining -= fill_qty

        if filled > 0:
            avg_price = total_cost / filled
        else:
            avg_price = 0.0

        return avg_price, filled


class OrderBookSimulator:
    """Simulate order book dynamics from OHLCV data."""

    def __init__(
        self,
        num_levels: int = 5,
        spread_bps: float = 10.0,
        depth_per_level: float = 1000.0,
        depth_decay: float = 0.8,
    ):
        """
        Initialize order book simulator.

        Args:
            num_levels: Number of price levels to simulate
            spread_bps: Typical spread in basis points
            depth_per_level: Base depth (volume) per level
            depth_decay: Decay factor for depth at each level
        """
        self.num_levels = num_levels
        self.spread_bps = spread_bps
        self.depth_per_level = depth_per_level
        self.depth_decay = depth_decay

    def generate_order_book(
        self,
        symbol: str,
        mid_price: float,
        volume: float,
        volatility: float = 0.02,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> OrderBook:
        """
        Generate synthetic order book.

        Args:
            symbol: Asset symbol
            mid_price: Mid price
            volume: Recent volume (affects depth)
            volatility: Price volatility (affects spread)
            timestamp: Current timestamp

        Returns:
            OrderBook
        """
        # Adjust spread based on volatility
        spread = mid_price * (self.spread_bps / 10000) * (1 + volatility * 10)
        half_spread = spread / 2

        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread

        # Calculate tick size (0.01 for most stocks)
        tick_size = 0.01 if mid_price < 1000 else 0.05

        # Generate bid levels
        bids = []
        for i in range(self.num_levels):
            price = best_bid - (i * tick_size)
            # Depth increases with volume and decreases with level
            depth = (self.depth_per_level * (volume / 1000000)) * (self.depth_decay ** i)
            bids.append(OrderBookLevel(price=price, quantity=depth))

        # Generate ask levels
        asks = []
        for i in range(self.num_levels):
            price = best_ask + (i * tick_size)
            depth = (self.depth_per_level * (volume / 1000000)) * (self.depth_decay ** i)
            asks.append(OrderBookLevel(price=price, quantity=depth))

        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=timestamp,
        )

    def generate_from_ohlcv(
        self,
        symbol: str,
        bar: pd.Series,
        timestamp: pd.Timestamp,
    ) -> OrderBook:
        """
        Generate order book from OHLCV bar.

        Args:
            symbol: Asset symbol
            bar: OHLCV bar (must have close, volume, high, low)
            timestamp: Current timestamp

        Returns:
            OrderBook
        """
        mid_price = bar['close']
        volume = bar.get('volume', 1000000)

        # Estimate volatility from high-low range
        if 'high' in bar and 'low' in bar and mid_price > 0:
            volatility = (bar['high'] - bar['low']) / mid_price
        else:
            volatility = 0.02

        return self.generate_order_book(
            symbol=symbol,
            mid_price=mid_price,
            volume=volume,
            volatility=volatility,
            timestamp=timestamp,
        )


class LimitOrderTracker:
    """Track and manage limit orders."""

    def __init__(self):
        """Initialize limit order tracker."""
        self.pending_orders: Dict[str, List] = defaultdict(list)

    def add_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        timestamp: pd.Timestamp,
    ):
        """Add limit order to tracker."""
        self.pending_orders[symbol].append({
            'order_id': order_id,
            'side': side,
            'quantity': quantity,
            'limit_price': limit_price,
            'timestamp': timestamp,
            'remaining': quantity,
        })

    def check_fills(
        self,
        symbol: str,
        order_book: OrderBook,
        current_timestamp: pd.Timestamp,
    ) -> List[Dict]:
        """
        Check if any limit orders can be filled.

        Returns:
            List of filled orders
        """
        fills = []

        if symbol not in self.pending_orders:
            return fills

        remaining_orders = []

        for order in self.pending_orders[symbol]:
            if order['side'] == 'buy':
                # Buy limit: fill if ask <= limit
                if order_book.best_ask and order_book.best_ask <= order['limit_price']:
                    avg_price, filled_qty = order_book.execute_limit_order(
                        'buy',
                        order['remaining'],
                        order['limit_price']
                    )

                    if filled_qty > 0:
                        fills.append({
                            'order_id': order['order_id'],
                            'symbol': symbol,
                            'side': order['side'],
                            'quantity': filled_qty,
                            'price': avg_price,
                            'timestamp': current_timestamp,
                        })

                        order['remaining'] -= filled_qty

                if order['remaining'] > 0:
                    remaining_orders.append(order)

            else:  # sell
                # Sell limit: fill if bid >= limit
                if order_book.best_bid and order_book.best_bid >= order['limit_price']:
                    avg_price, filled_qty = order_book.execute_limit_order(
                        'sell',
                        order['remaining'],
                        order['limit_price']
                    )

                    if filled_qty > 0:
                        fills.append({
                            'order_id': order['order_id'],
                            'symbol': symbol,
                            'side': order['side'],
                            'quantity': filled_qty,
                            'price': avg_price,
                            'timestamp': current_timestamp,
                        })

                        order['remaining'] -= filled_qty

                if order['remaining'] > 0:
                    remaining_orders.append(order)

        self.pending_orders[symbol] = remaining_orders
        return fills

    def cancel_order(self, order_id: str) -> bool:
        """Cancel limit order."""
        for symbol in self.pending_orders:
            orders = self.pending_orders[symbol]
            self.pending_orders[symbol] = [o for o in orders if o['order_id'] != order_id]
            if len(orders) != len(self.pending_orders[symbol]):
                return True
        return False

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get pending orders."""
        if symbol:
            return self.pending_orders.get(symbol, [])
        else:
            all_orders = []
            for orders in self.pending_orders.values():
                all_orders.extend(orders)
            return all_orders
