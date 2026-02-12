"""Strategy composition for multi-strategy portfolios."""
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from loguru import logger

from src.strategy.base import BaseStrategy, Order, OrderSide, OrderType, Position


class StrategyComposer:
    """
    Combine multiple strategies with dynamic allocation.

    Supports different allocation methods:
    - Static: Fixed weights
    - Dynamic: Rebalance based on performance
    - Risk-parity: Allocate based on volatility
    - Momentum: Allocate more to winning strategies
    """

    def __init__(
        self,
        strategies: Dict[str, BaseStrategy],
        allocation_method: str = "static",
        weights: Optional[Dict[str, float]] = None,
        rebalance_frequency: int = 20,
    ):
        """
        Initialize strategy composer.

        Args:
            strategies: Dict mapping strategy names to strategy instances
            allocation_method: Allocation method (static, dynamic, risk_parity, momentum)
            weights: Initial strategy weights (must sum to 1.0)
            rebalance_frequency: Days between rebalancing
        """
        self.strategies = strategies
        self.allocation_method = allocation_method
        self.rebalance_frequency = rebalance_frequency

        # Initialize weights
        if weights is None:
            # Equal weight
            n = len(strategies)
            self.weights = {name: 1.0 / n for name in strategies.keys()}
        else:
            # Validate weights sum to 1.0
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-6:
                logger.warning(f"Weights sum to {total}, normalizing to 1.0")
                self.weights = {k: v/total for k, v in weights.items()}
            else:
                self.weights = weights

        # Performance tracking
        self.strategy_returns: Dict[str, List[float]] = {name: [] for name in strategies.keys()}
        self.weight_history: List[Dict[str, float]] = []
        self.days_since_rebalance = 0

        logger.info(f"Initialized composer with {len(strategies)} strategies")
        logger.info(f"Initial weights: {self.weights}")

    def on_start(self, initial_capital: float):
        """Initialize all strategies."""
        for name, strategy in self.strategies.items():
            allocated_capital = initial_capital * self.weights[name]
            strategy.on_start(allocated_capital)

    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Run all strategies on new data."""
        # Update each strategy
        for name, strategy in self.strategies.items():
            strategy._current_timestamp = timestamp
            strategy._current_data = data
            strategy.on_data(timestamp, data)
            strategy.update_equity_curve()

        # Track strategy returns
        for name, strategy in self.strategies.items():
            if len(strategy.equity_curve) >= 2:
                ret = (strategy.equity_curve[-1] / strategy.equity_curve[-2]) - 1
                self.strategy_returns[name].append(ret)

        # Rebalance if needed
        self.days_since_rebalance += 1
        if self.days_since_rebalance >= self.rebalance_frequency:
            self.rebalance(timestamp)
            self.days_since_rebalance = 0

    def on_bar_close(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Call bar close hook for all strategies."""
        for strategy in self.strategies.values():
            strategy.on_bar_close(timestamp, data)

    def rebalance(self, timestamp: pd.Timestamp):
        """Rebalance strategy allocations."""
        if self.allocation_method == "static":
            # Keep current weights
            pass

        elif self.allocation_method == "dynamic":
            # Rebalance to target weights based on current values
            total_value = self.get_total_portfolio_value()
            for name, strategy in self.strategies.items():
                target_value = total_value * self.weights[name]
                current_value = strategy.portfolio_value

                if abs(current_value - target_value) / target_value > 0.05:  # 5% threshold
                    # Transfer capital between strategies
                    logger.info(
                        f"Rebalancing {name}: "
                        f"${current_value:.2f} -> ${target_value:.2f}"
                    )

        elif self.allocation_method == "risk_parity":
            # Allocate inverse to volatility
            self.weights = self._calculate_risk_parity_weights()
            logger.info(f"Risk parity weights: {self.weights}")

        elif self.allocation_method == "momentum":
            # Allocate more to winning strategies
            self.weights = self._calculate_momentum_weights()
            logger.info(f"Momentum weights: {self.weights}")

        self.weight_history.append(self.weights.copy())

    def _calculate_risk_parity_weights(self) -> Dict[str, float]:
        """Calculate risk parity weights (inverse volatility)."""
        vols = {}

        for name, returns in self.strategy_returns.items():
            if len(returns) < 20:
                vols[name] = 1.0  # Default
            else:
                vols[name] = np.std(returns[-60:])  # 60-day volatility

        # Inverse volatility weighting
        inv_vols = {name: 1.0 / vol if vol > 0 else 0 for name, vol in vols.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol == 0:
            # Fallback to equal weight
            n = len(self.strategies)
            return {name: 1.0 / n for name in self.strategies.keys()}

        weights = {name: inv_vol / total_inv_vol for name, inv_vol in inv_vols.items()}
        return weights

    def _calculate_momentum_weights(self, lookback: int = 60) -> Dict[str, float]:
        """Calculate momentum-based weights."""
        scores = {}

        for name, returns in self.strategy_returns.items():
            if len(returns) < lookback:
                scores[name] = 0.0
            else:
                # Use cumulative return as momentum score
                cum_return = np.prod([1 + r for r in returns[-lookback:]]) - 1
                scores[name] = max(0, cum_return)  # Only positive momentum

        total_score = sum(scores.values())

        if total_score == 0:
            # Fallback to equal weight
            n = len(self.strategies)
            return {name: 1.0 / n for name in self.strategies.keys()}

        weights = {name: score / total_score for name, score in scores.items()}
        return weights

    def get_total_portfolio_value(self) -> float:
        """Get total value across all strategies."""
        return sum(strategy.portfolio_value for strategy in self.strategies.values())

    def get_all_positions(self) -> Dict[str, List[Position]]:
        """Get all positions across strategies."""
        all_positions = {}
        for name, strategy in self.strategies.items():
            all_positions[name] = list(strategy.positions.values())
        return all_positions

    def get_consolidated_positions(self) -> Dict[str, Position]:
        """Get consolidated positions across all strategies."""
        consolidated = {}

        for strategy in self.strategies.values():
            for symbol, position in strategy.positions.items():
                if symbol in consolidated:
                    # Merge positions
                    existing = consolidated[symbol]
                    total_qty = existing.quantity + position.quantity
                    avg_price = (
                        (existing.entry_price * existing.quantity +
                         position.entry_price * position.quantity) / total_qty
                    )
                    existing.quantity = total_qty
                    existing.entry_price = avg_price
                else:
                    consolidated[symbol] = position

        return consolidated

    def get_performance_summary(self) -> Dict[str, any]:
        """Get performance summary for all strategies."""
        summary = {
            'total_value': self.get_total_portfolio_value(),
            'strategies': {},
        }

        for name, strategy in self.strategies.items():
            summary['strategies'][name] = {
                'weight': self.weights[name],
                'value': strategy.portfolio_value,
                'performance': strategy.get_performance_summary(),
            }

        return summary

    def on_stop(self):
        """Stop all strategies."""
        for strategy in self.strategies.values():
            strategy.on_stop()
