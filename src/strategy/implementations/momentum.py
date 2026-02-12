"""Momentum and trend following strategies."""
import pandas as pd
import numpy as np
from typing import Optional, List
from loguru import logger

from src.strategy.base import BaseStrategy, OrderType


class MomentumStrategy(BaseStrategy):
    """
    Simple momentum strategy.

    Buys assets with strong recent performance.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        momentum_threshold: float = 0.05,
        position_size: float = 0.1,
        **kwargs
    ):
        """
        Initialize momentum strategy.

        Args:
            lookback_period: Period for momentum calculation
            momentum_threshold: Minimum momentum for entry
            position_size: Position size as fraction of portfolio
        """
        super().__init__(**kwargs)
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.position_size = position_size

    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Process new data."""
        if len(data) < self.lookback_period:
            return

        # Calculate momentum
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-self.lookback_period]
        momentum = (current_price - past_price) / past_price

        # Get symbol name (handles both single and multi-symbol data)
        symbol = self.get_symbol(data)

        # Trading logic
        if momentum > self.momentum_threshold and not self.has_position(symbol):
            # Strong positive momentum - BUY
            self.buy(symbol, size=self.position_size)
            logger.info(f"BUY signal: Momentum = {momentum:.2%}")

        elif momentum < -self.momentum_threshold and self.has_position(symbol):
            # Momentum has turned negative - SELL
            self.close_position(symbol)
            logger.info(f"SELL signal: Momentum = {momentum:.2%}")


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following using moving average crossover.

    Classic dual moving average strategy.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        position_size: float = 0.1,
        use_atr_stops: bool = True,
        atr_multiplier: float = 2.0,
        **kwargs
    ):
        """
        Initialize trend following strategy.

        Args:
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            position_size: Position size as fraction of portfolio
            use_atr_stops: Whether to use ATR-based stops
            atr_multiplier: ATR multiplier for stops
        """
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size
        self.use_atr_stops = use_atr_stops
        self.atr_multiplier = atr_multiplier

        self.prev_signal = None

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(data) < period:
            return 0.0

        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range.iloc[-period:].mean()

    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Process new data."""
        if len(data) < self.slow_period:
            return

        # Calculate moving averages
        fast_ma = data['close'].iloc[-self.fast_period:].mean()
        slow_ma = data['close'].iloc[-self.slow_period:].mean()

        # Determine signal
        current_signal = "long" if fast_ma > slow_ma else "short"
        # Get symbol name (handles both single and multi-symbol data)
        symbol = self.get_symbol(data)
        current_price = data['close'].iloc[-1]

        # Trading logic
        if current_signal == "long" and self.prev_signal != "long":
            # Golden cross - BUY
            if self.has_position(symbol):
                self.close_position(symbol)

            self.buy(symbol, size=self.position_size)

            # Set ATR-based stop loss
            if self.use_atr_stops:
                atr = self.calculate_atr(data)
                stop_price = current_price - (self.atr_multiplier * atr)
                self.set_stop_loss(symbol, stop_price)

            logger.info(f"GOLDEN CROSS: Fast MA ({fast_ma:.2f}) > Slow MA ({slow_ma:.2f})")

        elif current_signal == "short" and self.prev_signal != "short":
            # Death cross - SELL
            if self.has_position(symbol):
                self.close_position(symbol)
                logger.info(f"DEATH CROSS: Fast MA ({fast_ma:.2f}) < Slow MA ({slow_ma:.2f})")

        self.prev_signal = current_signal


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy using Donchian Channels.

    Buys on breakouts above the high, sells on breakouts below the low.
    """

    def __init__(
        self,
        breakout_period: int = 20,
        exit_period: int = 10,
        position_size: float = 0.1,
        use_volume_filter: bool = True,
        volume_multiplier: float = 1.5,
        **kwargs
    ):
        """
        Initialize breakout strategy.

        Args:
            breakout_period: Period for breakout detection
            exit_period: Period for exit detection
            position_size: Position size as fraction of portfolio
            use_volume_filter: Whether to require volume confirmation
            volume_multiplier: Volume multiplier for confirmation
        """
        super().__init__(**kwargs)
        self.breakout_period = breakout_period
        self.exit_period = exit_period
        self.position_size = position_size
        self.use_volume_filter = use_volume_filter
        self.volume_multiplier = volume_multiplier

    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Process new data."""
        if len(data) < self.breakout_period:
            return

        # Calculate Donchian Channels
        high_channel = data['high'].iloc[-self.breakout_period:-1].max()
        low_channel = data['low'].iloc[-self.breakout_period:-1].min()

        exit_low = data['low'].iloc[-self.exit_period:-1].min()

        current_price = data['close'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]

        # Get symbol name (handles both single and multi-symbol data)
        symbol = self.get_symbol(data)

        # Volume filter
        volume_confirmed = True
        if self.use_volume_filter and 'volume' in data.columns:
            avg_volume = data['volume'].iloc[-20:-1].mean()
            current_volume = data['volume'].iloc[-1]
            volume_confirmed = current_volume >= (avg_volume * self.volume_multiplier)

        # Trading logic
        if current_high > high_channel and not self.has_position(symbol):
            if volume_confirmed:
                # Breakout above - BUY
                self.buy(symbol, size=self.position_size)
                self.set_stop_loss(symbol, low_channel)
                logger.info(
                    f"BREAKOUT: Price {current_price:.2f} > High {high_channel:.2f}"
                )

        elif self.has_position(symbol) and current_low < exit_low:
            # Exit on breakdown
            self.close_position(symbol)
            logger.info(f"EXIT: Price {current_price:.2f} < Exit {exit_low:.2f}")


class MultiFactorMomentumStrategy(BaseStrategy):
    """
    Multi-factor momentum strategy combining price, volume, and volatility.

    Uses composite score from multiple momentum factors.
    """

    def __init__(
        self,
        lookback_periods: List[int] = [5, 10, 20, 60],
        volume_lookback: int = 20,
        vol_lookback: int = 20,
        entry_threshold: float = 0.6,
        exit_threshold: float = 0.3,
        position_size: float = 0.1,
        **kwargs
    ):
        """
        Initialize multi-factor momentum strategy.

        Args:
            lookback_periods: Periods for momentum calculation
            volume_lookback: Period for volume analysis
            vol_lookback: Period for volatility analysis
            entry_threshold: Composite score threshold for entry
            exit_threshold: Composite score threshold for exit
            position_size: Position size as fraction of portfolio
        """
        super().__init__(**kwargs)
        self.lookback_periods = lookback_periods
        self.volume_lookback = volume_lookback
        self.vol_lookback = vol_lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size

    def calculate_composite_score(self, data: pd.DataFrame) -> float:
        """Calculate composite momentum score."""
        scores = []

        # Price momentum
        for period in self.lookback_periods:
            if len(data) >= period:
                momentum = data['close'].pct_change(period).iloc[-1]
                # Normalize to 0-1
                score = 1 / (1 + np.exp(-momentum * 10))  # Sigmoid
                scores.append(score)

        # Volume trend
        if 'volume' in data.columns and len(data) >= self.volume_lookback:
            vol_ma = data['volume'].iloc[-self.volume_lookback:].mean()
            current_vol = data['volume'].iloc[-1]
            vol_score = min(current_vol / vol_ma, 2.0) / 2.0  # Cap at 2x
            scores.append(vol_score)

        # Volatility (inverse - prefer lower volatility)
        if len(data) >= self.vol_lookback:
            returns = data['close'].pct_change()
            vol = returns.iloc[-self.vol_lookback:].std()
            avg_vol = returns.std()
            if avg_vol > 0:
                vol_score = 1.0 - min(vol / avg_vol, 2.0) / 2.0
                scores.append(vol_score)

        # Composite score (average)
        if len(scores) == 0:
            return 0.5

        return np.mean(scores)

    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Process new data."""
        if len(data) < max(self.lookback_periods):
            return

        # Calculate composite score
        score = self.calculate_composite_score(data)

        # Get symbol name (handles both single and multi-symbol data)
        symbol = self.get_symbol(data)

        # Trading logic
        if score > self.entry_threshold and not self.has_position(symbol):
            # Strong momentum - BUY
            self.buy(symbol, size=self.position_size)
            logger.info(f"BUY signal: Composite score = {score:.3f}")

        elif score < self.exit_threshold and self.has_position(symbol):
            # Weak momentum - SELL
            self.close_position(symbol)
            logger.info(f"SELL signal: Composite score = {score:.3f}")
