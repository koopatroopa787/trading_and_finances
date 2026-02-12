"""Mean reversion strategies."""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from loguru import logger
from scipy import stats

from src.strategy.base import BaseStrategy, OrderType


class MeanReversionStrategy(BaseStrategy):
    """
    Simple mean reversion strategy using Z-score.

    Buys when price is below mean - threshold*std
    Sells when price is above mean + threshold*std
    """

    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        position_size: float = 0.1,
        **kwargs
    ):
        """
        Initialize mean reversion strategy.

        Args:
            lookback_period: Period for calculating mean and std
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            position_size: Position size as fraction of portfolio
        """
        super().__init__(**kwargs)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size

    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Process new data."""
        if len(data) < self.lookback_period:
            return

        # Calculate Z-score
        close_prices = data['close'].iloc[-self.lookback_period:]
        mean_price = close_prices.mean()
        std_price = close_prices.std()

        if std_price == 0:
            return

        current_price = data['close'].iloc[-1]
        z_score = (current_price - mean_price) / std_price

        # Get symbol name (handles both single and multi-symbol data)
        symbol = self.get_symbol(data)

        # Trading logic
        if z_score < -self.entry_threshold and not self.has_position(symbol):
            # Price is significantly below mean - BUY
            self.buy(symbol, size=self.position_size)
            logger.info(f"BUY signal: Z-score = {z_score:.2f}")

        elif z_score > self.entry_threshold and not self.has_position(symbol):
            # Price is significantly above mean - SELL (short)
            # For long-only, we skip this
            pass

        elif self.has_position(symbol):
            # Exit logic
            if abs(z_score) < self.exit_threshold:
                # Price has reverted to mean
                self.close_position(symbol)
                logger.info(f"EXIT signal: Z-score = {z_score:.2f}")


class PairsTradingStrategy(BaseStrategy):
    """
    Pairs trading strategy using cointegration.

    Trades the spread between two cointegrated assets.
    """

    def __init__(
        self,
        asset1: str,
        asset2: str,
        lookback_period: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        position_size: float = 0.1,
        **kwargs
    ):
        """
        Initialize pairs trading strategy.

        Args:
            asset1: First asset symbol
            asset2: Second asset symbol
            lookback_period: Period for cointegration analysis
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            position_size: Position size as fraction of portfolio
        """
        super().__init__(**kwargs)
        self.asset1 = asset1
        self.asset2 = asset2
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size

        self.hedge_ratio = None

    def calculate_hedge_ratio(self, prices1: pd.Series, prices2: pd.Series) -> float:
        """Calculate hedge ratio using OLS regression."""
        from sklearn.linear_model import LinearRegression

        X = prices2.values.reshape(-1, 1)
        y = prices1.values

        model = LinearRegression()
        model.fit(X, y)

        return model.coef_[0]

    def calculate_spread(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """Calculate spread between two assets."""
        return prices1 - hedge_ratio * prices2

    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Process new data."""
        # Expect multi-index DataFrame with both assets
        if self.asset1 not in data.columns.get_level_values(0):
            return
        if self.asset2 not in data.columns.get_level_values(0):
            return

        if len(data) < self.lookback_period:
            return

        # Get price series
        prices1 = data[self.asset1]['close'].iloc[-self.lookback_period:]
        prices2 = data[self.asset2]['close'].iloc[-self.lookback_period:]

        # Calculate hedge ratio
        self.hedge_ratio = self.calculate_hedge_ratio(prices1, prices2)

        # Calculate spread
        spread = self.calculate_spread(prices1, prices2, self.hedge_ratio)

        # Calculate Z-score of spread
        spread_mean = spread.mean()
        spread_std = spread.std()

        if spread_std == 0:
            return

        current_spread = spread.iloc[-1]
        z_score = (current_spread - spread_mean) / spread_std

        # Trading logic
        has_pair = self.has_position(self.asset1) or self.has_position(self.asset2)

        if z_score > self.entry_threshold and not has_pair:
            # Spread is too high - short asset1, long asset2
            self.sell(self.asset1, size=self.position_size)
            self.buy(self.asset2, size=self.position_size * self.hedge_ratio)
            logger.info(f"PAIRS TRADE: Short {self.asset1}, Long {self.asset2}, Z={z_score:.2f}")

        elif z_score < -self.entry_threshold and not has_pair:
            # Spread is too low - long asset1, short asset2
            self.buy(self.asset1, size=self.position_size)
            self.sell(self.asset2, size=self.position_size * self.hedge_ratio)
            logger.info(f"PAIRS TRADE: Long {self.asset1}, Short {self.asset2}, Z={z_score:.2f}")

        elif has_pair and abs(z_score) < self.exit_threshold:
            # Spread has converged - close positions
            self.close_position(self.asset1)
            self.close_position(self.asset2)
            logger.info(f"PAIRS EXIT: Z={z_score:.2f}")


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Multi-asset statistical arbitrage using PCA.

    Identifies mean-reverting principal components.
    """

    def __init__(
        self,
        symbols: List[str],
        lookback_period: int = 60,
        n_components: int = 3,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        position_size: float = 0.05,
        **kwargs
    ):
        """
        Initialize statistical arbitrage strategy.

        Args:
            symbols: List of asset symbols
            lookback_period: Period for PCA analysis
            n_components: Number of principal components
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            position_size: Position size per asset
        """
        super().__init__(**kwargs)
        self.symbols = symbols
        self.lookback_period = lookback_period
        self.n_components = n_components
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size

        self.pca = None
        self.components = None

    def fit_pca(self, returns: pd.DataFrame):
        """Fit PCA on returns."""
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=self.n_components)
        self.components = self.pca.fit_transform(returns)

    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Process new data."""
        if len(data) < self.lookback_period:
            return

        # Calculate returns
        returns_data = {}
        for symbol in self.symbols:
            if symbol in data.columns.get_level_values(0):
                returns_data[symbol] = data[symbol]['close'].pct_change()

        if len(returns_data) < 2:
            return

        returns_df = pd.DataFrame(returns_data).iloc[-self.lookback_period:]
        returns_df = returns_df.dropna()

        if len(returns_df) < self.lookback_period // 2:
            return

        # Fit PCA
        self.fit_pca(returns_df)

        # Get latest principal component values
        latest_pc = self.components[-1]

        # Calculate Z-scores for each PC
        for i in range(self.n_components):
            pc_series = self.components[:, i]
            pc_mean = pc_series.mean()
            pc_std = pc_series.std()

            if pc_std == 0:
                continue

            z_score = (latest_pc[i] - pc_mean) / pc_std

            # Trading logic for this component
            if abs(z_score) > self.entry_threshold:
                # Get loadings for this component
                loadings = self.pca.components_[i]

                # Trade based on loadings
                for j, symbol in enumerate(self.symbols):
                    if abs(loadings[j]) > 0.2:  # Significant loading
                        if z_score > 0 and loadings[j] > 0:
                            # PC is high, loading is positive - sell
                            if not self.has_position(symbol):
                                self.sell(symbol, size=self.position_size)
                        elif z_score < 0 and loadings[j] > 0:
                            # PC is low, loading is positive - buy
                            if not self.has_position(symbol):
                                self.buy(symbol, size=self.position_size)

            elif abs(z_score) < self.exit_threshold:
                # Close positions
                for symbol in self.symbols:
                    if self.has_position(symbol):
                        self.close_position(symbol)
