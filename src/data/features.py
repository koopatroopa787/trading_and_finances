"""Feature engineering for trading strategies."""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Callable
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Generate technical indicators and custom features."""

    @staticmethod
    def add_sma(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        df = df.copy()
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, periods: List[int] = [12, 26, 50, 200]) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        df = df.copy()
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)."""
        df = df.copy()
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df = df.copy()
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        df['bb_upper'] = df['bb_middle'] + (std_dev * std)
        df['bb_lower'] = df['bb_middle'] - (std_dev * std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        df = df.copy()
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        df['atr'] = true_range.rolling(window=period).mean()
        df['atr_pct'] = df['atr'] / df['close']
        return df

    @staticmethod
    def add_stochastic(
        df: pd.DataFrame,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        df = df.copy()
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_k'] = df['stoch_k'].rolling(window=smooth_k).mean()
        df['stoch_d'] = df['stoch_k'].rolling(window=smooth_d).mean()
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index."""
        df = df.copy()

        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = ranges.max(axis=1).rolling(window=period).mean()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        df = df.copy()
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        df['obv_ema'] = obv.ewm(span=20).mean()
        return df

    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Weighted Average Price."""
        df = df.copy()
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return df

    @staticmethod
    def add_momentum(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """Add Momentum indicators."""
        df = df.copy()
        for period in periods:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'roc_{period}'] = 100 * (df['close'] / df['close'].shift(period) - 1)
        return df

    @staticmethod
    def add_volatility(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """Add Volatility measures."""
        df = df.copy()
        returns = df['close'].pct_change()

        for period in periods:
            # Historical volatility (annualized)
            df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)

            # Parkinson volatility (uses high-low)
            hl_ratio = np.log(df['high'] / df['low'])
            df[f'parkinson_vol_{period}'] = np.sqrt(
                hl_ratio.pow(2).rolling(window=period).mean() / (4 * np.log(2))
            ) * np.sqrt(252)

        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume-based features."""
        df = df.copy()

        # Volume ratios
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Volume price trend
        df['vpt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()

        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()

        df['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow))

        return df

    @staticmethod
    def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Add Price pattern features."""
        df = df.copy()

        # Higher highs, lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

        # Gap detection
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)

        # Inside/outside bars
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) &
                            (df['low'] > df['low'].shift(1))).astype(int)
        df['outside_bar'] = ((df['high'] > df['high'].shift(1)) &
                             (df['low'] < df['low'].shift(1))).astype(int)

        # Body and wick ratios
        df['body'] = np.abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['body_ratio'] = df['body'] / (df['high'] - df['low'])

        return df

    @staticmethod
    def add_statistical_features(df: pd.DataFrame, periods: List[int] = [20, 60]) -> pd.DataFrame:
        """Add Statistical features."""
        df = df.copy()
        returns = df['close'].pct_change()

        for period in periods:
            # Rolling statistics
            df[f'returns_mean_{period}'] = returns.rolling(window=period).mean()
            df[f'returns_std_{period}'] = returns.rolling(window=period).std()
            df[f'returns_skew_{period}'] = returns.rolling(window=period).skew()
            df[f'returns_kurt_{period}'] = returns.rolling(window=period).kurt()

            # Z-score of price
            mean = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'price_zscore_{period}'] = (df['close'] - mean) / std

        return df

    @staticmethod
    def add_fourier_features(df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """Add Fourier transform features for cycle detection."""
        df = df.copy()
        close_fft = np.fft.fft(df['close'].ffill())

        for i in range(1, n_components + 1):
            df[f'fft_real_{i}'] = np.real(close_fft)[i]
            df[f'fft_imag_{i}'] = np.imag(close_fft)[i]

        return df

    @staticmethod
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators."""
        df = df.copy()
        returns = df['close'].pct_change()

        # Trend strength
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']

        # Volatility regime
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        df['vol_regime'] = vol_20 / vol_60

        # Range expansion/contraction
        atr_14 = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        atr_50 = df['high'].rolling(50).max() - df['low'].rolling(50).min()
        df['range_regime'] = atr_14 / atr_50

        return df

    @staticmethod
    def add_all_features(
        df: pd.DataFrame,
        include_advanced: bool = True
    ) -> pd.DataFrame:
        """Add all available features."""
        logger.info("Adding all features...")

        # Basic indicators
        df = FeatureEngineer.add_sma(df)
        df = FeatureEngineer.add_ema(df)
        df = FeatureEngineer.add_rsi(df)
        df = FeatureEngineer.add_macd(df)
        df = FeatureEngineer.add_bollinger_bands(df)
        df = FeatureEngineer.add_atr(df)
        df = FeatureEngineer.add_stochastic(df)
        df = FeatureEngineer.add_adx(df)

        # Volume features
        df = FeatureEngineer.add_obv(df)
        df = FeatureEngineer.add_vwap(df)
        df = FeatureEngineer.add_volume_features(df)

        # Momentum and volatility
        df = FeatureEngineer.add_momentum(df)
        df = FeatureEngineer.add_volatility(df)

        # Price patterns
        df = FeatureEngineer.add_price_patterns(df)

        if include_advanced:
            # Advanced features
            df = FeatureEngineer.add_statistical_features(df)
            df = FeatureEngineer.add_fourier_features(df)
            df = FeatureEngineer.add_regime_features(df)

        logger.info(f"Total features: {len(df.columns)}")
        return df

    @staticmethod
    def select_features(
        df: pd.DataFrame,
        target: pd.Series,
        method: str = "mutual_info",
        k: int = 20
    ) -> List[str]:
        """
        Select top k features using specified method.

        Args:
            df: DataFrame with features
            target: Target variable
            method: Selection method ('mutual_info', 'f_regression', 'correlation')
            k: Number of features to select

        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest

        # Remove NaN values
        valid_idx = ~(df.isna().any(axis=1) | target.isna())
        X = df[valid_idx]
        y = target[valid_idx]

        if method == "mutual_info":
            scores = mutual_info_regression(X, y)
        elif method == "f_regression":
            scores, _ = f_regression(X, y)
        elif method == "correlation":
            scores = np.abs(X.corrwith(y))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Get top k features
        top_indices = np.argsort(scores)[-k:]
        selected_features = X.columns[top_indices].tolist()

        logger.info(f"Selected {k} features using {method}")
        return selected_features

    @staticmethod
    def create_lagged_features(
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """Create lagged versions of features."""
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return df

    @staticmethod
    def create_rolling_features(
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20],
        functions: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """Create rolling window features."""
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for window in windows:
                for func in functions:
                    feature_name = f'{col}_rolling_{window}_{func}'
                    if func == 'mean':
                        df[feature_name] = df[col].rolling(window).mean()
                    elif func == 'std':
                        df[feature_name] = df[col].rolling(window).std()
                    elif func == 'min':
                        df[feature_name] = df[col].rolling(window).min()
                    elif func == 'max':
                        df[feature_name] = df[col].rolling(window).max()

        return df
