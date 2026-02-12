"""Market regime detection using HMM and ML."""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from sklearn.preprocessing import StandardScaler
from loguru import logger


class RegimeDetector:
    """Base class for regime detection."""

    def fit(self, data: pd.DataFrame):
        """Fit the regime detector."""
        raise NotImplementedError

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regimes."""
        raise NotImplementedError


class HMMRegimeDetector(RegimeDetector):
    """Hidden Markov Model for regime detection."""

    def __init__(self, n_regimes: int = 3, n_iter: int = 100):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of market regimes
            n_iter: Number of iterations for EM algorithm
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, returns: pd.Series):
        """
        Fit HMM to returns data.

        Args:
            returns: Return series
        """
        from hmmlearn import hmm

        # Prepare features
        features = self._prepare_features(returns)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=42,
        )

        self.model.fit(features_scaled)

        logger.info(f"HMM fitted with {self.n_regimes} regimes")

        # Interpret regimes
        self._interpret_regimes(features_scaled)

    def predict(self, returns: pd.Series) -> np.ndarray:
        """
        Predict regimes.

        Args:
            returns: Return series

        Returns:
            Array of regime labels
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        features = self._prepare_features(returns)
        features_scaled = self.scaler.transform(features)

        regimes = self.model.predict(features_scaled)

        return regimes

    def _prepare_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare features for HMM."""
        # Create features: returns and volatility
        returns_arr = returns.values.reshape(-1, 1)

        # Rolling volatility
        volatility = returns.rolling(window=20).std().bfill().values.reshape(-1, 1)

        # Combine features
        features = np.hstack([returns_arr, volatility])

        return features

    def _interpret_regimes(self, features: np.ndarray):
        """Interpret regime characteristics."""
        regimes = self.model.predict(features)

        regime_stats = {}

        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            regime_features = features[regime_mask]

            if len(regime_features) > 0:
                avg_return = regime_features[:, 0].mean()
                avg_vol = regime_features[:, 1].mean()

                regime_stats[regime] = {
                    'avg_return': avg_return,
                    'avg_volatility': avg_vol,
                    'frequency': regime_mask.sum() / len(regimes),
                }

        # Label regimes
        regime_labels = {}
        sorted_regimes = sorted(
            regime_stats.items(),
            key=lambda x: (x[1]['avg_return'], -x[1]['avg_volatility'])
        )

        if self.n_regimes == 2:
            labels = ['Bear', 'Bull']
        elif self.n_regimes == 3:
            labels = ['Bear', 'Neutral', 'Bull']
        elif self.n_regimes == 4:
            labels = ['Bear', 'Low Vol', 'Bull', 'High Vol']
        else:
            labels = [f'Regime {i}' for i in range(self.n_regimes)]

        for i, (regime, stats) in enumerate(sorted_regimes):
            regime_labels[regime] = labels[i]
            logger.info(
                f"{labels[i]}: Return={stats['avg_return']:.4f}, "
                f"Vol={stats['avg_volatility']:.4f}, "
                f"Freq={stats['frequency']:.2%}"
            )

        self.regime_labels = regime_labels


class MLRegimeDetector(RegimeDetector):
    """ML-based regime detection using clustering."""

    def __init__(
        self,
        n_regimes: int = 3,
        method: str = "kmeans",
        lookback: int = 60,
    ):
        """
        Initialize ML regime detector.

        Args:
            n_regimes: Number of regimes
            method: Clustering method (kmeans, gmm)
            lookback: Lookback period for features
        """
        self.n_regimes = n_regimes
        self.method = method
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, data: pd.DataFrame):
        """
        Fit clustering model.

        Args:
            data: DataFrame with OHLCV data
        """
        # Prepare features
        features = self._prepare_features(data)

        # Remove NaN
        features = features.dropna()

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Fit model
        if self.method == "kmeans":
            from sklearn.cluster import KMeans
            self.model = KMeans(n_clusters=self.n_regimes, random_state=42)
        elif self.method == "gmm":
            from sklearn.mixture import GaussianMixture
            self.model = GaussianMixture(n_components=self.n_regimes, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.model.fit(features_scaled)

        logger.info(f"Regime detector fitted with {self.n_regimes} regimes")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict regimes.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Array of regime labels
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        features = self._prepare_features(data)
        features = features.ffill().fillna(0)

        features_scaled = self.scaler.transform(features)

        regimes = self.model.predict(features_scaled)

        return regimes

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime detection."""
        features = pd.DataFrame(index=data.index)

        # Returns
        returns = data['close'].pct_change()
        features['return_1d'] = returns
        features['return_5d'] = data['close'].pct_change(5)
        features['return_20d'] = data['close'].pct_change(20)

        # Volatility
        features['volatility_20d'] = returns.rolling(20).std()
        features['volatility_60d'] = returns.rolling(60).std()

        # Trend
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        features['trend'] = (features['sma_20'] - features['sma_50']) / features['sma_50']

        # Volume
        if 'volume' in data.columns:
            features['volume_ratio'] = (
                data['volume'] / data['volume'].rolling(20).mean()
            )

        # Range
        if 'high' in data.columns and 'low' in data.columns:
            features['range'] = (data['high'] - data['low']) / data['close']

        return features
