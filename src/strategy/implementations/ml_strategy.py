"""Machine learning-based trading strategies."""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

from src.strategy.base import BaseStrategy, OrderType
from src.data.features import FeatureEngineer


class MLStrategy(BaseStrategy):
    """
    Machine learning strategy with rolling window training.

    Trains models on historical data and predicts future returns.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        features: Optional[List[str]] = None,
        lookback: int = 252,
        retrain_frequency: int = 20,
        prediction_horizon: int = 1,
        entry_threshold: float = 0.01,
        position_size: float = 0.1,
        use_feature_selection: bool = True,
        n_features: int = 20,
        min_training_samples: int = 50,
        **model_params
    ):
        """
        Initialize ML strategy.

        Args:
            model_type: Model type (xgboost, lightgbm, catboost, random_forest)
            features: List of feature names (None = auto-generate)
            lookback: Training window size
            retrain_frequency: Days between retraining
            prediction_horizon: Days ahead to predict
            entry_threshold: Minimum predicted return for entry
            position_size: Position size as fraction of portfolio
            use_feature_selection: Whether to use feature selection
            n_features: Number of features to select
            min_training_samples: Minimum samples required for training
            **model_params: Additional model parameters
        """
        super().__init__(**model_params)
        self.model_type = model_type
        self.features = features
        self.lookback = lookback
        self.retrain_frequency = retrain_frequency
        self.prediction_horizon = prediction_horizon
        self.entry_threshold = entry_threshold
        self.position_size = position_size
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features
        self.min_training_samples = min_training_samples
        self.model_params = model_params

        self.model = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.days_since_retrain = 0
        self.feature_engineer = FeatureEngineer()

        # Historical data buffer
        self.data_buffer = []

    def _create_model(self):
        """Create ML model based on type."""
        if self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 5),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                random_state=42,
            )
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 5),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                random_state=42,
                verbose=-1,
            )
        elif self.model_type == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=self.model_params.get('n_estimators', 100),
                depth=self.model_params.get('max_depth', 5),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                random_state=42,
                verbose=False,
            )
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 5),
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features from raw data."""
        # Add all features
        data_with_features = self.feature_engineer.add_all_features(
            data.copy(),
            include_advanced=True
        )

        # Use specified features or all
        if self.features:
            feature_cols = [col for col in self.features if col in data_with_features.columns]
        else:
            # Exclude OHLCV columns
            exclude = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
            feature_cols = [col for col in data_with_features.columns if col not in exclude]

        return data_with_features[feature_cols]

    def _prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variable (future returns)."""
        returns = data['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        return returns

    def train_model(self, data: pd.DataFrame):
        """Train the ML model."""
        logger.info(f"Training {self.model_type} model with {len(data)} rows...")

        # Prepare features
        X = self._prepare_features(data)
        y = self._prepare_target(data)

        # Handle NaN values - forward fill features, then drop remaining NaNs
        X = X.ffill().bfill()  # Forward fill, then backward fill

        # Remove rows where target is NaN
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Final check: remove any remaining NaN rows
        final_valid_idx = ~X.isna().any(axis=1)
        X = X[final_valid_idx]
        y = y[final_valid_idx]

        logger.info(f"After cleaning: {len(X)} valid samples for training")

        if len(X) < self.min_training_samples:
            logger.warning(
                f"Not enough data for training: {len(X)} < {self.min_training_samples}. "
                f"Need at least {self.min_training_samples - len(X)} more samples."
            )
            return

        # Feature selection
        if self.use_feature_selection and self.selected_features is None:
            from sklearn.feature_selection import mutual_info_regression

            mi_scores = mutual_info_regression(X, y)
            top_indices = np.argsort(mi_scores)[-self.n_features:]
            self.selected_features = X.columns[top_indices].tolist()
            logger.info(f"Selected features: {self.selected_features}")

        if self.selected_features:
            X = X[self.selected_features]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)

        # Evaluate on training data
        train_score = self.model.score(X_scaled, y)
        logger.info(f"Model trained. R² score: {train_score:.4f}")

        self.days_since_retrain = 0

    def predict(self, data: pd.DataFrame) -> float:
        """Predict future return."""
        if self.model is None:
            return 0.0

        try:
            # Prepare features
            X = self._prepare_features(data)

            # Forward fill NaNs
            X = X.ffill().bfill()

            # Get latest observation
            X_latest = X.iloc[[-1]]

            # Handle missing features
            missing_features = set(self.selected_features or []) - set(X_latest.columns)
            if missing_features:
                for feat in missing_features:
                    X_latest[feat] = 0.0

            if self.selected_features:
                X_latest = X_latest[self.selected_features]

            # Fill any remaining NaN with 0
            X_latest = X_latest.fillna(0)

            # Scale
            X_scaled = self.scaler.transform(X_latest)

            # Predict
            prediction = self.model.predict(X_scaled)[0]

            return prediction

        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return 0.0

    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Process new data."""
        # Buffer data - keep more data to account for NaN removal in feature engineering
        self.data_buffer.append(data.iloc[-1])
        max_buffer_size = max(self.lookback * 3, 500)  # Keep at least 500 rows or 3x lookback
        if len(self.data_buffer) > max_buffer_size:
            self.data_buffer.pop(0)

        # Create DataFrame from buffer
        buffered_data = pd.DataFrame(self.data_buffer)

        # Need enough data for training (accounting for feature engineering warmup)
        min_buffer_size = max(self.lookback, 300)  # At least 300 rows or lookback period
        if len(buffered_data) < min_buffer_size:
            return

        # Retrain model if needed
        if self.model is None or self.days_since_retrain >= self.retrain_frequency:
            train_data = buffered_data  # Use all buffered data, not just last lookback
            self.train_model(train_data)

        # Make prediction
        pred_return = self.predict(buffered_data)

        # Get symbol name (handles both single and multi-symbol data)
        symbol = self.get_symbol(data)

        # Trading logic
        if pred_return > self.entry_threshold and not self.has_position(symbol):
            # Positive prediction - BUY
            self.buy(symbol, size=self.position_size)
            logger.info(f"ML BUY: Predicted return = {pred_return:.2%}")

        elif pred_return < -self.entry_threshold and self.has_position(symbol):
            # Negative prediction - SELL
            self.close_position(symbol)
            logger.info(f"ML SELL: Predicted return = {pred_return:.2%}")

        self.days_since_retrain += 1


class EnsembleMLStrategy(BaseStrategy):
    """
    Ensemble ML strategy combining multiple models.

    Uses voting/averaging from multiple ML models.
    """

    def __init__(
        self,
        model_types: List[str] = ["xgboost", "lightgbm", "random_forest"],
        lookback: int = 252,
        retrain_frequency: int = 20,
        prediction_horizon: int = 1,
        entry_threshold: float = 0.01,
        position_size: float = 0.1,
        ensemble_method: str = "average",
        **kwargs
    ):
        """
        Initialize ensemble ML strategy.

        Args:
            model_types: List of model types to ensemble
            lookback: Training window size
            retrain_frequency: Days between retraining
            prediction_horizon: Days ahead to predict
            entry_threshold: Minimum predicted return for entry
            position_size: Position size as fraction of portfolio
            ensemble_method: Ensemble method (average, weighted, voting)
        """
        super().__init__(**kwargs)
        self.model_types = model_types
        self.lookback = lookback
        self.retrain_frequency = retrain_frequency
        self.prediction_horizon = prediction_horizon
        self.entry_threshold = entry_threshold
        self.position_size = position_size
        self.ensemble_method = ensemble_method

        # Create individual ML strategies
        self.strategies = {}
        for model_type in model_types:
            self.strategies[model_type] = MLStrategy(
                model_type=model_type,
                lookback=lookback,
                retrain_frequency=retrain_frequency,
                prediction_horizon=prediction_horizon,
                entry_threshold=entry_threshold,
                position_size=position_size,
                min_training_samples=50,  # Lower threshold for ensemble
            )

        self.model_weights = {model: 1.0 / len(model_types) for model in model_types}

    def on_data(self, timestamp: pd.Timestamp, data: pd.DataFrame):
        """Process new data."""
        # Get predictions from all models
        predictions = {}

        for name, strategy in self.strategies.items():
            # Update strategy's buffer - keep more data for feature engineering
            strategy.data_buffer.append(data.iloc[-1])
            max_buffer_size = max(strategy.lookback * 3, 500)
            if len(strategy.data_buffer) > max_buffer_size:
                strategy.data_buffer.pop(0)

            buffered_data = pd.DataFrame(strategy.data_buffer)

            # Need enough data for training
            min_buffer_size = max(strategy.lookback, 300)
            if len(buffered_data) < min_buffer_size:
                continue

            # Retrain if needed
            if strategy.model is None or strategy.days_since_retrain >= strategy.retrain_frequency:
                train_data = buffered_data  # Use all buffered data
                strategy.train_model(train_data)

            # Get prediction
            if strategy.model is not None:
                pred = strategy.predict(buffered_data)
                predictions[name] = pred
                strategy.days_since_retrain += 1

        if len(predictions) == 0:
            return

        # Ensemble predictions
        if self.ensemble_method == "average":
            final_prediction = np.mean(list(predictions.values()))
        elif self.ensemble_method == "weighted":
            final_prediction = sum(
                pred * self.model_weights[name]
                for name, pred in predictions.items()
            )
        elif self.ensemble_method == "voting":
            # Vote: buy if majority predicts positive
            votes = [1 if pred > 0 else -1 for pred in predictions.values()]
            final_prediction = np.sign(sum(votes)) * abs(np.mean(list(predictions.values())))
        else:
            final_prediction = np.mean(list(predictions.values()))

        # Get symbol name (handles both single and multi-symbol data)
        symbol = self.get_symbol(data)

        # Trading logic
        if final_prediction > self.entry_threshold and not self.has_position(symbol):
            self.buy(symbol, size=self.position_size)
            logger.info(
                f"ENSEMBLE BUY: Prediction = {final_prediction:.2%} "
                f"(models: {predictions})"
            )

        elif final_prediction < -self.entry_threshold and self.has_position(symbol):
            self.close_position(symbol)
            logger.info(
                f"ENSEMBLE SELL: Prediction = {final_prediction:.2%} "
                f"(models: {predictions})"
            )
