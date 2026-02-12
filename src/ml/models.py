"""ML model training and selection utilities."""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from loguru import logger


class FeatureSelector:
    """Feature selection utilities."""

    @staticmethod
    def select_by_importance(
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "mutual_info",
        k: int = 20,
    ) -> List[str]:
        """
        Select top k features by importance.

        Args:
            X: Feature DataFrame
            y: Target series
            method: Selection method (mutual_info, f_score, correlation)
            k: Number of features to select

        Returns:
            List of selected feature names
        """
        # Remove NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        if len(X_clean) < k:
            logger.warning(f"Not enough samples ({len(X_clean)}) for selection")
            return X.columns.tolist()[:k]

        # Calculate scores
        if method == "mutual_info":
            scores = mutual_info_regression(X_clean, y_clean, random_state=42)
        elif method == "f_score":
            scores, _ = f_regression(X_clean, y_clean)
        elif method == "correlation":
            scores = np.abs(X_clean.corrwith(y_clean))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Get top k
        top_indices = np.argsort(scores)[-k:]
        selected = X.columns[top_indices].tolist()

        logger.info(f"Selected {k} features using {method}")
        return selected

    @staticmethod
    def select_by_model(
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "random_forest",
        k: int = 20,
    ) -> List[str]:
        """
        Select features using model-based importance.

        Args:
            X: Feature DataFrame
            y: Target series
            model_type: Model type (random_forest, gradient_boosting)
            k: Number of features to select

        Returns:
            List of selected feature names
        """
        # Remove NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        # Train model
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_type}")

        model.fit(X_clean, y_clean)

        # Get feature importances
        importances = model.feature_importances_

        # Select top k
        top_indices = np.argsort(importances)[-k:]
        selected = X.columns[top_indices].tolist()

        logger.info(f"Selected {k} features using {model_type}")
        return selected


class ModelTrainer:
    """Model training with cross-validation."""

    @staticmethod
    def train_with_cv(
        X: pd.DataFrame,
        y: pd.Series,
        model,
        n_splits: int = 5,
        purge_gap: int = 5,
    ) -> Tuple[object, Dict[str, float]]:
        """
        Train model with time series cross-validation.

        Args:
            X: Features
            y: Target
            model: Model instance
            n_splits: Number of CV splits
            purge_gap: Gap between train and test to prevent leakage

        Returns:
            Tuple of (fitted model, cv_scores dict)
        """
        # Remove NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_clean, y_clean,
            cv=tscv,
            scoring='r2',
            n_jobs=-1,
        )

        # Train on full data
        model.fit(X_clean, y_clean)

        scores = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max(),
        }

        logger.info(
            f"CV R² = {scores['cv_mean']:.4f} (+/- {scores['cv_std']:.4f})"
        )

        return model, scores

    @staticmethod
    def walk_forward_optimize(
        X: pd.DataFrame,
        y: pd.Series,
        model_class,
        train_size: int = 252,
        test_size: int = 60,
        step_size: int = 20,
    ) -> List[Dict]:
        """
        Walk-forward optimization.

        Args:
            X: Features
            y: Target
            model_class: Model class to instantiate
            train_size: Training window size
            test_size: Test window size
            step_size: Step size for rolling window

        Returns:
            List of results for each window
        """
        results = []

        for i in range(0, len(X) - train_size - test_size, step_size):
            # Split data
            train_start = i
            train_end = i + train_size
            test_start = train_end
            test_end = test_start + test_size

            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]

            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            # Remove NaN
            train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
            test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())

            X_train = X_train[train_valid]
            y_train = y_train[train_valid]
            X_test = X_test[test_valid]
            y_test = y_test[test_valid]

            if len(X_train) < 100 or len(X_test) < 10:
                continue

            # Train model
            model = model_class()
            model.fit(X_train, y_train)

            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            results.append({
                'window': i // step_size,
                'train_score': train_score,
                'test_score': test_score,
                'train_size': len(X_train),
                'test_size': len(X_test),
            })

        logger.info(f"Walk-forward completed: {len(results)} windows")

        return results
