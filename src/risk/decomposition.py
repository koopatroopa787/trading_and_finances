"""Risk decomposition and attribution."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import linalg


class RiskDecomposition:
    """Decompose portfolio risk into components."""

    @staticmethod
    def calculate_marginal_var(
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate marginal VaR for each position.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            Marginal VaR for each position
        """
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        if portfolio_vol == 0:
            return np.zeros_like(weights)

        # Marginal contribution to risk
        marginal_var = (cov_matrix @ weights) / portfolio_vol

        return marginal_var

    @staticmethod
    def calculate_component_var(
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate component VaR (contribution to total portfolio VaR).

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            Component VaR for each position
        """
        marginal_var = RiskDecomposition.calculate_marginal_var(weights, cov_matrix)
        component_var = weights * marginal_var

        return component_var

    @staticmethod
    def calculate_risk_contribution(
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate risk contribution metrics.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            Dict with marginal_var, component_var, contribution_pct
        """
        marginal_var = RiskDecomposition.calculate_marginal_var(weights, cov_matrix)
        component_var = RiskDecomposition.calculate_component_var(weights, cov_matrix)

        total_var = component_var.sum()
        contribution_pct = component_var / total_var if total_var != 0 else np.zeros_like(component_var)

        return {
            'marginal_var': marginal_var,
            'component_var': component_var,
            'contribution_pct': contribution_pct,
        }

    @staticmethod
    def decompose_by_factor(
        returns_df: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Decompose returns by risk factors.

        Args:
            returns_df: Asset returns
            factor_returns: Factor returns (e.g., market, size, value)

        Returns:
            Dict with factor_loadings, factor_returns, residuals
        """
        from sklearn.linear_model import LinearRegression

        assets = returns_df.columns
        factors = factor_returns.columns

        loadings = pd.DataFrame(index=assets, columns=factors)
        residuals = pd.DataFrame(index=returns_df.index, columns=assets)

        for asset in assets:
            y = returns_df[asset].values
            X = factor_returns.values

            # Remove NaN
            mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            y_clean = y[mask]
            X_clean = X[mask]

            if len(y_clean) < len(factors):
                continue

            # Fit regression
            model = LinearRegression()
            model.fit(X_clean, y_clean)

            loadings.loc[asset] = model.coef_

            # Calculate residuals
            predictions = model.predict(X)
            residuals[asset] = y - predictions

        return {
            'factor_loadings': loadings,
            'factor_returns': factor_returns,
            'residuals': residuals,
        }

    @staticmethod
    def calculate_concentration_risk(weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate concentration risk metrics.

        Args:
            weights: Portfolio weights (should sum to 1)

        Returns:
            Dict with herfindahl_index, effective_n, max_weight
        """
        # Herfindahl-Hirschman Index
        hhi = (weights ** 2).sum()

        # Effective number of positions
        effective_n = 1 / hhi if hhi > 0 else 0

        # Maximum weight
        max_weight = weights.max()

        return {
            'herfindahl_index': hhi,
            'effective_n': effective_n,
            'max_weight': max_weight,
        }

    @staticmethod
    def calculate_diversification_ratio(
        weights: np.ndarray,
        volatilities: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """
        Calculate diversification ratio.

        Ratio of weighted average volatility to portfolio volatility.
        Higher is better (more diversified).

        Args:
            weights: Portfolio weights
            volatilities: Individual asset volatilities
            cov_matrix: Covariance matrix

        Returns:
            Diversification ratio
        """
        weighted_vol = (weights * volatilities).sum()
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

        if portfolio_vol == 0:
            return 0.0

        div_ratio = weighted_vol / portfolio_vol

        return div_ratio
