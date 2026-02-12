"""Portfolio optimization algorithms."""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from loguru import logger


class MarkowitzOptimizer:
    """
    Mean-Variance (Markowitz) portfolio optimization.

    Finds the portfolio that maximizes Sharpe ratio or achieves
    target return/volatility with minimum risk.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        max_weight: float = 0.3,
        min_weight: float = 0.0,
        allow_short: bool = False,
    ):
        """
        Initialize Markowitz optimizer.

        Args:
            risk_free_rate: Annual risk-free rate
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            allow_short: Allow short positions
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight if not allow_short else -max_weight
        self.allow_short = allow_short

    def optimize(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        objective: str = "max_sharpe",
    ) -> pd.Series:
        """
        Optimize portfolio weights.

        Args:
            returns: DataFrame of asset returns
            target_return: Target return (if objective is min_vol)
            target_volatility: Target volatility (if objective is max_return)
            objective: Optimization objective (max_sharpe, min_vol, max_return)

        Returns:
            Series of optimal weights
        """
        # Calculate statistics
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized

        n_assets = len(mean_returns)

        # Define optimization variables
        weights = cp.Variable(n_assets)

        # Portfolio return and risk
        port_return = mean_returns.values @ weights
        port_risk = cp.quad_form(weights, cov_matrix.values)

        # Constraints
        constraints = [cp.sum(weights) == 1]

        if not self.allow_short:
            constraints.append(weights >= self.min_weight)

        constraints.append(weights <= self.max_weight)

        # Add target constraints
        if target_return is not None:
            constraints.append(port_return >= target_return)

        if target_volatility is not None:
            constraints.append(port_risk <= target_volatility ** 2)

        # Define objective
        if objective == "max_sharpe":
            # Maximize Sharpe ratio
            # Equivalent to minimizing negative Sharpe
            objective_func = cp.Minimize(-port_return / cp.sqrt(port_risk))

        elif objective == "min_vol":
            # Minimum volatility
            objective_func = cp.Minimize(port_risk)

        elif objective == "max_return":
            # Maximum return
            objective_func = cp.Maximize(port_return)

        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Solve
        problem = cp.Problem(objective_func, constraints)

        try:
            problem.solve()

            if weights.value is None:
                logger.error("Optimization failed")
                # Return equal weights as fallback
                return pd.Series(np.ones(n_assets) / n_assets, index=returns.columns)

            optimal_weights = pd.Series(weights.value, index=returns.columns)

            # Clean small weights
            optimal_weights[optimal_weights.abs() < 1e-4] = 0
            optimal_weights = optimal_weights / optimal_weights.sum()

            logger.info(f"Optimization successful: {objective}")
            return optimal_weights

        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return pd.Series(np.ones(n_assets) / n_assets, index=returns.columns)

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.

        Args:
            returns: DataFrame of asset returns
            n_points: Number of points on frontier

        Returns:
            DataFrame with return, volatility, and weights
        """
        mean_returns = returns.mean() * 252
        min_return = mean_returns.min()
        max_return = mean_returns.max()

        target_returns = np.linspace(min_return, max_return, n_points)

        frontier = []

        for target_ret in target_returns:
            weights = self.optimize(
                returns,
                target_return=target_ret,
                objective="min_vol"
            )

            port_return = (mean_returns * weights).sum()
            port_vol = np.sqrt(weights @ returns.cov() @ weights * 252)

            frontier.append({
                'return': port_return,
                'volatility': port_vol,
                'sharpe': (port_return - self.risk_free_rate) / port_vol,
                **{f'weight_{asset}': weight for asset, weight in weights.items()}
            })

        return pd.DataFrame(frontier)


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization.

    Combines market equilibrium with investor views.
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize Black-Litterman optimizer.

        Args:
            risk_aversion: Risk aversion parameter
            tau: Uncertainty in prior estimate
            risk_free_rate: Risk-free rate
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate

    def optimize(
        self,
        returns: pd.DataFrame,
        market_caps: pd.Series,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Calculate Black-Litterman optimal weights.

        Args:
            returns: Historical returns
            market_caps: Market capitalizations
            views: Dict of asset -> expected return views
            view_confidences: Dict of asset -> confidence (0-1)

        Returns:
            Optimal weights
        """
        # Calculate inputs
        cov_matrix = returns.cov() * 252
        assets = returns.columns

        # Market equilibrium weights (proportional to market cap)
        market_weights = market_caps / market_caps.sum()

        # Implied equilibrium returns (reverse optimization)
        implied_returns = self.risk_aversion * cov_matrix @ market_weights

        # If no views, return market weights
        if views is None or len(views) == 0:
            return market_weights

        # Incorporate views
        P = []  # Pick matrix
        Q = []  # View returns

        for asset, view_return in views.items():
            if asset not in assets:
                continue

            pick_vector = np.zeros(len(assets))
            pick_vector[assets.get_loc(asset)] = 1
            P.append(pick_vector)
            Q.append(view_return)

        P = np.array(P)
        Q = np.array(Q)

        # View uncertainty (Omega)
        if view_confidences is None:
            # Default: proportional to variance
            omega = np.diag(np.diag(P @ (self.tau * cov_matrix) @ P.T))
        else:
            # Use confidence levels
            omega_diag = []
            for asset in views.keys():
                if asset not in assets:
                    continue
                conf = view_confidences.get(asset, 0.5)
                asset_var = cov_matrix.loc[asset, asset]
                omega_diag.append(self.tau * asset_var / conf)
            omega = np.diag(omega_diag)

        # Black-Litterman formula
        tau_cov = self.tau * cov_matrix
        M_inv = np.linalg.inv(tau_cov)
        P_T = P.T

        # Posterior returns
        bl_returns = implied_returns + (
            tau_cov @ P_T @ np.linalg.inv(P @ tau_cov @ P_T + omega) @
            (Q - P @ implied_returns)
        )

        # Optimal weights
        bl_weights = (
            np.linalg.inv(self.risk_aversion * cov_matrix) @ bl_returns
        )

        # Normalize
        bl_weights = bl_weights / bl_weights.sum()
        bl_weights = pd.Series(bl_weights, index=assets)

        # Constrain weights
        bl_weights = bl_weights.clip(0, 0.3)
        bl_weights = bl_weights / bl_weights.sum()

        return bl_weights


class HRPOptimizer:
    """
    Hierarchical Risk Parity (HRP) portfolio optimization.

    Uses hierarchical clustering to build diversified portfolios.
    """

    def __init__(self):
        """Initialize HRP optimizer."""
        pass

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate HRP optimal weights.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Optimal weights
        """
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)

        # Hierarchical clustering
        links = linkage(squareform(distance_matrix.values), method='single')

        # Get quasi-diagonal matrix
        sorted_indices = self._get_quasi_diag(links)
        sorted_indices = [corr_matrix.index[i] for i in sorted_indices]

        # Calculate weights
        weights = self._get_recursive_bisection(
            returns[sorted_indices],
            corr_matrix.loc[sorted_indices, sorted_indices]
        )

        return weights

    def _get_quasi_diag(self, links):
        """Get quasi-diagonal ordering from linkage."""
        n = links.shape[0] + 1
        sorted_indices = []

        def recursive_sort(node):
            if node < n:
                sorted_indices.append(node)
            else:
                left = int(links[node - n, 0])
                right = int(links[node - n, 1])
                recursive_sort(left)
                recursive_sort(right)

        recursive_sort(2 * n - 2)
        return sorted_indices

    def _get_recursive_bisection(self, returns, corr):
        """Recursive bisection to allocate weights."""
        weights = pd.Series(1.0, index=returns.columns)
        clustered = [returns.columns.tolist()]

        while len(clustered) > 0:
            clustered = [
                cluster[start:end]
                for cluster in clustered
                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                if len(cluster) > 1
            ]

            for i in range(0, len(clustered), 2):
                cluster1 = clustered[i]
                cluster2 = clustered[i + 1] if i + 1 < len(clustered) else []

                if len(cluster2) == 0:
                    continue

                # Calculate cluster variance
                var1 = self._get_cluster_var(returns[cluster1], corr.loc[cluster1, cluster1])
                var2 = self._get_cluster_var(returns[cluster2], corr.loc[cluster2, cluster2])

                # Allocate weight
                alpha = 1 - var1 / (var1 + var2)

                weights[cluster1] *= alpha
                weights[cluster2] *= (1 - alpha)

        return weights

    def _get_cluster_var(self, returns, corr):
        """Calculate cluster variance."""
        cov = returns.cov()
        weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
        cluster_var = weights @ cov @ weights
        return cluster_var


class RiskParityOptimizer:
    """Risk Parity portfolio optimization."""

    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Initialize risk parity optimizer.

        Args:
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate risk parity weights.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Optimal weights
        """
        cov_matrix = returns.cov() * 252

        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)

        # Risk contribution for each asset should be equal
        # RC_i = w_i * (Cov @ w)_i

        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        risk_contrib = cp.multiply(weights, cov_matrix.values @ weights)

        # Minimize difference in risk contributions
        objective = cp.Minimize(
            cp.sum_squares(risk_contrib - portfolio_variance / n_assets)
        )

        constraints = [
            weights >= 0,
            cp.sum(weights) == 1,
        ]

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve()

            if weights.value is None:
                logger.warning("Risk parity optimization failed, using equal weights")
                return pd.Series(np.ones(n_assets) / n_assets, index=returns.columns)

            optimal_weights = pd.Series(weights.value, index=returns.columns)
            optimal_weights = optimal_weights / optimal_weights.sum()

            return optimal_weights

        except Exception as e:
            logger.error(f"Risk parity optimization error: {e}")
            return pd.Series(np.ones(n_assets) / n_assets, index=returns.columns)
