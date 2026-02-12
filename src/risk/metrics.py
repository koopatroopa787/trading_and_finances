"""Risk metrics calculation."""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from scipy import stats
from loguru import logger


class RiskMetrics:
    """Calculate various risk metrics."""

    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Return series
            confidence: Confidence level
            method: Calculation method (historical, parametric, cornish_fisher)

        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0

        if method == "historical":
            # Historical VaR
            return -np.percentile(returns, (1 - confidence) * 100)

        elif method == "parametric":
            # Parametric VaR (assumes normal distribution)
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(confidence)
            return -(mean - z_score * std)

        elif method == "cornish_fisher":
            # Cornish-Fisher expansion (adjusts for skewness and kurtosis)
            mean = returns.mean()
            std = returns.std()
            skew = returns.skew()
            kurt = returns.kurt()

            z = stats.norm.ppf(confidence)
            z_cf = (z +
                    (z**2 - 1) * skew / 6 +
                    (z**3 - 3*z) * kurt / 24 -
                    (2*z**3 - 5*z) * (skew**2) / 36)

            return -(mean - z_cf * std)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0

        var = RiskMetrics.calculate_var(returns, confidence, method="historical")
        # CVaR is the average of returns worse than VaR
        cvar = -returns[returns <= -var].mean()

        return cvar if not np.isnan(cvar) else 0.0

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, any]:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Equity curve series

        Returns:
            Dict with max_drawdown, max_drawdown_pct, start, end, duration
        """
        if len(equity_curve) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'start': None,
                'end': None,
                'duration': 0,
            }

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = equity_curve - running_max
        drawdown_pct = drawdown / running_max

        # Find maximum drawdown
        max_dd_idx = drawdown_pct.idxmin()
        max_dd_pct = drawdown_pct.min()
        max_dd = drawdown.min()

        # Find start of drawdown (when we were at the peak)
        dd_start = running_max[:max_dd_idx].idxmax()

        # Find recovery (if any)
        after_max_dd = equity_curve[max_dd_idx:]
        peak_value = running_max[max_dd_idx]
        recovery = after_max_dd[after_max_dd >= peak_value]

        if len(recovery) > 0:
            dd_end = recovery.index[0]
        else:
            dd_end = equity_curve.index[-1]

        # Calculate duration
        duration = (dd_end - dd_start).days if hasattr(dd_end - dd_start, 'days') else 0

        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'start': dd_start,
            'end': dd_end,
            'duration': duration,
        }

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe Ratio.

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)

        return sharpe

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation).

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0

        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)

        return sortino

    @staticmethod
    def calculate_calmar_ratio(
        returns: pd.Series,
        equity_curve: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown).

        Args:
            returns: Return series
            equity_curve: Equity curve
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0

        annual_return = returns.mean() * periods_per_year
        max_dd = RiskMetrics.calculate_max_drawdown(equity_curve)

        if abs(max_dd['max_drawdown_pct']) < 1e-6:
            return 0.0

        calmar = annual_return / abs(max_dd['max_drawdown_pct'])

        return calmar

    @staticmethod
    def calculate_omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega Ratio.

        Args:
            returns: Return series
            threshold: Threshold return

        Returns:
            Omega ratio
        """
        if len(returns) == 0:
            return 0.0

        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns < threshold]

        gains = returns_above.sum()
        losses = returns_below.sum()

        if losses == 0:
            return np.inf if gains > 0 else 0.0

        omega = gains / losses

        return omega

    @staticmethod
    def calculate_greeks(
        position: Dict,
        underlying_price: float,
        volatility: float,
        risk_free_rate: float = 0.02,
        time_to_expiry: float = 0.25,
    ) -> Dict[str, float]:
        """
        Calculate option Greeks (Delta, Gamma, Vega, Theta, Rho).

        Args:
            position: Position dict with option details
            underlying_price: Current underlying price
            volatility: Implied volatility
            risk_free_rate: Risk-free rate
            time_to_expiry: Time to expiration in years

        Returns:
            Dict with Greeks
        """
        from scipy.stats import norm

        strike = position.get('strike', underlying_price)
        option_type = position.get('option_type', 'call')
        quantity = position.get('quantity', 1)

        # Black-Scholes d1 and d2
        d1 = (np.log(underlying_price / strike) +
              (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))

        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1) * quantity
        else:  # put
            delta = (norm.cdf(d1) - 1) * quantity

        # Gamma
        gamma = (norm.pdf(d1) / (underlying_price * volatility * np.sqrt(time_to_expiry))) * quantity

        # Vega
        vega = (underlying_price * norm.pdf(d1) * np.sqrt(time_to_expiry)) * quantity / 100

        # Theta
        if option_type == 'call':
            theta = (-(underlying_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry)) -
                     risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:  # put
            theta = (-(underlying_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry)) +
                     risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2))

        theta = theta * quantity / 365

        # Rho
        if option_type == 'call':
            rho = (strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:  # put
            rho = (-strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2))

        rho = rho * quantity / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
        }

    @staticmethod
    def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets."""
        return returns_df.corr()

    @staticmethod
    def calculate_beta(
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """Calculate beta relative to market."""
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 0.0

        covariance = asset_returns.cov(market_returns)
        market_variance = market_returns.var()

        if market_variance == 0:
            return 0.0

        beta = covariance / market_variance

        return beta
