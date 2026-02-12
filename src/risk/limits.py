"""Risk limits and controls."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class RiskLimits:
    """Monitor and enforce risk limits."""

    def __init__(
        self,
        max_position_size: float = 0.2,
        max_sector_exposure: float = 0.4,
        max_leverage: float = 1.0,
        daily_loss_limit: float = 0.05,
        max_drawdown_limit: float = 0.25,
        max_correlation: float = 0.7,
        var_limit: Optional[float] = None,
    ):
        """
        Initialize risk limits.

        Args:
            max_position_size: Maximum position as fraction of portfolio
            max_sector_exposure: Maximum sector exposure
            max_leverage: Maximum portfolio leverage
            daily_loss_limit: Maximum daily loss
            max_drawdown_limit: Maximum drawdown before stopping
            max_correlation: Maximum correlation between positions
            var_limit: Maximum VaR
        """
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_leverage = max_leverage
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.max_correlation = max_correlation
        self.var_limit = var_limit

        self.violations: List[Dict] = []

    def check_position_size(
        self,
        position_value: float,
        portfolio_value: float
    ) -> bool:
        """Check if position size exceeds limit."""
        if portfolio_value == 0:
            return False

        position_pct = position_value / portfolio_value

        if position_pct > self.max_position_size:
            self.violations.append({
                'type': 'position_size',
                'value': position_pct,
                'limit': self.max_position_size,
            })
            logger.warning(
                f"Position size limit violated: {position_pct:.2%} > {self.max_position_size:.2%}"
            )
            return False

        return True

    def check_leverage(
        self,
        positions_value: float,
        portfolio_value: float
    ) -> bool:
        """Check if leverage exceeds limit."""
        if portfolio_value == 0:
            return False

        leverage = positions_value / portfolio_value

        if leverage > self.max_leverage:
            self.violations.append({
                'type': 'leverage',
                'value': leverage,
                'limit': self.max_leverage,
            })
            logger.warning(
                f"Leverage limit violated: {leverage:.2f}x > {self.max_leverage:.2f}x"
            )
            return False

        return True

    def check_daily_loss(
        self,
        current_value: float,
        start_of_day_value: float
    ) -> bool:
        """Check if daily loss exceeds limit."""
        if start_of_day_value == 0:
            return True

        daily_return = (current_value - start_of_day_value) / start_of_day_value

        if daily_return < -self.daily_loss_limit:
            self.violations.append({
                'type': 'daily_loss',
                'value': daily_return,
                'limit': -self.daily_loss_limit,
            })
            logger.error(
                f"Daily loss limit violated: {daily_return:.2%} < {-self.daily_loss_limit:.2%}"
            )
            return False

        return True

    def check_drawdown(
        self,
        current_value: float,
        peak_value: float
    ) -> bool:
        """Check if drawdown exceeds limit."""
        if peak_value == 0:
            return True

        drawdown = (current_value - peak_value) / peak_value

        if drawdown < -self.max_drawdown_limit:
            self.violations.append({
                'type': 'max_drawdown',
                'value': drawdown,
                'limit': -self.max_drawdown_limit,
            })
            logger.error(
                f"Maximum drawdown limit violated: {drawdown:.2%} < {-self.max_drawdown_limit:.2%}"
            )
            return False

        return True

    def check_correlation(
        self,
        returns_df: pd.DataFrame,
        new_asset: str,
        existing_assets: List[str]
    ) -> bool:
        """Check if adding new asset violates correlation limit."""
        if new_asset not in returns_df.columns:
            return True

        for asset in existing_assets:
            if asset not in returns_df.columns:
                continue

            corr = returns_df[new_asset].corr(returns_df[asset])

            if abs(corr) > self.max_correlation:
                self.violations.append({
                    'type': 'correlation',
                    'assets': [new_asset, asset],
                    'value': corr,
                    'limit': self.max_correlation,
                })
                logger.warning(
                    f"Correlation limit violated: {new_asset}-{asset} = {corr:.2f} > {self.max_correlation:.2f}"
                )
                return False

        return True

    def get_violations(self) -> List[Dict]:
        """Get all risk limit violations."""
        return self.violations

    def reset_violations(self):
        """Clear violation history."""
        self.violations = []
