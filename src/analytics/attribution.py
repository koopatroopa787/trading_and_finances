"""Attribution analysis for portfolio returns."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class AttributionAnalyzer:
    """Analyze return attribution by strategy, sector, or factor."""

    @staticmethod
    def strategy_attribution(
        strategy_returns: Dict[str, pd.Series],
        strategy_weights: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Calculate return attribution by strategy.

        Args:
            strategy_returns: Dict mapping strategy name to return series
            strategy_weights: Dict mapping strategy name to weight

        Returns:
            DataFrame with attribution results
        """
        # Align all return series
        returns_df = pd.DataFrame(strategy_returns)

        # Calculate weighted returns
        weighted_returns = {}
        for strategy, weight in strategy_weights.items():
            if strategy in returns_df.columns:
                weighted_returns[strategy] = returns_df[strategy] * weight

        weighted_df = pd.DataFrame(weighted_returns)

        # Calculate attribution
        attribution = {
            'strategy': list(weighted_df.columns),
            'weight': [strategy_weights.get(s, 0) for s in weighted_df.columns],
            'total_return': [weighted_df[s].sum() for s in weighted_df.columns],
            'avg_return': [weighted_df[s].mean() for s in weighted_df.columns],
            'contribution': [weighted_df[s].sum() for s in weighted_df.columns],
            'volatility': [returns_df[s].std() * np.sqrt(252) for s in weighted_df.columns],
            'sharpe': [
                (returns_df[s].mean() / returns_df[s].std() * np.sqrt(252))
                if returns_df[s].std() > 0 else 0
                for s in weighted_df.columns
            ],
        }

        attribution_df = pd.DataFrame(attribution)

        # Calculate contribution percentages
        total_contribution = attribution_df['contribution'].sum()
        if total_contribution != 0:
            attribution_df['contribution_pct'] = (
                attribution_df['contribution'] / total_contribution
            )
        else:
            attribution_df['contribution_pct'] = 0

        return attribution_df

    @staticmethod
    def sector_attribution(
        positions: pd.DataFrame,
        sector_mapping: Dict[str, str],
        returns_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate return attribution by sector.

        Args:
            positions: DataFrame with position data
            sector_mapping: Dict mapping symbol to sector
            returns_data: DataFrame with return data

        Returns:
            DataFrame with sector attribution
        """
        if 'symbol' not in positions.columns:
            return pd.DataFrame()

        # Add sector to positions
        positions['sector'] = positions['symbol'].map(sector_mapping)

        # Group by sector
        sector_groups = positions.groupby('sector')

        attribution = []

        for sector, group in sector_groups:
            # Calculate sector metrics
            symbols = group['symbol'].unique()

            sector_return = 0
            sector_weight = 0

            for symbol in symbols:
                if symbol in returns_data.columns:
                    symbol_returns = returns_data[symbol]
                    symbol_positions = group[group['symbol'] == symbol]

                    if len(symbol_positions) > 0:
                        avg_weight = symbol_positions['market_value'].mean()
                        sector_return += symbol_returns.mean() * avg_weight
                        sector_weight += avg_weight

            attribution.append({
                'sector': sector,
                'weight': sector_weight,
                'return': sector_return,
                'num_positions': len(symbols),
            })

        attribution_df = pd.DataFrame(attribution)

        # Normalize weights
        total_weight = attribution_df['weight'].sum()
        if total_weight > 0:
            attribution_df['weight_pct'] = attribution_df['weight'] / total_weight
        else:
            attribution_df['weight_pct'] = 0

        return attribution_df

    @staticmethod
    def factor_attribution(
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_loadings: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Calculate return attribution by factors.

        Args:
            returns: Portfolio return series
            factor_returns: DataFrame of factor returns
            factor_loadings: Factor loadings (if None, will be estimated)

        Returns:
            Dict with factor attribution
        """
        from sklearn.linear_model import LinearRegression

        # Align data
        aligned = pd.concat([returns, factor_returns], axis=1, join='inner')
        y = aligned.iloc[:, 0].values
        X = aligned.iloc[:, 1:].values

        # Estimate factor loadings if not provided
        if factor_loadings is None:
            model = LinearRegression()
            model.fit(X, y)
            loadings = pd.Series(model.coef_, index=factor_returns.columns)
        else:
            loadings = factor_loadings

        # Calculate factor contributions
        factor_contrib = {}

        for factor in factor_returns.columns:
            factor_ret = factor_returns[factor].mean()
            loading = loadings.get(factor, 0)
            contribution = loading * factor_ret

            factor_contrib[factor] = contribution

        # Residual (alpha)
        total_factor_contrib = sum(factor_contrib.values())
        total_return = returns.mean()
        factor_contrib['alpha'] = total_return - total_factor_contrib

        return factor_contrib

    @staticmethod
    def time_period_attribution(
        equity_curve: pd.Series,
        periods: List[str] = ['1M', '3M', '6M', '1Y', 'YTD', 'All']
    ) -> pd.DataFrame:
        """
        Calculate returns for different time periods.

        Args:
            equity_curve: Equity curve series
            periods: List of period labels

        Returns:
            DataFrame with period returns
        """
        current_date = equity_curve.index[-1]
        current_value = equity_curve.iloc[-1]

        period_returns = []

        for period in periods:
            if period == '1M':
                start_date = current_date - pd.DateOffset(months=1)
            elif period == '3M':
                start_date = current_date - pd.DateOffset(months=3)
            elif period == '6M':
                start_date = current_date - pd.DateOffset(months=6)
            elif period == '1Y':
                start_date = current_date - pd.DateOffset(years=1)
            elif period == 'YTD':
                start_date = pd.Timestamp(current_date.year, 1, 1)
            elif period == 'All':
                start_date = equity_curve.index[0]
            else:
                continue

            # Get value at start of period
            period_data = equity_curve[equity_curve.index >= start_date]

            if len(period_data) > 0:
                start_value = period_data.iloc[0]
                period_return = (current_value - start_value) / start_value

                period_returns.append({
                    'period': period,
                    'start_date': start_date,
                    'start_value': start_value,
                    'end_value': current_value,
                    'return': period_return,
                })

        return pd.DataFrame(period_returns)
