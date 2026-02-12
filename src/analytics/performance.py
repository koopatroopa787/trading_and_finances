"""Performance analysis and metrics calculation."""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
from loguru import logger

from src.risk.metrics import RiskMetrics


class PerformanceAnalyzer:
    """Analyze strategy performance."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        self.risk_metrics = RiskMetrics()

    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            equity_curve: Equity curve series
            trades: DataFrame of trades
            benchmark: Benchmark returns

        Returns:
            Dict of performance metrics
        """
        if len(equity_curve) == 0:
            return {}

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Basic metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        num_periods = len(equity_curve)

        # Annualized return
        years = num_periods / 252  # Assuming daily data
        if years > 0:
            cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
        else:
            cagr = 0.0

        # Volatility
        annual_vol = returns.std() * np.sqrt(252)

        # Risk-adjusted returns
        sharpe = self.risk_metrics.calculate_sharpe_ratio(
            returns, self.risk_free_rate, periods_per_year=252
        )
        sortino = self.risk_metrics.calculate_sortino_ratio(
            returns, self.risk_free_rate, periods_per_year=252
        )
        calmar = self.risk_metrics.calculate_calmar_ratio(returns, equity_curve, periods_per_year=252)
        omega = self.risk_metrics.calculate_omega_ratio(returns, threshold=0.0)

        # Risk metrics
        max_dd = self.risk_metrics.calculate_max_drawdown(equity_curve)
        var_95 = self.risk_metrics.calculate_var(returns, confidence=0.95, method="historical")
        var_99 = self.risk_metrics.calculate_var(returns, confidence=0.99, method="historical")
        cvar_95 = self.risk_metrics.calculate_cvar(returns, confidence=0.95)

        # Win rate and other trade statistics
        trade_stats = {}
        if trades is not None and len(trades) > 0:
            trade_stats = self._calculate_trade_statistics(trades)

        # Benchmark comparison
        benchmark_stats = {}
        if benchmark is not None:
            benchmark_stats = self._calculate_benchmark_stats(returns, benchmark)

        # Rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(equity_curve, returns)

        metrics = {
            # Returns
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': annual_vol,

            # Risk-adjusted returns
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega,

            # Risk metrics
            'max_drawdown': max_dd['max_drawdown'],
            'max_drawdown_pct': max_dd['max_drawdown_pct'],
            'max_drawdown_duration': max_dd['duration'],
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,

            # Distribution
            'skewness': returns.skew(),
            'kurtosis': returns.kurt(),

            # Trade statistics
            **trade_stats,

            # Benchmark
            **benchmark_stats,

            # Rolling metrics
            **rolling_metrics,

            # Additional info
            'num_periods': num_periods,
            'start_date': equity_curve.index[0],
            'end_date': equity_curve.index[-1],
        }

        return metrics

    def _calculate_trade_statistics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade-level statistics."""
        if 'effective_price' not in trades.columns:
            return {}

        # Calculate P&L per trade (simple approximation)
        buy_trades = trades[trades['side'] == 'buy']
        sell_trades = trades[trades['side'] == 'sell']

        num_trades = len(trades)
        num_winning = 0
        num_losing = 0
        total_profit = 0.0
        total_loss = 0.0

        # Match buy and sell trades (simplified)
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol].sort_values('timestamp')

            position = 0
            entry_price = 0
            trade_pnl = []

            for _, trade in symbol_trades.iterrows():
                if trade['side'] == 'buy':
                    if position > 0:
                        # Adding to position
                        entry_price = (entry_price * position + trade['effective_price'] * trade['quantity']) / (position + trade['quantity'])
                    else:
                        entry_price = trade['effective_price']
                    position += trade['quantity']
                else:  # sell
                    if position > 0:
                        pnl = (trade['effective_price'] - entry_price) * trade['quantity']
                        trade_pnl.append(pnl)

                        if pnl > 0:
                            num_winning += 1
                            total_profit += pnl
                        else:
                            num_losing += 1
                            total_loss += abs(pnl)

                    position -= trade['quantity']

        win_rate = num_winning / (num_winning + num_losing) if (num_winning + num_losing) > 0 else 0

        avg_win = total_profit / num_winning if num_winning > 0 else 0
        avg_loss = total_loss / num_losing if num_losing > 0 else 0

        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        return {
            'num_trades': num_trades,
            'num_winning_trades': num_winning,
            'num_losing_trades': num_losing,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss,
        }

    def _calculate_benchmark_stats(
        self,
        returns: pd.Series,
        benchmark: pd.Series
    ) -> Dict[str, Any]:
        """Calculate statistics relative to benchmark."""
        # Align returns
        aligned = pd.DataFrame({'strategy': returns, 'benchmark': benchmark}).dropna()

        if len(aligned) == 0:
            return {}

        strategy_ret = aligned['strategy']
        benchmark_ret = aligned['benchmark']

        # Alpha and Beta
        beta = self.risk_metrics.calculate_beta(strategy_ret, benchmark_ret)
        alpha = (strategy_ret.mean() - beta * benchmark_ret.mean()) * 252

        # Tracking error
        tracking_error = (strategy_ret - benchmark_ret).std() * np.sqrt(252)

        # Information ratio
        information_ratio = (
            (strategy_ret.mean() - benchmark_ret.mean()) /
            (strategy_ret - benchmark_ret).std() * np.sqrt(252)
            if (strategy_ret - benchmark_ret).std() > 0 else 0
        )

        # Correlation
        correlation = strategy_ret.corr(benchmark_ret)

        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'correlation_with_benchmark': correlation,
        }

    def _calculate_rolling_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
        window: int = 60
    ) -> Dict[str, Any]:
        """Calculate rolling metrics."""
        if len(returns) < window:
            return {}

        # Rolling Sharpe
        rolling_sharpe = (
            returns.rolling(window).mean() /
            returns.rolling(window).std() *
            np.sqrt(252)
        ).dropna()

        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)

        return {
            'avg_rolling_sharpe_60d': rolling_sharpe.mean(),
            'current_rolling_sharpe_60d': rolling_sharpe.iloc[-1] if len(rolling_sharpe) > 0 else 0,
            'avg_rolling_vol_60d': rolling_vol.mean(),
            'current_rolling_vol_60d': rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0,
        }

    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate text report from metrics."""
        report = "=" * 60 + "\n"
        report += "PERFORMANCE REPORT\n"
        report += "=" * 60 + "\n\n"

        report += "RETURNS:\n"
        report += f"  Total Return: {metrics.get('total_return', 0):.2%}\n"
        report += f"  CAGR: {metrics.get('cagr', 0):.2%}\n"
        report += f"  Annual Volatility: {metrics.get('annual_volatility', 0):.2%}\n\n"

        report += "RISK-ADJUSTED RETURNS:\n"
        report += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
        report += f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}\n"
        report += f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}\n"
        report += f"  Omega Ratio: {metrics.get('omega_ratio', 0):.3f}\n\n"

        report += "RISK METRICS:\n"
        report += f"  Maximum Drawdown: {metrics.get('max_drawdown_pct', 0):.2%}\n"
        report += f"  VaR (95%): {metrics.get('var_95', 0):.2%}\n"
        report += f"  CVaR (95%): {metrics.get('cvar_95', 0):.2%}\n\n"

        if 'num_trades' in metrics:
            report += "TRADE STATISTICS:\n"
            report += f"  Number of Trades: {metrics.get('num_trades', 0)}\n"
            report += f"  Win Rate: {metrics.get('win_rate', 0):.2%}\n"
            report += f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            report += f"  Average Win: ${metrics.get('avg_win', 0):.2f}\n"
            report += f"  Average Loss: ${metrics.get('avg_loss', 0):.2f}\n\n"

        if 'beta' in metrics:
            report += "BENCHMARK COMPARISON:\n"
            report += f"  Alpha: {metrics.get('alpha', 0):.2%}\n"
            report += f"  Beta: {metrics.get('beta', 0):.3f}\n"
            report += f"  Information Ratio: {metrics.get('information_ratio', 0):.3f}\n"
            report += f"  Correlation: {metrics.get('correlation_with_benchmark', 0):.3f}\n\n"

        report += "=" * 60 + "\n"

        return report
