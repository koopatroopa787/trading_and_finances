"""Visualization utilities for performance analysis."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict


class PerformanceVisualizer:
    """Create visualizations for performance analysis."""

    def __init__(self, style: str = 'darkgrid'):
        """
        Initialize visualizer.

        Args:
            style: Seaborn style
        """
        sns.set_style(style)
        self.colors = px.colors.qualitative.Plotly

    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Equity Curve"
    ) -> go.Figure:
        """Plot equity curve with optional benchmark."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            name='Strategy',
            line=dict(color='#2E86AB', width=2),
        ))

        if benchmark is not None:
            # Normalize benchmark to same starting value
            benchmark_normalized = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]

            fig.add_trace(go.Scatter(
                x=benchmark_normalized.index,
                y=benchmark_normalized.values,
                name='Benchmark',
                line=dict(color='#A23B72', width=2, dash='dash'),
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_dark',
            height=500,
        )

        return fig

    def plot_drawdown(
        self,
        equity_curve: pd.Series,
        title: str = "Drawdown"
    ) -> go.Figure:
        """Plot drawdown chart."""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#E63946', width=2),
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_dark',
            height=400,
        )

        return fig

    def plot_monthly_returns(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap"
    ) -> go.Figure:
        """Plot monthly returns heatmap."""
        # Resample to monthly
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create pivot table
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values,
        })

        pivot = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')

        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns,
            y=month_names,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values * 100, 2),
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Return (%)"),
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title='Month',
            template='plotly_dark',
            height=500,
        )

        return fig

    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 60,
        title: str = "Rolling Metrics"
    ) -> go.Figure:
        """Plot rolling Sharpe ratio and volatility."""
        rolling_sharpe = (
            returns.rolling(window).mean() /
            returns.rolling(window).std() *
            np.sqrt(252)
        )

        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility (%)'),
            vertical_spacing=0.15,
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name='Sharpe Ratio',
                line=dict(color='#06FFA5', width=2),
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name='Volatility',
                line=dict(color='#FFBA08', width=2),
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)

        fig.update_layout(
            title_text=title,
            template='plotly_dark',
            height=600,
            showlegend=False,
        )

        return fig

    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution"
    ) -> go.Figure:
        """Plot returns distribution histogram."""
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns.values * 100,
            nbinsx=50,
            name='Returns',
            marker_color='#2E86AB',
        ))

        # Add normal distribution overlay
        from scipy import stats
        mu, sigma = returns.mean() * 100, returns.std() * 100
        x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        y = stats.norm.pdf(x, mu, sigma) * len(returns) * (returns.max() - returns.min()) * 100 / 50

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            name='Normal Distribution',
            line=dict(color='#E63946', width=2, dash='dash'),
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=400,
        )

        return fig

    def plot_efficient_frontier(
        self,
        frontier_data: pd.DataFrame,
        current_portfolio: Optional[Dict] = None,
        title: str = "Efficient Frontier"
    ) -> go.Figure:
        """Plot efficient frontier."""
        fig = go.Figure()

        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_data['volatility'] * 100,
            y=frontier_data['return'] * 100,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#06FFA5', width=3),
        ))

        # Current portfolio
        if current_portfolio:
            fig.add_trace(go.Scatter(
                x=[current_portfolio['volatility'] * 100],
                y=[current_portfolio['return'] * 100],
                mode='markers',
                name='Current Portfolio',
                marker=dict(color='#E63946', size=15, symbol='star'),
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Volatility (%)',
            yaxis_title='Return (%)',
            template='plotly_dark',
            height=500,
        )

        return fig

    def plot_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """Plot correlation matrix heatmap."""
        corr = returns_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
        ))

        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=600,
        )

        return fig

    def plot_position_allocation(
        self,
        positions: pd.DataFrame,
        title: str = "Position Allocation"
    ) -> go.Figure:
        """Plot position allocation pie chart."""
        fig = go.Figure(data=[go.Pie(
            labels=positions['symbol'],
            values=positions['market_value'],
            hole=.3,
            marker=dict(colors=self.colors),
        )])

        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=500,
        )

        return fig

    def plot_trade_analysis(
        self,
        trades: pd.DataFrame,
        title: str = "Trade Analysis"
    ) -> go.Figure:
        """Plot trade analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Trades Over Time',
                'Trade Size Distribution',
                'Profit/Loss by Trade',
                'Cumulative P&L'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
        )

        # Trades over time
        fig.add_trace(
            go.Scatter(
                x=trades['timestamp'],
                y=trades['price'],
                mode='markers',
                name='Trade Price',
                marker=dict(size=8, color=self.colors[0]),
            ),
            row=1, col=1
        )

        # Trade size distribution
        fig.add_trace(
            go.Histogram(
                x=trades['quantity'],
                name='Trade Size',
                marker_color=self.colors[1],
            ),
            row=1, col=2
        )

        # Calculate P&L per trade (simplified)
        trades['pnl'] = trades.apply(
            lambda x: x['quantity'] * x['price'] * (1 if x['side'] == 'sell' else -1),
            axis=1
        )

        # P&L by trade
        fig.add_trace(
            go.Bar(
                x=list(range(len(trades))),
                y=trades['pnl'],
                name='P&L',
                marker_color=[self.colors[2] if pnl > 0 else self.colors[3] for pnl in trades['pnl']],
            ),
            row=2, col=1
        )

        # Cumulative P&L
        cumulative_pnl = trades['pnl'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=trades['timestamp'],
                y=cumulative_pnl,
                name='Cumulative P&L',
                line=dict(color=self.colors[4], width=2),
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text=title,
            template='plotly_dark',
            height=800,
            showlegend=False,
        )

        return fig
