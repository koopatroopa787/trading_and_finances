"""Streamlit dashboard for backtesting results."""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory (helps with relative imports)
os.chdir(str(project_root))

try:
    from src.data.loaders import YahooFinanceLoader
except ImportError as e:
    st.error(f"""
    **Import Error**: Could not import required modules.

    Please install the package first:
    ```bash
    cd {project_root}
    pip install -e .
    ```

    Or ensure you're running from the project root:
    ```bash
    cd {project_root}
    streamlit run dashboard/app.py
    ```

    Error: {e}
    """)
    st.stop()

from src.data.loaders import YahooFinanceLoader
from src.data.processors import DataProcessor
from src.data.features import FeatureEngineer
from src.strategy.implementations.momentum import MomentumStrategy, TrendFollowingStrategy
from src.strategy.implementations.mean_reversion import MeanReversionStrategy
from src.strategy.implementations.ml_strategy import MLStrategy, EnsembleMLStrategy
from src.execution.simulator import ExecutionSimulator
from src.analytics.performance import PerformanceAnalyzer
from src.analytics.visualization import PerformanceVisualizer
from src.optimization.portfolio import MarkowitzOptimizer, HRPOptimizer
from src.risk.metrics import RiskMetrics

# Page config
st.set_page_config(
    page_title="Quantitative Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #06FFA5;
    }
    h2, h3 {
        color: #2E86AB;
    }
    .metric-container {
        background-color: #1a1d29;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #06FFA5;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Multi-Strategy Backtesting Engine")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Data Selection
    st.subheader("📊 Data")
    symbols = st.multiselect(
        "Select Symbols",
        ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'SPY', 'QQQ'],
        default=['AAPL', 'GOOGL', 'MSFT']
    )

    start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
    end_date = st.date_input("End Date", pd.to_datetime('2023-12-31'))

    # Strategy Selection
    st.subheader("🎯 Strategy")
    strategy_type = st.selectbox(
        "Select Strategy",
        ['Momentum', 'Mean Reversion', 'Trend Following', 'ML-Based', 'Multi-Strategy']
    )

    # Execution Parameters
    st.subheader("💰 Execution")
    initial_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
    commission_rate = st.number_input("Commission Rate (%)", value=0.1, step=0.01) / 100

    # Run Backtest Button
    run_backtest = st.button("🚀 Run Backtest", type="primary")

# Main content
if run_backtest and len(symbols) > 0:
    with st.spinner("Loading data..."):
        # Load data
        loader = YahooFinanceLoader()
        try:
            data = loader.load(
                symbols,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )

            # Process data
            processor = DataProcessor()
            data = processor.clean_data(data)

            st.success(f"✅ Loaded {len(data)} rows of data for {len(symbols)} symbols")

        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    with st.spinner("Running backtest..."):
        # Create strategy
        if strategy_type == 'Momentum':
            strategy = MomentumStrategy(
                lookback_period=20,
                momentum_threshold=0.05,
                position_size=0.1
            )
        elif strategy_type == 'Mean Reversion':
            strategy = MeanReversionStrategy(
                lookback_period=20,
                entry_threshold=2.0,
                exit_threshold=0.5,
                position_size=0.1
            )
        elif strategy_type == 'Trend Following':
            strategy = TrendFollowingStrategy(
                fast_period=20,
                slow_period=50,
                position_size=0.1
            )
        elif strategy_type == 'ML-Based':
            strategy = MLStrategy(
                model_type='xgboost',
                lookback=252,
                retrain_frequency=20,
                position_size=0.1
            )
        elif strategy_type == 'Multi-Strategy':
            strategy = EnsembleMLStrategy(
                model_types=['xgboost', 'lightgbm', 'random_forest'],
                lookback=252,
                retrain_frequency=20,
                position_size=0.1,
                ensemble_method='average'
            )

        # Create executor
        executor = ExecutionSimulator(
            initial_capital=initial_capital,
            use_order_book=True,
            partial_fills=True,
        )

        # Run backtest
        try:
            # For single symbol, pass data directly
            if len(symbols) == 1:
                result = executor.run(strategy, data)
            else:
                # For first symbol (multi-symbol support to be added)
                symbol_data = data[symbols[0]] if hasattr(data.columns, 'levels') else data
                result = executor.run(strategy, symbol_data)

            st.success("✅ Backtest completed!")

        except Exception as e:
            st.error(f"Error running backtest: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

    # Display Results
    st.markdown("---")
    st.header("📊 Results")

    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Return",
            f"{result.metrics.get('total_return', 0):.2%}",
            delta=f"{result.metrics.get('cagr', 0):.2%} CAGR"
        )

    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{result.metrics.get('sharpe_ratio', 0):.2f}",
            delta="Risk-Adjusted"
        )

    with col3:
        st.metric(
            "Max Drawdown",
            f"{result.metrics.get('max_drawdown_pct', 0):.2%}",
            delta=f"{result.metrics.get('max_drawdown_duration', 0)} days",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            "Win Rate",
            f"{result.metrics.get('win_rate', 0):.2%}",
            delta=f"{result.metrics.get('num_trades', 0)} trades"
        )

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance",
        "📉 Risk Analysis",
        "💼 Positions",
        "📋 Trades",
        "📊 Statistics"
    ])

    visualizer = PerformanceVisualizer()

    with tab1:
        st.subheader("Equity Curve")
        if len(result.equity_curve) > 0:
            fig = visualizer.plot_equity_curve(result.equity_curve)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Drawdown")
            if len(result.equity_curve) > 0:
                fig = visualizer.plot_drawdown(result.equity_curve)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Returns Distribution")
            if len(result.equity_curve) > 0:
                returns = result.equity_curve.pct_change().dropna()
                fig = visualizer.plot_returns_distribution(returns)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Risk Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Value at Risk**")
            st.metric("VaR (95%)", f"{result.metrics.get('var_95', 0):.2%}")
            st.metric("VaR (99%)", f"{result.metrics.get('var_99', 0):.2%}")
            st.metric("CVaR (95%)", f"{result.metrics.get('cvar_95', 0):.2%}")

        with col2:
            st.markdown("**Risk-Adjusted Returns**")
            st.metric("Sharpe Ratio", f"{result.metrics.get('sharpe_ratio', 0):.2f}")
            st.metric("Sortino Ratio", f"{result.metrics.get('sortino_ratio', 0):.2f}")
            st.metric("Calmar Ratio", f"{result.metrics.get('calmar_ratio', 0):.2f}")

        with col3:
            st.markdown("**Distribution**")
            st.metric("Skewness", f"{result.metrics.get('skewness', 0):.3f}")
            st.metric("Kurtosis", f"{result.metrics.get('kurtosis', 0):.3f}")
            st.metric("Volatility", f"{result.metrics.get('annual_volatility', 0):.2%}")

        st.subheader("Rolling Metrics")
        if len(result.equity_curve) > 0:
            returns = result.equity_curve.pct_change().dropna()
            if len(returns) > 60:
                fig = visualizer.plot_rolling_metrics(returns, window=60)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Current Positions")
        if len(result.positions) > 0:
            latest_positions = result.positions[
                result.positions['timestamp'] == result.positions['timestamp'].max()
            ]

            if len(latest_positions) > 0:
                st.dataframe(
                    latest_positions[[
                        'symbol', 'quantity', 'entry_price', 'current_price',
                        'market_value', 'unrealized_pnl', 'unrealized_pnl_pct'
                    ]].style.format({
                        'entry_price': '${:.2f}',
                        'current_price': '${:.2f}',
                        'market_value': '${:.2f}',
                        'unrealized_pnl': '${:.2f}',
                        'unrealized_pnl_pct': '{:.2%}',
                    }),
                    use_container_width=True
                )

                # Position allocation
                if len(latest_positions) > 1:
                    fig = visualizer.plot_position_allocation(latest_positions)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No open positions")
        else:
            st.info("No position data available")

    with tab4:
        st.subheader("Trade History")
        if len(result.trades) > 0:
            st.dataframe(
                result.trades.style.format({
                    'price': '${:.2f}',
                    'effective_price': '${:.2f}',
                    'commission': '${:.2f}',
                    'slippage': '${:.2f}',
                    'total_cost': '${:.2f}',
                }),
                use_container_width=True
            )

            # Trade analysis
            st.subheader("Trade Analysis")
            if len(result.trades) > 1:
                fig = visualizer.plot_trade_analysis(result.trades)
                st.plotly_chart(fig, use_container_width=True)

            # Download trades
            csv = result.trades.to_csv(index=False)
            st.download_button(
                label="📥 Download Trades CSV",
                data=csv,
                file_name="trades.csv",
                mime="text/csv",
            )
        else:
            st.info("No trades executed")

    with tab5:
        st.subheader("Detailed Statistics")

        # Create comprehensive stats table
        stats_data = []

        # Returns
        stats_data.extend([
            ("Returns", "Total Return", f"{result.metrics.get('total_return', 0):.2%}"),
            ("Returns", "CAGR", f"{result.metrics.get('cagr', 0):.2%}"),
            ("Returns", "Annual Volatility", f"{result.metrics.get('annual_volatility', 0):.2%}"),
        ])

        # Risk-Adjusted
        stats_data.extend([
            ("Risk-Adjusted", "Sharpe Ratio", f"{result.metrics.get('sharpe_ratio', 0):.3f}"),
            ("Risk-Adjusted", "Sortino Ratio", f"{result.metrics.get('sortino_ratio', 0):.3f}"),
            ("Risk-Adjusted", "Calmar Ratio", f"{result.metrics.get('calmar_ratio', 0):.3f}"),
            ("Risk-Adjusted", "Omega Ratio", f"{result.metrics.get('omega_ratio', 0):.3f}"),
        ])

        # Risk
        stats_data.extend([
            ("Risk", "Maximum Drawdown", f"{result.metrics.get('max_drawdown_pct', 0):.2%}"),
            ("Risk", "Drawdown Duration", f"{result.metrics.get('max_drawdown_duration', 0)} days"),
            ("Risk", "VaR (95%)", f"{result.metrics.get('var_95', 0):.2%}"),
            ("Risk", "CVaR (95%)", f"{result.metrics.get('cvar_95', 0):.2%}"),
        ])

        # Trading
        if 'num_trades' in result.metrics:
            stats_data.extend([
                ("Trading", "Number of Trades", f"{result.metrics.get('num_trades', 0)}"),
                ("Trading", "Win Rate", f"{result.metrics.get('win_rate', 0):.2%}"),
                ("Trading", "Profit Factor", f"{result.metrics.get('profit_factor', 0):.2f}"),
            ])

        stats_df = pd.DataFrame(stats_data, columns=["Category", "Metric", "Value"])

        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
        )

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the Multi-Strategy Backtesting Engine! 🎯

    This professional-grade quantitative trading system allows you to:

    - 📊 **Backtest** multiple trading strategies with realistic execution
    - 📈 **Analyze** comprehensive performance metrics and risk-adjusted returns
    - 💼 **Optimize** portfolio allocation using advanced algorithms
    - 📉 **Visualize** equity curves, drawdowns, and risk decomposition
    - 🤖 **Compare** ML-based strategies with traditional approaches

    ### Quick Start:
    1. Select symbols from the sidebar
    2. Choose date range
    3. Pick a strategy
    4. Configure execution parameters
    5. Click "Run Backtest"

    ### Features:
    - ✅ Realistic slippage and transaction costs
    - ✅ Order book simulation
    - ✅ Risk management with VaR and CVaR
    - ✅ Portfolio optimization (Markowitz, HRP, Risk Parity)
    - ✅ Machine learning strategies with rolling window training
    - ✅ Comprehensive performance attribution

    **Get started by configuring the backtest in the sidebar!** 👈
    """)

    # Show example visualization
    st.subheader("Example: Efficient Frontier")

    # Create sample data
    np.random.seed(42)
    n_assets = 5
    returns = pd.DataFrame(
        np.random.randn(252, n_assets) * 0.01,
        columns=[f'Asset {i+1}' for i in range(n_assets)]
    )

    try:
        optimizer = MarkowitzOptimizer()
        frontier = optimizer.efficient_frontier(returns, n_points=30)

        fig = visualizer.plot_efficient_frontier(frontier)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Configure and run a backtest to see results!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Multi-Strategy Backtesting Engine | Built with Streamlit, NumPy, Pandas, and Plotly</p>
</div>
""", unsafe_allow_html=True)
