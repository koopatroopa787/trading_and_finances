"""Portfolio optimization example."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import YahooFinanceLoader
from src.data.processors import DataProcessor
from src.optimization.portfolio import MarkowitzOptimizer, HRPOptimizer, RiskParityOptimizer
from src.analytics.visualization import PerformanceVisualizer


def main():
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    # Load data for multiple assets
    print("\n📊 Loading data...")
    loader = YahooFinanceLoader()

    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

    data = loader.load(
        symbols=symbols,
        start='2020-01-01',
        end='2023-12-31',
    )

    print(f"✅ Loaded data for {len(symbols)} assets")

    # Calculate returns
    print("\n📈 Calculating returns...")
    returns = {}
    for symbol in symbols:
        if hasattr(data.columns, 'levels'):
            symbol_data = data[symbol]['close']
        else:
            symbol_data = data['close']

        returns[symbol] = symbol_data.pct_change().dropna()

    returns_df = pd.DataFrame(returns)

    # Remove NaN
    returns_df = returns_df.dropna()

    print(f"Returns shape: {returns_df.shape}")

    # 1. Markowitz Optimization
    print("\n1️⃣ Markowitz Optimization (Maximum Sharpe Ratio)")
    markowitz = MarkowitzOptimizer(risk_free_rate=0.02)

    weights_markowitz = markowitz.optimize(
        returns_df,
        objective="max_sharpe"
    )

    print("\nOptimal weights (Markowitz):")
    for symbol, weight in weights_markowitz.items():
        if weight > 0.01:
            print(f"  {symbol}: {weight:.2%}")

    # Calculate efficient frontier
    print("\n📊 Calculating efficient frontier...")
    frontier = markowitz.efficient_frontier(returns_df, n_points=30)

    # 2. Hierarchical Risk Parity
    print("\n2️⃣ Hierarchical Risk Parity (HRP)")
    hrp = HRPOptimizer()

    weights_hrp = hrp.optimize(returns_df)

    print("\nOptimal weights (HRP):")
    for symbol, weight in weights_hrp.items():
        if weight > 0.01:
            print(f"  {symbol}: {weight:.2%}")

    # 3. Risk Parity
    print("\n3️⃣ Risk Parity")
    rp = RiskParityOptimizer()

    weights_rp = rp.optimize(returns_df)

    print("\nOptimal weights (Risk Parity):")
    for symbol, weight in weights_rp.items():
        if weight > 0.01:
            print(f"  {symbol}: {weight:.2%}")

    # Compare portfolios
    print("\n📊 Portfolio Comparison:")

    portfolios = {
        'Equal Weight': pd.Series([1/len(symbols)] * len(symbols), index=symbols),
        'Markowitz': weights_markowitz,
        'HRP': weights_hrp,
        'Risk Parity': weights_rp,
    }

    print("\n" + "=" * 80)
    print(f"{'Portfolio':<20} {'Return':<12} {'Volatility':<12} {'Sharpe':<12}")
    print("=" * 80)

    for name, weights in portfolios.items():
        port_return = (returns_df @ weights).mean() * 252
        port_vol = (returns_df @ weights).std() * (252 ** 0.5)
        sharpe = (port_return - 0.02) / port_vol

        print(f"{name:<20} {port_return:>10.2%}  {port_vol:>10.2%}  {sharpe:>10.2f}")

    print("=" * 80)

    # Visualize
    print("\n📈 Creating visualization...")
    visualizer = PerformanceVisualizer()

    # Calculate current portfolio stats
    current_stats = {
        'return': (returns_df @ weights_markowitz).mean() * 252,
        'volatility': (returns_df @ weights_markowitz).std() * (252 ** 0.5),
    }

    fig = visualizer.plot_efficient_frontier(frontier, current_stats)

    # Save figure
    import plotly.io as pio
    pio.write_html(fig, 'efficient_frontier.html')

    print("\n✅ Visualization saved to efficient_frontier.html")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
