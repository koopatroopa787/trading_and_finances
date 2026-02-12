"""Simple backtest example using momentum strategy."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import YahooFinanceLoader
from src.data.processors import DataProcessor
from src.strategy.implementations.momentum import MomentumStrategy
from src.execution.simulator import ExecutionSimulator
from src.analytics.performance import PerformanceAnalyzer


def main():
    print("=" * 60)
    print("SIMPLE MOMENTUM BACKTEST")
    print("=" * 60)

    # 1. Load data
    print("\n📊 Loading data...")
    loader = YahooFinanceLoader()
    data = loader.load(
        symbols='AAPL',
        start='2020-01-01',
        end='2023-12-31',
    )

    # 2. Clean data
    print("🧹 Cleaning data...")
    processor = DataProcessor()
    data = processor.clean_data(data)

    print(f"✅ Loaded {len(data)} days of data")

    # 3. Create strategy
    print("\n🎯 Creating momentum strategy...")
    strategy = MomentumStrategy(
        lookback_period=20,
        momentum_threshold=0.05,
        position_size=0.1,
        name="Momentum 20-day"
    )

    # 4. Run backtest
    print("\n🚀 Running backtest...")
    executor = ExecutionSimulator(
        initial_capital=100000,
        use_order_book=True,
        partial_fills=True,
    )

    result = executor.run(strategy, data)

    # 5. Analyze results
    print("\n📈 Analyzing performance...")
    analyzer = PerformanceAnalyzer()

    # Print report
    report = analyzer.generate_report(result.metrics)
    print("\n" + report)

    # 6. Show trades
    print(f"\n💼 Total trades executed: {len(result.trades)}")
    if len(result.trades) > 0:
        print("\nFirst 5 trades:")
        print(result.trades[['timestamp', 'side', 'quantity', 'price', 'total_cost']].head())

    # 7. Export results
    print("\n💾 Exporting results...")
    result.equity_curve.to_csv('equity_curve.csv')
    result.trades.to_csv('trades.csv')

    print("\n✅ Results exported to CSV files")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
