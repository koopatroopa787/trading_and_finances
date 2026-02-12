"""Multi-strategy backtest with portfolio composition."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import YahooFinanceLoader
from src.data.processors import DataProcessor
from src.strategy.implementations.momentum import MomentumStrategy
from src.strategy.implementations.mean_reversion import MeanReversionStrategy
from src.strategy.implementations.ml_strategy import MLStrategy
from src.strategy.composer import StrategyComposer
from src.execution.simulator import ExecutionSimulator


def main():
    print("=" * 60)
    print("MULTI-STRATEGY PORTFOLIO BACKTEST")
    print("=" * 60)

    # Load data
    print("\n📊 Loading data...")
    loader = YahooFinanceLoader()
    data = loader.load(
        symbols='SPY',
        start='2020-01-01',
        end='2023-12-31',
    )

    processor = DataProcessor()
    data = processor.clean_data(data)

    # Create strategies
    print("\n🎯 Creating strategies...")

    strategies = {
        'momentum': MomentumStrategy(
            lookback_period=20,
            momentum_threshold=0.05,
            position_size=0.1,
            name="Momentum"
        ),
        'mean_reversion': MeanReversionStrategy(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5,
            position_size=0.1,
            name="Mean Reversion"
        ),
        'ml_xgboost': MLStrategy(
            model_type='xgboost',
            lookback=252,
            retrain_frequency=20,
            position_size=0.1,
            name="ML XGBoost"
        ),
    }

    # Create composer with dynamic allocation
    print("\n🎭 Composing strategies...")
    composer = StrategyComposer(
        strategies=strategies,
        allocation_method="risk_parity",
        weights={'momentum': 0.4, 'mean_reversion': 0.3, 'ml_xgboost': 0.3},
        rebalance_frequency=20,
    )

    # Initialize composer
    composer.on_start(initial_capital=100000)

    # Run through data
    print("\n🚀 Running multi-strategy backtest...")

    for i, (timestamp, row) in enumerate(data.iterrows()):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(data)} ({i/len(data)*100:.1f}%)")

        # Update composer
        composer.on_data(timestamp, data.iloc[:i+1])

        if i % 20 == 0:  # Bar close
            composer.on_bar_close(timestamp, data.iloc[:i+1])

    composer.on_stop()

    # Get performance summary
    print("\n📈 Performance Summary:")
    summary = composer.get_performance_summary()

    print(f"\n💰 Total Portfolio Value: ${summary['total_value']:,.2f}")

    print("\n📊 Strategy Performance:")
    for name, stats in summary['strategies'].items():
        print(f"\n  {name}:")
        print(f"    Weight: {stats['weight']:.2%}")
        print(f"    Value: ${stats['value']:,.2f}")
        perf = stats['performance']
        if perf:
            print(f"    Return: {perf.get('total_return', 0):.2%}")
            print(f"    Sharpe: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"    Trades: {perf.get('num_trades', 0)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
