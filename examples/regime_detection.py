"""Market regime detection example."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import YahooFinanceLoader
from src.data.processors import DataProcessor
from src.ml.regime_detector import HMMRegimeDetector, MLRegimeDetector


def main():
    print("=" * 60)
    print("MARKET REGIME DETECTION")
    print("=" * 60)

    # Load data
    print("\n📊 Loading data...")
    loader = YahooFinanceLoader()
    data = loader.load(
        symbols='SPY',
        start='2015-01-01',
        end='2023-12-31',
    )

    processor = DataProcessor()
    data = processor.clean_data(data)

    print(f"✅ Loaded {len(data)} days of data")

    # Calculate returns
    returns = data['close'].pct_change().dropna()

    # 1. HMM Regime Detection
    print("\n1️⃣ Hidden Markov Model Regime Detection")
    print("Training HMM with 3 regimes...")

    hmm_detector = HMMRegimeDetector(n_regimes=3, n_iter=100)
    hmm_detector.fit(returns)

    # Predict regimes
    regimes_hmm = hmm_detector.predict(returns)

    print(f"\n✅ HMM regimes detected")

    # Analyze regime transitions
    regime_changes = np.diff(regimes_hmm)
    n_transitions = np.count_nonzero(regime_changes)

    print(f"Number of regime transitions: {n_transitions}")
    print(f"Average regime duration: {len(regimes_hmm) / n_transitions:.1f} days")

    # 2. ML-based Regime Detection
    print("\n2️⃣ ML-based Regime Detection (K-Means)")
    print("Training K-Means clustering...")

    ml_detector = MLRegimeDetector(n_regimes=3, method="kmeans")
    ml_detector.fit(data)

    regimes_ml = ml_detector.predict(data)

    print(f"\n✅ ML regimes detected")

    # Regime performance analysis
    print("\n📊 Regime Performance Analysis (HMM):")

    regime_stats = []

    for regime in range(3):
        regime_mask = regimes_hmm == regime
        regime_returns = returns[regime_mask[1:]]  # Offset by 1 due to pct_change

        if len(regime_returns) > 0:
            regime_stats.append({
                'regime': regime,
                'frequency': regime_mask.sum() / len(regimes_hmm),
                'avg_return': regime_returns.mean() * 252,
                'volatility': regime_returns.std() * np.sqrt(252),
                'sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252),
            })

    stats_df = pd.DataFrame(regime_stats)

    print("\n" + "=" * 70)
    print(f"{'Regime':<10} {'Frequency':<12} {'Avg Return':<14} {'Volatility':<14} {'Sharpe':<10}")
    print("=" * 70)

    for _, row in stats_df.iterrows():
        print(
            f"{int(row['regime']):<10} "
            f"{row['frequency']:>10.2%}  "
            f"{row['avg_return']:>12.2%}  "
            f"{row['volatility']:>12.2%}  "
            f"{row['sharpe']:>8.2f}"
        )

    print("=" * 70)

    # Regime-based strategy performance
    print("\n💡 Regime-Based Strategy Suggestion:")

    best_regime = stats_df.loc[stats_df['sharpe'].idxmax()]

    print(f"\nBest performing regime: {int(best_regime['regime'])}")
    print(f"  - Occurs {best_regime['frequency']:.1%} of the time")
    print(f"  - Sharpe Ratio: {best_regime['sharpe']:.2f}")
    print(f"  - Annual Return: {best_regime['avg_return']:.2%}")

    print("\n💡 Strategy: Increase position size during regime {}, reduce during others".format(
        int(best_regime['regime'])
    ))

    # Save results
    print("\n💾 Saving regime data...")

    regime_data = pd.DataFrame({
        'date': data.index[1:],  # Offset for returns
        'close': data['close'].iloc[1:].values,
        'return': returns.values,
        'regime_hmm': regimes_hmm,
        'regime_ml': regimes_ml[1:],
    })

    regime_data.to_csv('regime_data.csv', index=False)

    print("✅ Regime data saved to regime_data.csv")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
