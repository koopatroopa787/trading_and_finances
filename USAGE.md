## Usage Guide

Comprehensive guide for using the Multi-Strategy Backtesting Engine.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run a Simple Backtest

```python
from src.data.loaders import YahooFinanceLoader
from src.strategy.implementations.momentum import MomentumStrategy
from src.execution.simulator import ExecutionSimulator

# Load data
loader = YahooFinanceLoader()
data = loader.load('AAPL', start='2020-01-01', end='2023-12-31')

# Create strategy
strategy = MomentumStrategy(lookback_period=20, position_size=0.1)

# Run backtest
executor = ExecutionSimulator(initial_capital=100000)
result = executor.run(strategy, data)

# Print results
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
```

### 3. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Examples

### Example 1: Simple Momentum Strategy

```bash
python examples/simple_backtest.py
```

This example demonstrates:
- Loading data from Yahoo Finance
- Creating a momentum strategy
- Running a backtest with realistic execution
- Analyzing performance metrics

### Example 2: Multi-Strategy Portfolio

```bash
python examples/multi_strategy.py
```

This example shows:
- Combining multiple strategies
- Dynamic portfolio allocation
- Risk parity weighting
- Strategy composition

### Example 3: Portfolio Optimization

```bash
python examples/portfolio_optimization.py
```

This example covers:
- Markowitz mean-variance optimization
- Hierarchical Risk Parity (HRP)
- Risk parity allocation
- Efficient frontier visualization

### Example 4: Regime Detection

```bash
python examples/regime_detection.py
```

This example includes:
- Hidden Markov Model regime detection
- ML-based clustering regimes
- Regime performance analysis
- Regime-based strategy adaptation

## Core Components

### 1. Data Pipeline

#### Load Data from Multiple Sources

```python
from src.data.loaders import YahooFinanceLoader, AlphaVantageLoader, CCXTLoader

# Yahoo Finance
yahoo_loader = YahooFinanceLoader()
data = yahoo_loader.load(['AAPL', 'GOOGL'], start='2020-01-01')

# Alpha Vantage (requires API key)
av_loader = AlphaVantageLoader(api_key='YOUR_KEY')
data = av_loader.load('AAPL', interval='daily')

# Cryptocurrency (via CCXT)
crypto_loader = CCXTLoader(exchange='binance')
data = crypto_loader.load('BTC/USDT', interval='1d')
```

#### Process and Clean Data

```python
from src.data.processors import DataProcessor

processor = DataProcessor()

# Clean data
data = processor.clean_data(data, fill_method='ffill')

# Adjust for splits
splits = {'2020-08-31': 4.0}  # 4-for-1 split
data = processor.adjust_for_splits(data, splits)

# Adjust for dividends
dividends = {'2020-11-06': 0.82}
data = processor.adjust_for_dividends(data, dividends)

# Resample to different frequency
data_weekly = processor.resample_data(data, freq='1W')
```

#### Feature Engineering

```python
from src.data.features import FeatureEngineer

fe = FeatureEngineer()

# Add all technical indicators
data = fe.add_all_features(data, include_advanced=True)

# Or add specific features
data = fe.add_rsi(data, period=14)
data = fe.add_macd(data)
data = fe.add_bollinger_bands(data)

# Feature selection
selected = fe.select_features(
    data,
    target=data['close'].pct_change().shift(-1),
    method='mutual_info',
    k=20
)
```

### 2. Trading Strategies

#### Create Custom Strategy

```python
from src.strategy.base import BaseStrategy, OrderType

class MyStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(**params)
        self.sma_period = params.get('sma_period', 20)

    def on_data(self, timestamp, data):
        # Calculate signal
        sma = data['close'].rolling(self.sma_period).mean().iloc[-1]
        current_price = data['close'].iloc[-1]

        symbol = 'ASSET'

        # Trading logic
        if current_price > sma and not self.has_position(symbol):
            self.buy(symbol, size=0.1)
        elif current_price < sma and self.has_position(symbol):
            self.close_position(symbol)
```

#### Use Built-in Strategies

```python
from src.strategy.implementations.momentum import MomentumStrategy, TrendFollowingStrategy
from src.strategy.implementations.mean_reversion import MeanReversionStrategy, PairsTradingStrategy
from src.strategy.implementations.ml_strategy import MLStrategy, EnsembleMLStrategy

# Momentum
momentum = MomentumStrategy(
    lookback_period=20,
    momentum_threshold=0.05,
    position_size=0.1
)

# Mean Reversion
mean_rev = MeanReversionStrategy(
    lookback_period=20,
    entry_threshold=2.0,
    exit_threshold=0.5,
    position_size=0.1
)

# ML-based
ml_strategy = MLStrategy(
    model_type='xgboost',
    lookback=252,
    retrain_frequency=20,
    position_size=0.1
)

# Ensemble
ensemble = EnsembleMLStrategy(
    model_types=['xgboost', 'lightgbm', 'random_forest'],
    ensemble_method='weighted'
)
```

### 3. Execution Simulation

```python
from src.execution.simulator import ExecutionSimulator
from src.execution.cost_models import VolumeCostModel

# Create custom cost model
cost_model = VolumeCostModel(
    commission_rate=0.001,
    base_slippage_bps=2.0,
    volatility_multiplier=10.0
)

# Create simulator
executor = ExecutionSimulator(
    initial_capital=100000,
    cost_model=cost_model,
    use_order_book=True,
    partial_fills=True,
    max_position_size=0.25,
    max_leverage=1.0
)

# Run backtest
result = executor.run(strategy, data, start_date='2020-01-01')
```

### 4. Risk Management

```python
from src.risk.metrics import RiskMetrics
from src.risk.limits import RiskLimits
from src.risk.decomposition import RiskDecomposition

# Calculate risk metrics
risk = RiskMetrics()

var_95 = risk.calculate_var(returns, confidence=0.95)
cvar_95 = risk.calculate_cvar(returns, confidence=0.95)
max_dd = risk.calculate_max_drawdown(equity_curve)

# Set risk limits
limits = RiskLimits(
    max_position_size=0.2,
    max_leverage=1.0,
    daily_loss_limit=0.05,
    max_drawdown_limit=0.25
)

# Check limits
is_valid = limits.check_position_size(position_value, portfolio_value)
is_valid = limits.check_leverage(positions_value, portfolio_value)

# Risk decomposition
decomp = RiskDecomposition()
risk_contrib = decomp.calculate_risk_contribution(weights, cov_matrix)
```

### 5. Portfolio Optimization

```python
from src.optimization.portfolio import (
    MarkowitzOptimizer,
    HRPOptimizer,
    BlackLittermanOptimizer
)

# Markowitz (Maximum Sharpe)
markowitz = MarkowitzOptimizer(risk_free_rate=0.02)
weights = markowitz.optimize(returns_df, objective="max_sharpe")

# HRP
hrp = HRPOptimizer()
weights = hrp.optimize(returns_df)

# Black-Litterman (with views)
bl = BlackLittermanOptimizer()
views = {'AAPL': 0.15, 'GOOGL': 0.12}  # Expected returns
weights = bl.optimize(returns_df, market_caps, views=views)

# Efficient frontier
frontier = markowitz.efficient_frontier(returns_df, n_points=50)
```

### 6. Performance Analytics

```python
from src.analytics.performance import PerformanceAnalyzer
from src.analytics.attribution import AttributionAnalyzer
from src.analytics.visualization import PerformanceVisualizer

# Analyze performance
analyzer = PerformanceAnalyzer()
metrics = analyzer.calculate_metrics(equity_curve, trades, benchmark)

# Generate report
report = analyzer.generate_report(metrics)
print(report)

# Attribution analysis
attribution = AttributionAnalyzer()
strategy_attr = attribution.strategy_attribution(
    strategy_returns,
    strategy_weights
)

# Visualize
visualizer = PerformanceVisualizer()
fig = visualizer.plot_equity_curve(equity_curve, benchmark)
fig.show()
```

### 7. Machine Learning

```python
from src.ml.regime_detector import HMMRegimeDetector, MLRegimeDetector
from src.ml.models import FeatureSelector, ModelTrainer

# Regime detection
hmm = HMMRegimeDetector(n_regimes=3)
hmm.fit(returns)
regimes = hmm.predict(returns)

# Feature selection
selector = FeatureSelector()
selected_features = selector.select_by_importance(
    X, y, method='mutual_info', k=20
)

# Model training with walk-forward
trainer = ModelTrainer()
results = trainer.walk_forward_optimize(
    X, y,
    model_class=XGBRegressor,
    train_size=252,
    test_size=60
)
```

## Configuration

Edit `config.py` to customize:

```python
from config import config

# Data settings
config.data.data_dir = Path("./my_data")
config.data.storage_format = "parquet"

# Backtest settings
config.backtest.initial_capital = 100000
config.backtest.commission_rate = 0.001
config.backtest.slippage_bps = 5

# Risk settings
config.risk.risk_free_rate = 0.02
config.risk.var_confidence = 0.95
config.risk.max_position_size = 0.15
```

## Advanced Topics

### Walk-Forward Optimization

```python
from src.ml.models import ModelTrainer

trainer = ModelTrainer()

# Walk-forward with XGBoost
results = trainer.walk_forward_optimize(
    X=features,
    y=target,
    model_class=XGBRegressor,
    train_size=252,
    test_size=60,
    step_size=20
)

# Analyze results
import pandas as pd
results_df = pd.DataFrame(results)
print(f"Average test R²: {results_df['test_score'].mean():.4f}")
```

### Multi-Strategy Composition

```python
from src.strategy.composer import StrategyComposer

strategies = {
    'momentum': MomentumStrategy(...),
    'mean_rev': MeanReversionStrategy(...),
    'ml': MLStrategy(...)
}

# Risk parity allocation
composer = StrategyComposer(
    strategies=strategies,
    allocation_method="risk_parity",
    rebalance_frequency=20
)

# Dynamic allocation based on performance
composer = StrategyComposer(
    strategies=strategies,
    allocation_method="momentum",
    rebalance_frequency=20
)
```

### Custom Cost Models

```python
from src.execution.cost_models import CostModel, TransactionCost

class CustomCostModel(CostModel):
    def calculate_cost(self, quantity, price, side, volatility=None, volume=None):
        # Custom logic
        commission = quantity * price * 0.001
        slippage = quantity * price * 0.0005

        return TransactionCost(
            commission=commission,
            slippage=slippage,
            market_impact=0.0,
            spread_cost=0.0
        )
```

## Tips and Best Practices

1. **Data Quality**: Always clean and validate your data before backtesting
2. **Overfitting**: Use walk-forward optimization and out-of-sample testing
3. **Transaction Costs**: Include realistic slippage and commissions
4. **Position Sizing**: Use risk-based position sizing (Kelly, Risk Parity)
5. **Risk Management**: Set stop-losses and position limits
6. **Diversification**: Combine multiple uncorrelated strategies
7. **Benchmarking**: Always compare against a buy-and-hold benchmark
8. **Regime Awareness**: Adapt strategies to different market regimes

## Troubleshooting

### Common Issues

**Issue**: "No data loaded"
- Check symbol spelling
- Verify date range
- Ensure internet connection for data download

**Issue**: "Optimization failed"
- Check for NaN values in returns
- Ensure sufficient data points
- Verify constraints are feasible

**Issue**: "Model training error"
- Check feature-target alignment
- Remove NaN values
- Ensure enough training samples

## API Reference

See individual module documentation:
- [Data Pipeline](src/data/README.md)
- [Strategies](src/strategy/README.md)
- [Execution](src/execution/README.md)
- [Risk Management](src/risk/README.md)
- [Optimization](src/optimization/README.md)
- [Analytics](src/analytics/README.md)
