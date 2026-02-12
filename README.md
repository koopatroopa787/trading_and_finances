# Multi-Strategy Backtesting Engine with Portfolio Optimization

A professional-grade quantitative trading system for developing, testing, and optimizing trading strategies with realistic market conditions.

## Requirements

- **Python 3.11 or higher** (tested on Python 3.11+)
- NumPy, Pandas, SciPy, scikit-learn
- Financial data libraries (yfinance, ccxt, alpha-vantage)
- ML frameworks (XGBoost, LightGBM, CatBoost)
- Visualization (Plotly, Streamlit, Matplotlib)

## Features

### Core System
- **Multi-Source Data Pipeline**: Ingest from Yahoo Finance, Alpha Vantage, CCXT (crypto)
- **Advanced Feature Engineering**: 100+ technical indicators, volatility metrics, custom features
- **Strategy Framework**: Extensible base class with lifecycle hooks
- **Realistic Execution**: Slippage models, transaction costs, partial fills, order book simulation
- **Risk Management**: VaR, CVaR, Greeks, drawdown control, position limits
- **Portfolio Optimization**: Markowitz, Black-Litterman, Hierarchical Risk Parity
- **Performance Analytics**: Sharpe, Sortino, Calmar, Omega ratios, attribution analysis

### Advanced Features
- **Regime Detection**: Hidden Markov Models for market state identification
- **ML-Based Strategies**: XGBoost, LightGBM, CatBoost with rolling window training
- **Reinforcement Learning**: PPO/SAC agents for optimal position sizing
- **Options Strategies**: Volatility trading, straddles, iron condors, delta hedging
- **Walk-Forward Optimization**: Prevent overfitting with out-of-sample validation

### Interactive Dashboard
- Real-time strategy performance monitoring
- Risk decomposition visualizations
- Portfolio efficient frontier
- Drawdown analysis
- Strategy comparison metrics

## Architecture

```
src/
├── data/              # Data ingestion and processing
├── strategy/          # Strategy implementations
├── execution/         # Order execution simulation
├── risk/              # Risk management
├── optimization/      # Portfolio optimization
├── analytics/         # Performance analysis
└── ml/                # Machine learning components
```

## 🚀 Quick Start

> **New to the project?** See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions!

### Installation

**Prerequisites**: Python 3.11 or higher

```bash
# Navigate to project directory
cd trading_and_finances

# Verify Python version
python --version  # Should show Python 3.11 or higher

# Install in editable mode (RECOMMENDED)
pip install -e .

# This allows Python to find the 'src' module from anywhere
```

**Alternative**: Install dependencies only
```bash
pip install -r requirements.txt
# Note: Must run scripts from project root directory
```

### Basic Usage

```python
from src.data.loaders import YahooFinanceLoader
from src.strategy.implementations.momentum import MomentumStrategy
from src.execution.simulator import ExecutionSimulator
from src.analytics.performance import PerformanceAnalyzer

# Load data
loader = YahooFinanceLoader()
data = loader.load(['AAPL', 'GOOGL', 'MSFT'], start='2020-01-01', end='2023-12-31')

# Create strategy
strategy = MomentumStrategy(lookback_period=20, position_size=0.1)

# Run backtest
simulator = ExecutionSimulator(initial_capital=100000)
results = simulator.run(strategy, data)

# Analyze performance
analyzer = PerformanceAnalyzer()
metrics = analyzer.calculate_metrics(results)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Strategy Examples

### Momentum Strategy
```python
from src.strategy.base import BaseStrategy

class MomentumStrategy(BaseStrategy):
    def on_data(self, timestamp, data):
        returns = data['close'].pct_change(self.lookback_period)
        if returns > 0.05:
            self.buy(size=self.position_size)
        elif returns < -0.05:
            self.sell(size=self.position_size)
```

### ML-Based Strategy
```python
from src.strategy.implementations.ml_strategy import MLStrategy

strategy = MLStrategy(
    model='xgboost',
    features=['rsi', 'macd', 'bb_width', 'volume_ratio'],
    lookback=252,
    retrain_frequency=20
)
```

## Performance Metrics

- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar, Omega
- **Risk Metrics**: VaR (95%, 99%), CVaR, Maximum Drawdown
- **Attribution**: Strategy contribution, sector exposure
- **Greeks**: Delta, Gamma, Vega, Theta (for options)

## Risk Management

- Position sizing with Kelly Criterion
- Dynamic stop-loss and take-profit
- Portfolio-level risk limits
- Correlation-based diversification
- Drawdown control mechanisms

## Portfolio Optimization

### Mean-Variance (Markowitz)
```python
from src.optimization.portfolio import MarkowitzOptimizer

optimizer = MarkowitzOptimizer()
weights = optimizer.optimize(returns, target_return=0.15)
```

### Hierarchical Risk Parity
```python
from src.optimization.portfolio import HRPOptimizer

optimizer = HRPOptimizer()
weights = optimizer.optimize(returns)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Solution 1** - Install the package (recommended):
```bash
cd trading_and_finances
pip install -e .
```

**Solution 2** - Run from project root:
```bash
cd trading_and_finances  # Ensure you're in the correct directory
python examples/simple_backtest.py
streamlit run dashboard/app.py
```

### "No module named 'pandas'" or other package errors

```bash
# Install all dependencies
pip install -r requirements.txt

# Or upgrade pip first if installation fails
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Dashboard won't start

```bash
# Install streamlit explicitly
pip install streamlit>=1.25.0

# Try different port if 8501 is busy
streamlit run dashboard/app.py --server.port 8502
```

### Import errors in examples

Make sure you're running from the project root:
```bash
# Check current directory
pwd  # Should end in 'trading_and_finances'

# If not, navigate there
cd path/to/trading_and_finances
python examples/simple_backtest.py
```

**For more troubleshooting**, see [QUICKSTART.md](QUICKSTART.md)

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## License

MIT License

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves risk of loss.
