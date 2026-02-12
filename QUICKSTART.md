# Quick Start Guide

Get up and running with the Multi-Strategy Backtesting Engine in 5 minutes!

## Prerequisites

- **Python 3.11 or higher** installed
- Internet connection (for downloading market data)

## Installation

### Option 1: Install in Editable Mode (Recommended for Development)

```bash
# Navigate to the project directory
cd trading_and_finances

# Install in editable mode
pip install -e .
```

This installs the package so Python can find the `src` module from anywhere.

### Option 2: Install from Requirements

```bash
# Install just the dependencies
pip install -r requirements.txt
```

**Note**: If using this option, you must run commands from the project root directory.

### Verify Installation

```bash
# Check Python version
python --version  # Should show 3.11.0 or higher

# Run verification script
python verify_python311.py
```

## Running the Dashboard

### Method 1: After Installing Package (Recommended)

```bash
# From anywhere
streamlit run dashboard/app.py
```

### Method 2: From Project Root

```bash
# Navigate to project root first
cd trading_and_finances

# Run dashboard
streamlit run dashboard/app.py
```

### Method 3: Specify Full Path

```bash
# From project root
cd trading_and_finances
python -m streamlit run dashboard/app.py
```

## Running Examples

All examples should be run from the **project root directory**:

```bash
# Navigate to project root
cd trading_and_finances

# Run simple backtest
python examples/simple_backtest.py

# Run multi-strategy backtest
python examples/multi_strategy.py

# Run portfolio optimization
python examples/portfolio_optimization.py

# Run regime detection
python examples/regime_detection.py
```

## Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'src'"

**Solution 1** - Install the package:
```bash
cd trading_and_finances
pip install -e .
```

**Solution 2** - Run from project root:
```bash
cd trading_and_finances  # Make sure you're in the right directory
python examples/simple_backtest.py
```

**Solution 3** - Check your Python path:
```python
import sys
print(sys.path)
```

### Issue 2: "No module named 'pandas'" or other packages

**Solution** - Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue 3: Package installation fails

**Solution** - Upgrade pip first:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Issue 4: Streamlit doesn't start

**Solution 1** - Install streamlit explicitly:
```bash
pip install streamlit>=1.25.0
```

**Solution 2** - Check if port is available:
```bash
streamlit run dashboard/app.py --server.port 8502
```

## First Time Usage

### 1. Test the Installation

```bash
cd trading_and_finances
python verify_python311.py
```

Expected output:
```
✅ Python version check passed
✅ All required packages found
✅ Pandas version compatible
✅ All basic imports successful
```

### 2. Run a Simple Backtest

```bash
python examples/simple_backtest.py
```

This will:
- Download AAPL stock data from Yahoo Finance
- Run a momentum strategy backtest
- Display performance metrics
- Export results to CSV

### 3. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

This opens http://localhost:8501 in your browser with an interactive UI.

### 4. Customize Parameters

Edit examples or create your own:

```python
from src.data.loaders import YahooFinanceLoader
from src.strategy.implementations.momentum import MomentumStrategy
from src.execution.simulator import ExecutionSimulator

# Load data
loader = YahooFinanceLoader()
data = loader.load('AAPL', start='2020-01-01', end='2023-12-31')

# Create strategy
strategy = MomentumStrategy(
    lookback_period=20,
    momentum_threshold=0.05,
    position_size=0.1
)

# Run backtest
executor = ExecutionSimulator(initial_capital=100000)
result = executor.run(strategy, data)

# Print results
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
```

## Project Structure

```
trading_and_finances/
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   ├── strategy/          # Trading strategies
│   ├── execution/         # Backtesting engine
│   ├── risk/              # Risk management
│   ├── optimization/      # Portfolio optimization
│   ├── analytics/         # Performance analysis
│   └── ml/                # Machine learning
│
├── dashboard/             # Streamlit dashboard
│   └── app.py
│
├── examples/              # Example scripts
│   ├── simple_backtest.py
│   ├── multi_strategy.py
│   ├── portfolio_optimization.py
│   └── regime_detection.py
│
├── tests/                 # Unit tests
│
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── pyproject.toml        # Modern packaging config
└── README.md             # Documentation
```

## Next Steps

1. **Read the documentation**:
   - `README.md` - Overview and features
   - `USAGE.md` - Comprehensive usage guide
   - `PYTHON311_COMPATIBILITY.md` - Compatibility details

2. **Try different strategies**:
   - Momentum
   - Mean Reversion
   - ML-based
   - Multi-strategy portfolios

3. **Explore optimization**:
   - Markowitz optimization
   - Hierarchical Risk Parity
   - Black-Litterman
   - Risk Parity

4. **Customize the dashboard**:
   - Edit `dashboard/app.py`
   - Add your own visualizations
   - Integrate custom strategies

## Tips

1. **Always run from project root**: Most commands expect to be run from the `trading_and_finances` directory

2. **Install in editable mode**: Use `pip install -e .` for development - changes take effect immediately

3. **Use virtual environments**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -e .
   ```

4. **Check Python version**: The system requires Python 3.11+

5. **Cache data**: Downloaded data is cached in `./cache` directory to speed up subsequent runs

## Getting Help

- Check `USAGE.md` for detailed examples
- Run `verify_python311.py` to diagnose issues
- Review example scripts in `examples/`
- Check error messages - they're designed to be helpful!

## Performance Tips

1. **Use cached data**: Set `use_cache=True` in data loaders
2. **Reduce data range**: Start with shorter date ranges for testing
3. **Limit symbols**: Test with 1-3 symbols before scaling up
4. **Python 3.11**: Ensure you're using Python 3.11+ for 10-25% better performance

## Ready to Start?

```bash
# 1. Install
cd trading_and_finances
pip install -e .

# 2. Verify
python verify_python311.py

# 3. Run example
python examples/simple_backtest.py

# 4. Launch dashboard
streamlit run dashboard/app.py
```

Happy backtesting! 📈🚀
