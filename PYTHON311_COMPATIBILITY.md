# Python 3.11+ Compatibility Guide

This document outlines the Python 3.11+ compatibility features and requirements for the Multi-Strategy Backtesting Engine.

## System Requirements

### Python Version
- **Minimum**: Python 3.11.0
- **Recommended**: Python 3.11.14 or later
- **Tested on**: Python 3.11.14

### Why Python 3.11+?

Python 3.11 provides significant benefits for this quantitative trading system:

1. **Performance**: Up to 25% faster than Python 3.10 (critical for backtesting)
2. **Better Error Messages**: Enhanced tracebacks for debugging
3. **Exception Groups**: Better error handling in async code
4. **TOML Support**: Built-in `tomllib` for configuration files
5. **Type Hints**: Improved type annotation capabilities

## Compatibility Changes Made

### 1. Deprecated Pandas Methods Removed

**Issue**: `fillna(method='ffill')` and `fillna(method='bfill')` deprecated in Pandas 2.0+

**Fixed in files**:
- `src/data/processors.py`
- `src/data/features.py`
- `src/ml/regime_detector.py`

**Changes**:
```python
# OLD (deprecated)
df.fillna(method='ffill')
df.fillna(method='bfill')

# NEW (Python 3.11 compatible)
df.ffill()
df.bfill()
```

### 2. Package Version Requirements

All dependencies updated to support Python 3.11:

```
numpy>=1.24.0        # Python 3.11 support added
pandas>=2.0.0        # Modern API, deprecated methods removed
scipy>=1.10.0        # Python 3.11 compatible
xgboost>=1.7.0       # Python 3.11 wheels available
lightgbm>=4.0.0      # Full Python 3.11 support
catboost>=1.2.0      # Python 3.11 compatible
```

### 3. Type Hints Compatibility

Type hints follow Python 3.10+ standards:

```python
from typing import Optional, List, Dict, Union

# These work in Python 3.11
Optional[str]  # Still valid
List[int]      # Still valid
Dict[str, Any] # Still valid

# Python 3.10+ also supports:
str | None     # Instead of Optional[str]
list[int]      # Instead of List[int]
dict[str, Any] # Instead of Dict[str, Any]
```

The codebase uses both styles for maximum compatibility.

### 4. Project Configuration Files

#### pyproject.toml
Modern Python packaging with explicit version requirements:
```toml
[project]
requires-python = ">=3.11"
```

#### setup.py
Traditional setup with version enforcement:
```python
python_requires=">=3.11"
```

## Verification

### Quick Check
```bash
# Verify Python version
python --version  # Should show 3.11.0 or higher

# Run compatibility verification
python verify_python311.py
```

### Expected Output
```
✅ Python version check passed
✅ All required packages found
✅ Pandas version compatible
✅ Modern fillna syntax works (ffill/bfill)
✅ All basic imports successful
✅ ALL CHECKS PASSED
```

## Installation

### Using pip (Recommended)
```bash
# Verify Python version first
python --version

# Install all dependencies
pip install -r requirements.txt
```

### Using setup.py
```bash
# Install in development mode
pip install -e .

# Or regular install
pip install .
```

### Using pyproject.toml (Modern)
```bash
# Install with pip
pip install .

# Or with build
python -m build
pip install dist/*.whl
```

## Package-Specific Notes

### NumPy
- Python 3.11 wheels available since NumPy 1.24.0
- Vectorized operations fully compatible
- No changes needed to existing code

### Pandas
- Pandas 2.0+ required for Python 3.11
- Deprecated methods removed (fillna with method parameter)
- All code updated to use modern API

### XGBoost / LightGBM / CatBoost
- All provide Python 3.11 wheels
- No compatibility issues
- ML training works as expected

### Streamlit
- Full Python 3.11 support since Streamlit 1.25.0
- Dashboard runs without issues

### PyTorch
- Python 3.11 support since PyTorch 2.0
- GPU and CPU versions both compatible

## Performance Benefits in Python 3.11

### Faster Backtesting
- **10-25% faster** execution for backtests
- Improved dictionary and list operations
- Faster function calls

### Memory Efficiency
- Better memory layout for objects
- Reduced memory overhead in long-running backtests

### Error Handling
- Better error messages for debugging
- More informative tracebacks

## Migration from Python 3.10

If migrating from Python 3.10:

1. **Update Python**:
   ```bash
   # Using pyenv
   pyenv install 3.11.14
   pyenv local 3.11.14

   # Or using conda
   conda create -n trading python=3.11
   conda activate trading
   ```

2. **Reinstall Dependencies**:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. **Verify Installation**:
   ```bash
   python verify_python311.py
   ```

## Known Limitations

### Optional Packages

Some optional packages may require compilation:

1. **TA-Lib** (removed from requirements):
   - Requires system libraries
   - Use `pandas-ta` instead (pure Python)

2. **PyTables**:
   - May need HDF5 libraries installed
   - Used for HDF5 storage (optional)

### Platform-Specific

- **Windows**: All packages available as wheels
- **macOS**: Works on both Intel and Apple Silicon
- **Linux**: All packages compatible

## Testing

### Run Tests
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Type Checking
```bash
# Run mypy
mypy src/ --python-version 3.11
```

### Linting
```bash
# Run ruff
ruff check src/

# Auto-fix
ruff check src/ --fix
```

## Future Compatibility

The codebase is designed to be forward-compatible:

- **Python 3.12**: No breaking changes expected
- **Python 3.13**: Should work without modifications
- **Pandas 3.0**: Will require minimal updates

## Support

For Python 3.11 compatibility issues:

1. Check this document first
2. Run `verify_python311.py`
3. Verify package versions
4. Check GitHub issues

## Summary

✅ **Fully compatible with Python 3.11+**
✅ **All deprecated methods removed**
✅ **Modern packaging with pyproject.toml**
✅ **Performance optimized**
✅ **Type hints compatible**
✅ **Comprehensive verification script included**

The system is production-ready for Python 3.11 and will remain compatible with future Python versions.
