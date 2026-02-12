"""Verify Python 3.11+ compatibility and basic functionality."""
import sys
import importlib.util

def check_python_version():
    """Check if Python version is 3.11 or higher."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ ERROR: Python 3.11 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False

    print("✅ Python version check passed")
    return True


def check_required_packages():
    """Check if required packages are available."""
    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'yfinance',
        'plotly',
        'streamlit',
        'cvxpy',
        'xgboost',
        'lightgbm',
        'loguru',
    ]

    missing_packages = []

    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
            print(f"❌ Missing: {package}")
        else:
            print(f"✅ Found: {package}")

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False

    print("\n✅ All required packages found")
    return True


def check_pandas_compatibility():
    """Check pandas version and deprecated method usage."""
    import pandas as pd

    print(f"\nPandas version: {pd.__version__}")

    # Check pandas version is 2.0+
    version = pd.__version__.split('.')
    major_version = int(version[0])

    if major_version < 2:
        print(f"⚠️  WARNING: Pandas 2.0+ recommended (current: {pd.__version__})")
        print("   Some features may not work correctly")
    else:
        print("✅ Pandas version compatible")

    # Test that fillna(method=) is not used (deprecated in pandas 2.0)
    test_series = pd.Series([1, None, 3, None, 5])

    try:
        # Test new syntax
        result = test_series.ffill()
        print("✅ Modern fillna syntax works (ffill/bfill)")
    except Exception as e:
        print(f"❌ Error with modern fillna: {e}")
        return False

    return True


def test_basic_imports():
    """Test basic imports from the package."""
    print("\nTesting basic imports...")

    try:
        from src.data.loaders import YahooFinanceLoader
        print("✅ Data loaders import successful")
    except Exception as e:
        print(f"❌ Data loaders import failed: {e}")
        return False

    try:
        from src.strategy.base import BaseStrategy
        print("✅ Strategy base import successful")
    except Exception as e:
        print(f"❌ Strategy base import failed: {e}")
        return False

    try:
        from src.execution.simulator import ExecutionSimulator
        print("✅ Execution simulator import successful")
    except Exception as e:
        print(f"❌ Execution simulator import failed: {e}")
        return False

    try:
        from src.risk.metrics import RiskMetrics
        print("✅ Risk metrics import successful")
    except Exception as e:
        print(f"❌ Risk metrics import failed: {e}")
        return False

    try:
        from src.optimization.portfolio import MarkowitzOptimizer
        print("✅ Portfolio optimization import successful")
    except Exception as e:
        print(f"❌ Portfolio optimization import failed: {e}")
        return False

    try:
        from src.analytics.performance import PerformanceAnalyzer
        print("✅ Analytics import successful")
    except Exception as e:
        print(f"❌ Analytics import failed: {e}")
        return False

    print("\n✅ All basic imports successful")
    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("PYTHON 3.11+ COMPATIBILITY VERIFICATION")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Pandas Compatibility", check_pandas_compatibility),
        ("Basic Imports", test_basic_imports),
    ]

    all_passed = True

    for check_name, check_func in checks:
        print(f"\n{'=' * 60}")
        print(f"Checking: {check_name}")
        print("=" * 60)

        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ Check failed with exception: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - System ready for Python 3.11+")
    else:
        print("❌ SOME CHECKS FAILED - Please review errors above")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
