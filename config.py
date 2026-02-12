"""Configuration settings for the backtesting engine."""
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


class DataConfig(BaseModel):
    """Data-related configuration."""
    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    cache_dir: Path = Field(default_factory=lambda: Path("./cache"))
    storage_format: str = "parquet"  # parquet, hdf5, csv

    def __init__(self, **data):
        super().__init__(**data)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    initial_capital: float = Field(default=100000.0, gt=0)
    commission_rate: float = Field(default=0.001, ge=0, le=1)
    slippage_bps: float = Field(default=5.0, ge=0)  # basis points
    min_commission: float = Field(default=1.0, ge=0)

    # Order execution
    default_order_type: str = "market"  # market, limit
    partial_fills: bool = True

    # Position sizing
    max_position_size: float = Field(default=0.2, ge=0, le=1)
    max_portfolio_leverage: float = Field(default=1.0, ge=0)

    # Risk management
    max_drawdown_pct: float = Field(default=0.25, ge=0, le=1)
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


class RiskConfig(BaseModel):
    """Risk management configuration."""
    risk_free_rate: float = Field(default=0.02, ge=0)

    # VaR settings
    var_confidence: float = Field(default=0.95, ge=0, le=1)
    var_lookback: int = Field(default=252, gt=0)

    # Position limits
    max_single_position: float = Field(default=0.15, ge=0, le=1)
    max_sector_exposure: float = Field(default=0.4, ge=0, le=1)
    max_correlation: float = Field(default=0.7, ge=0, le=1)

    # Drawdown control
    daily_loss_limit: float = Field(default=0.05, ge=0, le=1)
    drawdown_stop: float = Field(default=0.20, ge=0, le=1)


class MLConfig(BaseModel):
    """Machine learning configuration."""
    train_split: float = Field(default=0.7, ge=0, le=1)
    validation_split: float = Field(default=0.15, ge=0, le=1)
    test_split: float = Field(default=0.15, ge=0, le=1)

    # Model training
    lookback_period: int = Field(default=252, gt=0)
    retrain_frequency: int = Field(default=20, gt=0)
    feature_selection_k: int = Field(default=20, gt=0)

    # Cross-validation
    cv_folds: int = Field(default=5, gt=0)
    cv_purge_gap: int = Field(default=5, ge=0)

    def __post_init__(self):
        assert abs(self.train_split + self.validation_split + self.test_split - 1.0) < 1e-6, \
            "Train, validation, and test splits must sum to 1.0"


class OptimizationConfig(BaseModel):
    """Portfolio optimization configuration."""
    method: str = "hrp"  # hrp, markowitz, black_litterman, risk_parity
    rebalance_frequency: int = Field(default=20, gt=0)  # trading days

    # Markowitz settings
    target_return: float | None = None
    target_volatility: float | None = None
    max_weight: float = Field(default=0.3, ge=0, le=1)
    min_weight: float = Field(default=0.0, ge=0, le=1)

    # Transaction costs
    consider_transaction_costs: bool = True
    turnover_penalty: float = Field(default=0.001, ge=0)


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    host: str = "localhost"
    port: int = Field(default=8501, ge=1024, le=65535)
    theme: str = "dark"  # dark, light

    # Update frequencies (seconds)
    refresh_rate: int = Field(default=5, gt=0)
    chart_update_rate: int = Field(default=1, gt=0)


class Config:
    """Main configuration class."""

    def __init__(self):
        self.data = DataConfig()
        self.backtest = BacktestConfig()
        self.risk = RiskConfig()
        self.ml = MLConfig()
        self.optimization = OptimizationConfig()
        self.dashboard = DashboardConfig()

        # API Keys
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": self.data.model_dump(),
            "backtest": self.backtest.model_dump(),
            "risk": self.risk.model_dump(),
            "ml": self.ml.model_dump(),
            "optimization": self.optimization.model_dump(),
            "dashboard": self.dashboard.model_dump(),
        }


# Global configuration instance
config = Config()
