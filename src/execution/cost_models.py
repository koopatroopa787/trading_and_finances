"""Transaction cost models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional


@dataclass
class TransactionCost:
    """Transaction cost breakdown."""
    commission: float
    slippage: float
    market_impact: float
    spread_cost: float

    @property
    def total(self) -> float:
        """Total transaction cost."""
        return self.commission + self.slippage + self.market_impact + self.spread_cost


class CostModel(ABC):
    """Base class for transaction cost models."""

    @abstractmethod
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        side: str,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> TransactionCost:
        """Calculate transaction costs."""
        pass


class FixedCostModel(CostModel):
    """Fixed transaction cost model."""

    def __init__(
        self,
        commission_rate: float = 0.001,
        min_commission: float = 1.0,
        slippage_bps: float = 5.0,
    ):
        """
        Initialize fixed cost model.

        Args:
            commission_rate: Commission as fraction of trade value
            min_commission: Minimum commission per trade
            slippage_bps: Slippage in basis points
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps

    def calculate_cost(
        self,
        quantity: float,
        price: float,
        side: str,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> TransactionCost:
        """Calculate transaction costs."""
        trade_value = quantity * price

        # Commission
        commission = max(trade_value * self.commission_rate, self.min_commission)

        # Slippage (in bps)
        slippage = trade_value * (self.slippage_bps / 10000)

        return TransactionCost(
            commission=commission,
            slippage=slippage,
            market_impact=0.0,
            spread_cost=0.0,
        )


class VolumeCostModel(CostModel):
    """Volume-based transaction cost model."""

    def __init__(
        self,
        commission_rate: float = 0.001,
        min_commission: float = 1.0,
        base_slippage_bps: float = 2.0,
        volatility_multiplier: float = 10.0,
        volume_impact_factor: float = 0.1,
        spread_bps: float = 2.0,
    ):
        """
        Initialize volume-based cost model.

        Args:
            commission_rate: Commission as fraction of trade value
            min_commission: Minimum commission per trade
            base_slippage_bps: Base slippage in basis points
            volatility_multiplier: Multiplier for volatility-based slippage
            volume_impact_factor: Factor for volume-based market impact
            spread_bps: Bid-ask spread in basis points
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.base_slippage_bps = base_slippage_bps
        self.volatility_multiplier = volatility_multiplier
        self.volume_impact_factor = volume_impact_factor
        self.spread_bps = spread_bps

    def calculate_cost(
        self,
        quantity: float,
        price: float,
        side: str,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> TransactionCost:
        """Calculate transaction costs."""
        trade_value = quantity * price

        # Commission
        commission = max(trade_value * self.commission_rate, self.min_commission)

        # Volatility-based slippage
        if volatility is not None:
            slippage_bps = self.base_slippage_bps + (volatility * self.volatility_multiplier * 10000)
        else:
            slippage_bps = self.base_slippage_bps

        slippage = trade_value * (slippage_bps / 10000)

        # Market impact (based on trade size relative to volume)
        market_impact = 0.0
        if volume is not None and volume > 0:
            participation_rate = quantity / volume
            # Square root impact model
            market_impact = trade_value * self.volume_impact_factor * np.sqrt(participation_rate)

        # Spread cost (pay half the spread)
        spread_cost = trade_value * (self.spread_bps / 10000) * 0.5

        return TransactionCost(
            commission=commission,
            slippage=slippage,
            market_impact=market_impact,
            spread_cost=spread_cost,
        )


class NonLinearImpactModel(CostModel):
    """Non-linear market impact model (more realistic for large orders)."""

    def __init__(
        self,
        commission_rate: float = 0.001,
        min_commission: float = 1.0,
        base_slippage_bps: float = 2.0,
        permanent_impact_coef: float = 0.1,
        temporary_impact_coef: float = 0.05,
        spread_bps: float = 2.0,
    ):
        """
        Initialize non-linear impact model.

        Args:
            commission_rate: Commission as fraction of trade value
            min_commission: Minimum commission per trade
            base_slippage_bps: Base slippage in basis points
            permanent_impact_coef: Coefficient for permanent market impact
            temporary_impact_coef: Coefficient for temporary market impact
            spread_bps: Bid-ask spread in basis points
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.base_slippage_bps = base_slippage_bps
        self.permanent_impact_coef = permanent_impact_coef
        self.temporary_impact_coef = temporary_impact_coef
        self.spread_bps = spread_bps

    def calculate_cost(
        self,
        quantity: float,
        price: float,
        side: str,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> TransactionCost:
        """Calculate transaction costs with non-linear impact."""
        trade_value = quantity * price

        # Commission
        commission = max(trade_value * self.commission_rate, self.min_commission)

        # Base slippage
        slippage = trade_value * (self.base_slippage_bps / 10000)

        # Market impact (non-linear)
        market_impact = 0.0
        if volume is not None and volume > 0:
            participation_rate = quantity / volume

            # Permanent impact (square root)
            permanent = (
                trade_value *
                self.permanent_impact_coef *
                np.sqrt(participation_rate)
            )

            # Temporary impact (linear)
            temporary = (
                trade_value *
                self.temporary_impact_coef *
                participation_rate
            )

            market_impact = permanent + temporary

        # Spread cost
        spread_cost = trade_value * (self.spread_bps / 10000) * 0.5

        return TransactionCost(
            commission=commission,
            slippage=slippage,
            market_impact=market_impact,
            spread_cost=spread_cost,
        )
