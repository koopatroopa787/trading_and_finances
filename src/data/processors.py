"""Data processing utilities for cleaning and corporate actions."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from loguru import logger
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """Process and clean market data."""

    @staticmethod
    def clean_data(
        df: pd.DataFrame,
        fill_method: str = "ffill",
        drop_na: bool = False,
        remove_outliers: bool = True,
        outlier_std: float = 5.0
    ) -> pd.DataFrame:
        """
        Clean market data.

        Args:
            df: Input DataFrame
            fill_method: Method to fill missing values (ffill, bfill, interpolate)
            drop_na: Whether to drop remaining NaN values
            remove_outliers: Whether to remove statistical outliers
            outlier_std: Number of standard deviations for outlier detection

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Remove duplicate indices
        df = df[~df.index.duplicated(keep='first')]

        # Sort by index
        df = df.sort_index()

        # Handle missing values
        if fill_method == "ffill":
            df = df.ffill()
        elif fill_method == "bfill":
            df = df.bfill()
        elif fill_method == "interpolate":
            df = df.interpolate(method='time')

        # Remove outliers (for returns, not prices)
        if remove_outliers and 'close' in df.columns:
            returns = df['close'].pct_change()
            mean_return = returns.mean()
            std_return = returns.std()

            outlier_mask = np.abs(returns - mean_return) > (outlier_std * std_return)
            if outlier_mask.sum() > 0:
                logger.warning(f"Removing {outlier_mask.sum()} outliers")
                df = df[~outlier_mask]

        # Drop remaining NaN if requested
        if drop_na:
            df = df.dropna()

        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df

    @staticmethod
    def adjust_for_splits(
        df: pd.DataFrame,
        splits: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Adjust prices for stock splits.

        Args:
            df: DataFrame with OHLCV data
            splits: Dict mapping dates to split ratios (e.g., {'2020-01-15': 2.0})

        Returns:
            Adjusted DataFrame
        """
        if splits is None or len(splits) == 0:
            return df

        df = df.copy()

        for split_date, ratio in splits.items():
            split_dt = pd.to_datetime(split_date)
            mask = df.index < split_dt

            # Adjust prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] / ratio

            # Adjust volume
            if 'volume' in df.columns:
                df.loc[mask, 'volume'] = df.loc[mask, 'volume'] * ratio

        logger.info(f"Applied {len(splits)} stock splits")
        return df

    @staticmethod
    def adjust_for_dividends(
        df: pd.DataFrame,
        dividends: Optional[Dict[str, float]] = None,
        adjust_method: str = "backward"
    ) -> pd.DataFrame:
        """
        Adjust prices for dividends.

        Args:
            df: DataFrame with OHLCV data
            dividends: Dict mapping dates to dividend amounts
            adjust_method: 'backward' or 'forward' adjustment

        Returns:
            Adjusted DataFrame
        """
        if dividends is None or len(dividends) == 0:
            return df

        df = df.copy()

        # Sort dividends by date
        sorted_divs = sorted(dividends.items(), key=lambda x: pd.to_datetime(x[0]))

        for div_date, div_amount in sorted_divs:
            div_dt = pd.to_datetime(div_date)

            # Find the close price on ex-dividend date
            if div_dt not in df.index:
                continue

            close_price = df.loc[div_dt, 'close']
            adjustment_factor = 1 - (div_amount / close_price)

            if adjust_method == "backward":
                mask = df.index < div_dt
            else:
                mask = df.index >= div_dt

            # Adjust prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] * adjustment_factor

        logger.info(f"Applied {len(dividends)} dividend adjustments")
        return df

    @staticmethod
    def resample_data(
        df: pd.DataFrame,
        freq: str = "1D",
        agg_method: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Resample data to different frequency.

        Args:
            df: Input DataFrame
            freq: Target frequency (e.g., '1D', '1H', '5T')
            agg_method: Dict mapping column names to aggregation methods

        Returns:
            Resampled DataFrame
        """
        if agg_method is None:
            agg_method = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }

        # Only resample columns that exist
        agg_method = {k: v for k, v in agg_method.items() if k in df.columns}

        df_resampled = df.resample(freq).agg(agg_method)
        df_resampled = df_resampled.dropna()

        logger.info(f"Resampled to {freq}: {len(df_resampled)} rows")
        return df_resampled

    @staticmethod
    def detect_gaps(
        df: pd.DataFrame,
        freq: str = "1D",
        max_gap: int = 5
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Detect gaps in time series data.

        Args:
            df: Input DataFrame
            freq: Expected frequency
            max_gap: Maximum acceptable gap in periods

        Returns:
            List of (start, end) tuples for gaps
        """
        df = df.sort_index()
        expected_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )

        missing_dates = expected_index.difference(df.index)
        gaps = []

        if len(missing_dates) > 0:
            # Group consecutive missing dates
            current_gap_start = missing_dates[0]
            current_gap_end = missing_dates[0]

            for i in range(1, len(missing_dates)):
                if (missing_dates[i] - missing_dates[i-1]).days <= 1:
                    current_gap_end = missing_dates[i]
                else:
                    gap_size = (current_gap_end - current_gap_start).days + 1
                    if gap_size >= max_gap:
                        gaps.append((current_gap_start, current_gap_end))
                    current_gap_start = missing_dates[i]
                    current_gap_end = missing_dates[i]

            # Add last gap
            gap_size = (current_gap_end - current_gap_start).days + 1
            if gap_size >= max_gap:
                gaps.append((current_gap_start, current_gap_end))

        if gaps:
            logger.warning(f"Found {len(gaps)} gaps in data")

        return gaps

    @staticmethod
    def align_data(
        dfs: List[pd.DataFrame],
        method: str = "inner",
        fill_method: str = "ffill"
    ) -> List[pd.DataFrame]:
        """
        Align multiple DataFrames to common time index.

        Args:
            dfs: List of DataFrames to align
            method: Join method ('inner', 'outer')
            fill_method: Method to fill missing values

        Returns:
            List of aligned DataFrames
        """
        if len(dfs) == 0:
            return []

        # Get common index
        if method == "inner":
            common_index = dfs[0].index
            for df in dfs[1:]:
                common_index = common_index.intersection(df.index)
        else:  # outer
            common_index = dfs[0].index
            for df in dfs[1:]:
                common_index = common_index.union(df.index)

        common_index = common_index.sort_values()

        # Reindex all DataFrames
        aligned_dfs = []
        for df in dfs:
            df_aligned = df.reindex(common_index)

            # Fill missing values
            if fill_method == 'ffill':
                df_aligned = df_aligned.ffill()
            elif fill_method == 'bfill':
                df_aligned = df_aligned.bfill()

            aligned_dfs.append(df_aligned)

        logger.info(f"Aligned {len(dfs)} DataFrames to {len(common_index)} rows")
        return aligned_dfs

    @staticmethod
    def calculate_returns(
        df: pd.DataFrame,
        method: str = "simple",
        periods: int = 1
    ) -> pd.Series:
        """
        Calculate returns from price data.

        Args:
            df: DataFrame with 'close' column
            method: 'simple' or 'log' returns
            periods: Number of periods for return calculation

        Returns:
            Series of returns
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        if method == "simple":
            returns = df['close'].pct_change(periods)
        elif method == "log":
            returns = np.log(df['close'] / df['close'].shift(periods))
        else:
            raise ValueError(f"Unknown method: {method}")

        return returns

    @staticmethod
    def save_to_parquet(
        df: pd.DataFrame,
        path: Path,
        compression: str = "snappy"
    ):
        """Save DataFrame to Parquet format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression=compression, index=True)
        logger.info(f"Saved to {path}")

    @staticmethod
    def save_to_hdf5(
        df: pd.DataFrame,
        path: Path,
        key: str = "data",
        mode: str = "w"
    ):
        """Save DataFrame to HDF5 format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_hdf(path, key=key, mode=mode, format='table')
        logger.info(f"Saved to {path}")

    @staticmethod
    def load_from_parquet(path: Path) -> pd.DataFrame:
        """Load DataFrame from Parquet format."""
        df = pd.read_parquet(path)
        logger.info(f"Loaded from {path}: {len(df)} rows")
        return df

    @staticmethod
    def load_from_hdf5(path: Path, key: str = "data") -> pd.DataFrame:
        """Load DataFrame from HDF5 format."""
        df = pd.read_hdf(path, key=key)
        logger.info(f"Loaded from {path}: {len(df)} rows")
        return df


class PointInTimeData:
    """
    Ensure point-in-time correctness to prevent look-ahead bias.

    This class manages data availability timestamps to ensure that
    backtesting only uses information that would have been available
    at the time of decision-making.
    """

    def __init__(self, df: pd.DataFrame, delay_minutes: int = 0):
        """
        Initialize point-in-time data manager.

        Args:
            df: DataFrame with market data
            delay_minutes: Data delay in minutes (e.g., 15 for delayed quotes)
        """
        self.df = df.copy()
        self.delay = pd.Timedelta(minutes=delay_minutes)

    def get_data_at(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        """
        Get data available at given timestamp.

        Args:
            timestamp: Query timestamp

        Returns:
            DataFrame with data available at timestamp
        """
        available_until = timestamp - self.delay
        return self.df[self.df.index <= available_until]

    def is_available(self, timestamp: pd.Timestamp, data_timestamp: pd.Timestamp) -> bool:
        """Check if data at data_timestamp is available at timestamp."""
        return data_timestamp <= (timestamp - self.delay)
