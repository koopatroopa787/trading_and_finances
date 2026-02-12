"""Data loaders for multiple sources."""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from loguru import logger
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import ccxt
from tqdm import tqdm
import time


class BaseLoader(ABC):
    """Base class for data loaders."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize loader with optional caching."""
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load(
        self,
        symbols: Union[str, List[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        **kwargs
    ) -> pd.DataFrame:
        """Load data for given symbols."""
        pass

    def _get_cache_path(self, symbol: str, interval: str) -> Path:
        """Get cache file path for symbol."""
        return self.cache_dir / f"{symbol}_{interval}.parquet"

    def _load_from_cache(
        self, symbol: str, interval: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(symbol, interval)
        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index)

            # Filter by date range
            start_dt = pd.to_datetime(start) if start else df.index.min()
            end_dt = pd.to_datetime(end) if end else df.index.max()
            df = df.loc[start_dt:end_dt]

            logger.debug(f"Loaded {symbol} from cache: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Error loading cache for {symbol}: {e}")
            return None

    def _save_to_cache(self, df: pd.DataFrame, symbol: str, interval: str):
        """Save data to cache."""
        try:
            cache_path = self._get_cache_path(symbol, interval)
            df.to_parquet(cache_path)
            logger.debug(f"Saved {symbol} to cache: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Error saving cache for {symbol}: {e}")


class YahooFinanceLoader(BaseLoader):
    """Loader for Yahoo Finance data."""

    def load(
        self,
        symbols: Union[str, List[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.

        Args:
            symbols: Single symbol or list of symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Set default date range
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        all_data = {}

        for symbol in tqdm(symbols, desc="Loading Yahoo Finance data"):
            # Try cache first
            if use_cache:
                cached_data = self._load_from_cache(symbol, interval, start, end)
                if cached_data is not None and len(cached_data) > 0:
                    all_data[symbol] = cached_data
                    continue

            # Download from Yahoo Finance
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, interval=interval)

                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                # Standardize column names
                df.columns = [col.lower() for col in df.columns]
                df.index.name = "date"

                # Save to cache
                if use_cache:
                    self._save_to_cache(df, symbol, interval)

                all_data[symbol] = df
                logger.info(f"Loaded {symbol}: {len(df)} rows")

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No data loaded for any symbols")

        # Combine data
        if len(all_data) == 1:
            return list(all_data.values())[0]
        else:
            # Multi-symbol: create MultiIndex DataFrame
            combined = pd.concat(all_data, axis=1, keys=all_data.keys())
            return combined


class AlphaVantageLoader(BaseLoader):
    """Loader for Alpha Vantage data."""

    def __init__(self, api_key: str, cache_dir: Optional[Path] = None):
        """
        Initialize Alpha Vantage loader.

        Args:
            api_key: Alpha Vantage API key
        """
        super().__init__(cache_dir)
        if not api_key:
            raise ValueError("Alpha Vantage API key required")
        self.ts = TimeSeries(key=api_key, output_format='pandas')

    def load(
        self,
        symbols: Union[str, List[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from Alpha Vantage.

        Args:
            symbols: Single symbol or list of symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        all_data = {}

        for symbol in tqdm(symbols, desc="Loading Alpha Vantage data"):
            # Try cache first
            if use_cache:
                cached_data = self._load_from_cache(symbol, interval, start or "1900-01-01", end or "2100-01-01")
                if cached_data is not None and len(cached_data) > 0:
                    all_data[symbol] = cached_data
                    continue

            # Download from Alpha Vantage
            try:
                if interval in ["1min", "5min", "15min", "30min", "60min"]:
                    df, meta = self.ts.get_intraday(symbol, interval=interval, outputsize='full')
                elif interval == "daily" or interval == "1d":
                    df, meta = self.ts.get_daily(symbol, outputsize='full')
                elif interval == "weekly" or interval == "1wk":
                    df, meta = self.ts.get_weekly(symbol)
                else:  # monthly
                    df, meta = self.ts.get_monthly(symbol)

                # Standardize column names
                df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]
                df.columns = [col.lower() for col in df.columns]
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                # Filter by date range
                if start:
                    df = df[df.index >= pd.to_datetime(start)]
                if end:
                    df = df[df.index <= pd.to_datetime(end)]

                # Save to cache
                if use_cache:
                    self._save_to_cache(df, symbol, interval)

                all_data[symbol] = df
                logger.info(f"Loaded {symbol}: {len(df)} rows")

                # Rate limiting (Alpha Vantage: 5 calls/min for free tier)
                time.sleep(12)

            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No data loaded for any symbols")

        # Combine data
        if len(all_data) == 1:
            return list(all_data.values())[0]
        else:
            combined = pd.concat(all_data, axis=1, keys=all_data.keys())
            return combined


class CCXTLoader(BaseLoader):
    """Loader for cryptocurrency data via CCXT."""

    def __init__(
        self,
        exchange: str = "binance",
        cache_dir: Optional[Path] = None,
        **exchange_kwargs
    ):
        """
        Initialize CCXT loader.

        Args:
            exchange: Exchange name (binance, coinbase, kraken, etc.)
            exchange_kwargs: Additional exchange parameters
        """
        super().__init__(cache_dir)
        exchange_class = getattr(ccxt, exchange)
        self.exchange = exchange_class(exchange_kwargs)

    def load(
        self,
        symbols: Union[str, List[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load cryptocurrency data from exchange.

        Args:
            symbols: Single symbol or list of symbols (e.g., 'BTC/USDT')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Convert interval to CCXT format
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
        }
        timeframe = interval_map.get(interval, "1d")

        all_data = {}

        for symbol in tqdm(symbols, desc=f"Loading {self.exchange.name} data"):
            # Try cache first
            if use_cache:
                cached_data = self._load_from_cache(
                    symbol.replace("/", "_"), interval,
                    start or "1900-01-01", end or "2100-01-01"
                )
                if cached_data is not None and len(cached_data) > 0:
                    all_data[symbol] = cached_data
                    continue

            # Download from exchange
            try:
                # Convert dates to timestamps
                since = None
                if start:
                    since = int(pd.to_datetime(start).timestamp() * 1000)

                until = None
                if end:
                    until = int(pd.to_datetime(end).timestamp() * 1000)

                # Fetch OHLCV data
                all_ohlcv = []
                current_since = since

                while True:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, timeframe, since=current_since, limit=1000
                    )

                    if not ohlcv:
                        break

                    all_ohlcv.extend(ohlcv)

                    # Check if we've reached the end
                    last_timestamp = ohlcv[-1][0]
                    if until and last_timestamp >= until:
                        break

                    # Move to next batch
                    current_since = last_timestamp + 1

                    # Rate limiting
                    time.sleep(self.exchange.rateLimit / 1000)

                    if len(ohlcv) < 1000:
                        break

                # Convert to DataFrame
                df = pd.DataFrame(
                    all_ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("date", inplace=True)
                df.drop("timestamp", axis=1, inplace=True)

                # Filter by end date
                if end:
                    df = df[df.index <= pd.to_datetime(end)]

                # Save to cache
                if use_cache:
                    self._save_to_cache(df, symbol.replace("/", "_"), interval)

                all_data[symbol] = df
                logger.info(f"Loaded {symbol}: {len(df)} rows")

            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No data loaded for any symbols")

        # Combine data
        if len(all_data) == 1:
            return list(all_data.values())[0]
        else:
            combined = pd.concat(all_data, axis=1, keys=all_data.keys())
            return combined


class DataAggregator:
    """Aggregate data from multiple sources."""

    def __init__(self):
        self.loaders: Dict[str, BaseLoader] = {}

    def add_loader(self, name: str, loader: BaseLoader):
        """Add a data loader."""
        self.loaders[name] = loader

    def load_all(
        self,
        sources: Dict[str, List[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from multiple sources.

        Args:
            sources: Dict mapping source name to list of symbols
            start: Start date
            end: End date
            interval: Data interval

        Returns:
            Dict mapping source name to DataFrame
        """
        results = {}

        for source_name, symbols in sources.items():
            if source_name not in self.loaders:
                logger.warning(f"No loader for {source_name}")
                continue

            try:
                data = self.loaders[source_name].load(
                    symbols, start=start, end=end, interval=interval
                )
                results[source_name] = data
            except Exception as e:
                logger.error(f"Error loading from {source_name}: {e}")

        return results
