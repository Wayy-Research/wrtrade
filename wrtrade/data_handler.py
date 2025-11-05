"""
Unified Data Handler for Mixed Asset Portfolios.
Supports both stocks/ETFs and crypto through Alpaca's different API methods.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import polars as pl
import numpy as np

from wrtrade.brokers_real import AlpacaBrokerAdapter, MarketData

logger = logging.getLogger(__name__)


class AssetType(Enum):
    """Asset type classification."""
    STOCK = "stock"
    ETF = "etf" 
    CRYPTO = "crypto"


@dataclass
class AssetInfo:
    """Information about an asset."""
    symbol: str
    asset_type: AssetType
    display_name: Optional[str] = None
    
    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.symbol


class UnifiedDataHandler:
    """
    Unified data handler for mixed asset portfolios.
    Handles both stocks/ETFs and crypto through appropriate Alpaca API methods.
    """
    
    def __init__(self, broker: AlpacaBrokerAdapter):
        self.broker = broker
        self.logger = logging.getLogger(f"{__name__}.UnifiedDataHandler")
        
        # Cache for recent data to avoid redundant API calls
        self._historical_cache = {}
        self._market_data_cache = {}
        self._cache_ttl_minutes = 5
        
    def define_mixed_portfolio(self, 
                              stocks: List[str] = None,
                              etfs: List[str] = None, 
                              crypto: List[str] = None) -> List[AssetInfo]:
        """
        Define a mixed portfolio with explicit asset type classification.
        
        Args:
            stocks: List of stock symbols (e.g., ['AAPL', 'TSLA'])
            etfs: List of ETF symbols (e.g., ['SPY', 'QQQ']) 
            crypto: List of crypto symbols (e.g., ['BTC/USD', 'ETH/USD'])
            
        Returns:
            List of AssetInfo objects for the portfolio
        """
        portfolio = []
        
        if stocks:
            portfolio.extend([
                AssetInfo(symbol, AssetType.STOCK, f"{symbol} Stock") 
                for symbol in stocks
            ])
            
        if etfs:
            portfolio.extend([
                AssetInfo(symbol, AssetType.ETF, f"{symbol} ETF") 
                for symbol in etfs
            ])
            
        if crypto:
            portfolio.extend([
                AssetInfo(symbol, AssetType.CRYPTO, f"{symbol} Crypto") 
                for symbol in crypto
            ])
            
        self.logger.info(f"Defined mixed portfolio with {len(portfolio)} assets:")
        for asset in portfolio:
            self.logger.info(f"  - {asset.display_name} ({asset.asset_type.value})")
            
        return portfolio
    
    async def get_historical_data(self, 
                                asset: AssetInfo,
                                start_date: datetime,
                                end_date: datetime,
                                timeframe: str = '1Day') -> pl.DataFrame:
        """
        Get historical data for any asset type using appropriate API method.
        
        Args:
            asset: AssetInfo object specifying the asset and its type
            start_date: Start date for historical data
            end_date: End date for historical data  
            timeframe: Timeframe ('1Day', '1Hour', '1Min')
            
        Returns:
            Polars DataFrame with historical data
        """
        # Check cache first
        cache_key = f"{asset.symbol}_{timeframe}_{start_date}_{end_date}"
        if cache_key in self._historical_cache:
            cache_time, data = self._historical_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=self._cache_ttl_minutes):
                self.logger.debug(f"Using cached data for {asset.symbol}")
                return data
        
        try:
            if asset.asset_type == AssetType.CRYPTO:
                data = await self._get_crypto_historical_data(asset, start_date, end_date, timeframe)
            else:  # STOCK or ETF
                data = await self._get_stock_historical_data(asset, start_date, end_date, timeframe)
            
            # Cache the result
            self._historical_cache[cache_key] = (datetime.now(), data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {asset.display_name}: {e}")
            return pl.DataFrame()
    
    async def _get_crypto_historical_data(self, 
                                        asset: AssetInfo,
                                        start_date: datetime, 
                                        end_date: datetime,
                                        timeframe: str) -> pl.DataFrame:
        """Get historical data for crypto using crypto-specific API."""
        if not self.broker.authenticated or not self.broker.client:
            raise RuntimeError("Broker not authenticated")
        
        await self.broker.rate_limiter.acquire()
        
        # Convert timeframe
        from alpaca_trade_api.rest import TimeFrame
        if timeframe == '1Day':
            tf = TimeFrame.Day
        elif timeframe == '1Hour':
            tf = TimeFrame.Hour
        elif timeframe == '1Min':
            tf = TimeFrame.Minute
        else:
            tf = TimeFrame.Day
        
        # Format dates for Alpaca API
        start_str = start_date.replace(microsecond=0).isoformat() + 'Z'
        end_str = end_date.replace(microsecond=0).isoformat() + 'Z'
        
        self.logger.info(f"Requesting crypto data: {asset.symbol} from {start_str} to {end_str}")
        
        # Use crypto-specific API method
        bars_request = self.broker.client.get_crypto_bars(
            asset.symbol,
            tf,
            start=start_str,
            end=end_str,
        )
        
        bars_df = bars_request.df
        self.logger.info(f"Alpaca returned {len(bars_df)} crypto bars for {asset.symbol}")
        
        if len(bars_df) > 0:
            self.logger.debug(f"Crypto data sample for {asset.symbol}: {bars_df.head()}")
            return pl.from_pandas(bars_df.reset_index())
        else:
            return pl.DataFrame()
    
    async def _get_stock_historical_data(self,
                                       asset: AssetInfo, 
                                       start_date: datetime,
                                       end_date: datetime,
                                       timeframe: str) -> pl.DataFrame:
        """Get historical data for stocks/ETFs using standard API."""
        # Use the existing broker method (which uses get_bars for stocks)
        return await self.broker.get_historical_data(asset.symbol, start_date, end_date, timeframe)
    
    async def get_market_data(self, assets: List[AssetInfo]) -> Dict[str, MarketData]:
        """
        Get real-time market data for mixed asset portfolio.
        
        Args:
            assets: List of AssetInfo objects
            
        Returns:
            Dictionary mapping symbol -> MarketData
        """
        # Check cache
        cache_key = "market_data_" + "_".join([a.symbol for a in assets])
        if cache_key in self._market_data_cache:
            cache_time, data = self._market_data_cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=30):  # 30 second cache for market data
                return data
        
        # Separate assets by type
        crypto_assets = [a for a in assets if a.asset_type == AssetType.CRYPTO]
        stock_assets = [a for a in assets if a.asset_type in (AssetType.STOCK, AssetType.ETF)]
        
        market_data = {}
        
        # Get crypto market data
        if crypto_assets:
            crypto_data = await self._get_crypto_market_data(crypto_assets)
            market_data.update(crypto_data)
        
        # Get stock market data  
        if stock_assets:
            stock_symbols = [a.symbol for a in stock_assets]
            stock_data = await self.broker.get_market_data(stock_symbols)
            market_data.update(stock_data)
        
        # Cache the result
        self._market_data_cache[cache_key] = (datetime.now(), market_data)
        
        self.logger.info(f"Retrieved market data for {len(market_data)} assets")
        return market_data
    
    async def _get_crypto_market_data(self, crypto_assets: List[AssetInfo]) -> Dict[str, MarketData]:
        """Get real-time market data for crypto assets."""
        if not self.broker.authenticated or not self.broker.client:
            raise RuntimeError("Broker not authenticated")
        
        try:
            await self.broker.rate_limiter.acquire()
            
            crypto_symbols = [asset.symbol for asset in crypto_assets]
            
            # Use crypto-specific API method
            snapshots = self.broker.client.get_crypto_snapshots(crypto_symbols)
            
            result = {}
            for symbol in crypto_symbols:
                if hasattr(snapshots, symbol.replace('/', '').lower()):
                    snapshot = getattr(snapshots, symbol.replace('/', '').lower())
                    if snapshot and hasattr(snapshot, 'latest_quote') and snapshot.latest_quote:
                        quote = snapshot.latest_quote
                        result[symbol] = MarketData(
                            symbol=symbol,
                            price=(quote.bid_price + quote.ask_price) / 2,
                            bid=quote.bid_price,
                            ask=quote.ask_price,
                            bid_size=quote.bid_size,
                            ask_size=quote.ask_size,
                            timestamp=datetime.now()
                        )
                        
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get crypto market data: {e}")
            return {}
    
    async def get_portfolio_historical_data(self,
                                          portfolio: List[AssetInfo],
                                          start_date: datetime,
                                          end_date: datetime,
                                          timeframe: str = '1Day') -> Dict[str, pl.DataFrame]:
        """
        Get historical data for entire mixed portfolio.
        
        Args:
            portfolio: List of AssetInfo objects
            start_date: Start date
            end_date: End date  
            timeframe: Timeframe
            
        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        self.logger.info(f"Fetching historical data for {len(portfolio)} assets in mixed portfolio")
        
        # Fetch data for all assets concurrently
        tasks = []
        for asset in portfolio:
            task = self.get_historical_data(asset, start_date, end_date, timeframe)
            tasks.append((asset.symbol, task))
        
        # Wait for all data to be fetched
        results = {}
        for symbol, task in tasks:
            try:
                data = await task
                results[symbol] = data
                if not data.is_empty():
                    self.logger.info(f"✅ {symbol}: {len(data)} data points")
                else:
                    self.logger.warning(f"❌ {symbol}: No data available")
            except Exception as e:
                self.logger.error(f"❌ {symbol}: Error fetching data - {e}")
                results[symbol] = pl.DataFrame()
            
            # Rate limiting between requests
            await asyncio.sleep(0.1)
        
        return results
    
    def clear_cache(self):
        """Clear all cached data."""
        self._historical_cache.clear()
        self._market_data_cache.clear()
        self.logger.info("Data cache cleared")