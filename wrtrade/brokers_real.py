"""
Real broker adapters with actual API communication for Alpaca and Robinhood.
This module implements production-ready broker integrations with proper error handling,
rate limiting, and comprehensive logging.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
import json
from enum import Enum
import os
from pathlib import Path

# Third-party broker SDKs
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Alpaca Trade API not installed. Install with: pip install alpaca-trade-api")

try:
    import robin_stocks.robinhood as rh
    ROBINHOOD_AVAILABLE = True
except ImportError:
    ROBINHOOD_AVAILABLE = False
    print("Robin Stocks not installed. Install with: pip install robin-stocks")


logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported across brokers."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Standardized order status across brokers."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    side: str = "long"  # 'long' or 'short'
    cost_basis: float = 0.0
    
    def __post_init__(self):
        if self.cost_basis == 0.0:
            self.cost_basis = abs(self.quantity) * self.avg_price
        if self.unrealized_pnl == 0.0:
            self.unrealized_pnl = self.market_value - self.cost_basis


@dataclass
class Order:
    """Represents a trading order with comprehensive parameters."""
    symbol: str
    quantity: float
    side: str  # 'buy' or 'sell'
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    extended_hours: bool = False
    
    # Order status tracking
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    avg_fill_price: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    fees: float = 0.0
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class AccountInfo:
    """Comprehensive account information."""
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    day_trade_buying_power: float = 0.0
    is_day_trader: bool = False
    day_trades_remaining: int = 3
    maintenance_margin: float = 0.0
    initial_margin: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class MarketData:
    """Real-time market data."""
    symbol: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    volume: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class BrokerConfig:
    """Configuration for broker adapters."""
    paper_trading: bool = True
    max_requests_per_minute: int = 200
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    enable_logging: bool = True
    log_level: str = "INFO"


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        # If we're at the limit, wait
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0]) + 0.1
            logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
            await asyncio.sleep(sleep_time)
            return await self.acquire()
        
        self.requests.append(now)


class BrokerAdapter(ABC):
    """
    Abstract base class for broker adapters with comprehensive functionality.
    """
    
    def __init__(self, api_credentials: Dict[str, str], config: Optional[BrokerConfig] = None):
        """Initialize broker adapter."""
        self.credentials = api_credentials
        self.config = config or BrokerConfig()
        self.authenticated = False
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute)
        
        # Setup logging
        if self.config.enable_logging:
            logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Connection health tracking
        self.last_heartbeat = None
        self.connection_errors = 0
        self.max_connection_errors = 5
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Establish connection to broker."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        await self.authenticate()
    
    async def disconnect(self):
        """Clean up connections."""
        if self.session:
            await self.session.close()
            self.session = None
        self.authenticated = False
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with broker API."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> bool:
        """Place a trading order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of an order."""
        pass
    
    @abstractmethod
    async def get_orders(self, symbol: Optional[str] = None, 
                        limit: int = 100) -> List[Order]:
        """Get order history."""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbols: Union[str, List[str]]) -> Dict[str, MarketData]:
        """Get real-time market data."""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = '1Day'
    ) -> pl.DataFrame:
        """Get historical price data."""
        pass
    
    async def health_check(self) -> bool:
        """Check broker connection health."""
        try:
            await self.get_account_info()
            self.last_heartbeat = datetime.now()
            self.connection_errors = 0
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.connection_errors += 1
            return False
    
    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """Validate order before submission."""
        if order.quantity <= 0:
            return False, "Order quantity must be positive"
        
        if order.side not in ['buy', 'sell']:
            return False, "Order side must be 'buy' or 'sell'"
        
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            return False, "Limit price required for limit orders"
        
        if order.order_type == OrderType.STOP and order.stop_price is None:
            return False, "Stop price required for stop orders"
        
        if order.order_type == OrderType.STOP_LIMIT:
            if order.limit_price is None or order.stop_price is None:
                return False, "Both limit and stop price required for stop limit orders"
        
        if order.order_type == OrderType.TRAILING_STOP and order.trail_percent is None:
            return False, "Trail percent required for trailing stop orders"
        
        return True, "Valid order"


class AlpacaBrokerAdapter(BrokerAdapter):
    """
    Alpaca broker adapter with comprehensive API integration.
    Supports both paper and live trading.
    """
    
    def __init__(self, api_credentials: Dict[str, str], config: Optional[BrokerConfig] = None):
        """Initialize Alpaca adapter."""
        super().__init__(api_credentials, config)
        
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca Trade API not available. Install with: pip install alpaca-trade-api")
        
        self.api_key = api_credentials.get('api_key')
        self.secret_key = api_credentials.get('secret_key')
        self.base_url = api_credentials.get('base_url', 
            'https://paper-api.alpaca.markets' if config and config.paper_trading 
            else 'https://api.alpaca.markets'
        )
        
        self.client: Optional[tradeapi.REST] = None
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret key are required")
    
    async def authenticate(self) -> bool:
        """Authenticate with Alpaca API."""
        try:
            await self.rate_limiter.acquire()
            
            self.client = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Test authentication by getting account info
            account = self.client.get_account()
            self.authenticated = True
            logger.info(f"Alpaca authentication successful. Account: {account.account_number}")
            return True
            
        except Exception as e:
            logger.error(f"Alpaca authentication failed: {e}")
            self.authenticated = False
            return False
    
    async def get_account_info(self) -> AccountInfo:
        """Get Alpaca account information."""
        if not self.authenticated or not self.client:
            raise RuntimeError("Not authenticated with Alpaca")
        
        try:
            await self.rate_limiter.acquire()
            account = self.client.get_account()
            
            return AccountInfo(
                account_id=account.account_number,
                buying_power=float(account.buying_power),
                cash=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                equity=float(account.equity),
                day_trade_buying_power=float(account.daytrading_buying_power),
                is_day_trader=account.pattern_day_trader,
                maintenance_margin=float(account.maintenance_margin or 0),
                initial_margin=float(account.initial_margin or 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get Alpaca account info: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get current Alpaca positions."""
        if not self.authenticated or not self.client:
            raise RuntimeError("Not authenticated with Alpaca")
        
        try:
            await self.rate_limiter.acquire()
            positions = self.client.list_positions()
            
            result = []
            for pos in positions:
                result.append(Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    avg_price=float(pos.avg_entry_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    side='long' if float(pos.qty) > 0 else 'short',
                    cost_basis=float(pos.cost_basis)
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get Alpaca positions: {e}")
            raise
    
    async def place_order(self, order: Order) -> bool:
        """Place order with Alpaca."""
        if not self.authenticated or not self.client:
            raise RuntimeError("Not authenticated with Alpaca")
        
        # Validate order
        is_valid, error_msg = self.validate_order(order)
        if not is_valid:
            logger.error(f"Invalid order: {error_msg}")
            order.status = OrderStatus.REJECTED
            return False
        
        try:
            await self.rate_limiter.acquire()
            
            # Convert our order to Alpaca format
            alpaca_order_params = {
                'symbol': order.symbol,
                'qty': int(order.quantity),
                'side': order.side,
                'type': order.order_type.value,
                'time_in_force': order.time_in_force.value,
                'extended_hours': order.extended_hours
            }
            
            # Add price parameters based on order type
            if order.order_type == OrderType.LIMIT:
                alpaca_order_params['limit_price'] = order.limit_price
            elif order.order_type == OrderType.STOP:
                alpaca_order_params['stop_price'] = order.stop_price
            elif order.order_type == OrderType.STOP_LIMIT:
                alpaca_order_params['limit_price'] = order.limit_price
                alpaca_order_params['stop_price'] = order.stop_price
            elif order.order_type == OrderType.TRAILING_STOP:
                alpaca_order_params['trail_percent'] = order.trail_percent
            
            # Submit order
            submitted_order = self.client.submit_order(**alpaca_order_params)
            
            # Update order with response data
            order.order_id = submitted_order.id
            order.status = OrderStatus(submitted_order.status.lower())
            order.created_at = submitted_order.created_at
            order.updated_at = submitted_order.updated_at
            
            # Update fill information if available
            if hasattr(submitted_order, 'filled_qty') and submitted_order.filled_qty:
                order.filled_quantity = float(submitted_order.filled_qty)
            if hasattr(submitted_order, 'filled_avg_price') and submitted_order.filled_avg_price:
                order.avg_fill_price = float(submitted_order.filled_avg_price)
            
            logger.info(f"Order placed successfully: {order.order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to place Alpaca order: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel Alpaca order."""
        if not self.authenticated or not self.client:
            raise RuntimeError("Not authenticated with Alpaca")
        
        try:
            await self.rate_limiter.acquire()
            self.client.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel Alpaca order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get Alpaca order status."""
        if not self.authenticated or not self.client:
            raise RuntimeError("Not authenticated with Alpaca")
        
        try:
            await self.rate_limiter.acquire()
            alpaca_order = self.client.get_order(order_id)
            
            order = Order(
                symbol=alpaca_order.symbol,
                quantity=float(alpaca_order.qty),
                side=alpaca_order.side,
                order_type=OrderType(alpaca_order.order_type),
                time_in_force=TimeInForce(alpaca_order.time_in_force),
                order_id=alpaca_order.id,
                status=OrderStatus(alpaca_order.status.lower()),
                filled_quantity=float(alpaca_order.filled_qty or 0),
                avg_fill_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                created_at=alpaca_order.created_at,
                updated_at=alpaca_order.updated_at
            )
            
            if alpaca_order.limit_price:
                order.limit_price = float(alpaca_order.limit_price)
            if alpaca_order.stop_price:
                order.stop_price = float(alpaca_order.stop_price)
            if hasattr(alpaca_order, 'trail_percent') and alpaca_order.trail_percent:
                order.trail_percent = float(alpaca_order.trail_percent)
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to get Alpaca order status {order_id}: {e}")
            return None
    
    async def get_orders(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get Alpaca order history."""
        if not self.authenticated or not self.client:
            raise RuntimeError("Not authenticated with Alpaca")
        
        try:
            await self.rate_limiter.acquire()
            
            # Get orders from Alpaca
            orders = self.client.list_orders(
                status='all',
                limit=limit,
                direction='desc'
            )
            
            result = []
            for alpaca_order in orders:
                # Filter by symbol if specified
                if symbol and alpaca_order.symbol != symbol:
                    continue
                
                order = Order(
                    symbol=alpaca_order.symbol,
                    quantity=float(alpaca_order.qty),
                    side=alpaca_order.side,
                    order_type=OrderType(alpaca_order.order_type),
                    time_in_force=TimeInForce(alpaca_order.time_in_force),
                    order_id=alpaca_order.id,
                    status=OrderStatus(alpaca_order.status.lower()),
                    filled_quantity=float(alpaca_order.filled_qty or 0),
                    avg_fill_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                    created_at=alpaca_order.created_at,
                    updated_at=alpaca_order.updated_at
                )
                
                if alpaca_order.limit_price:
                    order.limit_price = float(alpaca_order.limit_price)
                if alpaca_order.stop_price:
                    order.stop_price = float(alpaca_order.stop_price)
                
                result.append(order)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get Alpaca orders: {e}")
            return []
    
    async def get_market_data(self, symbols: Union[str, List[str]]) -> Dict[str, MarketData]:
        """Get real-time market data from Alpaca."""
        if not self.authenticated or not self.client:
            raise RuntimeError("Not authenticated with Alpaca")
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        try:
            await self.rate_limiter.acquire()
            
            # Get latest quotes
            quotes = self.client.get_latest_quotes(symbols)
            
            result = {}
            for symbol in symbols:
                if symbol in quotes:
                    quote = quotes[symbol]
                    result[symbol] = MarketData(
                        symbol=symbol,
                        price=(quote.bid_price + quote.ask_price) / 2,  # Mid price
                        bid=quote.bid_price,
                        ask=quote.ask_price,
                        bid_size=quote.bid_size,
                        ask_size=quote.ask_size,
                        timestamp=quote.timestamp
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get Alpaca market data: {e}")
            return {}
    
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = '1Day'
    ) -> pl.DataFrame:
        """Get historical data from Alpaca."""
        if not self.authenticated or not self.client:
            raise RuntimeError("Not authenticated with Alpaca")
        
        try:
            await self.rate_limiter.acquire()
            
            # Convert timeframe
            if timeframe == '1Day':
                tf = TimeFrame.Day
            elif timeframe == '1Hour':
                tf = TimeFrame.Hour
            elif timeframe == '1Min':
                tf = TimeFrame.Minute
            else:
                tf = TimeFrame.Day
            
            # Format dates for Alpaca API (RFC3339 format)
            start_str = start_date.replace(microsecond=0).isoformat() + 'Z'
            end_str = end_date.replace(microsecond=0).isoformat() + 'Z'
            
            logger.info(f"Requesting {symbol} data from {start_str} to {end_str} (timeframe: {timeframe})")
            
            # Get bars
            bars_request = self.client.get_bars(
                symbol,
                tf,
                start=start_str,
                end=end_str,
                adjustment='raw'
            )
            
            bars_df = bars_request.df
            logger.info(f"Alpaca returned {len(bars_df)} bars for {symbol}")
            
            if len(bars_df) > 0:
                logger.debug(f"Sample data for {symbol}: {bars_df.head()}")
            
            # Convert to Polars DataFrame
            return pl.from_pandas(bars_df.reset_index())
            
        except Exception as e:
            logger.error(f"Failed to get Alpaca historical data for {symbol}: {e}")
            logger.error(f"Request details - Symbol: {symbol}, Start: {start_str}, End: {end_str}, Timeframe: {timeframe}")
            return pl.DataFrame()


class RobinhoodBrokerAdapter(BrokerAdapter):
    """
    Robinhood broker adapter with comprehensive API integration.
    Note: Robinhood has limited official API support, this uses robin-stocks.
    """
    
    def __init__(self, api_credentials: Dict[str, str], config: Optional[BrokerConfig] = None):
        """Initialize Robinhood adapter."""
        super().__init__(api_credentials, config)
        
        if not ROBINHOOD_AVAILABLE:
            raise ImportError("Robin Stocks not available. Install with: pip install robin-stocks")
        
        self.username = api_credentials.get('username')
        self.password = api_credentials.get('password')
        self.totp = api_credentials.get('totp')  # Optional TOTP for 2FA
        
        if not self.username or not self.password:
            raise ValueError("Robinhood username and password are required")
    
    async def authenticate(self) -> bool:
        """Authenticate with Robinhood."""
        try:
            await self.rate_limiter.acquire()
            
            login_result = rh.authentication.login(
                username=self.username,
                password=self.password,
                expiresIn=86400,  # 24 hours
                scope='internal',
                by_sms=True,
                store_session=True,
                mfa_code=self.totp
            )
            
            if login_result:
                self.authenticated = True
                logger.info("Robinhood authentication successful")
                return True
            else:
                logger.error("Robinhood authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"Robinhood authentication failed: {e}")
            self.authenticated = False
            return False
    
    async def get_account_info(self) -> AccountInfo:
        """Get Robinhood account information."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Robinhood")
        
        try:
            await self.rate_limiter.acquire()
            
            # Get account profile and positions
            profile = rh.profiles.load_account_profile()
            user_data = rh.profiles.load_user()
            
            # Calculate portfolio value
            positions = rh.account.get_open_stock_positions()
            portfolio_value = 0.0
            for pos in positions:
                portfolio_value += float(pos.get('market_value', 0))
            
            return AccountInfo(
                account_id=profile.get('account_number', ''),
                buying_power=float(profile.get('buying_power', 0)),
                cash=float(profile.get('cash', 0)),
                portfolio_value=portfolio_value,
                equity=portfolio_value + float(profile.get('cash', 0)),
                is_day_trader=profile.get('is_day_trader', False)
            )
            
        except Exception as e:
            logger.error(f"Failed to get Robinhood account info: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get current Robinhood positions."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Robinhood")
        
        try:
            await self.rate_limiter.acquire()
            positions_data = rh.account.get_open_stock_positions()
            
            result = []
            for pos_data in positions_data:
                symbol = pos_data.get('symbol', '')
                quantity = float(pos_data.get('quantity', 0))
                
                if quantity != 0:  # Only include non-zero positions
                    result.append(Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=float(pos_data.get('average_buy_price', 0)),
                        market_value=float(pos_data.get('market_value', 0)),
                        unrealized_pnl=float(pos_data.get('total_return_today', 0)),
                        side='long' if quantity > 0 else 'short'
                    ))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get Robinhood positions: {e}")
            raise
    
    async def place_order(self, order: Order) -> bool:
        """Place order with Robinhood."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Robinhood")
        
        # Validate order
        is_valid, error_msg = self.validate_order(order)
        if not is_valid:
            logger.error(f"Invalid order: {error_msg}")
            order.status = OrderStatus.REJECTED
            return False
        
        try:
            await self.rate_limiter.acquire()
            
            # Convert order type and parameters
            rh_params = {
                'symbol': order.symbol,
                'quantity': int(order.quantity),
                'side': order.side,
                'timeInForce': 'gfd' if order.time_in_force == TimeInForce.DAY else 'gtc',
                'extendedHours': order.extended_hours
            }
            
            # Place order based on type
            if order.order_type == OrderType.MARKET:
                if order.side == 'buy':
                    result = rh.orders.order_buy_market(**rh_params)
                else:
                    result = rh.orders.order_sell_market(**rh_params)
            elif order.order_type == OrderType.LIMIT:
                rh_params['price'] = order.limit_price
                if order.side == 'buy':
                    result = rh.orders.order_buy_limit(**rh_params)
                else:
                    result = rh.orders.order_sell_limit(**rh_params)
            else:
                logger.error(f"Order type {order.order_type} not supported by Robinhood adapter")
                return False
            
            if result:
                order.order_id = result.get('id')
                order.status = OrderStatus.SUBMITTED
                order.created_at = datetime.now()
                logger.info(f"Robinhood order placed: {order.order_id}")
                return True
            else:
                order.status = OrderStatus.REJECTED
                return False
                
        except Exception as e:
            logger.error(f"Failed to place Robinhood order: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel Robinhood order."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Robinhood")
        
        try:
            await self.rate_limiter.acquire()
            result = rh.orders.cancel_stock_order(order_id)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to cancel Robinhood order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get Robinhood order status."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Robinhood")
        
        try:
            await self.rate_limiter.acquire()
            # Note: Robinhood API limitations - this is a simplified implementation
            orders = rh.orders.get_all_stock_orders()
            
            for rh_order in orders:
                if rh_order.get('id') == order_id:
                    order = Order(
                        symbol=rh.stocks.get_symbol_by_url(rh_order.get('instrument')),
                        quantity=float(rh_order.get('quantity', 0)),
                        side=rh_order.get('side'),
                        order_type=OrderType.MARKET if rh_order.get('type') == 'market' else OrderType.LIMIT,
                        order_id=order_id,
                        status=OrderStatus(rh_order.get('state', 'unknown').lower()),
                        filled_quantity=float(rh_order.get('executed_quantity', 0)),
                        created_at=datetime.fromisoformat(rh_order.get('created_at', '').replace('Z', '+00:00'))
                    )
                    
                    if rh_order.get('price'):
                        order.limit_price = float(rh_order.get('price'))
                    
                    return order
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get Robinhood order status {order_id}: {e}")
            return None
    
    async def get_orders(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get Robinhood order history."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Robinhood")
        
        try:
            await self.rate_limiter.acquire()
            rh_orders = rh.orders.get_all_stock_orders()
            
            result = []
            count = 0
            
            for rh_order in rh_orders:
                if count >= limit:
                    break
                
                order_symbol = rh.stocks.get_symbol_by_url(rh_order.get('instrument'))
                
                # Filter by symbol if specified
                if symbol and order_symbol != symbol:
                    continue
                
                order = Order(
                    symbol=order_symbol,
                    quantity=float(rh_order.get('quantity', 0)),
                    side=rh_order.get('side'),
                    order_type=OrderType.MARKET if rh_order.get('type') == 'market' else OrderType.LIMIT,
                    order_id=rh_order.get('id'),
                    status=OrderStatus(rh_order.get('state', 'unknown').lower()),
                    filled_quantity=float(rh_order.get('executed_quantity', 0)),
                    created_at=datetime.fromisoformat(rh_order.get('created_at', '').replace('Z', '+00:00'))
                )
                
                if rh_order.get('price'):
                    order.limit_price = float(rh_order.get('price'))
                
                result.append(order)
                count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get Robinhood orders: {e}")
            return []
    
    async def get_market_data(self, symbols: Union[str, List[str]]) -> Dict[str, MarketData]:
        """Get real-time market data from Robinhood."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Robinhood")
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        try:
            await self.rate_limiter.acquire()
            quotes_data = rh.stocks.get_quotes(symbols)
            
            result = {}
            for i, symbol in enumerate(symbols):
                if i < len(quotes_data):
                    quote = quotes_data[i]
                    result[symbol] = MarketData(
                        symbol=symbol,
                        price=float(quote.get('last_trade_price', 0)),
                        bid=float(quote.get('bid_price', 0)),
                        ask=float(quote.get('ask_price', 0)),
                        bid_size=int(quote.get('bid_size', 0)),
                        ask_size=int(quote.get('ask_size', 0)),
                        timestamp=datetime.now()
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get Robinhood market data: {e}")
            return {}
    
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = '1Day'
    ) -> pl.DataFrame:
        """Get historical data from Robinhood."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Robinhood")
        
        try:
            await self.rate_limiter.acquire()
            
            # Convert timeframe for Robinhood
            if timeframe == '1Day':
                interval = 'day'
                span = 'year'
            elif timeframe == '1Hour':
                interval = 'hour'
                span = 'week'
            else:
                interval = 'day'
                span = 'year'
            
            # Get historical data
            historicals = rh.stocks.get_stock_historicals(
                symbol,
                interval=interval,
                span=span
            )
            
            # Convert to DataFrame format
            if historicals:
                data = []
                for hist in historicals:
                    data.append({
                        'timestamp': datetime.fromisoformat(hist['begins_at'].replace('Z', '+00:00')),
                        'open': float(hist['open_price']),
                        'high': float(hist['high_price']),
                        'low': float(hist['low_price']),
                        'close': float(hist['close_price']),
                        'volume': int(hist['volume']) if hist['volume'] else 0
                    })
                
                df = pl.DataFrame(data)
                
                # Filter by date range
                df = df.filter(
                    (pl.col('timestamp') >= start_date) & 
                    (pl.col('timestamp') <= end_date)
                )
                
                return df
            
            return pl.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get Robinhood historical data: {e}")
            return pl.DataFrame()


class BrokerFactory:
    """Factory for creating broker adapters."""
    
    @staticmethod
    def create_broker(
        broker_name: str, 
        api_credentials: Dict[str, str], 
        config: Optional[BrokerConfig] = None
    ) -> BrokerAdapter:
        """Create broker adapter by name."""
        broker_name = broker_name.lower()
        
        if broker_name == 'alpaca':
            return AlpacaBrokerAdapter(api_credentials, config)
        elif broker_name == 'robinhood':
            return RobinhoodBrokerAdapter(api_credentials, config)
        else:
            raise ValueError(f"Unknown broker: {broker_name}")
    
    @staticmethod
    def list_available_brokers() -> List[str]:
        """List available brokers."""
        brokers = []
        if ALPACA_AVAILABLE:
            brokers.append('alpaca')
        if ROBINHOOD_AVAILABLE:
            brokers.append('robinhood')
        return brokers