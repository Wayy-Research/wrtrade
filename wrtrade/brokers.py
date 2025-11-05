import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import asyncio
import time
from datetime import datetime, timedelta
import requests
import json

from wrtrade.components import PortfolioComponent, CompositePortfolio


logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    side: str  # 'long' or 'short'


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    quantity: float
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop', etc.
    time_in_force: str = 'day'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    extended_hours: bool = False


@dataclass
class OrderStatus:
    """Status of a submitted order."""
    order_id: str
    status: str  # 'pending', 'filled', 'cancelled', 'rejected'
    filled_quantity: float
    remaining_quantity: float
    average_fill_price: Optional[float]
    created_at: datetime
    updated_at: datetime


@dataclass
class AccountInfo:
    """Account information."""
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    day_trade_buying_power: float
    is_day_trader: bool
    positions: List[Position]


class BrokerAdapter(ABC):
    """
    Abstract base class for broker adapters.
    Provides unified interface for different brokers.
    """
    
    def __init__(self, api_credentials: Dict[str, str], paper_trading: bool = True):
        """
        Initialize broker adapter.
        
        Args:
            api_credentials: Dictionary containing API keys/tokens
            paper_trading: Whether to use paper trading mode
        """
        self.credentials = api_credentials
        self.paper_trading = paper_trading
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
    async def place_order(self, order: Order) -> OrderStatus:
        """Place a trading order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get status of an order."""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get real-time market data."""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = '1D'
    ) -> pl.DataFrame:
        """Get historical price data."""
        pass
    
    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """
        Validate order before submission.
        
        Args:
            order: Order to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if order.quantity <= 0:
            return False, "Order quantity must be positive"
        
        if order.side not in ['buy', 'sell']:
            return False, "Order side must be 'buy' or 'sell'"
        
        if order.order_type == 'limit' and order.limit_price is None:
            return False, "Limit price required for limit orders"
        
        if order.order_type == 'stop' and order.stop_price is None:
            return False, "Stop price required for stop orders"
        
        return True, ""
    
    def log_trade(self, order: Order, status: OrderStatus) -> None:
        """Log trade execution for audit trail."""
        log_entry = {
            'timestamp': datetime.now(),
            'symbol': order.symbol,
            'quantity': order.quantity,
            'side': order.side,
            'order_type': order.order_type,
            'status': status.status,
            'fill_price': status.average_fill_price,
            'order_id': status.order_id,
            'paper_trading': self.paper_trading
        }
        
        logger.info(f"Trade executed: {json.dumps(log_entry, default=str)}")


class AlpacaBrokerAdapter(BrokerAdapter):
    """
    Alpaca Markets broker adapter.
    Supports both paper and live trading through Alpaca's API.
    """
    
    def __init__(self, api_credentials: Dict[str, str], paper_trading: bool = True):
        """
        Initialize Alpaca adapter.
        
        Args:
            api_credentials: Must contain 'api_key' and 'secret_key'
            paper_trading: Use paper trading endpoint
        """
        super().__init__(api_credentials, paper_trading)
        
        required_keys = ['api_key', 'secret_key']
        for key in required_keys:
            if key not in api_credentials:
                raise ValueError(f"Missing required credential: {key}")
        
        self.base_url = "https://paper-api.alpaca.markets/v2" if paper_trading else "https://api.alpaca.markets/v2"
        self.data_url = "https://data.alpaca.markets/v2"
        
        self.headers = {
            'APCA-API-KEY-ID': self.credentials['api_key'],
            'APCA-API-SECRET-KEY': self.credentials['secret_key'],
            'Content-Type': 'application/json'
        }
        
    async def authenticate(self) -> bool:
        """Test authentication with Alpaca API."""
        try:
            response = requests.get(f"{self.base_url}/account", headers=self.headers)
            if response.status_code == 200:
                self.authenticated = True
                logger.info("Successfully authenticated with Alpaca")
                return True
            else:
                logger.error(f"Alpaca authentication failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Alpaca authentication error: {e}")
            return False
    
    async def get_account_info(self) -> AccountInfo:
        """Get Alpaca account information."""
        if not self.authenticated:
            await self.authenticate()
        
        try:
            # Get account details
            account_response = requests.get(f"{self.base_url}/account", headers=self.headers)
            account_response.raise_for_status()
            account_data = account_response.json()
            
            # Get positions
            positions = await self.get_positions()
            
            return AccountInfo(
                account_id=account_data.get('account_number', ''),
                buying_power=float(account_data.get('buying_power', 0)),
                cash=float(account_data.get('cash', 0)),
                portfolio_value=float(account_data.get('portfolio_value', 0)),
                day_trade_buying_power=float(account_data.get('daytrading_buying_power', 0)),
                is_day_trader=account_data.get('pattern_day_trader', False),
                positions=positions
            )
            
        except Exception as e:
            logger.error(f"Error getting Alpaca account info: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get current Alpaca positions."""
        if not self.authenticated:
            await self.authenticate()
        
        try:
            response = requests.get(f"{self.base_url}/positions", headers=self.headers)
            response.raise_for_status()
            positions_data = response.json()
            
            positions = []
            for pos_data in positions_data:
                position = Position(
                    symbol=pos_data['symbol'],
                    quantity=float(pos_data['qty']),
                    average_price=float(pos_data['avg_cost']),
                    market_value=float(pos_data['market_value']),
                    unrealized_pnl=float(pos_data['unrealized_pl']),
                    side='long' if float(pos_data['qty']) > 0 else 'short'
                )
                positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting Alpaca positions: {e}")
            raise
    
    async def place_order(self, order: Order) -> OrderStatus:
        """Place order with Alpaca."""
        if not self.authenticated:
            await self.authenticate()
        
        # Validate order
        is_valid, error_msg = self.validate_order(order)
        if not is_valid:
            raise ValueError(f"Invalid order: {error_msg}")
        
        # Build order payload
        order_data = {
            'symbol': order.symbol,
            'qty': str(order.quantity),
            'side': order.side,
            'type': order.order_type,
            'time_in_force': order.time_in_force,
            'extended_hours': order.extended_hours
        }
        
        if order.limit_price is not None:
            order_data['limit_price'] = str(order.limit_price)
        
        if order.stop_price is not None:
            order_data['stop_price'] = str(order.stop_price)
        
        try:
            response = requests.post(
                f"{self.base_url}/orders",
                headers=self.headers,
                json=order_data
            )
            response.raise_for_status()
            order_response = response.json()
            
            order_status = OrderStatus(
                order_id=order_response['id'],
                status=order_response['status'],
                filled_quantity=float(order_response.get('filled_qty', 0)),
                remaining_quantity=float(order_response.get('qty', 0)) - float(order_response.get('filled_qty', 0)),
                average_fill_price=float(order_response.get('filled_avg_price', 0)) if order_response.get('filled_avg_price') else None,
                created_at=datetime.fromisoformat(order_response['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(order_response['updated_at'].replace('Z', '+00:00'))
            )
            
            # Log the trade
            self.log_trade(order, order_status)
            
            return order_status
            
        except Exception as e:
            logger.error(f"Error placing Alpaca order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel Alpaca order."""
        if not self.authenticated:
            await self.authenticate()
        
        try:
            response = requests.delete(f"{self.base_url}/orders/{order_id}", headers=self.headers)
            return response.status_code == 204
        except Exception as e:
            logger.error(f"Error cancelling Alpaca order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get Alpaca order status."""
        if not self.authenticated:
            await self.authenticate()
        
        try:
            response = requests.get(f"{self.base_url}/orders/{order_id}", headers=self.headers)
            response.raise_for_status()
            order_data = response.json()
            
            return OrderStatus(
                order_id=order_data['id'],
                status=order_data['status'],
                filled_quantity=float(order_data.get('filled_qty', 0)),
                remaining_quantity=float(order_data.get('qty', 0)) - float(order_data.get('filled_qty', 0)),
                average_fill_price=float(order_data.get('filled_avg_price', 0)) if order_data.get('filled_avg_price') else None,
                created_at=datetime.fromisoformat(order_data['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(order_data['updated_at'].replace('Z', '+00:00'))
            )
            
        except Exception as e:
            logger.error(f"Error getting Alpaca order status {order_id}: {e}")
            raise
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get real-time market data from Alpaca."""
        try:
            symbols_str = ','.join(symbols)
            response = requests.get(
                f"{self.data_url}/stocks/quotes/latest",
                headers=self.headers,
                params={'symbols': symbols_str}
            )
            response.raise_for_status()
            
            quotes_data = response.json()['quotes']
            
            market_data = {}
            for symbol, quote in quotes_data.items():
                market_data[symbol] = {
                    'bid': float(quote['bp']),
                    'ask': float(quote['ap']),
                    'bid_size': int(quote['bs']),
                    'ask_size': int(quote['as']),
                    'timestamp': quote['t']
                }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting Alpaca market data: {e}")
            return {}
    
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = '1Day'
    ) -> pl.DataFrame:
        """Get historical data from Alpaca."""
        try:
            params = {
                'symbols': symbol,
                'timeframe': timeframe,
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'adjustment': 'raw'
            }
            
            response = requests.get(
                f"{self.data_url}/stocks/bars",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            bars = data['bars'].get(symbol, [])
            
            if not bars:
                return pl.DataFrame()
            
            # Convert to polars DataFrame
            df_data = {
                'timestamp': [bar['t'] for bar in bars],
                'open': [float(bar['o']) for bar in bars],
                'high': [float(bar['h']) for bar in bars],
                'low': [float(bar['l']) for bar in bars],
                'close': [float(bar['c']) for bar in bars],
                'volume': [int(bar['v']) for bar in bars],
                'vwap': [float(bar['vw']) for bar in bars],
                'trade_count': [int(bar['n']) for bar in bars]
            }
            
            return pl.DataFrame(df_data)
            
        except Exception as e:
            logger.error(f"Error getting Alpaca historical data: {e}")
            return pl.DataFrame()


class RobinhoodBrokerAdapter(BrokerAdapter):
    """
    Robinhood broker adapter.
    Note: Robinhood API is not officially public, this is a placeholder
    for when/if they provide official API access.
    """
    
    def __init__(self, api_credentials: Dict[str, str], paper_trading: bool = True):
        """
        Initialize Robinhood adapter.
        
        Note: This is a placeholder implementation.
        """
        super().__init__(api_credentials, paper_trading)
        
        # Robinhood doesn't have an official API yet
        logger.warning("Robinhood API is not officially available. This is a placeholder implementation.")
        
        self.base_url = "https://api.robinhood.com"  # Unofficial
        
    async def authenticate(self) -> bool:
        """Placeholder authentication."""
        logger.warning("Robinhood authentication not implemented - no official API")
        return False
    
    async def get_account_info(self) -> AccountInfo:
        """Placeholder account info."""
        raise NotImplementedError("Robinhood API not officially available")
    
    async def get_positions(self) -> List[Position]:
        """Placeholder positions."""
        raise NotImplementedError("Robinhood API not officially available")
    
    async def place_order(self, order: Order) -> OrderStatus:
        """Placeholder order placement."""
        raise NotImplementedError("Robinhood API not officially available")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Placeholder order cancellation."""
        raise NotImplementedError("Robinhood API not officially available")
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Placeholder order status."""
        raise NotImplementedError("Robinhood API not officially available")
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Placeholder market data."""
        raise NotImplementedError("Robinhood API not officially available")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = '1D'
    ) -> pl.DataFrame:
        """Placeholder historical data."""
        raise NotImplementedError("Robinhood API not officially available")


class BrokerFactory:
    """Factory for creating broker adapter instances."""
    
    @staticmethod
    def create_broker(
        broker_name: str, 
        api_credentials: Dict[str, str], 
        paper_trading: bool = True
    ) -> BrokerAdapter:
        """
        Create broker adapter instance.
        
        Args:
            broker_name: Name of broker ('alpaca', 'robinhood')
            api_credentials: API credentials dictionary
            paper_trading: Use paper trading mode
            
        Returns:
            BrokerAdapter instance
        """
        if broker_name.lower() == 'alpaca':
            return AlpacaBrokerAdapter(api_credentials, paper_trading)
        elif broker_name.lower() == 'robinhood':
            return RobinhoodBrokerAdapter(api_credentials, paper_trading)
        else:
            raise ValueError(f"Unknown broker: {broker_name}")
    
    @staticmethod
    def get_supported_brokers() -> List[str]:
        """Get list of supported broker names."""
        return ['alpaca', 'robinhood']


class TradingSession:
    """
    Manages a trading session with a specific broker.
    Handles order execution, position management, and risk controls.
    """
    
    def __init__(
        self, 
        broker: BrokerAdapter,
        max_position_size: float = 0.1,  # 10% of portfolio
        max_daily_loss: float = 0.05,    # 5% daily loss limit
        require_confirmation: bool = True
    ):
        """
        Initialize trading session.
        
        Args:
            broker: Broker adapter instance
            max_position_size: Maximum position size as fraction of portfolio
            max_daily_loss: Maximum daily loss as fraction of portfolio
            require_confirmation: Whether to require manual confirmation for trades
        """
        self.broker = broker
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.require_confirmation = require_confirmation
        
        self.session_start = datetime.now()
        self.daily_pnl = 0.0
        self.executed_orders = []
        self.risk_breaches = []
        
    async def execute_portfolio_rebalance(
        self, 
        portfolio: CompositePortfolio,
        target_weights: Dict[str, float],
        symbols_map: Dict[str, str]  # Component name -> trading symbol
    ) -> List[OrderStatus]:
        """
        Execute portfolio rebalancing through broker.
        
        Args:
            portfolio: Portfolio to rebalance
            target_weights: Target weights by component name
            symbols_map: Mapping from component names to trading symbols
            
        Returns:
            List of order statuses
        """
        logger.info("Starting portfolio rebalancing")
        
        # Get current account info
        account_info = await self.broker.get_account_info()
        current_positions = {pos.symbol: pos for pos in account_info.positions}
        
        # Calculate target positions in dollar amounts
        portfolio_value = account_info.portfolio_value
        target_positions = {}
        
        for component_name, weight in target_weights.items():
            if component_name in symbols_map:
                symbol = symbols_map[component_name]
                target_value = portfolio_value * weight
                target_positions[symbol] = target_value
        
        # Generate orders to reach target positions
        orders_to_place = []
        
        for symbol, target_value in target_positions.items():
            current_pos = current_positions.get(symbol)
            current_value = current_pos.market_value if current_pos else 0.0
            
            value_diff = target_value - current_value
            
            if abs(value_diff) > 100:  # Only rebalance if difference > $100
                # Get current market price
                market_data = await self.broker.get_market_data([symbol])
                if symbol not in market_data:
                    logger.warning(f"No market data for {symbol}, skipping")
                    continue
                
                current_price = (market_data[symbol]['bid'] + market_data[symbol]['ask']) / 2
                shares_to_trade = value_diff / current_price
                
                if abs(shares_to_trade) >= 1:  # Only trade whole shares for now
                    order = Order(
                        symbol=symbol,
                        quantity=abs(shares_to_trade),
                        side='buy' if shares_to_trade > 0 else 'sell',
                        order_type='market'
                    )
                    orders_to_place.append(order)
        
        # Execute orders with risk checks
        executed_orders = []
        
        for order in orders_to_place:
            try:
                # Risk check
                if not self._check_risk_limits(order, account_info):
                    logger.warning(f"Risk check failed for order: {order.symbol}")
                    continue
                
                # Confirmation if required
                if self.require_confirmation:
                    confirmation = input(f"Execute {order.side} {order.quantity} {order.symbol}? (y/n): ")
                    if confirmation.lower() != 'y':
                        logger.info(f"Order cancelled by user: {order.symbol}")
                        continue
                
                # Place order
                order_status = await self.broker.place_order(order)
                executed_orders.append(order_status)
                self.executed_orders.append(order_status)
                
                # Brief pause between orders
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to execute order for {order.symbol}: {e}")
                
        logger.info(f"Rebalancing completed. Executed {len(executed_orders)} orders")
        return executed_orders
    
    def _check_risk_limits(self, order: Order, account_info: AccountInfo) -> bool:
        """
        Check if order passes risk management rules.
        
        Args:
            order: Order to check
            account_info: Current account information
            
        Returns:
            True if order passes risk checks
        """
        # Check position size limit
        order_value = order.quantity  # This would need current price
        max_order_value = account_info.portfolio_value * self.max_position_size
        
        if order_value > max_order_value:
            self.risk_breaches.append(f"Position size limit exceeded for {order.symbol}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss * account_info.portfolio_value:
            self.risk_breaches.append(f"Daily loss limit exceeded: ${self.daily_pnl:.2f}")
            return False
        
        # Check buying power
        if order.side == 'buy' and order_value > account_info.buying_power:
            self.risk_breaches.append(f"Insufficient buying power for {order.symbol}")
            return False
        
        return True
    
    async def monitor_positions(self, check_interval: int = 60) -> None:
        """
        Monitor positions and update daily P&L.
        
        Args:
            check_interval: How often to check positions (seconds)
        """
        logger.info("Starting position monitoring")
        
        while True:
            try:
                account_info = await self.broker.get_account_info()
                
                # Calculate daily P&L
                total_unrealized_pnl = sum(pos.unrealized_pnl for pos in account_info.positions)
                self.daily_pnl = total_unrealized_pnl  # Simplified - should track realized too
                
                # Log status
                logger.info(f"Portfolio value: ${account_info.portfolio_value:.2f}, "
                          f"Daily P&L: ${self.daily_pnl:.2f}, "
                          f"Positions: {len(account_info.positions)}")
                
                # Check for risk breaches
                if self.daily_pnl < -self.max_daily_loss * account_info.portfolio_value:
                    logger.warning("DAILY LOSS LIMIT EXCEEDED - Consider closing positions")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(check_interval)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of trading session."""
        return {
            'session_start': self.session_start,
            'session_duration': datetime.now() - self.session_start,
            'orders_executed': len(self.executed_orders),
            'current_daily_pnl': self.daily_pnl,
            'risk_breaches': self.risk_breaches.copy(),
            'paper_trading': self.broker.paper_trading
        }