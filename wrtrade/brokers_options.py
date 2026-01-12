"""
Alpaca Options Broker Adapter for wrtrade.

Extends the base BrokerAdapter to support options trading on Alpaca.

Features:
- Single-leg options orders (buy/sell calls and puts)
- Multi-leg orders (spreads, straddles, iron condors)
- Options position management with Greeks
- Exercise and assignment handling
- Paper trading support

API Docs: https://docs.alpaca.markets/docs/options-trading

Trading Levels:
- Level 1: Covered calls, cash-secured puts
- Level 2: Long calls and puts
- Level 3: Spreads (requires approval)
"""

import polars as pl
import requests
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
import logging
import asyncio
import re

from wrtrade.brokers_real import (
    BrokerAdapter,
    BrokerConfig,
    Order,
    Position,
    AccountInfo,
    MarketData,
    OrderType,
    OrderStatus,
    TimeInForce,
    RateLimiter,
)

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


class OptionSide(Enum):
    """Option position side."""
    BUY_TO_OPEN = "buy_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"
    SELL_TO_CLOSE = "sell_to_close"


class OptionStrategy(Enum):
    """Multi-leg option strategies."""
    SINGLE = "single"
    COVERED_CALL = "covered_call"
    VERTICAL_SPREAD = "vertical_spread"
    CALENDAR_SPREAD = "calendar_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"


@dataclass
class OptionLeg:
    """Single leg of an options order."""
    contract_symbol: str  # OCC symbol (e.g., SPY240119C00500000)
    quantity: int
    side: OptionSide

    # Computed from OCC symbol
    underlying: Optional[str] = None
    expiration: Optional[date] = None
    option_type: Optional[OptionType] = None
    strike: Optional[float] = None


@dataclass
class OptionOrder:
    """Options order with one or more legs."""
    legs: List[OptionLeg]
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None  # Net debit/credit for multi-leg
    stop_price: Optional[float] = None

    # Strategy classification
    strategy: OptionStrategy = OptionStrategy.SINGLE

    # Order tracking
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    fees: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class OptionPosition:
    """Options position with Greeks."""
    contract_symbol: str
    underlying: str
    expiration: date
    option_type: OptionType
    strike: float

    quantity: int  # Positive = long, negative = short
    avg_price: float
    market_value: float

    # Current market data
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    underlying_price: Optional[float] = None

    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None

    # P&L
    unrealized_pnl: float = 0.0
    cost_basis: float = 0.0

    # Days to expiration
    dte: Optional[int] = None

    def __post_init__(self):
        if self.expiration:
            self.dte = (self.expiration - date.today()).days
        if self.cost_basis == 0.0:
            self.cost_basis = abs(self.quantity) * self.avg_price * 100  # Contract multiplier


@dataclass
class OptionQuote:
    """Real-time option quote."""
    contract_symbol: str
    underlying: str
    expiration: date
    option_type: str
    strike: float

    bid: float
    ask: float
    last_price: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None

    underlying_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2


class AlpacaOptionsBrokerAdapter(BrokerAdapter):
    """
    Alpaca broker adapter for options trading.

    Supports both paper and live trading for US equity options.

    Features:
    - Single-leg options orders
    - Multi-leg spreads (Level 3 required)
    - Position management with Greeks
    - Exercise and assignment handling

    Example:
        >>> adapter = AlpacaOptionsBrokerAdapter(
        ...     api_credentials={
        ...         'api_key': 'your_key',
        ...         'secret_key': 'your_secret',
        ...     },
        ...     config=BrokerConfig(paper_trading=True)
        ... )
        >>> async with adapter:
        ...     positions = await adapter.get_option_positions()
    """

    # API endpoints
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"

    # OCC symbol pattern
    OCC_PATTERN = re.compile(r'^([A-Z]{1,6})(\d{6})([CP])(\d{8})$')

    def __init__(
        self,
        api_credentials: Dict[str, str],
        config: Optional[BrokerConfig] = None
    ):
        """
        Initialize Alpaca options adapter.

        Args:
            api_credentials: Dict with 'api_key' and 'secret_key'
            config: BrokerConfig with paper_trading flag
        """
        super().__init__(api_credentials, config)

        self.api_key = api_credentials.get('api_key')
        self.secret_key = api_credentials.get('secret_key')

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret key are required")

        # Set base URL based on paper/live mode
        self.base_url = (
            self.PAPER_URL if self.config.paper_trading
            else self.LIVE_URL
        )

        # Authentication headers
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

        # Trading level (set after authentication)
        self.options_trading_level: Optional[int] = None

    async def authenticate(self) -> bool:
        """Authenticate with Alpaca and check options trading status."""
        try:
            await self.rate_limiter.acquire()

            url = f"{self.base_url}/v2/account"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            account = response.json()
            self.authenticated = True

            # Check options trading level
            self.options_trading_level = account.get('options_trading_level', 0)

            logger.info(
                f"Alpaca authentication successful. "
                f"Account: {account.get('account_number')} "
                f"Options Level: {self.options_trading_level}"
            )

            return True

        except Exception as e:
            logger.error(f"Alpaca authentication failed: {e}")
            self.authenticated = False
            return False

    async def get_account_info(self) -> AccountInfo:
        """Get Alpaca account information."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        try:
            await self.rate_limiter.acquire()

            url = f"{self.base_url}/v2/account"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            account = response.json()

            return AccountInfo(
                account_id=account.get('account_number', ''),
                buying_power=float(account.get('options_buying_power', 0) or account.get('buying_power', 0)),
                cash=float(account.get('cash', 0)),
                portfolio_value=float(account.get('portfolio_value', 0)),
                equity=float(account.get('equity', 0)),
                day_trade_buying_power=float(account.get('daytrading_buying_power', 0)),
                is_day_trader=account.get('pattern_day_trader', False),
                maintenance_margin=float(account.get('maintenance_margin', 0) or 0),
                initial_margin=float(account.get('initial_margin', 0) or 0),
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise

    async def get_positions(self) -> List[Position]:
        """Get all positions (equity + options)."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        try:
            await self.rate_limiter.acquire()

            url = f"{self.base_url}/v2/positions"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            positions = response.json()

            result = []
            for pos in positions:
                result.append(Position(
                    symbol=pos.get('symbol', ''),
                    quantity=float(pos.get('qty', 0)),
                    avg_price=float(pos.get('avg_entry_price', 0)),
                    market_value=float(pos.get('market_value', 0)),
                    unrealized_pnl=float(pos.get('unrealized_pl', 0)),
                    side='long' if float(pos.get('qty', 0)) > 0 else 'short',
                    cost_basis=float(pos.get('cost_basis', 0)),
                ))

            return result

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise

    async def get_option_positions(self) -> List[OptionPosition]:
        """
        Get options positions with Greeks.

        Returns:
            List of OptionPosition objects with Greeks data
        """
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        try:
            await self.rate_limiter.acquire()

            url = f"{self.base_url}/v2/positions"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            positions = response.json()

            option_positions = []
            option_symbols = []

            # First pass: identify options and collect symbols
            for pos in positions:
                symbol = pos.get('symbol', '')
                asset_class = pos.get('asset_class', '')

                if asset_class == 'us_option' or self._is_option_symbol(symbol):
                    option_symbols.append(symbol)

            # Fetch Greeks for all option positions
            greeks_map = {}
            if option_symbols:
                greeks_map = await self._fetch_option_greeks(option_symbols)

            # Second pass: build OptionPosition objects
            for pos in positions:
                symbol = pos.get('symbol', '')
                asset_class = pos.get('asset_class', '')

                if asset_class == 'us_option' or self._is_option_symbol(symbol):
                    parsed = self._parse_occ_symbol(symbol)
                    if not parsed:
                        continue

                    underlying, exp_date, opt_type, strike = parsed
                    greeks = greeks_map.get(symbol, {})

                    option_positions.append(OptionPosition(
                        contract_symbol=symbol,
                        underlying=underlying,
                        expiration=exp_date,
                        option_type=OptionType.CALL if opt_type == 'C' else OptionType.PUT,
                        strike=strike,
                        quantity=int(float(pos.get('qty', 0))),
                        avg_price=float(pos.get('avg_entry_price', 0)),
                        market_value=float(pos.get('market_value', 0)),
                        unrealized_pnl=float(pos.get('unrealized_pl', 0)),
                        cost_basis=float(pos.get('cost_basis', 0)),
                        last_price=float(pos.get('current_price', 0)) if pos.get('current_price') else None,
                        delta=greeks.get('delta'),
                        gamma=greeks.get('gamma'),
                        theta=greeks.get('theta'),
                        vega=greeks.get('vega'),
                        implied_volatility=greeks.get('implied_volatility'),
                        underlying_price=greeks.get('underlying_price'),
                    ))

            return option_positions

        except Exception as e:
            logger.error(f"Failed to get option positions: {e}")
            raise

    async def place_order(self, order: Order) -> bool:
        """
        Place an equity order (inherited from base).

        For options orders, use place_option_order() instead.
        """
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        # Validate order
        is_valid, error_msg = self.validate_order(order)
        if not is_valid:
            logger.error(f"Invalid order: {error_msg}")
            order.status = OrderStatus.REJECTED
            return False

        try:
            await self.rate_limiter.acquire()

            payload = {
                "symbol": order.symbol,
                "qty": str(int(order.quantity)),
                "side": order.side,
                "type": order.order_type.value,
                "time_in_force": order.time_in_force.value,
            }

            if order.order_type == OrderType.LIMIT:
                payload["limit_price"] = str(order.limit_price)
            elif order.order_type == OrderType.STOP:
                payload["stop_price"] = str(order.stop_price)
            elif order.order_type == OrderType.STOP_LIMIT:
                payload["limit_price"] = str(order.limit_price)
                payload["stop_price"] = str(order.stop_price)

            url = f"{self.base_url}/v2/orders"
            response = requests.post(
                url, headers=self.headers, json=payload, timeout=30
            )
            response.raise_for_status()

            result = response.json()
            order.order_id = result.get('id')
            order.status = OrderStatus(result.get('status', 'pending').lower())
            order.created_at = datetime.fromisoformat(
                result.get('created_at', '').replace('Z', '+00:00')
            ) if result.get('created_at') else datetime.now()

            logger.info(f"Order placed: {order.order_id}")
            return True

        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed to place order: {e.response.text}")
            order.status = OrderStatus.REJECTED
            return False
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            order.status = OrderStatus.REJECTED
            return False

    async def place_option_order(self, order: OptionOrder) -> bool:
        """
        Place an options order (single or multi-leg).

        Args:
            order: OptionOrder with legs and pricing

        Returns:
            True if order placed successfully
        """
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        # Validate options trading level
        if len(order.legs) > 1 and self.options_trading_level < 3:
            logger.error("Multi-leg orders require Level 3 options approval")
            order.status = OrderStatus.REJECTED
            return False

        # Validate limit price for limit orders
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            logger.error("Limit price required for limit orders")
            order.status = OrderStatus.REJECTED
            return False

        # Validate stop price for stop orders
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            logger.error("Stop price required for stop orders")
            order.status = OrderStatus.REJECTED
            return False

        # Validate contract expiration dates
        today = date.today()
        for leg in order.legs:
            parsed = self._parse_occ_symbol(leg.contract_symbol)
            if parsed:
                _, exp_date, _, _ = parsed
                if exp_date < today:
                    logger.error(f"Contract {leg.contract_symbol} expired on {exp_date}")
                    order.status = OrderStatus.REJECTED
                    return False

        try:
            await self.rate_limiter.acquire()

            # Build order payload
            if len(order.legs) == 1:
                # Single-leg order
                payload = self._build_single_leg_payload(order)
            else:
                # Multi-leg order
                payload = self._build_multi_leg_payload(order)

            url = f"{self.base_url}/v2/orders"
            response = requests.post(
                url, headers=self.headers, json=payload, timeout=30
            )
            response.raise_for_status()

            result = response.json()
            order.order_id = result.get('id')
            order.status = OrderStatus(result.get('status', 'pending').lower())
            order.created_at = datetime.fromisoformat(
                result.get('created_at', '').replace('Z', '+00:00')
            ) if result.get('created_at') else datetime.now()

            logger.info(f"Option order placed: {order.order_id}")
            return True

        except requests.exceptions.HTTPError as e:
            error_detail = e.response.text
            logger.error(f"Failed to place option order: {error_detail}")
            order.status = OrderStatus.REJECTED
            return False
        except Exception as e:
            logger.error(f"Failed to place option order: {e}")
            order.status = OrderStatus.REJECTED
            return False

    def _build_single_leg_payload(self, order: OptionOrder) -> Dict[str, Any]:
        """Build payload for single-leg option order."""
        leg = order.legs[0]

        # Determine side based on OptionSide
        if leg.side in [OptionSide.BUY_TO_OPEN, OptionSide.BUY_TO_CLOSE]:
            side = "buy"
        else:
            side = "sell"

        payload = {
            "symbol": leg.contract_symbol,
            "qty": str(leg.quantity),
            "side": side,
            "type": order.order_type.value,
            "time_in_force": order.time_in_force.value,
        }

        if order.order_type == OrderType.LIMIT and order.limit_price:
            payload["limit_price"] = str(order.limit_price)
        elif order.order_type == OrderType.STOP and order.stop_price:
            payload["stop_price"] = str(order.stop_price)
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.limit_price:
                payload["limit_price"] = str(order.limit_price)
            if order.stop_price:
                payload["stop_price"] = str(order.stop_price)

        return payload

    def _build_multi_leg_payload(self, order: OptionOrder) -> Dict[str, Any]:
        """Build payload for multi-leg option order."""
        legs = []
        underlyings = set()
        expirations = set()

        for leg in order.legs:
            # Parse contract to validate
            parsed = self._parse_occ_symbol(leg.contract_symbol)
            if not parsed:
                raise ValueError(f"Invalid OCC symbol: {leg.contract_symbol}")

            underlying, exp_date, opt_type, strike = parsed
            underlyings.add(underlying)
            expirations.add(exp_date)

            # Determine side
            if leg.side in [OptionSide.BUY_TO_OPEN, OptionSide.BUY_TO_CLOSE]:
                side = "buy"
            else:
                side = "sell"

            legs.append({
                "symbol": leg.contract_symbol,
                "qty": str(leg.quantity),
                "side": side,
            })

        # Validate all legs have same underlying
        if len(underlyings) > 1:
            raise ValueError(f"Multi-leg orders must have same underlying. Found: {underlyings}")

        # Validate same expiration for most strategies (except calendar spreads)
        if order.strategy != OptionStrategy.CALENDAR_SPREAD and len(expirations) > 1:
            raise ValueError(f"Strategy {order.strategy.value} requires same expiration. Found: {expirations}")

        payload = {
            "order_class": "mleg",  # Multi-leg order class
            "legs": legs,
            "type": order.order_type.value,
            "time_in_force": order.time_in_force.value,
        }

        if order.order_type == OrderType.LIMIT and order.limit_price:
            # Format price with proper precision for options
            payload["limit_price"] = f"{order.limit_price:.2f}"

        return payload

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        try:
            await self.rate_limiter.acquire()

            url = f"{self.base_url}/v2/orders/{order_id}"
            response = requests.delete(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get status of an order."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        try:
            await self.rate_limiter.acquire()

            url = f"{self.base_url}/v2/orders/{order_id}"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            order = Order(
                symbol=data.get('symbol', ''),
                quantity=float(data.get('qty', 0)),
                side=data.get('side', ''),
                order_type=OrderType(data.get('type', 'market')),
                time_in_force=TimeInForce(data.get('time_in_force', 'day')),
                order_id=data.get('id'),
                status=OrderStatus(data.get('status', 'pending').lower()),
                filled_quantity=float(data.get('filled_qty', 0) or 0),
                avg_fill_price=float(data.get('filled_avg_price')) if data.get('filled_avg_price') else None,
            )

            if data.get('limit_price'):
                order.limit_price = float(data.get('limit_price'))
            if data.get('stop_price'):
                order.stop_price = float(data.get('stop_price'))

            return order

        except Exception as e:
            logger.error(f"Failed to get order status {order_id}: {e}")
            return None

    async def get_orders(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        try:
            await self.rate_limiter.acquire()

            url = f"{self.base_url}/v2/orders"
            params = {
                "status": "all",
                "limit": limit,
                "direction": "desc",
            }

            if symbol:
                params["symbols"] = symbol

            response = requests.get(
                url, headers=self.headers, params=params, timeout=30
            )
            response.raise_for_status()

            orders_data = response.json()

            result = []
            for data in orders_data:
                order = Order(
                    symbol=data.get('symbol', ''),
                    quantity=float(data.get('qty', 0)),
                    side=data.get('side', ''),
                    order_type=OrderType(data.get('type', 'market')),
                    time_in_force=TimeInForce(data.get('time_in_force', 'day')),
                    order_id=data.get('id'),
                    status=OrderStatus(data.get('status', 'pending').lower()),
                    filled_quantity=float(data.get('filled_qty', 0) or 0),
                    avg_fill_price=float(data.get('filled_avg_price')) if data.get('filled_avg_price') else None,
                )

                if data.get('limit_price'):
                    order.limit_price = float(data.get('limit_price'))
                if data.get('stop_price'):
                    order.stop_price = float(data.get('stop_price'))

                result.append(order)

            return result

        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    async def get_market_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Dict[str, MarketData]:
        """Get real-time market data for equities."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        if isinstance(symbols, str):
            symbols = [symbols]

        try:
            await self.rate_limiter.acquire()

            url = f"{self.DATA_URL}/v2/stocks/quotes/latest"
            params = {"symbols": ",".join(symbols)}

            response = requests.get(
                url, headers=self.headers, params=params, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            quotes = data.get('quotes', {})

            result = {}
            for symbol in symbols:
                if symbol in quotes:
                    q = quotes[symbol]
                    result[symbol] = MarketData(
                        symbol=symbol,
                        price=(q.get('bp', 0) + q.get('ap', 0)) / 2,
                        bid=q.get('bp'),
                        ask=q.get('ap'),
                        bid_size=q.get('bs'),
                        ask_size=q.get('as'),
                        timestamp=datetime.fromisoformat(
                            q.get('t', '').replace('Z', '+00:00')
                        ) if q.get('t') else datetime.now(),
                    )

            return result

        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return {}

    async def get_option_quotes(
        self,
        symbols: Union[str, List[str]]
    ) -> Dict[str, OptionQuote]:
        """
        Get real-time quotes for option contracts.

        Args:
            symbols: OCC option symbol(s)

        Returns:
            Dict mapping symbols to OptionQuote objects
        """
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        if isinstance(symbols, str):
            symbols = [symbols]

        try:
            await self.rate_limiter.acquire()

            # Get latest quotes
            url = f"{self.DATA_URL}/v1beta1/options/quotes/latest"
            params = {"symbols": ",".join(symbols)}

            response = requests.get(
                url, headers=self.headers, params=params, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            quotes = data.get('quotes', {})

            # Also get snapshots for Greeks
            greeks_map = await self._fetch_option_greeks(symbols)

            result = {}
            for symbol in symbols:
                if symbol in quotes:
                    q = quotes[symbol]
                    parsed = self._parse_occ_symbol(symbol)

                    if parsed:
                        underlying, exp_date, opt_type, strike = parsed
                        greeks = greeks_map.get(symbol, {})

                        result[symbol] = OptionQuote(
                            contract_symbol=symbol,
                            underlying=underlying,
                            expiration=exp_date,
                            option_type='call' if opt_type == 'C' else 'put',
                            strike=strike,
                            bid=float(q.get('bp', 0)),
                            ask=float(q.get('ap', 0)),
                            delta=greeks.get('delta'),
                            gamma=greeks.get('gamma'),
                            theta=greeks.get('theta'),
                            vega=greeks.get('vega'),
                            implied_volatility=greeks.get('implied_volatility'),
                            underlying_price=greeks.get('underlying_price'),
                            timestamp=datetime.fromisoformat(
                                q.get('t', '').replace('Z', '+00:00')
                            ) if q.get('t') else datetime.now(),
                        )

            return result

        except Exception as e:
            logger.error(f"Failed to get option quotes: {e}")
            return {}

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1Day'
    ) -> pl.DataFrame:
        """Get historical data for equity."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        try:
            await self.rate_limiter.acquire()

            url = f"{self.DATA_URL}/v2/stocks/{symbol}/bars"

            params = {
                "start": start_date.isoformat() + 'Z',
                "end": end_date.isoformat() + 'Z',
                "timeframe": timeframe,
                "limit": 10000,
            }

            response = requests.get(
                url, headers=self.headers, params=params, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            bars = data.get('bars', [])

            if not bars:
                return pl.DataFrame()

            records = []
            for bar in bars:
                records.append({
                    'timestamp': bar.get('t'),
                    'open': float(bar.get('o', 0)),
                    'high': float(bar.get('h', 0)),
                    'low': float(bar.get('l', 0)),
                    'close': float(bar.get('c', 0)),
                    'volume': int(bar.get('v', 0)),
                    'vwap': float(bar.get('vw', 0)),
                })

            return pl.DataFrame(records)

        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pl.DataFrame()

    async def exercise_option(self, contract_symbol: str) -> bool:
        """
        Submit exercise instruction for an option position.

        Args:
            contract_symbol: OCC option symbol to exercise

        Returns:
            True if exercise submitted successfully
        """
        if not self.authenticated:
            raise RuntimeError("Not authenticated with Alpaca")

        try:
            await self.rate_limiter.acquire()

            url = f"{self.base_url}/v2/positions/{contract_symbol}/exercise"
            response = requests.post(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            logger.info(f"Exercise submitted for {contract_symbol}")
            return True

        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed to exercise {contract_symbol}: {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Failed to exercise {contract_symbol}: {e}")
            return False

    async def _fetch_option_greeks(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch Greeks for option symbols from snapshot endpoint."""
        if not symbols:
            return {}

        try:
            # Group symbols by underlying to use snapshots endpoint
            underlyings = set()
            for symbol in symbols:
                parsed = self._parse_occ_symbol(symbol)
                if parsed:
                    underlyings.add(parsed[0])

            greeks_map = {}

            for underlying in underlyings:
                url = f"{self.DATA_URL}/v1beta1/options/snapshots/{underlying}"
                params = {"feed": "indicative", "limit": 1000}

                response = requests.get(
                    url, headers=self.headers, params=params, timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    snapshots = data.get('snapshots', {})

                    for contract, snapshot in snapshots.items():
                        if contract in symbols:
                            greeks = snapshot.get('greeks', {})
                            greeks_map[contract] = {
                                'delta': greeks.get('delta'),
                                'gamma': greeks.get('gamma'),
                                'theta': greeks.get('theta'),
                                'vega': greeks.get('vega'),
                                'implied_volatility': snapshot.get('impliedVolatility'),
                                'underlying_price': snapshot.get('underlyingPrice'),
                            }

            return greeks_map

        except Exception as e:
            logger.warning(f"Failed to fetch Greeks: {e}")
            return {}

    def _is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol is an OCC option symbol."""
        return bool(self.OCC_PATTERN.match(symbol))

    def _parse_occ_symbol(self, symbol: str) -> Optional[Tuple[str, date, str, float]]:
        """
        Parse OCC option symbol.

        Returns:
            Tuple of (underlying, expiration, type, strike) or None
        """
        match = self.OCC_PATTERN.match(symbol)
        if not match:
            return None

        underlying, date_str, opt_type, strike_str = match.groups()

        try:
            exp_date = datetime.strptime(date_str, '%y%m%d').date()
            strike = float(strike_str) / 1000.0
            return (underlying, exp_date, opt_type, strike)
        except ValueError:
            return None

    @staticmethod
    def build_occ_symbol(
        underlying: str,
        expiration: date,
        option_type: str,
        strike: float
    ) -> str:
        """
        Build OCC option symbol from components.

        Args:
            underlying: Stock ticker
            expiration: Expiration date
            option_type: "call" or "put"
            strike: Strike price

        Returns:
            OCC symbol string
        """
        date_str = expiration.strftime('%y%m%d')
        opt_char = 'C' if option_type.lower() == 'call' else 'P'
        strike_int = int(strike * 1000)
        return f"{underlying.upper()}{date_str}{opt_char}{strike_int:08d}"

    # Strategy builders

    def build_vertical_spread(
        self,
        underlying: str,
        expiration: date,
        option_type: str,
        long_strike: float,
        short_strike: float,
        quantity: int = 1
    ) -> OptionOrder:
        """
        Build a vertical spread order.

        Args:
            underlying: Stock ticker
            expiration: Expiration date
            option_type: "call" or "put"
            long_strike: Strike to buy
            short_strike: Strike to sell
            quantity: Number of spreads

        Returns:
            OptionOrder with two legs
        """
        long_symbol = self.build_occ_symbol(underlying, expiration, option_type, long_strike)
        short_symbol = self.build_occ_symbol(underlying, expiration, option_type, short_strike)

        return OptionOrder(
            legs=[
                OptionLeg(
                    contract_symbol=long_symbol,
                    quantity=quantity,
                    side=OptionSide.BUY_TO_OPEN,
                ),
                OptionLeg(
                    contract_symbol=short_symbol,
                    quantity=quantity,
                    side=OptionSide.SELL_TO_OPEN,
                ),
            ],
            strategy=OptionStrategy.VERTICAL_SPREAD,
        )

    def build_straddle(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        quantity: int = 1,
        side: str = "buy"
    ) -> OptionOrder:
        """
        Build a straddle order (same strike call + put).

        Args:
            underlying: Stock ticker
            expiration: Expiration date
            strike: Strike price
            quantity: Number of straddles
            side: "buy" or "sell"

        Returns:
            OptionOrder with two legs
        """
        call_symbol = self.build_occ_symbol(underlying, expiration, "call", strike)
        put_symbol = self.build_occ_symbol(underlying, expiration, "put", strike)

        if side == "buy":
            option_side = OptionSide.BUY_TO_OPEN
        else:
            option_side = OptionSide.SELL_TO_OPEN

        return OptionOrder(
            legs=[
                OptionLeg(
                    contract_symbol=call_symbol,
                    quantity=quantity,
                    side=option_side,
                ),
                OptionLeg(
                    contract_symbol=put_symbol,
                    quantity=quantity,
                    side=option_side,
                ),
            ],
            strategy=OptionStrategy.STRADDLE,
        )

    def build_strangle(
        self,
        underlying: str,
        expiration: date,
        call_strike: float,
        put_strike: float,
        quantity: int = 1,
        side: str = "buy"
    ) -> OptionOrder:
        """
        Build a strangle order (different strike call + put).

        Args:
            underlying: Stock ticker
            expiration: Expiration date
            call_strike: Call strike price
            put_strike: Put strike price
            quantity: Number of strangles
            side: "buy" or "sell"

        Returns:
            OptionOrder with two legs
        """
        call_symbol = self.build_occ_symbol(underlying, expiration, "call", call_strike)
        put_symbol = self.build_occ_symbol(underlying, expiration, "put", put_strike)

        if side == "buy":
            option_side = OptionSide.BUY_TO_OPEN
        else:
            option_side = OptionSide.SELL_TO_OPEN

        return OptionOrder(
            legs=[
                OptionLeg(
                    contract_symbol=call_symbol,
                    quantity=quantity,
                    side=option_side,
                ),
                OptionLeg(
                    contract_symbol=put_symbol,
                    quantity=quantity,
                    side=option_side,
                ),
            ],
            strategy=OptionStrategy.STRANGLE,
        )

    def build_iron_condor(
        self,
        underlying: str,
        expiration: date,
        put_long_strike: float,
        put_short_strike: float,
        call_short_strike: float,
        call_long_strike: float,
        quantity: int = 1
    ) -> OptionOrder:
        """
        Build an iron condor order.

        Args:
            underlying: Stock ticker
            expiration: Expiration date
            put_long_strike: Lower put strike (buy)
            put_short_strike: Inner put strike (sell)
            call_short_strike: Inner call strike (sell)
            call_long_strike: Upper call strike (buy)
            quantity: Number of iron condors

        Returns:
            OptionOrder with four legs
        """
        return OptionOrder(
            legs=[
                OptionLeg(
                    contract_symbol=self.build_occ_symbol(underlying, expiration, "put", put_long_strike),
                    quantity=quantity,
                    side=OptionSide.BUY_TO_OPEN,
                ),
                OptionLeg(
                    contract_symbol=self.build_occ_symbol(underlying, expiration, "put", put_short_strike),
                    quantity=quantity,
                    side=OptionSide.SELL_TO_OPEN,
                ),
                OptionLeg(
                    contract_symbol=self.build_occ_symbol(underlying, expiration, "call", call_short_strike),
                    quantity=quantity,
                    side=OptionSide.SELL_TO_OPEN,
                ),
                OptionLeg(
                    contract_symbol=self.build_occ_symbol(underlying, expiration, "call", call_long_strike),
                    quantity=quantity,
                    side=OptionSide.BUY_TO_OPEN,
                ),
            ],
            strategy=OptionStrategy.IRON_CONDOR,
        )
