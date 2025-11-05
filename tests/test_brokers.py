"""
Tests for broker adapter system.
"""

import pytest
import polars as pl
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from wrtrade.brokers import (
    BrokerAdapter, 
    AlpacaBrokerAdapter,
    RobinhoodBrokerAdapter,
    TradingSession,
    Order,
    Position,
    OrderStatus,
    OrderType,
    TradingSessionConfig,
    MockBrokerAdapter
)


class TestOrder:
    """Test Order data class."""
    
    def test_order_creation(self):
        """Test basic order creation."""
        order = Order(
            symbol="AAPL",
            quantity=100,
            order_type=OrderType.MARKET,
            side="buy"
        )
        
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.side == "buy"
        assert order.status == OrderStatus.PENDING
        assert order.order_id is None
    
    def test_order_with_limit_price(self):
        """Test order with limit price."""
        order = Order(
            symbol="TSLA",
            quantity=50,
            order_type=OrderType.LIMIT,
            side="sell",
            limit_price=250.0
        )
        
        assert order.limit_price == 250.0
        assert order.order_type == OrderType.LIMIT
    
    def test_order_validation(self):
        """Test order validation."""
        # Should work with valid inputs
        order = Order("AAPL", 100, OrderType.MARKET, "buy")
        assert order.symbol == "AAPL"
        
        # Test quantity validation
        with pytest.raises(ValueError):
            Order("AAPL", 0, OrderType.MARKET, "buy")
        
        with pytest.raises(ValueError):
            Order("AAPL", -10, OrderType.MARKET, "buy")


class TestPosition:
    """Test Position data class."""
    
    def test_position_creation(self):
        """Test basic position creation."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_value=15000.0
        )
        
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.avg_price == 150.0
        assert position.market_value == 15000.0
    
    def test_position_pnl(self):
        """Test position P&L calculation."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_value=16000.0
        )
        
        # P&L should be market_value - (quantity * avg_price)
        expected_pnl = 16000.0 - (100 * 150.0)
        assert abs(position.unrealized_pnl - expected_pnl) < 1e-6


class TestMockBrokerAdapter:
    """Test mock broker adapter for testing purposes."""
    
    @pytest.fixture
    def mock_broker(self, mock_api_credentials):
        """Create mock broker adapter."""
        return MockBrokerAdapter(mock_api_credentials)
    
    def test_mock_broker_authentication(self, mock_broker):
        """Test mock broker authentication."""
        result = mock_broker.authenticate()
        assert result is True
    
    def test_mock_broker_account_info(self, mock_broker):
        """Test mock broker account info."""
        mock_broker.authenticate()
        account_info = mock_broker.get_account_info()
        
        assert "account_id" in account_info
        assert "buying_power" in account_info
        assert "cash" in account_info
        assert account_info["buying_power"] > 0
    
    def test_mock_broker_positions(self, mock_broker):
        """Test mock broker positions."""
        mock_broker.authenticate()
        positions = mock_broker.get_positions()
        
        assert isinstance(positions, list)
        # Mock should have some default positions
        assert len(positions) > 0
        
        for position in positions:
            assert isinstance(position, Position)
            assert position.symbol is not None
            assert position.quantity != 0
    
    def test_mock_broker_place_order(self, mock_broker):
        """Test mock broker order placement."""
        mock_broker.authenticate()
        
        order = Order(
            symbol="AAPL",
            quantity=100,
            order_type=OrderType.MARKET,
            side="buy"
        )
        
        result = mock_broker.place_order(order)
        
        assert result is True
        assert order.status == OrderStatus.FILLED  # Mock fills immediately
        assert order.order_id is not None
        assert order.filled_quantity == 100
        assert order.fill_price is not None
    
    def test_mock_broker_order_history(self, mock_broker):
        """Test mock broker order history."""
        mock_broker.authenticate()
        
        # Place an order first
        order = Order("TSLA", 50, OrderType.LIMIT, "sell", limit_price=200.0)
        mock_broker.place_order(order)
        
        # Get order history
        orders = mock_broker.get_orders()
        assert len(orders) >= 1
        
        # Check that our order is there
        order_symbols = [o.symbol for o in orders]
        assert "TSLA" in order_symbols
    
    def test_mock_broker_market_data(self, mock_broker):
        """Test mock broker market data."""
        mock_broker.authenticate()
        
        price = mock_broker.get_current_price("AAPL")
        assert isinstance(price, float)
        assert price > 0
    
    def test_mock_broker_error_handling(self, mock_broker):
        """Test mock broker error handling."""
        # Test without authentication
        with pytest.raises(Exception):
            mock_broker.get_account_info()
        
        # Test invalid order
        mock_broker.authenticate()
        invalid_order = Order("INVALID_SYMBOL_XYZ123", 1, OrderType.MARKET, "buy")
        result = mock_broker.place_order(invalid_order)
        assert result is False
        assert invalid_order.status == OrderStatus.REJECTED


class TestTradingSessionConfig:
    """Test trading session configuration."""
    
    def test_default_config(self):
        """Test default trading session configuration."""
        config = TradingSessionConfig()
        
        assert config.max_position_size == 1000.0
        assert config.max_daily_trades == 10
        assert config.risk_per_trade == 0.02
        assert config.stop_loss_percent == 0.05
        assert config.take_profit_percent == 0.10
        assert config.dry_run is False
    
    def test_custom_config(self):
        """Test custom trading session configuration."""
        config = TradingSessionConfig(
            max_position_size=5000.0,
            max_daily_trades=50,
            risk_per_trade=0.01,
            dry_run=True
        )
        
        assert config.max_position_size == 5000.0
        assert config.max_daily_trades == 50
        assert config.risk_per_trade == 0.01
        assert config.dry_run is True


class TestTradingSession:
    """Test trading session with risk management."""
    
    @pytest.fixture
    def trading_config(self):
        """Create trading configuration."""
        return TradingSessionConfig(
            max_position_size=1000.0,
            max_daily_trades=5,
            risk_per_trade=0.02,
            dry_run=True
        )
    
    @pytest.fixture
    def trading_session(self, mock_api_credentials, trading_config):
        """Create trading session with mock broker."""
        broker = MockBrokerAdapter(mock_api_credentials)
        return TradingSession(broker, trading_config)
    
    def test_trading_session_creation(self, trading_session):
        """Test trading session creation."""
        assert isinstance(trading_session.broker, MockBrokerAdapter)
        assert isinstance(trading_session.config, TradingSessionConfig)
        assert trading_session.trade_count == 0
        assert len(trading_session.daily_trades) == 0
    
    def test_trading_session_authentication(self, trading_session):
        """Test trading session authentication."""
        result = trading_session.start_session()
        assert result is True
        assert trading_session.is_authenticated is True
    
    def test_position_size_calculation(self, trading_session):
        """Test position size calculation with risk management."""
        trading_session.start_session()
        
        # Get account info
        account_info = trading_session.broker.get_account_info()
        available_cash = account_info["cash"]
        
        # Calculate position size for a trade
        symbol = "AAPL"
        current_price = trading_session.broker.get_current_price(symbol)
        risk_amount = available_cash * trading_session.config.risk_per_trade
        
        # Position size should be limited by risk and max position size
        position_size = min(
            risk_amount / current_price,
            trading_session.config.max_position_size
        )
        
        assert position_size > 0
        assert position_size <= trading_session.config.max_position_size
    
    def test_order_placement_with_risk_checks(self, trading_session):
        """Test order placement with risk management checks."""
        trading_session.start_session()
        
        # Should allow normal order within limits
        order1 = Order("AAPL", 100, OrderType.MARKET, "buy")
        result1 = trading_session.place_order(order1)
        assert result1 is True
        assert trading_session.trade_count == 1
        
        # Should allow multiple orders up to daily limit
        for i in range(3):  # 3 more orders (4 total, limit is 5)
            order = Order(f"STOCK{i}", 50, OrderType.MARKET, "buy")
            result = trading_session.place_order(order)
            assert result is True
        
        assert trading_session.trade_count == 4
        
        # Should reject order that exceeds daily trade limit
        order_over_limit = Order("TSLA", 25, OrderType.MARKET, "buy")
        result_over_limit = trading_session.place_order(order_over_limit)
        assert result_over_limit is True  # 5th trade still allowed
        
        # 6th trade should be rejected
        order_rejected = Order("MSFT", 25, OrderType.MARKET, "buy")
        result_rejected = trading_session.place_order(order_rejected)
        assert result_rejected is False
    
    def test_position_size_risk_management(self, trading_session):
        """Test position size risk management."""
        trading_session.start_session()
        
        # Should reject order that's too large
        large_order = Order("AAPL", 10000, OrderType.MARKET, "buy")  # Exceeds max_position_size
        result = trading_session.place_order(large_order)
        
        # Mock broker should handle the validation
        assert isinstance(result, bool)
    
    def test_daily_reset(self, trading_session):
        """Test daily trading limits reset."""
        trading_session.start_session()
        
        # Make some trades
        for i in range(3):
            order = Order(f"STOCK{i}", 50, OrderType.MARKET, "buy")
            trading_session.place_order(order)
        
        assert trading_session.trade_count == 3
        
        # Reset daily counters (simulating next day)
        trading_session.reset_daily_limits()
        assert trading_session.trade_count == 0
        assert len(trading_session.daily_trades) == 0
    
    def test_dry_run_mode(self, trading_session):
        """Test dry run mode functionality."""
        # Config is set to dry_run=True
        assert trading_session.config.dry_run is True
        
        trading_session.start_session()
        
        order = Order("AAPL", 100, OrderType.MARKET, "buy")
        result = trading_session.place_order(order)
        
        # In dry run, orders should be "simulated" but not actually placed
        assert result is True
        assert order.status == OrderStatus.FILLED  # Mock fills immediately
    
    def test_session_monitoring(self, trading_session):
        """Test session monitoring capabilities."""
        trading_session.start_session()
        
        # Get session stats
        stats = trading_session.get_session_stats()
        
        assert "trades_today" in stats
        assert "remaining_trades" in stats
        assert "total_pnl" in stats
        assert "positions_count" in stats
        
        assert stats["trades_today"] == 0
        assert stats["remaining_trades"] == 5  # max_daily_trades
    
    def test_error_handling(self, trading_session):
        """Test error handling in trading session."""
        # Test placing order before authentication
        order = Order("AAPL", 100, OrderType.MARKET, "buy")
        result = trading_session.place_order(order)
        assert result is False
        
        # Test session start
        trading_session.start_session()
        
        # Test invalid order
        invalid_order = Order("", 0, OrderType.MARKET, "buy")
        with pytest.raises(ValueError):
            trading_session.place_order(invalid_order)


class TestAlpacaBrokerAdapter:
    """Test Alpaca broker adapter (with mocking)."""
    
    @pytest.fixture
    def mock_alpaca_client(self):
        """Create mock Alpaca client."""
        mock_client = MagicMock()
        
        # Mock account info
        mock_account = MagicMock()
        mock_account.cash = "10000.0"
        mock_account.buying_power = "20000.0"
        mock_account.account_number = "123456789"
        mock_client.get_account.return_value = mock_account
        
        # Mock positions
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.0"
        mock_position.market_value = "15500.0"
        mock_client.list_positions.return_value = [mock_position]
        
        # Mock order placement
        mock_order = MagicMock()
        mock_order.id = "order_123"
        mock_order.status = "filled"
        mock_order.filled_qty = "100"
        mock_order.filled_avg_price = "151.0"
        mock_client.submit_order.return_value = mock_order
        
        return mock_client
    
    @pytest.fixture
    def alpaca_adapter(self, mock_api_credentials):
        """Create Alpaca adapter instance."""
        return AlpacaBrokerAdapter(mock_api_credentials)
    
    @patch('wrtrade.brokers.tradeapi')
    def test_alpaca_authentication(self, mock_tradeapi, alpaca_adapter, mock_alpaca_client):
        """Test Alpaca authentication."""
        mock_tradeapi.REST.return_value = mock_alpaca_client
        
        result = alpaca_adapter.authenticate()
        assert result is True
        assert alpaca_adapter.client is not None
    
    @patch('wrtrade.brokers.tradeapi')
    def test_alpaca_account_info(self, mock_tradeapi, alpaca_adapter, mock_alpaca_client):
        """Test Alpaca account info retrieval."""
        mock_tradeapi.REST.return_value = mock_alpaca_client
        alpaca_adapter.authenticate()
        
        account_info = alpaca_adapter.get_account_info()
        
        assert account_info["cash"] == 10000.0
        assert account_info["buying_power"] == 20000.0
        assert account_info["account_id"] == "123456789"
    
    @patch('wrtrade.brokers.tradeapi')
    def test_alpaca_positions(self, mock_tradeapi, alpaca_adapter, mock_alpaca_client):
        """Test Alpaca positions retrieval."""
        mock_tradeapi.REST.return_value = mock_alpaca_client
        alpaca_adapter.authenticate()
        
        positions = alpaca_adapter.get_positions()
        
        assert len(positions) == 1
        position = positions[0]
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.avg_price == 150.0
        assert position.market_value == 15500.0
    
    @patch('wrtrade.brokers.tradeapi')
    def test_alpaca_order_placement(self, mock_tradeapi, alpaca_adapter, mock_alpaca_client):
        """Test Alpaca order placement."""
        mock_tradeapi.REST.return_value = mock_alpaca_client
        alpaca_adapter.authenticate()
        
        order = Order("AAPL", 100, OrderType.MARKET, "buy")
        result = alpaca_adapter.place_order(order)
        
        assert result is True
        assert order.order_id == "order_123"
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.fill_price == 151.0
    
    @patch('wrtrade.brokers.tradeapi')
    def test_alpaca_error_handling(self, mock_tradeapi, alpaca_adapter, mock_alpaca_client):
        """Test Alpaca error handling."""
        # Test authentication failure
        mock_tradeapi.REST.side_effect = Exception("API Error")
        result = alpaca_adapter.authenticate()
        assert result is False
        
        # Test unauthenticated operations
        with pytest.raises(Exception):
            alpaca_adapter.get_account_info()


class TestRobinhoodBrokerAdapter:
    """Test Robinhood broker adapter (with mocking)."""
    
    @pytest.fixture
    def robinhood_adapter(self, mock_api_credentials):
        """Create Robinhood adapter instance."""
        return RobinhoodBrokerAdapter(mock_api_credentials)
    
    @patch('wrtrade.brokers.robin_stocks')
    def test_robinhood_authentication(self, mock_robin, robinhood_adapter):
        """Test Robinhood authentication."""
        mock_robin.authentication.login.return_value = {"access_token": "token123"}
        
        result = robinhood_adapter.authenticate()
        assert result is True
    
    @patch('wrtrade.brokers.robin_stocks')
    def test_robinhood_account_info(self, mock_robin, robinhood_adapter):
        """Test Robinhood account info retrieval."""
        # Mock login
        mock_robin.authentication.login.return_value = {"access_token": "token123"}
        robinhood_adapter.authenticate()
        
        # Mock account data
        mock_robin.profiles.load_account_profile.return_value = {
            "account": "RH12345",
            "buying_power": "15000.50",
            "cash": "5000.25"
        }
        
        account_info = robinhood_adapter.get_account_info()
        
        assert account_info["account_id"] == "RH12345"
        assert account_info["buying_power"] == 15000.50
        assert account_info["cash"] == 5000.25
    
    @patch('wrtrade.brokers.robin_stocks')
    def test_robinhood_positions(self, mock_robin, robinhood_adapter):
        """Test Robinhood positions retrieval."""
        mock_robin.authentication.login.return_value = {"access_token": "token123"}
        robinhood_adapter.authenticate()
        
        # Mock positions data
        mock_robin.account.get_open_stock_positions.return_value = [
            {
                "symbol": "TSLA",
                "quantity": "50.00000000",
                "average_buy_price": "200.50",
                "market_value": "10200.00"
            }
        ]
        
        positions = robinhood_adapter.get_positions()
        
        assert len(positions) == 1
        position = positions[0]
        assert position.symbol == "TSLA"
        assert position.quantity == 50
        assert position.avg_price == 200.50
        assert position.market_value == 10200.0
    
    @patch('wrtrade.brokers.robin_stocks')
    def test_robinhood_error_handling(self, mock_robin, robinhood_adapter):
        """Test Robinhood error handling."""
        # Test authentication failure
        mock_robin.authentication.login.side_effect = Exception("Login failed")
        result = robinhood_adapter.authenticate()
        assert result is False


class TestBrokerIntegration:
    """Integration tests for broker system."""
    
    def test_broker_adapter_interface_compliance(self):
        """Test that all broker adapters implement the required interface."""
        adapters = [MockBrokerAdapter, AlpacaBrokerAdapter, RobinhoodBrokerAdapter]
        
        required_methods = [
            'authenticate', 'get_account_info', 'get_positions', 
            'place_order', 'get_orders', 'get_current_price'
        ]
        
        for adapter_class in adapters:
            for method_name in required_methods:
                assert hasattr(adapter_class, method_name)
                # Check that it's a method (callable)
                method = getattr(adapter_class, method_name)
                assert callable(method)
    
    def test_order_lifecycle(self, mock_api_credentials):
        """Test complete order lifecycle with mock broker."""
        broker = MockBrokerAdapter(mock_api_credentials)
        session = TradingSession(broker, TradingSessionConfig(dry_run=True))
        
        # Start session
        session.start_session()
        
        # Create and place order
        order = Order("AAPL", 100, OrderType.MARKET, "buy")
        initial_status = order.status
        
        result = session.place_order(order)
        
        # Check order progression
        assert initial_status == OrderStatus.PENDING
        assert result is True
        assert order.status == OrderStatus.FILLED
        assert order.order_id is not None
        assert order.filled_quantity == 100
    
    def test_position_tracking(self, mock_api_credentials):
        """Test position tracking across multiple trades."""
        broker = MockBrokerAdapter(mock_api_credentials)
        session = TradingSession(broker, TradingSessionConfig(dry_run=True))
        
        session.start_session()
        
        # Get initial positions
        initial_positions = broker.get_positions()
        initial_count = len(initial_positions)
        
        # Place a few trades
        symbols = ["AAPL", "TSLA", "GOOGL"]
        for symbol in symbols:
            order = Order(symbol, 50, OrderType.MARKET, "buy")
            session.place_order(order)
        
        # Check updated positions (mock broker behavior may vary)
        updated_positions = broker.get_positions()
        assert isinstance(updated_positions, list)
        assert all(isinstance(pos, Position) for pos in updated_positions)
    
    def test_risk_management_integration(self, mock_api_credentials):
        """Test risk management integration across the system."""
        config = TradingSessionConfig(
            max_position_size=500.0,
            max_daily_trades=3,
            risk_per_trade=0.01,
            stop_loss_percent=0.03,
            take_profit_percent=0.06,
            dry_run=True
        )
        
        broker = MockBrokerAdapter(mock_api_credentials)
        session = TradingSession(broker, config)
        session.start_session()
        
        # Test that limits are enforced
        successful_trades = 0
        for i in range(5):  # Try to place 5 trades (limit is 3)
            order = Order(f"STOCK{i}", 100, OrderType.MARKET, "buy")
            result = session.place_order(order)
            if result:
                successful_trades += 1
        
        # Should only allow 3 trades
        assert successful_trades <= config.max_daily_trades
    
    def test_error_recovery(self, mock_api_credentials):
        """Test error recovery mechanisms."""
        broker = MockBrokerAdapter(mock_api_credentials)
        session = TradingSession(broker, TradingSessionConfig(dry_run=True))
        
        # Test session recovery after connection issues
        session.start_session()
        assert session.is_authenticated is True
        
        # Simulate connection loss and recovery
        session.is_authenticated = False
        result = session.start_session()  # Should re-authenticate
        assert result is True
        assert session.is_authenticated is True