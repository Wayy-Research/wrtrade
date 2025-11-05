"""
Tests for simplified deployment system.
"""

import pytest
import polars as pl
import os
from unittest.mock import Mock, patch, AsyncMock

from wrtrade.deploy import DeployConfig, validate_strategy, deploy
from wrtrade.components import SignalComponent, CompositePortfolio
from wrtrade.ndimensional_portfolio import NDimensionalPortfolioBuilder


class TestDeployConfig:
    """Test deployment configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeployConfig()
        assert config.broker == "alpaca"
        assert config.paper == True
        assert config.max_position_pct == 0.10
        assert config.max_daily_loss_pct == 0.05
        assert config.validate == True
        assert config.min_sortino == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeployConfig(
            broker="alpaca",
            paper=False,
            max_position_pct=0.05,
            validate=False
        )
        assert config.broker == "alpaca"
        assert config.paper == False
        assert config.max_position_pct == 0.05
        assert config.validate == False

    def test_environment_credentials(self):
        """Test loading credentials from environment."""
        os.environ['ALPACA_API_KEY'] = 'test_key'
        os.environ['ALPACA_SECRET_KEY'] = 'test_secret'

        config = DeployConfig(broker="alpaca")
        assert config.api_key == 'test_key'
        assert config.secret_key == 'test_secret'

        # Cleanup
        del os.environ['ALPACA_API_KEY']
        del os.environ['ALPACA_SECRET_KEY']

    def test_explicit_credentials(self):
        """Test explicit credential override."""
        config = DeployConfig(
            broker="alpaca",
            api_key="explicit_key",
            secret_key="explicit_secret"
        )
        assert config.api_key == "explicit_key"
        assert config.secret_key == "explicit_secret"


class TestValidateStrategy:
    """Test strategy validation."""

    def test_validate_good_strategy(self, sample_prices, ma_crossover_signal):
        """Test validation of a good strategy."""
        builder = NDimensionalPortfolioBuilder()
        component = builder.create_signal_component("MA", ma_crossover_signal)
        portfolio = builder.create_portfolio("Test", [component])

        # Should pass validation with min_sortino=0
        result = validate_strategy(portfolio, sample_prices, min_sortino=0.0)
        assert isinstance(result, bool)

    def test_validate_high_threshold(self, sample_prices, ma_crossover_signal):
        """Test validation fails with very high threshold."""
        builder = NDimensionalPortfolioBuilder()
        component = builder.create_signal_component("MA", ma_crossover_signal)
        portfolio = builder.create_portfolio("Test", [component])

        # Should fail with unrealistic threshold
        result = validate_strategy(portfolio, sample_prices, min_sortino=10.0)
        assert result == False

    def test_validate_with_composite_portfolio(self, sample_prices, ma_crossover_signal, momentum_signal):
        """Test validation with composite portfolio."""
        builder = NDimensionalPortfolioBuilder()

        ma_comp = builder.create_signal_component("MA", ma_crossover_signal, weight=0.5)
        mom_comp = builder.create_signal_component("Mom", momentum_signal, weight=0.5)

        portfolio = builder.create_portfolio("Multi", [ma_comp, mom_comp])

        result = validate_strategy(portfolio, sample_prices, min_sortino=0.0)
        assert isinstance(result, bool)


class TestDeploy:
    """Test deployment function."""

    @pytest.mark.asyncio
    async def test_deploy_basic_mock(self, sample_prices, ma_crossover_signal):
        """Test basic deployment with mocked broker."""
        builder = NDimensionalPortfolioBuilder()
        component = builder.create_signal_component("MA", ma_crossover_signal)
        portfolio = builder.create_portfolio("Test", [component])

        symbols = {"MA": "AAPL"}

        # Mock the broker factory and broker
        with patch('wrtrade.brokers.BrokerFactory.create_broker') as mock_factory:
            mock_broker = AsyncMock()
            mock_broker.authenticate = AsyncMock(return_value=True)

            # Mock account info
            from wrtrade.brokers import AccountInfo, Position
            mock_account = AccountInfo(
                account_id="test_account",
                buying_power=10000.0,
                cash=10000.0,
                portfolio_value=10000.0,
                day_trade_buying_power=40000.0,
                is_day_trader=False,
                positions=[]
            )
            mock_broker.get_account_info = AsyncMock(return_value=mock_account)
            mock_factory.return_value = mock_broker

            config = DeployConfig(
                broker="alpaca",
                paper=True,
                api_key="test_key",
                secret_key="test_secret"
            )

            deployment_id = await deploy(portfolio, symbols, config)

            # Check deployment ID is returned
            assert isinstance(deployment_id, str)
            assert len(deployment_id) > 0

            # Verify broker was created and authenticated
            mock_factory.assert_called_once()
            mock_broker.authenticate.assert_called_once()
            mock_broker.get_account_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_authentication_failure(self, sample_prices, ma_crossover_signal):
        """Test deployment fails gracefully when authentication fails."""
        builder = NDimensionalPortfolioBuilder()
        component = builder.create_signal_component("MA", ma_crossover_signal)
        portfolio = builder.create_portfolio("Test", [component])

        symbols = {"MA": "AAPL"}

        with patch('wrtrade.brokers.BrokerFactory.create_broker') as mock_factory:
            mock_broker = AsyncMock()
            mock_broker.authenticate = AsyncMock(return_value=False)
            mock_factory.return_value = mock_broker

            config = DeployConfig(
                broker="alpaca",
                paper=True,
                api_key="test_key",
                secret_key="test_secret"
            )

            # Should raise RuntimeError when authentication fails
            with pytest.raises(RuntimeError, match="Failed to authenticate"):
                await deploy(portfolio, symbols, config)

    @pytest.mark.asyncio
    async def test_deploy_multiple_symbols(self, sample_prices, ma_crossover_signal, momentum_signal):
        """Test deployment with multiple symbol mappings."""
        builder = NDimensionalPortfolioBuilder()

        ma_comp = builder.create_signal_component("MA", ma_crossover_signal, weight=0.5)
        mom_comp = builder.create_signal_component("Mom", momentum_signal, weight=0.5)

        portfolio = builder.create_portfolio("Multi", [ma_comp, mom_comp])

        symbols = {
            "MA": "AAPL",
            "Mom": "TSLA"
        }

        with patch('wrtrade.brokers.BrokerFactory.create_broker') as mock_factory:
            mock_broker = AsyncMock()
            mock_broker.authenticate = AsyncMock(return_value=True)

            from wrtrade.brokers import AccountInfo
            mock_account = AccountInfo(
                account_id="test",
                buying_power=10000.0,
                cash=10000.0,
                portfolio_value=10000.0,
                day_trade_buying_power=40000.0,
                is_day_trader=False,
                positions=[]
            )
            mock_broker.get_account_info = AsyncMock(return_value=mock_account)
            mock_factory.return_value = mock_broker

            config = DeployConfig(
                broker="alpaca",
                paper=True,
                api_key="test_key",
                secret_key="test_secret"
            )

            deployment_id = await deploy(portfolio, symbols, config)
            assert isinstance(deployment_id, str)


class TestDeployIntegration:
    """Integration tests for deployment."""

    def test_config_defaults_are_sensible(self):
        """Test that default configuration values are reasonable."""
        config = DeployConfig()

        # Paper trading by default
        assert config.paper == True

        # Conservative risk limits
        assert config.max_position_pct <= 0.20  # At most 20%
        assert config.max_daily_loss_pct <= 0.10  # At most 10%

        # Validation enabled by default
        assert config.validate == True

        # Reasonable Sortino threshold
        assert config.min_sortino >= 0.5

    def test_validation_before_deploy_workflow(self, sample_prices, ma_crossover_signal):
        """Test the recommended workflow: validate then deploy."""
        builder = NDimensionalPortfolioBuilder()
        component = builder.create_signal_component("MA", ma_crossover_signal)
        portfolio = builder.create_portfolio("Test", [component])

        # Step 1: Validate
        is_valid = validate_strategy(portfolio, sample_prices, min_sortino=0.0)
        assert isinstance(is_valid, bool)

        # Step 2: Only deploy if valid (we'll skip actual deploy in test)
        if is_valid:
            # Would call deploy here in real code
            pass

    def test_simple_api_usage(self, sample_prices):
        """Test that the API is simple and intuitive."""
        # Define a simple signal
        def simple_signal(prices):
            return pl.Series([1] * len(prices))

        # Build portfolio - should be simple
        builder = NDimensionalPortfolioBuilder()
        component = builder.create_signal_component("Simple", simple_signal)
        portfolio = builder.create_portfolio("SimpleStrategy", [component])

        # Validate - should be simple
        valid = validate_strategy(portfolio, sample_prices, min_sortino=0.0)

        # Configure deployment - should be simple
        config = DeployConfig(broker="alpaca", paper=True)

        # Symbols mapping - should be simple
        symbols = {"Simple": "AAPL"}

        # All the pieces are there and simple to use
        assert portfolio is not None
        assert isinstance(valid, bool)
        assert config.broker == "alpaca"
        assert "Simple" in symbols
