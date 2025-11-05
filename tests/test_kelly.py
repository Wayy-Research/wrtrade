"""
Tests for Kelly optimization system.
"""

import pytest
import polars as pl
import numpy as np
from typing import List, Dict, Any

from wrtrade.kelly import (
    KellyOptimizer, 
    HierarchicalKellyOptimizer, 
    KellyConfig
)
from wrtrade.components import SignalComponent, CompositePortfolio


class TestKellyConfig:
    """Test Kelly optimization configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = KellyConfig()
        
        assert config.min_weight == 0.0
        assert config.max_weight == 1.0
        assert config.max_leverage == 1.0
        assert config.risk_free_rate == 0.02
        assert config.lookback_window == 252
        assert config.rebalance_frequency == 21
        assert config.regularization_lambda == 0.01
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = KellyConfig(
            min_weight=-0.5,
            max_weight=2.0,
            max_leverage=3.0,
            risk_free_rate=0.05
        )
        
        assert config.min_weight == -0.5
        assert config.max_weight == 2.0
        assert config.max_leverage == 3.0
        assert config.risk_free_rate == 0.05


class TestKellyOptimizer:
    """Test individual Kelly criterion calculations."""
    
    @pytest.fixture
    def optimizer(self):
        """Create Kelly optimizer instance."""
        return KellyOptimizer()
    
    @pytest.fixture
    def constrained_optimizer(self):
        """Create Kelly optimizer with constraints."""
        config = KellyConfig(
            min_weight=0.1,
            max_weight=0.9,
            max_leverage=1.0,
            allow_short=False
        )
        return KellyOptimizer(config)
    
    def test_single_asset_kelly_basic(self, optimizer, sample_prices):
        """Test basic single asset Kelly calculation."""
        returns = sample_prices.log().diff().fill_null(0)
        
        kelly_fraction = optimizer.calculate_discrete_kelly(
            returns, risk_free_rate=0.02
        )
        
        # Kelly fraction should be finite
        assert np.isfinite(kelly_fraction)
        # For reasonable returns, Kelly should be bounded
        assert -10 <= kelly_fraction <= 10
    
    def test_single_asset_kelly_edge_cases(self, optimizer):
        """Test single asset Kelly with edge cases."""
        # All zero returns
        zero_returns = pl.Series([0.0] * 100)
        kelly = optimizer.calculate_discrete_kelly(zero_returns)
        assert kelly == 0.0
        
        # Very high volatility, low return
        np.random.seed(42)
        high_vol_returns = pl.Series(np.random.normal(-0.001, 0.1, 100))
        kelly = optimizer.calculate_discrete_kelly(high_vol_returns)
        # Should be negative or very small positive
        assert kelly < 0.5
    
    def test_single_asset_kelly_with_risk_free_rate(self, optimizer, sample_prices):
        """Test Kelly calculation with different risk-free rates."""
        returns = sample_prices.log().diff().fill_null(0)
        
        kelly_0 = optimizer.calculate_discrete_kelly(returns, risk_free_rate=0.0)
        kelly_5 = optimizer.calculate_discrete_kelly(returns, risk_free_rate=0.05)
        
        # Results should be finite
        assert np.isfinite(kelly_0)
        assert np.isfinite(kelly_5)
    
    def test_portfolio_kelly_calculation(self, optimizer, multi_market_prices):
        """Test portfolio Kelly calculation."""
        # Convert prices to returns
        returns_list = []
        for prices in multi_market_prices:
            returns = prices.log().diff().fill_null(0)
            returns_list.append(returns.to_numpy())
        
        returns_matrix = np.column_stack(returns_list)
        
        kelly_weights = optimizer.calculate_portfolio_kelly(
            returns_matrix, risk_free_rate=0.02
        )
        
        # Should have weights for each asset
        assert len(kelly_weights) == len(multi_market_prices)
        # All weights should be finite
        assert all(np.isfinite(w) for w in kelly_weights.values())
        # Should return asset_0, asset_1, asset_2 keys
        expected_keys = {f"asset_{i}" for i in range(len(multi_market_prices))}
        assert set(kelly_weights.keys()) == expected_keys
    
    def test_portfolio_kelly_with_names(self, optimizer, multi_market_prices):
        """Test portfolio Kelly calculation with custom asset names."""
        returns_list = []
        for prices in multi_market_prices:
            returns = prices.log().diff().fill_null(0)
            returns_list.append(returns.to_numpy())
        
        returns_matrix = np.column_stack(returns_list)
        names = ['AAPL', 'TSLA', 'GOOGL']
        
        kelly_weights = optimizer.calculate_portfolio_kelly(
            returns_matrix, names=names, risk_free_rate=0.02
        )
        
        assert set(kelly_weights.keys()) == set(names)
        assert all(np.isfinite(w) for w in kelly_weights.values())
    
    def test_portfolio_kelly_singular_matrix(self, optimizer):
        """Test portfolio Kelly with singular covariance matrix."""
        # Create perfectly correlated returns (singular covariance)
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, 100)
        returns_matrix = np.column_stack([
            base_returns,
            base_returns * 1.5,  # Perfectly correlated
            base_returns * 0.8   # Perfectly correlated
        ])
        
        # Should handle singular matrix gracefully
        kelly_weights = optimizer.calculate_portfolio_kelly(returns_matrix)
        
        # Should return some reasonable result (not crash)
        assert len(kelly_weights) == 3
        assert all(np.isfinite(w) for w in kelly_weights.values())
    
    def test_constrained_kelly_optimization(self, constrained_optimizer, multi_market_prices):
        """Test Kelly optimization with constraints."""
        returns_list = []
        for prices in multi_market_prices:
            returns = prices.log().diff().fill_null(0)
            returns_list.append(returns.to_numpy())
        
        returns_matrix = np.column_stack(returns_list)
        
        # Test with constraints
        constrained_weights = constrained_optimizer.calculate_portfolio_kelly(returns_matrix)
        
        # Check constraints are satisfied (allowing for small numerical errors)
        for weight in constrained_weights.values():
            assert weight >= 0.1 - 1e-6  # min_weight constraint
            assert weight <= 0.9 + 1e-6  # max_weight constraint
        
        # Check leverage constraint
        total_leverage = sum(abs(w) for w in constrained_weights.values())
        assert total_leverage <= 1.0 + 1e-6
    
    def test_insufficient_data_handling(self, optimizer):
        """Test handling of insufficient data."""
        # Create very short return series
        short_returns = np.random.normal(0.001, 0.02, 10)  # Less than min_observations
        
        # Single asset
        kelly_single = optimizer.calculate_discrete_kelly(pl.Series(short_returns))
        assert kelly_single == 0.0
        
        # Portfolio
        short_matrix = short_returns.reshape(-1, 1)
        kelly_portfolio = optimizer.calculate_portfolio_kelly(short_matrix)
        assert len(kelly_portfolio) == 1
        assert kelly_portfolio['asset_0'] == 1.0  # Equal weight fallback


class TestHierarchicalKellyOptimizer:
    """Test hierarchical Kelly optimization for N-dimensional portfolios."""
    
    @pytest.fixture
    def hierarchical_optimizer(self):
        """Create hierarchical Kelly optimizer."""
        config = KellyConfig(
            min_weight=0.0,
            max_weight=1.0,
            lookback_window=100  # Shorter for testing
        )
        return HierarchicalKellyOptimizer(config)
    
    def test_optimizer_creation(self, hierarchical_optimizer):
        """Test hierarchical optimizer creation."""
        assert isinstance(hierarchical_optimizer, HierarchicalKellyOptimizer)
        assert isinstance(hierarchical_optimizer.optimizer, KellyOptimizer)
        assert hierarchical_optimizer.config.lookback_window == 100
    
    def test_single_component_optimization_interface(self, hierarchical_optimizer, sample_signal_components):
        """Test that hierarchical optimizer can handle single components."""
        component = sample_signal_components[0]
        
        # Check that component has necessary methods
        assert hasattr(component, 'generate_signals')
        assert hasattr(component, 'calculate_returns')
        assert hasattr(component, 'name')
    
    def test_portfolio_interface_check(self, hierarchical_optimizer, sample_signal_components, long_sample_prices):
        """Test that portfolios have necessary interfaces for optimization."""
        from wrtrade.components import CompositePortfolio
        
        # Create a simple composite portfolio
        portfolio = CompositePortfolio(
            name="test_portfolio",
            components=sample_signal_components[:2],
            weight=1.0
        )
        
        # Check interfaces exist
        assert hasattr(portfolio, 'generate_signals')
        assert hasattr(portfolio, 'calculate_returns')
        assert hasattr(portfolio, 'component_list')
        assert hasattr(portfolio, 'name')
        
        # Test signal generation works
        signals = portfolio.generate_signals(long_sample_prices)
        assert isinstance(signals, pl.Series)
        assert len(signals) == len(long_sample_prices)
        
        # Test returns calculation works
        returns = portfolio.calculate_returns(long_sample_prices)
        assert isinstance(returns, pl.Series)
        assert hasattr(portfolio, '_returns')


class TestKellyIntegration:
    """Integration tests for Kelly optimization with other systems."""
    
    def test_kelly_with_simple_portfolio(self, sample_signal_components, long_sample_prices):
        """Test Kelly optimization with a simple portfolio."""
        from wrtrade.components import CompositePortfolio
        
        # Create portfolio
        portfolio = CompositePortfolio(
            name="kelly_test",
            components=sample_signal_components[:2],
            weight=1.0
        )
        
        # Generate returns
        portfolio.generate_signals(long_sample_prices)
        portfolio.calculate_returns(long_sample_prices)
        
        # Basic optimization with individual components
        optimizer = KellyOptimizer()
        
        # Test that we can extract returns from components
        component_returns = []
        for component in portfolio.component_list:
            component.generate_signals(long_sample_prices)
            component.calculate_returns(long_sample_prices)
            if hasattr(component, '_returns') and component._returns is not None:
                component_returns.append(component._returns.to_numpy())
        
        if component_returns:
            returns_matrix = np.column_stack(component_returns)
            kelly_weights = optimizer.calculate_portfolio_kelly(returns_matrix)
            
            assert len(kelly_weights) == len(component_returns)
            assert all(np.isfinite(w) for w in kelly_weights.values())
    
    def test_kelly_config_variations(self, sample_signal_components, long_sample_prices):
        """Test different Kelly configurations."""
        from wrtrade.components import CompositePortfolio
        
        # Test configurations
        configs = [
            KellyConfig(max_leverage=0.5, allow_short=False),
            KellyConfig(max_leverage=2.0, allow_short=True),
            KellyConfig(regularization_lambda=0.1, use_constraints=True),
            KellyConfig(regularization_lambda=0.001, use_constraints=False)
        ]
        
        portfolio = CompositePortfolio(
            name="config_test",
            components=sample_signal_components[:2],
            weight=1.0
        )
        
        portfolio.generate_signals(long_sample_prices)
        portfolio.calculate_returns(long_sample_prices)
        
        for i, config in enumerate(configs):
            optimizer = KellyOptimizer(config)
            
            # Test single asset optimization
            returns = long_sample_prices.log().diff().fill_null(0)
            kelly_single = optimizer.calculate_discrete_kelly(returns)
            
            assert np.isfinite(kelly_single), f"Config {i} failed single asset test"
            
            # Verify constraints are respected
            if not config.allow_short:
                assert kelly_single >= config.min_weight - 1e-6
            assert abs(kelly_single) <= config.max_weight * config.max_leverage + 1e-6
    
    def test_kelly_with_different_data_sizes(self):
        """Test Kelly optimization with different data sizes."""
        optimizer = KellyOptimizer()
        
        # Test various data sizes
        for n_obs in [50, 100, 500, 1000]:
            np.random.seed(42)  # Consistent results
            returns = pl.Series(np.random.normal(0.001, 0.02, n_obs))
            
            kelly = optimizer.calculate_discrete_kelly(returns)
            
            if n_obs >= optimizer.config.min_observations:
                assert np.isfinite(kelly)
            else:
                assert kelly == 0.0  # Insufficient data
    
    def test_kelly_numerical_stability(self):
        """Test Kelly optimization numerical stability."""
        optimizer = KellyOptimizer()
        
        # Test with extreme values
        extreme_cases = [
            np.random.normal(0.1, 0.01, 252),    # High return, low vol
            np.random.normal(-0.1, 0.01, 252),   # Negative return, low vol
            np.random.normal(0.001, 0.5, 252),   # Low return, high vol
            np.random.normal(0.0, 0.01, 252),    # Zero return
        ]
        
        for i, returns_data in enumerate(extreme_cases):
            returns = pl.Series(returns_data)
            kelly = optimizer.calculate_discrete_kelly(returns)
            
            # Should always return finite value
            assert np.isfinite(kelly), f"Extreme case {i} produced non-finite result"
            
            # Should be bounded by configuration
            assert abs(kelly) <= optimizer.config.max_weight * optimizer.config.max_leverage + 1e-6