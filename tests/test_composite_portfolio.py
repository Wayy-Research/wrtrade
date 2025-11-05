"""
Pytest tests for CompositePortfolio and N-dimensional portfolio functionality.
"""

import pytest
import polars as pl
import numpy as np
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wrtrade'))

from wrtrade.components import SignalComponent, CompositePortfolio, AllocationWeights
from wrtrade.ndimensional_portfolio import NDimensionalPortfolioBuilder, PortfolioBuilderConfig


class TestCompositePortfolio:
    """Test CompositePortfolio functionality."""
    
    @pytest.fixture
    def sample_signal_components(self, ma_crossover_signal, momentum_signal, buy_hold_signal):
        """Create sample signal components for testing."""
        return [
            SignalComponent("ma_signal", ma_crossover_signal, weight=0.4),
            SignalComponent("momentum_signal", momentum_signal, weight=0.3),
            SignalComponent("buy_hold_signal", buy_hold_signal, weight=0.3)
        ]
    
    def test_basic_creation(self, sample_signal_components):
        """Test basic CompositePortfolio creation."""
        portfolio = CompositePortfolio(
            name="test_portfolio",
            components=sample_signal_components,
            weight=1.0
        )
        
        assert portfolio.name == "test_portfolio"
        assert portfolio.weight == 1.0
        assert len(portfolio.component_list) == 3
        assert len(portfolio.components) == 3
        
        # Check component names are mapped correctly
        assert "ma_signal" in portfolio.components
        assert "momentum_signal" in portfolio.components  
        assert "buy_hold_signal" in portfolio.components
    
    def test_component_management(self, sample_signal_components):
        """Test adding and removing components."""
        portfolio = CompositePortfolio(
            name="mgmt_test",
            components=sample_signal_components[:2],  # Start with 2 components
            weight=1.0
        )
        
        assert len(portfolio.component_list) == 2
        
        # Add component
        new_component = sample_signal_components[2]
        portfolio.add_component(new_component)
        
        assert len(portfolio.component_list) == 3
        assert new_component.name in portfolio.components
        
        # Remove component
        portfolio.remove_component("ma_signal")
        
        assert len(portfolio.component_list) == 2
        assert "ma_signal" not in portfolio.components
        assert len(portfolio.component_list) == len(portfolio.components)  # Consistency check
    
    def test_signal_generation(self, sample_prices, sample_signal_components):
        """Test composite signal generation."""
        portfolio = CompositePortfolio(
            name="signal_test",
            components=sample_signal_components,
            weight=1.0
        )
        
        signals = portfolio.generate_signals(sample_prices)
        
        # Check signal properties
        assert len(signals) == len(sample_prices)
        assert isinstance(signals, pl.Series)
        
        # Signals should be bounded between -1 and 1 (weighted average)
        assert signals.min() >= -1.0, f"Signal too negative: {signals.min()}"
        assert signals.max() <= 1.0, f"Signal too positive: {signals.max()}"
        
        # Should have some variation
        assert signals.std() > 0, "Signals have no variation"
    
    def test_return_calculation(self, sample_prices, sample_signal_components):
        """Test composite return calculation."""
        portfolio = CompositePortfolio(
            name="return_test",
            components=sample_signal_components,
            weight=1.0
        )
        
        returns = portfolio.calculate_returns(sample_prices)
        
        # Check return properties
        assert len(returns) == len(sample_prices)
        assert isinstance(returns, pl.Series)
        assert returns.is_finite().all(), "All returns should be finite"
        
        # Check that internal state is set
        assert portfolio._returns is not None
    
    def test_performance_metrics(self, sample_prices, sample_signal_components):
        """Test composite portfolio performance metrics."""
        portfolio = CompositePortfolio(
            name="metrics_test",
            components=sample_signal_components,
            weight=1.0
        )
        
        # Calculate returns first
        portfolio.calculate_returns(sample_prices)
        metrics = portfolio.get_performance_metrics()
        
        # Check portfolio-level metrics exist
        required_metrics = ['volatility', 'sortino_ratio', 'gain_to_pain_ratio', 'max_drawdown']
        for metric in required_metrics:
            assert metric in metrics, f"Missing portfolio metric: {metric}"
        
        # Check component-level metrics are included
        for component in sample_signal_components:
            component_prefix = f"{component.name}_"
            component_metrics = [key for key in metrics.keys() if key.startswith(component_prefix)]
            assert len(component_metrics) > 0, f"No metrics found for component {component.name}"
    
    def test_weight_management(self, sample_signal_components):
        """Test component weight management."""
        portfolio = CompositePortfolio(
            name="weight_test",
            components=sample_signal_components,
            weight=1.0
        )
        
        # Get initial weights
        initial_weights = portfolio.get_component_weights()
        assert len(initial_weights) == 3
        assert initial_weights["ma_signal"] == 0.4
        assert initial_weights["momentum_signal"] == 0.3
        assert initial_weights["buy_hold_signal"] == 0.3
        
        # Rebalance components
        new_weights = {
            "ma_signal": 0.5,
            "momentum_signal": 0.3,
            "buy_hold_signal": 0.2
        }
        portfolio.rebalance_components(new_weights)
        
        # Check weights were updated
        updated_weights = portfolio.get_component_weights()
        for name, expected_weight in new_weights.items():
            assert abs(updated_weights[name] - expected_weight) < 1e-10
        
        # Check allocation history
        allocation_history = portfolio.get_allocation_history()
        assert len(allocation_history) == 1
        assert isinstance(allocation_history[0], AllocationWeights)
    
    def test_empty_portfolio(self, sample_prices):
        """Test behavior with empty portfolio."""
        empty_portfolio = CompositePortfolio(
            name="empty",
            components=[],
            weight=1.0
        )
        
        # Should handle empty portfolio gracefully
        signals = empty_portfolio.generate_signals(sample_prices)
        assert len(signals) == len(sample_prices)
        assert all(s == 0 for s in signals.to_list())
        
        returns = empty_portfolio.calculate_returns(sample_prices)
        assert len(returns) == len(sample_prices)
        assert all(r == 0.0 for r in returns.to_list())
    
    def test_nested_portfolios(self, sample_prices, sample_signal_components):
        """Test portfolio of portfolios (nesting)."""
        # Create sub-portfolios
        sub_portfolio_1 = CompositePortfolio(
            name="sub_portfolio_1",
            components=sample_signal_components[:2],  # First 2 components
            weight=0.6
        )
        
        sub_portfolio_2 = CompositePortfolio(
            name="sub_portfolio_2", 
            components=sample_signal_components[2:],  # Last component
            weight=0.4
        )
        
        # Create meta-portfolio
        meta_portfolio = CompositePortfolio(
            name="meta_portfolio",
            components=[sub_portfolio_1, sub_portfolio_2],
            weight=1.0
        )
        
        # Test signal generation works
        signals = meta_portfolio.generate_signals(sample_prices)
        assert len(signals) == len(sample_prices)
        assert isinstance(signals, pl.Series)
        
        # Test return calculation works
        returns = meta_portfolio.calculate_returns(sample_prices)
        assert len(returns) == len(sample_prices)
        assert returns.is_finite().all()
        
        # Test performance metrics include nested components
        metrics = meta_portfolio.get_performance_metrics()
        assert "sub_portfolio_1_" in str(metrics) or any(key.startswith("sub_portfolio_1") for key in metrics.keys())
    
    def test_print_structure(self, sample_signal_components, capfd):
        """Test portfolio structure printing."""
        # Create nested structure
        sub_portfolio = CompositePortfolio(
            name="sub_portfolio",
            components=sample_signal_components[:2],
            weight=0.7
        )
        
        main_portfolio = CompositePortfolio(
            name="main_portfolio",
            components=[sub_portfolio, sample_signal_components[2]],
            weight=1.0
        )
        
        # Print structure
        main_portfolio.print_structure()
        
        # Capture output
        captured = capfd.readouterr()
        
        # Check that structure is printed
        assert "main_portfolio" in captured.out
        assert "sub_portfolio" in captured.out
        assert "buy_hold_signal" in captured.out


class TestNDimensionalPortfolioBuilder:
    """Test NDimensionalPortfolioBuilder functionality."""
    
    def test_builder_creation(self):
        """Test basic builder creation."""
        builder = NDimensionalPortfolioBuilder()
        assert builder.config.default_weight == 1.0
        assert builder.config.max_depth == 10
        assert builder.config.kelly_optimization == True
        
        # Test with custom config
        config = PortfolioBuilderConfig(
            default_weight=0.5,
            max_depth=5,
            kelly_optimization=False
        )
        custom_builder = NDimensionalPortfolioBuilder(config)
        assert custom_builder.config.default_weight == 0.5
        assert custom_builder.config.max_depth == 5
        assert custom_builder.config.kelly_optimization == False
    
    def test_signal_registration(self, ma_crossover_signal, momentum_signal):
        """Test signal function registration and reuse."""
        builder = NDimensionalPortfolioBuilder()
        
        # Register signals
        builder.register_signal("ma_cross", ma_crossover_signal)
        builder.register_signal("momentum", momentum_signal)
        
        assert "ma_cross" in builder._signal_registry
        assert "momentum" in builder._signal_registry
        
        # Create components using registered signals
        component1 = builder.create_signal_component("test1", "ma_cross")
        component2 = builder.create_signal_component("test2", "momentum")
        
        assert component1.name == "test1"
        assert component2.name == "test2"
        assert component1.signal_func == ma_crossover_signal
        assert component2.signal_func == momentum_signal
    
    def test_component_creation(self, ma_crossover_signal):
        """Test component creation methods."""
        builder = NDimensionalPortfolioBuilder()
        
        # Create signal component directly
        component = builder.create_signal_component(
            name="direct_signal",
            signal_func_or_name=ma_crossover_signal,
            weight=0.75,
            max_position=0.5
        )
        
        assert isinstance(component, SignalComponent)
        assert component.name == "direct_signal"
        assert component.weight == 0.75
        assert component.max_position == 0.5
        
        # Create component with default weight
        default_component = builder.create_signal_component(
            name="default_signal",
            signal_func_or_name=ma_crossover_signal
        )
        assert default_component.weight == builder.config.default_weight
    
    def test_portfolio_creation(self, sample_signal_components):
        """Test portfolio creation methods."""
        builder = NDimensionalPortfolioBuilder()
        
        # Create basic portfolio
        portfolio = builder.create_portfolio(
            name="test_portfolio",
            components=sample_signal_components,
            weight=0.8
        )
        
        assert isinstance(portfolio, CompositePortfolio)
        assert portfolio.name == "test_portfolio"
        assert portfolio.weight == 0.8
        assert len(portfolio.component_list) == 3
        assert portfolio.kelly_optimization == builder.config.kelly_optimization
        
        # Create portfolio with custom settings
        custom_portfolio = builder.create_portfolio(
            name="custom_portfolio",
            components=sample_signal_components[:2],
            rebalance_frequency=30,
            kelly_optimization=False
        )
        assert custom_portfolio.rebalance_frequency == 30
        assert custom_portfolio.kelly_optimization == False
    
    def test_balanced_portfolio_creation(self, sample_signal_components):
        """Test balanced portfolio creation."""
        builder = NDimensionalPortfolioBuilder()
        
        # Create balanced portfolio
        balanced_portfolio = builder.create_balanced_portfolio(
            name="balanced_test",
            components=sample_signal_components
        )
        
        # Check that weights are equal
        expected_weight = 1.0 / len(sample_signal_components)
        for component in balanced_portfolio.component_list:
            assert abs(component.weight - expected_weight) < 1e-10
    
    def test_portfolio_validation(self, sample_signal_components):
        """Test portfolio structure validation."""
        builder = NDimensionalPortfolioBuilder()
        
        # Create valid portfolio
        valid_portfolio = builder.create_portfolio(
            name="valid",
            components=sample_signal_components
        )
        
        # Should validate successfully
        assert builder.validate_portfolio_structure(valid_portfolio) == True
        
        # Test with duplicate names (should fail)
        duplicate_components = [
            SignalComponent("duplicate", lambda p: pl.Series([0] * len(p)), 1.0),
            SignalComponent("duplicate", lambda p: pl.Series([1] * len(p)), 1.0)
        ]
        
        duplicate_portfolio = CompositePortfolio("invalid", duplicate_components)
        
        with pytest.raises(ValueError, match="Duplicate component name"):
            builder.validate_portfolio_structure(duplicate_portfolio)
    
    def test_max_depth_validation(self):
        """Test maximum depth validation."""
        config = PortfolioBuilderConfig(max_depth=2)
        builder = NDimensionalPortfolioBuilder(config)
        
        # Create deeply nested structure that exceeds max depth
        # depth=0: level4_portfolio
        # depth=1: level3_portfolio 
        # depth=2: level2_portfolio
        # depth=3: level1_portfolio  <- This should exceed max_depth=2
        level1_component = SignalComponent("level1", lambda p: pl.Series([0] * len(p)), 1.0)
        level1_portfolio = CompositePortfolio("level1_portfolio", [level1_component])  # Add another portfolio level
        level2_portfolio = CompositePortfolio("level2", [level1_portfolio])
        level3_portfolio = CompositePortfolio("level3", [level2_portfolio])
        level4_portfolio = CompositePortfolio("level4", [level3_portfolio])  # This should exceed max_depth
        
        # Should raise error for exceeding max depth
        with pytest.raises(ValueError, match="Portfolio depth exceeds maximum"):
            builder.validate_portfolio_structure(level4_portfolio)


class TestComplexPortfolioScenarios:
    """Test complex multi-level portfolio scenarios."""
    
    def test_three_level_portfolio(self, sample_prices, ma_crossover_signal, momentum_signal, buy_hold_signal):
        """Test a three-level portfolio hierarchy."""
        builder = NDimensionalPortfolioBuilder()
        
        # Level 1: Individual signals
        signal1 = builder.create_signal_component("signal1", ma_crossover_signal, weight=0.5)
        signal2 = builder.create_signal_component("signal2", momentum_signal, weight=0.5)
        signal3 = builder.create_signal_component("signal3", buy_hold_signal, weight=1.0)
        
        # Level 2: Sub-portfolios
        sub_portfolio_1 = builder.create_portfolio("sub_port_1", [signal1, signal2], weight=0.6)
        sub_portfolio_2 = builder.create_portfolio("sub_port_2", [signal3], weight=0.4)
        
        # Level 3: Meta-portfolio
        meta_portfolio = builder.create_portfolio("meta_port", [sub_portfolio_1, sub_portfolio_2])
        
        # Test that the entire structure works
        signals = meta_portfolio.generate_signals(sample_prices)
        returns = meta_portfolio.calculate_returns(sample_prices)
        metrics = meta_portfolio.get_performance_metrics()
        
        assert len(signals) == len(sample_prices)
        assert len(returns) == len(sample_prices)
        assert len(metrics) > 0
        
        # Test structure validation
        assert builder.validate_portfolio_structure(meta_portfolio) == True
    
    def test_portfolio_consistency(self, sample_prices, ma_crossover_signal):
        """Test that portfolio calculations are consistent across multiple runs."""
        builder = NDimensionalPortfolioBuilder()
        
        # Create identical portfolios
        component1 = builder.create_signal_component("test", ma_crossover_signal, weight=1.0)
        component2 = builder.create_signal_component("test", ma_crossover_signal, weight=1.0)
        
        portfolio1 = builder.create_portfolio("p1", [component1])
        portfolio2 = builder.create_portfolio("p2", [component2])
        
        # Calculate returns
        returns1 = portfolio1.calculate_returns(sample_prices)
        returns2 = portfolio2.calculate_returns(sample_prices)
        
        # Should be nearly identical (allowing for floating point errors)
        diff = (returns1 - returns2).abs().max()
        assert diff < 1e-10, f"Portfolio calculations not consistent: max diff = {diff}"
    
    def test_mixed_weight_portfolio(self, long_sample_prices):
        """Test portfolio with mixed positive and negative weights (long/short)."""
        builder = NDimensionalPortfolioBuilder()
        
        # Create components with mixed weights
        long_component = builder.create_signal_component(
            "long_signal", 
            lambda p: pl.Series([1] * len(p)),  # Always long
            weight=1.5
        )
        
        short_component = builder.create_signal_component(
            "short_signal",
            lambda p: pl.Series([1] * len(p)),  # Always long signal, but negative weight
            weight=-0.5  # Short position
        )
        
        mixed_portfolio = builder.create_portfolio(
            "mixed_portfolio",
            [long_component, short_component]
        )
        
        # Calculate returns
        returns = mixed_portfolio.calculate_returns(long_sample_prices)
        
        # Should work without errors
        assert len(returns) == len(long_sample_prices)
        assert returns.is_finite().all()
        
        # Net weight should be 1.0 (1.5 - 0.5)
        net_weight = sum(comp.weight for comp in mixed_portfolio.component_list)
        assert abs(net_weight - 1.0) < 1e-10
    
    def test_portfolio_attribution(self, sample_prices, sample_signal_components):
        """Test that portfolio attribution adds up correctly."""
        portfolio = CompositePortfolio(
            name="attribution_test",
            components=sample_signal_components,
            weight=1.0
        )
        
        # Calculate portfolio returns
        portfolio_returns = portfolio.calculate_returns(sample_prices)
        portfolio_total = portfolio_returns.sum()
        
        # Calculate individual component contributions
        component_contributions = 0.0
        for component in sample_signal_components:
            component_returns = component.calculate_returns(sample_prices)
            component_contributions += component_returns.sum()
        
        # Portfolio total should approximately equal sum of component contributions
        # (allowing for small numerical differences)
        diff = abs(portfolio_total - component_contributions)
        relative_diff = diff / abs(portfolio_total) if portfolio_total != 0 else diff
        
        assert relative_diff < 0.01, f"Attribution error too large: {relative_diff:.4f}"