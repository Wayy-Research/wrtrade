"""
Pytest tests for PortfolioComponent interface and SignalComponent.
"""

import pytest
import polars as pl
import numpy as np
from typing import Dict, Any

# Add the wrtrade package to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wrtrade'))

from wrtrade.components import PortfolioComponent, SignalComponent, AllocationWeights


class TestPortfolioComponentInterface:
    """Test the abstract PortfolioComponent interface."""
    
    def test_signal_component_is_portfolio_component(self, sample_prices, ma_crossover_signal):
        """Test that SignalComponent properly inherits from PortfolioComponent."""
        component = SignalComponent(
            name="test_component",
            signal_func=ma_crossover_signal,
            weight=1.0
        )
        
        assert isinstance(component, PortfolioComponent)
    
    def test_required_methods_exist(self, sample_prices, ma_crossover_signal):
        """Test that all required abstract methods are implemented."""
        component = SignalComponent(
            name="test_component",
            signal_func=ma_crossover_signal,
            weight=1.0
        )
        
        # Test all required methods exist and are callable
        assert hasattr(component, 'generate_signals')
        assert hasattr(component, 'calculate_returns')
        assert hasattr(component, 'get_performance_metrics')
        assert hasattr(component, 'set_weight')
        assert hasattr(component, 'get_weight')
        
        # Test methods return expected types
        signals = component.generate_signals(sample_prices)
        assert isinstance(signals, pl.Series)
        
        returns = component.calculate_returns(sample_prices)
        assert isinstance(returns, pl.Series)
        
        metrics = component.get_performance_metrics()
        assert isinstance(metrics, dict)
    
    def test_weight_management(self, sample_prices, ma_crossover_signal):
        """Test weight getting and setting functionality."""
        initial_weight = 0.75
        component = SignalComponent(
            name="weight_test",
            signal_func=ma_crossover_signal,
            weight=initial_weight
        )
        
        # Test initial weight
        assert component.get_weight() == initial_weight
        
        # Test weight setting
        new_weight = 1.25
        component.set_weight(new_weight)
        assert component.get_weight() == new_weight
        
        # Test negative weight (should be allowed for short positions)
        component.set_weight(-0.5)
        assert component.get_weight() == -0.5


class TestSignalComponent:
    """Test SignalComponent functionality."""
    
    def test_basic_creation(self, ma_crossover_signal):
        """Test basic SignalComponent creation."""
        component = SignalComponent(
            name="basic_test",
            signal_func=ma_crossover_signal,
            weight=1.0
        )
        
        assert component.name == "basic_test"
        assert component.weight == 1.0
        assert component.signal_func == ma_crossover_signal
        assert component.max_position == float('inf')
        assert component.take_profit is None
        assert component.stop_loss is None
    
    def test_signal_generation(self, sample_prices, ma_crossover_signal):
        """Test signal generation functionality."""
        component = SignalComponent(
            name="signal_test",
            signal_func=ma_crossover_signal,
            weight=1.0
        )
        
        signals = component.generate_signals(sample_prices)
        
        # Check signal properties
        assert len(signals) == len(sample_prices)
        assert isinstance(signals, pl.Series)
        
        # Check signal values are in expected range
        unique_signals = set(signals.to_numpy())
        assert unique_signals.issubset({-1, 0, 1}), f"Unexpected signal values: {unique_signals}"
        
        # Should have some non-zero signals for MA crossover
        non_zero_signals = len(signals.filter(signals != 0))
        assert non_zero_signals > 0, "MA crossover should generate some signals"
    
    def test_return_calculation(self, sample_prices, ma_crossover_signal):
        """Test return calculation functionality."""
        component = SignalComponent(
            name="return_test",
            signal_func=ma_crossover_signal,
            weight=1.0
        )
        
        returns = component.calculate_returns(sample_prices)
        
        # Check return properties
        assert len(returns) == len(sample_prices)
        assert isinstance(returns, pl.Series)
        
        # Returns should be finite numbers
        assert returns.is_finite().all(), "All returns should be finite"
        
        # Check that internal state is set
        assert component._returns is not None
        assert component._positions is not None
    
    def test_performance_metrics(self, sample_prices, ma_crossover_signal):
        """Test performance metrics calculation."""
        component = SignalComponent(
            name="metrics_test",
            signal_func=ma_crossover_signal,
            weight=1.0
        )
        
        # Calculate returns first
        component.calculate_returns(sample_prices)
        metrics = component.get_performance_metrics()
        
        # Check required metrics exist
        required_metrics = ['volatility', 'sortino_ratio', 'gain_to_pain_ratio', 'max_drawdown']
        for metric in required_metrics:
            assert metric in metrics, f"Missing required metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be numeric"
            assert np.isfinite(metrics[metric]), f"Metric {metric} should be finite"
    
    def test_weight_scaling(self, sample_prices, ma_crossover_signal):
        """Test that component weight properly scales returns."""
        base_component = SignalComponent(
            name="base",
            signal_func=ma_crossover_signal,
            weight=1.0
        )
        
        scaled_component = SignalComponent(
            name="scaled",
            signal_func=ma_crossover_signal,
            weight=2.0
        )
        
        base_returns = base_component.calculate_returns(sample_prices)
        scaled_returns = scaled_component.calculate_returns(sample_prices)
        
        # Scaled returns should be approximately 2x base returns
        ratio = scaled_returns.sum() / base_returns.sum() if base_returns.sum() != 0 else 0
        assert abs(ratio - 2.0) < 0.01, f"Weight scaling not working: ratio = {ratio}"
    
    def test_position_limits(self, sample_prices, buy_hold_signal):
        """Test position limiting functionality."""
        max_pos = 0.5
        component = SignalComponent(
            name="limited",
            signal_func=buy_hold_signal,  # Always long signal
            weight=1.0,
            max_position=max_pos
        )
        
        component.calculate_returns(sample_prices)
        positions = component._positions
        
        # Check position limits are respected
        max_absolute_position = positions.abs().max()
        assert max_absolute_position <= max_pos, f"Position limit violated: {max_absolute_position} > {max_pos}"
    
    def test_different_signal_types(self, sample_prices, ma_crossover_signal, momentum_signal, buy_hold_signal):
        """Test with different types of signal functions."""
        signals_config = [
            ("ma_crossover", ma_crossover_signal),
            ("momentum", momentum_signal),
            ("buy_hold", buy_hold_signal)
        ]
        
        components = {}
        results = {}
        
        for name, signal_func in signals_config:
            component = SignalComponent(
                name=name,
                signal_func=signal_func,
                weight=1.0
            )
            
            returns = component.calculate_returns(sample_prices)
            metrics = component.get_performance_metrics()
            
            components[name] = component
            results[name] = {
                'total_return': returns.sum(),
                'volatility': metrics['volatility'],
                'n_signals': (component.generate_signals(sample_prices) != 0).sum()
            }
        
        # All should work without errors
        for name, result in results.items():
            assert np.isfinite(result['total_return']), f"{name} has non-finite return"
            assert np.isfinite(result['volatility']), f"{name} has non-finite volatility"
            assert result['n_signals'] >= 0, f"{name} has negative signal count"
        
        # Buy-hold should have most signals (all 1's)
        assert results['buy_hold']['n_signals'] >= results['ma_crossover']['n_signals']


class TestAllocationWeights:
    """Test AllocationWeights utility class."""
    
    def test_basic_creation(self):
        """Test basic AllocationWeights creation."""
        weights = AllocationWeights(
            weights={'A': 0.4, 'B': 0.6},
            timestamp='2024-01-01',
            rebalance_reason='initial'
        )
        
        assert weights.weights == {'A': 0.4, 'B': 0.6}
        assert weights.timestamp == '2024-01-01'
        assert weights.rebalance_reason == 'initial'
    
    def test_normalization(self):
        """Test weight normalization functionality."""
        # Test normalization of weights that don't sum to 1
        weights = AllocationWeights(
            weights={'A': 0.8, 'B': 1.2, 'C': 0.6}  # Sum = 2.6
        )
        
        normalized = weights.normalize()
        
        # Check normalization
        total = sum(normalized.weights.values())
        assert abs(total - 1.0) < 1e-10, f"Normalized weights don't sum to 1: {total}"
        
        # Check proportions are maintained
        original_ratio = weights.weights['A'] / weights.weights['B']
        normalized_ratio = normalized.weights['A'] / normalized.weights['B']
        assert abs(original_ratio - normalized_ratio) < 1e-10, "Proportions not maintained"
    
    def test_zero_sum_weights(self):
        """Test handling of zero-sum weights."""
        weights = AllocationWeights(weights={'A': 0.0, 'B': 0.0})
        normalized = weights.normalize()
        
        # Should return original when sum is zero
        assert normalized.weights == weights.weights


class TestSignalComponentIntegration:
    """Integration tests for SignalComponent with realistic scenarios."""
    
    def test_multiple_components_correlation(self, long_sample_prices):
        """Test correlation between different signal components."""
        # Create components with different parameters
        components = {
            'fast_ma': SignalComponent(
                name="fast_ma",
                signal_func=lambda p: self._ma_signal(p, 5, 15),
                weight=1.0
            ),
            'slow_ma': SignalComponent(
                name="slow_ma", 
                signal_func=lambda p: self._ma_signal(p, 20, 50),
                weight=1.0
            ),
            'momentum': SignalComponent(
                name="momentum",
                signal_func=lambda p: self._momentum_signal(p, 20),
                weight=1.0
            )
        }
        
        # Calculate returns for all components
        returns_data = {}
        for name, component in components.items():
            returns_data[name] = component.calculate_returns(long_sample_prices).to_numpy()
        
        # Calculate correlations
        correlations = {}
        names = list(returns_data.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name_i, name_j = names[i], names[j]
                corr = np.corrcoef(returns_data[name_i], returns_data[name_j])[0, 1]
                correlations[f"{name_i}_vs_{name_j}"] = corr
        
        # Correlations should be reasonable (not perfect, not zero)
        for pair, corr in correlations.items():
            assert abs(corr) < 0.99, f"Correlation too high for {pair}: {corr}"
            assert np.isfinite(corr), f"Invalid correlation for {pair}: {corr}"
    
    def _ma_signal(self, prices: pl.Series, fast: int, slow: int) -> pl.Series:
        """Helper method for MA signal."""
        if len(prices) < slow:
            return pl.Series([0] * len(prices))
        
        fast_ma = prices.rolling_mean(window_size=fast)
        slow_ma = prices.rolling_mean(window_size=slow)
        
        signals = []
        for i in range(len(prices)):
            if i < slow:
                signals.append(0)
            elif fast_ma[i] > slow_ma[i]:
                signals.append(1)
            else:
                signals.append(-1)
        
        return pl.Series(signals)
    
    def _momentum_signal(self, prices: pl.Series, lookback: int) -> pl.Series:
        """Helper method for momentum signal."""
        signals = []
        for i in range(len(prices)):
            if i < lookback:
                signals.append(0)
            else:
                momentum = (prices[i] - prices[i-lookback]) / prices[i-lookback]
                signals.append(1 if momentum > 0.02 else -1 if momentum < -0.02 else 0)
        
        return pl.Series(signals)
    
    def test_edge_cases(self, ma_crossover_signal):
        """Test edge cases and error conditions."""
        # Test with very short price series
        short_prices = pl.Series([100.0, 101.0, 99.0])
        
        component = SignalComponent(
            name="edge_test",
            signal_func=ma_crossover_signal,
            weight=1.0
        )
        
        # Should handle short series gracefully
        signals = component.generate_signals(short_prices)
        assert len(signals) == len(short_prices)
        
        returns = component.calculate_returns(short_prices)
        assert len(returns) == len(short_prices)
        
        # Test with single price point
        single_price = pl.Series([100.0])
        single_signals = component.generate_signals(single_price)
        assert len(single_signals) == 1
        
        single_returns = component.calculate_returns(single_price)
        assert len(single_returns) == 1
        assert single_returns[0] == 0.0  # No return possible with single price
    
    def test_performance_consistency(self, sample_prices, ma_crossover_signal):
        """Test that performance calculations are consistent."""
        component = SignalComponent(
            name="consistency_test",
            signal_func=ma_crossover_signal,
            weight=1.0
        )
        
        # Calculate returns multiple times
        returns1 = component.calculate_returns(sample_prices)
        returns2 = component.calculate_returns(sample_prices)
        
        # Should be identical
        assert returns1.equals(returns2), "Returns calculation not consistent"
        
        # Metrics should be identical
        metrics1 = component.get_performance_metrics()
        metrics2 = component.get_performance_metrics()
        
        for key in metrics1.keys():
            assert abs(metrics1[key] - metrics2[key]) < 1e-10, f"Inconsistent metric: {key}"