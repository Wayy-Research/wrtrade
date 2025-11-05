"""
Pytest configuration and fixtures for WRTrade testing.
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime
from typing import Callable


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)  # Reproducible tests
    n_days = 252
    returns = np.random.normal(0.0005, 0.02, n_days)
    
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    return pl.Series(prices[:-1])


@pytest.fixture
def long_sample_prices():
    """Generate longer sample price data for extended testing."""
    np.random.seed(42)
    n_days = 500
    returns = np.random.normal(0.0005, 0.015, n_days)
    
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    return pl.Series(prices[:-1])


@pytest.fixture
def ma_crossover_signal():
    """Simple moving average crossover signal function."""
    def signal_func(prices: pl.Series, fast_period: int = 10, slow_period: int = 30) -> pl.Series:
        if len(prices) < slow_period:
            return pl.Series([0] * len(prices))
        
        fast_ma = prices.rolling_mean(window_size=fast_period)
        slow_ma = prices.rolling_mean(window_size=slow_period)
        
        signals = []
        for i in range(len(prices)):
            if i < slow_period:
                signals.append(0)
            else:
                fast_val = fast_ma[i] if fast_ma[i] is not None else 0
                slow_val = slow_ma[i] if slow_ma[i] is not None else 0
                prev_fast = fast_ma[i-1] if i > 0 and fast_ma[i-1] is not None else 0
                prev_slow = slow_ma[i-1] if i > 0 and slow_ma[i-1] is not None else 0
                
                if fast_val > slow_val and prev_fast <= prev_slow:
                    signals.append(1)
                elif fast_val < slow_val and prev_fast >= prev_slow:
                    signals.append(-1)
                else:
                    signals.append(0)
        
        return pl.Series(signals)
    
    return signal_func


@pytest.fixture
def momentum_signal():
    """Simple momentum signal function."""
    def signal_func(prices: pl.Series, lookback: int = 20, threshold: float = 0.05) -> pl.Series:
        if len(prices) < lookback:
            return pl.Series([0] * len(prices))
        
        signals = []
        for i in range(len(prices)):
            if i < lookback:
                signals.append(0)
            else:
                current_price = prices[i]
                past_price = prices[i - lookback]
                momentum = (current_price - past_price) / past_price
                
                if momentum > threshold:
                    signals.append(1)
                elif momentum < -threshold:
                    signals.append(-1)
                else:
                    signals.append(0)
        
        return pl.Series(signals)
    
    return signal_func


@pytest.fixture
def buy_hold_signal():
    """Simple buy and hold signal."""
    def signal_func(prices: pl.Series) -> pl.Series:
        return pl.Series([1] * len(prices))
    
    return signal_func


@pytest.fixture
def multi_market_prices():
    """Generate correlated price data for multiple markets."""
    np.random.seed(42)
    n_days = 300
    n_markets = 3
    
    # Generate correlated returns
    correlation_matrix = np.array([
        [1.0, 0.6, 0.4],
        [0.6, 1.0, 0.3],
        [0.4, 0.3, 1.0]
    ])
    
    mean_returns = np.array([0.0005, 0.0003, 0.0007])
    volatilities = np.array([0.02, 0.015, 0.025])
    
    # Generate correlated random returns
    random_returns = np.random.multivariate_normal(
        mean=np.zeros(n_markets),
        cov=correlation_matrix,
        size=n_days
    )
    
    # Scale by volatilities and add mean
    scaled_returns = random_returns * volatilities + mean_returns
    
    # Convert to prices
    markets_data = []
    start_prices = [100.0, 50.0, 200.0]
    
    for market_idx in range(n_markets):
        prices = [start_prices[market_idx]]
        for ret in scaled_returns[:, market_idx]:
            prices.append(prices[-1] * (1 + ret))
        markets_data.append(pl.Series(prices[:-1]))
    
    return markets_data


@pytest.fixture
def mock_api_credentials():
    """Mock API credentials for testing."""
    return {
        'api_key': 'test_api_key_12345',
        'secret_key': 'test_secret_key_67890'
    }


@pytest.fixture
def sample_symbols_map():
    """Sample mapping from component names to trading symbols."""
    return {
        'MA_Signal': 'AAPL',
        'Momentum_Signal': 'TSLA',
        'Mean_Reversion': 'MSFT',
        'Trend_Following': 'GOOGL'
    }


@pytest.fixture
def sample_signal_components(ma_crossover_signal, momentum_signal, buy_hold_signal):
    """Create sample signal components for testing."""
    from wrtrade.components import SignalComponent
    
    return [
        SignalComponent("ma_signal", ma_crossover_signal, weight=0.4),
        SignalComponent("momentum_signal", momentum_signal, weight=0.3),
        SignalComponent("buy_hold_signal", buy_hold_signal, weight=0.3)
    ]