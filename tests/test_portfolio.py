import pytest
import polars as pl
import numpy as np
from wrtrade.portfolio import Portfolio
from wrtrade.metrics import volatility, sortino_ratio, gain_to_pain_ratio, max_drawdown


@pytest.fixture
def sample_prices():
    """Generate sample price data using Polars."""
    np.random.seed(42)
    return pl.Series('price', np.random.normal(100, 10, 100))


@pytest.fixture
def sample_signals():
    """Generate sample signal data using Polars.""" 
    np.random.seed(42)
    return pl.Series('signal', np.random.choice([-1, 0, 1], 100))


@pytest.fixture
def sample_portfolio(sample_prices, sample_signals):
    """Create a sample portfolio."""
    return Portfolio(sample_prices, sample_signals, max_position=5)


def test_portfolio_initialization(sample_portfolio):
    """Test portfolio initialization."""
    assert isinstance(sample_portfolio, Portfolio)
    assert isinstance(sample_portfolio.prices, pl.Series)
    assert isinstance(sample_portfolio.signals, pl.Series)
    assert sample_portfolio.max_position == 5
    assert sample_portfolio.take_profit is None
    assert sample_portfolio.stop_loss is None


def test_portfolio_lazy_evaluation(sample_portfolio):
    """Test that metrics are calculated lazily."""
    # Before calculation
    assert sample_portfolio._positions is None
    assert sample_portfolio._returns is None
    assert sample_portfolio._cumulative_returns is None
    assert sample_portfolio._metrics is None
    
    # Access positions - should trigger calculation
    positions = sample_portfolio.positions
    assert sample_portfolio._positions is not None
    assert isinstance(positions, pl.Series)
    assert len(positions) == 100


def test_calculate_performance(sample_portfolio):
    """Test performance calculation."""
    results = sample_portfolio.calculate_performance()
    
    assert isinstance(results, dict)
    assert 'total_return' in results
    assert 'positions' in results
    assert 'returns' in results
    assert 'cumulative_returns' in results
    assert 'metrics' in results
    
    assert isinstance(results['total_return'], (float, np.floating))
    assert isinstance(results['positions'], pl.Series)
    assert isinstance(results['returns'], pl.Series)
    assert isinstance(results['cumulative_returns'], pl.Series)
    assert isinstance(results['metrics'], dict)


def test_position_constraints(sample_prices, sample_signals):
    """Test max position constraints are respected."""
    max_pos = 3
    portfolio = Portfolio(sample_prices, sample_signals, max_position=max_pos)
    positions = portfolio.positions
    
    assert positions.max() <= max_pos
    assert positions.min() >= -max_pos


def test_metrics_calculation(sample_portfolio):
    """Test that all required metrics are calculated."""
    metrics = sample_portfolio.metrics
    
    required_metrics = ['volatility', 'sortino_ratio', 'gain_to_pain_ratio', 'max_drawdown']
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (float, np.floating))


def test_portfolio_properties(sample_portfolio):
    """Test portfolio properties return correct types."""
    # Test positions
    positions = sample_portfolio.positions
    assert isinstance(positions, pl.Series)
    assert len(positions) == 100
    
    # Test returns
    returns = sample_portfolio.returns
    assert isinstance(returns, pl.Series)
    assert len(returns) == 100
    
    # Test cumulative returns
    cum_returns = sample_portfolio.cumulative_returns
    assert isinstance(cum_returns, pl.Series)
    assert len(cum_returns) == 100


def test_take_profit_stop_loss_initialization():
    """Test portfolio with take profit and stop loss."""
    prices = pl.Series('price', [100, 105, 110, 95, 90])
    signals = pl.Series('signal', [1, 0, 0, 0, -1])
    
    portfolio = Portfolio(prices, signals, take_profit=0.05, stop_loss=0.02)
    assert portfolio.take_profit == 0.05
    assert portfolio.stop_loss == 0.02
    
    # Should still calculate without errors
    results = portfolio.calculate_performance()
    assert isinstance(results, dict)


def test_empty_signals():
    """Test portfolio with all zero signals."""
    prices = pl.Series('price', [100, 101, 102, 103, 104])
    signals = pl.Series('signal', [0, 0, 0, 0, 0])
    
    portfolio = Portfolio(prices, signals)
    results = portfolio.calculate_performance()
    
    # All positions should be zero
    assert portfolio.positions.sum() == 0
    # All returns should be zero
    assert portfolio.returns.sum() == 0


def test_single_direction_signals():
    """Test portfolio with only buy signals."""
    prices = pl.Series('price', [100, 101, 102, 103, 104])
    signals = pl.Series('signal', [1, 1, 1, 1, 1])
    
    portfolio = Portfolio(prices, signals, max_position=10)
    positions = portfolio.positions
    
    # Positions should increase monotonically (within limits)
    assert positions[-1] == 5  # Should be 5 (length of series)
    assert all(positions[i] <= positions[i+1] for i in range(len(positions)-1))


def test_portfolio_with_nan_prices():
    """Test portfolio handles NaN values gracefully."""
    prices_data = [100, 101, np.nan, 103, 104]
    # Convert to float series to handle NaN properly
    prices = pl.Series('price', prices_data, dtype=pl.Float64)
    signals = pl.Series('signal', [1, 0, 1, 0, -1])
    
    portfolio = Portfolio(prices, signals)
    # Should not raise an error
    results = portfolio.calculate_performance()
    assert isinstance(results, dict)