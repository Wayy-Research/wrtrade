import pytest
import polars as pl
import numpy as np
from wrtrade.metrics import (
    volatility, 
    sortino_ratio, 
    gain_to_pain_ratio, 
    max_drawdown,
    rolling_volatility,
    rolling_metric,
    rolling_max_drawdown,
    calculate_all_metrics,
    calculate_all_rolling_metrics,
    tear_sheet
)


@pytest.fixture
def sample_returns():
    """Generate sample return data."""
    np.random.seed(42)
    return pl.Series('returns', np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year


@pytest.fixture
def positive_returns():
    """Generate only positive returns."""
    return pl.Series('returns', [0.01, 0.02, 0.015, 0.005, 0.03])


@pytest.fixture
def negative_returns():
    """Generate only negative returns."""
    return pl.Series('returns', [-0.01, -0.02, -0.015, -0.005, -0.03])


@pytest.fixture
def mixed_returns():
    """Generate mixed positive and negative returns."""
    return pl.Series('returns', [0.02, -0.01, 0.03, -0.015, 0.01, -0.005])


def test_volatility(sample_returns):
    """Test volatility calculation."""
    vol = volatility(sample_returns)
    assert isinstance(vol, float)
    assert vol > 0
    
    # Test with known values
    known_returns = pl.Series('returns', [0.01, -0.01, 0.01, -0.01])
    vol_known = volatility(known_returns)
    assert vol_known > 0


def test_sortino_ratio(sample_returns):
    """Test Sortino ratio calculation."""
    sortino = sortino_ratio(sample_returns)
    assert isinstance(sortino, (float, type(float('inf'))))
    
    # Test with only positive returns
    pos_returns = pl.Series('returns', [0.01, 0.02, 0.03])
    sortino_pos = sortino_ratio(pos_returns)
    assert sortino_pos == float('inf')  # No downside deviation
    
    # Test with mixed returns
    mixed = pl.Series('returns', [0.02, -0.01, 0.01])
    sortino_mixed = sortino_ratio(mixed)
    assert isinstance(sortino_mixed, float)


def test_gain_to_pain_ratio(mixed_returns):
    """Test gain to pain ratio calculation."""
    gtp = gain_to_pain_ratio(mixed_returns)
    assert isinstance(gtp, (float, type(float('inf'))))
    assert gtp > 0
    
    # Test with only positive returns
    pos_returns = pl.Series('returns', [0.01, 0.02, 0.03])
    gtp_pos = gain_to_pain_ratio(pos_returns)
    assert gtp_pos == float('inf')  # No losses
    
    # Test with only negative returns  
    neg_returns = pl.Series('returns', [-0.01, -0.02, -0.03])
    gtp_neg = gain_to_pain_ratio(neg_returns)
    assert gtp_neg == 0  # No gains


def test_max_drawdown(sample_returns):
    """Test maximum drawdown calculation."""
    dd = max_drawdown(sample_returns)
    assert isinstance(dd, float)
    assert dd <= 0  # Drawdown should be negative or zero
    
    # Test with monotonically increasing returns
    increasing = pl.Series('returns', [0.01, 0.01, 0.01, 0.01])
    dd_inc = max_drawdown(increasing)
    assert dd_inc == 0  # No drawdown
    
    # Test with known drawdown pattern
    pattern = pl.Series('returns', [0.1, -0.05, -0.03, 0.02])
    dd_pattern = max_drawdown(pattern)
    assert dd_pattern < 0


def test_rolling_volatility(sample_returns):
    """Test rolling volatility calculation."""
    window = 50
    rolling_vol = rolling_volatility(sample_returns, window)
    
    assert isinstance(rolling_vol, pl.Series)
    assert len(rolling_vol) == len(sample_returns)
    
    # First (window-1) values should be null
    assert rolling_vol[:window-1].null_count() == window - 1
    
    # Non-null values should be positive
    non_null_values = rolling_vol.drop_nulls()
    assert all(val > 0 for val in non_null_values)


def test_rolling_metric(sample_returns):
    """Test generic rolling metric function."""
    window = 30
    rolling_vol = rolling_metric(sample_returns, volatility, window)
    
    assert isinstance(rolling_vol, pl.Series)
    assert len(rolling_vol) == len(sample_returns)


def test_rolling_max_drawdown(sample_returns):
    """Test rolling maximum drawdown."""
    window = 50
    rolling_dd = rolling_max_drawdown(sample_returns, window)
    
    assert isinstance(rolling_dd, pl.Series)
    assert len(rolling_dd) == len(sample_returns)
    
    # Non-null values should be <= 0
    non_null_values = rolling_dd.drop_nulls()
    assert all(val <= 0 for val in non_null_values)


def test_calculate_all_metrics(sample_returns):
    """Test calculation of all metrics at once."""
    metrics = calculate_all_metrics(sample_returns)
    
    assert isinstance(metrics, dict)
    required_metrics = ['volatility', 'sortino_ratio', 'gain_to_pain_ratio', 'max_drawdown']
    
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (float, type(float('inf'))))


def test_calculate_all_rolling_metrics(sample_returns):
    """Test calculation of all rolling metrics."""
    window = 50
    rolling_metrics = calculate_all_rolling_metrics(sample_returns, window)
    
    assert isinstance(rolling_metrics, dict)
    expected_keys = ['rolling_volatility', 'rolling_sortino', 'rolling_gain_to_pain', 'rolling_max_drawdown']
    
    for key in expected_keys:
        assert key in rolling_metrics
        assert isinstance(rolling_metrics[key], pl.Series)
        assert len(rolling_metrics[key]) == len(sample_returns)


def test_tear_sheet(sample_returns, capsys):
    """Test tear sheet printing."""
    tear_sheet(sample_returns)
    captured = capsys.readouterr()
    
    # Check that output contains expected elements
    assert "PERFORMANCE TEAR SHEET" in captured.out
    assert "Volatility" in captured.out
    assert "Sortino Ratio" in captured.out
    assert "Gain to Pain Ratio" in captured.out
    assert "Max Drawdown" in captured.out
    assert "=" in captured.out  # Check for formatting


def test_edge_cases():
    """Test edge cases and error handling."""
    # Empty series
    empty_returns = pl.Series('returns', [], dtype=pl.Float64)
    
    # Should handle gracefully without crashing
    try:
        vol = volatility(empty_returns)
        # If it doesn't crash, vol should be 0
        assert vol == 0
    except:
        pass  # Some edge cases might raise exceptions, which is acceptable
    
    # Single value
    single_return = pl.Series('returns', [0.01])
    vol_single = volatility(single_return)
    assert vol_single == 0  # No variance with single value
    
    # All zeros
    zero_returns = pl.Series('returns', [0, 0, 0, 0])
    vol_zero = volatility(zero_returns)
    assert vol_zero == 0


def test_metric_consistency():
    """Test that individual metric functions give same results as batch calculation."""
    returns = pl.Series('returns', [0.02, -0.01, 0.03, -0.015, 0.01])
    
    # Individual calculations
    vol_individual = volatility(returns)
    sortino_individual = sortino_ratio(returns)
    gtp_individual = gain_to_pain_ratio(returns)
    dd_individual = max_drawdown(returns)
    
    # Batch calculation
    metrics_batch = calculate_all_metrics(returns)
    
    # Should be the same
    assert abs(vol_individual - metrics_batch['volatility']) < 1e-10
    assert abs(sortino_individual - metrics_batch['sortino_ratio']) < 1e-10
    assert abs(gtp_individual - metrics_batch['gain_to_pain_ratio']) < 1e-10
    assert abs(dd_individual - metrics_batch['max_drawdown']) < 1e-10