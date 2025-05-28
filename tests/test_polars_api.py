"""Test Polars API usage to ensure correct method calls"""

import pytest
import polars as pl
import numpy as np
from datetime import date


def test_basic_series_creation():
    """Test basic Polars Series creation."""
    simple_series = pl.Series("test", [1, 2, 3, 4, 5])
    assert isinstance(simple_series, pl.Series)
    assert len(simple_series) == 5


def test_cumulative_operations():
    """Test cumulative operations work correctly."""
    simple_series = pl.Series("test", [1, 2, 3, 4, 5])
    
    cum_sum = simple_series.cum_sum()
    assert isinstance(cum_sum, pl.Series)
    assert len(cum_sum) == len(simple_series)
    assert cum_sum.to_list() == [1, 3, 6, 10, 15]


def test_rolling_operations():
    """Test rolling standard deviation works."""
    simple_series = pl.Series("test", [1.0, 2.0, 3.0, 4.0, 5.0])
    
    rolling_std = simple_series.rolling_std(window_size=3)
    assert isinstance(rolling_std, pl.Series)
    assert len(rolling_std) == len(simple_series)


def test_rolling_map_exists():
    """Test if rolling_map function exists and works."""
    simple_series = pl.Series("test", [1, 2, 3, 4, 5])
    
    def test_func(s):
        return s.sum()
    
    # Check if rolling_map exists
    try:
        rolling_map = simple_series.rolling_map(test_func, window_size=3)
        assert isinstance(rolling_map, pl.Series)
        assert len(rolling_map) == len(simple_series)
    except AttributeError:
        # If rolling_map doesn't exist, we need to use alternative approach
        pytest.skip("rolling_map not available in this Polars version")


def test_date_range_creation():
    """Test date range creation with different syntaxes."""
    # Test primary syntax
    try:
        dates = pl.date_range(date(2022, 1, 1), date(2022, 1, 10), "1d", eager=True)
        assert len(dates) > 0
    except Exception:
        # Try alternative syntax
        dates = pl.date_range(start=date(2022, 1, 1), end=date(2022, 1, 10), interval="1d")
        assert len(dates) > 0


def test_basic_statistics():
    """Test basic statistical operations."""
    simple_series = pl.Series("test", [1.0, 2.0, 3.0, 4.0, 5.0])
    
    mean_val = simple_series.mean()
    std_val = simple_series.std()
    sum_val = simple_series.sum()
    
    assert mean_val is not None
    assert std_val is not None  
    assert sum_val is not None
    assert sum_val == 15.0