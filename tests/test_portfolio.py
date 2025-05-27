import pytest
import pandas as pd
import numpy as np
from wrtrade.portfolio import Portfolio

@pytest.fixture
def sample_market_data():
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    return pd.Series(np.random.normal(100, 10, 100), index=dates)

@pytest.fixture
def sample_signals():
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    return pd.Series(np.random.choice([-1, 0, 1], 100), index=dates)

@pytest.fixture
def sample_portfolio(sample_market_data, sample_signals):
    return Portfolio(sample_market_data, sample_signals)

def test_portfolio_initialization(sample_portfolio):
    assert isinstance(sample_portfolio, Portfolio)
    assert isinstance(sample_portfolio.market_data, pd.Series)
    assert isinstance(sample_portfolio.signals, pd.Series)
    assert isinstance(sample_portfolio.params, dict)
    assert 'stop' in sample_portfolio.params
    assert 'tp' in sample_portfolio.params

def test_calculate_market_returns(sample_portfolio):
    sample_portfolio.calculate_market_returns()
    assert isinstance(sample_portfolio.market_returns, pd.Series)
    assert len(sample_portfolio.market_returns) == len(sample_portfolio.market_data)
    assert np.isclose(sample_portfolio.market_returns.iloc[0], 0)

def test_calculate_portfolio_returns(sample_portfolio):
    sample_portfolio.calculate_portfolio_returns()
    assert isinstance(sample_portfolio.cumulative_returns, pd.Series)
    assert len(sample_portfolio.cumulative_returns) == len(sample_portfolio.market_data)
    assert isinstance(sample_portfolio.position, pd.Series)
    assert len(sample_portfolio.position) == len(sample_portfolio.market_data)

def test_get_results(sample_portfolio):
    sample_portfolio.calculate_portfolio_returns()
    results = sample_portfolio.get_results()
    assert isinstance(results, dict)
    assert 'cumulative_return' in results
    assert isinstance(results['cumulative_return'], float)
    assert 'trades' in results
    assert isinstance(results['trades'], pd.DataFrame)
