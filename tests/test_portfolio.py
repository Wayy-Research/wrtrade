import pytest
import pandas as pd
from datetime import datetime
from wrtrade.portfolio import Portfolio

@pytest.fixture
def sample_portfolio():
    return Portfolio(initial_capital=10000)

@pytest.fixture
def sample_market_data():
    return pd.Series([150.0, 160.0, 170.0])

@pytest.fixture
def sample_signals():
    return pd.Series([1, -1, 1])

def test_portfolio_initialization(sample_portfolio):
    assert sample_portfolio.initial_capital == 10000
    assert sample_portfolio.current_capital == 10000
    assert sample_portfolio.trades == []
    assert sample_portfolio.position == []

def test_market_returns(sample_portfolio, sample_market_data, sample_signals):
    sample_portfolio.generate_trades(sample_market_data, sample_signals)
    sample_portfolio.calculate_returns(sample_market_data)
    assert sample_portfolio.market_returns[0] == 0
    assert round(sample_portfolio.market_returns[1], 2) == 0.06
    assert round(sample_portfolio.market_returns[2], 2) == 0.06


def test_trade_creation(sample_portfolio, sample_market_data, sample_signals):
    sample_portfolio.generate_trades(sample_market_data, sample_signals)
    sample_portfolio.calculate_returns(sample_market_data)
    assert sample_portfolio.trades[0] == {'dt': 0, 'price': 150.0, 'direction': 1}
    assert sample_portfolio.trades[1] == {'dt': 1, 'price': 160.0, 'direction': -1}
    assert sample_portfolio.trades[2] == {'dt': 2, 'price': 170.0, 'direction': 1}


def test_generate_position(sample_portfolio, sample_market_data, sample_signals):
    sample_portfolio.generate_trades(sample_market_data, sample_signals)
    sample_portfolio.calculate_returns(sample_market_data)
    assert sample_portfolio.position[0] == 1
    assert sample_portfolio.position[1] == 0
    assert sample_portfolio.position[2] == 1


def test_cumulative_creation(sample_portfolio, sample_market_data, sample_signals):
    sample_portfolio.generate_trades(sample_market_data, sample_signals)
    sample_portfolio.calculate_returns(sample_market_data)
    assert sample_portfolio.cumulative[0] == 0
    assert sample_portfolio.cumulative[1] == 0
    assert round(sample_portfolio.cumulative[2], 2) == 0.06


def test_notional_creation(sample_portfolio, sample_market_data, sample_signals):
    sample_portfolio.generate_trades(sample_market_data, sample_signals)
    sample_portfolio.calculate_returns(sample_market_data)
    assert sample_portfolio.notional[0] == 10000
    assert sample_portfolio.notional[1] == 10000
    assert round(sample_portfolio.notional[2], 2) == 10606.25