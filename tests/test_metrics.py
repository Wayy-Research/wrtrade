import pytest
import pandas as pd
import numpy as np
from wrtrade.metrics import Metrics

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    market_returns = pd.Series(np.random.normal(0.0005, 0.015, 100), index=dates)
    return returns, market_returns

class TestMetrics:
    def test_initialization(self, sample_data):
        returns, market_returns = sample_data
        metrics = Metrics(returns, market_returns)
        assert isinstance(metrics, Metrics)
        assert metrics.returns.equals(returns)
        assert metrics.market_returns.equals(market_returns)

    def test_total_return(self, sample_data):
        returns, _ = sample_data
        metrics = Metrics(returns, _)
        total_return = metrics.total_return()
        assert isinstance(total_return, float)
        assert np.isclose(total_return, (1 + returns).prod() - 1, rtol=1e-5)

    def test_annualized_return(self, sample_data):
        returns, _ = sample_data
        metrics = Metrics(returns, _)
        annualized_return = metrics.annualized_return()
        assert isinstance(annualized_return, float)
        expected = (1 + metrics.total_return()) ** (252 / len(returns)) - 1
        assert np.isclose(annualized_return, expected, rtol=1e-5)

    def test_sharpe_ratio(self, sample_data):
        returns, _ = sample_data
        metrics = Metrics(returns, _)
        sharpe = metrics.sharpe_ratio()
        assert isinstance(sharpe, float)
        expected = np.sqrt(252) * returns.mean() / returns.std()
        assert np.isclose(sharpe, expected, rtol=1e-5)

    def test_sortino_ratio(self, sample_data):
        returns, _ = sample_data
        metrics = Metrics(returns, _)
        sortino = metrics.sortino_ratio()
        assert isinstance(sortino, float)
        downside_returns = returns[returns < 0]
        expected = np.sqrt(252) * returns.mean() / downside_returns.std()
        assert np.isclose(sortino, expected, rtol=1e-5)

    def test_information_criterion(self, sample_data):
        returns, market_returns = sample_data
        metrics = Metrics(returns, market_returns)
        ic = metrics.information_criterion()
        assert isinstance(ic, float)
        expected = (returns - market_returns).mean() / (returns - market_returns).std()
        assert np.isclose(ic, expected, rtol=1e-5)

    def test_max_drawdown(self, sample_data):
        returns, _ = sample_data
        metrics = Metrics(returns, _)
        max_dd = metrics.max_drawdown()
        assert isinstance(max_dd, float)
        assert max_dd <= 0
        cumulative_returns = (1 + returns).cumprod()
        expected = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        assert np.isclose(max_dd, expected, rtol=1e-5)

    def test_win_rate(self, sample_data):
        returns, _ = sample_data
        metrics = Metrics(returns, _)
        win_rate = metrics.win_rate()
        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1
        expected = (returns > 0).mean()
        assert np.isclose(win_rate, expected, rtol=1e-5)

    def test_profit_factor(self, sample_data):
        returns, _ = sample_data
        metrics = Metrics(returns, _)
        pf = metrics.profit_factor()
        assert isinstance(pf, float)
        assert pf >= 0
        expected = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
        assert np.isclose(pf, expected, rtol=1e-5)

    def test_alpha(self, sample_data):
        returns, market_returns = sample_data
        metrics = Metrics(returns, market_returns)
        alpha = metrics.alpha()
        assert isinstance(alpha, float)
        beta = metrics.beta()
        expected = returns.mean() - (0.05/252 + beta * (market_returns.mean() - 0.05/252))
        assert np.isclose(alpha, expected, rtol=1e-5)

    def test_beta(self, sample_data):
        returns, market_returns = sample_data
        metrics = Metrics(returns, market_returns)
        beta = metrics.beta()
        assert isinstance(beta, float)
        expected = np.cov(returns, market_returns)[0, 1] / np.var(market_returns)
        assert np.isclose(beta, expected, rtol=1e-5)

    def test_calculate_metrics(self, sample_data):
        returns, market_returns = sample_data
        metrics = Metrics(returns, market_returns)
        results = metrics.calculate_metrics()
        assert isinstance(results, dict)
        expected_keys = ['total_return', 'annualized_return', 'sharpe_ratio', 'sortino_ratio',
                         'information_criterion', 'max_drawdown', 'win_rate', 'profit_factor', 'alpha', 'beta']
        assert all(key in results for key in expected_keys)
        for key, value in results.items():
            assert isinstance(value, float)

    @pytest.mark.parametrize("window", [20, 50])
    def test_rolling_metrics(self, sample_data, window):
        returns, market_returns = sample_data
        metrics = Metrics(returns, market_returns)
        
        rolling_sharpe = metrics.rolling_sharpe_ratio(window=window)
        rolling_sortino = metrics.rolling_sortino_ratio(window=window)
        rolling_ic = metrics.rolling_information_ratio(window=window)
        
        for rolling_metric in [rolling_sharpe, rolling_sortino, rolling_ic]:
            assert isinstance(rolling_metric, pd.Series)
            assert len(rolling_metric) == len(returns)
            assert rolling_metric.iloc[:window-1].isna().all()
            assert not rolling_metric.iloc[window:].isna().any()
            assert all(isinstance(x, float) for x in rolling_metric.dropna())

    def test_edge_cases(self):
        # Test with empty data
        empty_returns = pd.Series()
        empty_market_returns = pd.Series()
        metrics = Metrics(empty_returns, empty_market_returns)
        
        with pytest.raises(ValueError):
            metrics.total_return()
        
        # Test with constant returns
        constant_returns = pd.Series([0.01] * 100)
        constant_market_returns = pd.Series([0.005] * 100)
        metrics = Metrics(constant_returns, constant_market_returns)
        
        assert metrics.sharpe_ratio() == float('inf')
        assert metrics.sortino_ratio() == float('inf')
        assert metrics.win_rate() == 1.0
        assert metrics.max_drawdown() == 0.0

    def test_input_validation(self):
        invalid_returns = [1, 2, 3]  # Not a pd.Series
        valid_market_returns = pd.Series([0.01, 0.02, 0.03])
        
        with pytest.raises(TypeError):
            Metrics(invalid_returns, valid_market_returns)
        
        mismatched_returns = pd.Series([0.01, 0.02])
        with pytest.raises(ValueError):
            Metrics(mismatched_returns, valid_market_returns)
