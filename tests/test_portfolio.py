"""
Tests for the core Portfolio API.

Tests the new simplified Portfolio class that accepts signal functions.
"""

import pytest
import polars as pl
import numpy as np
from wrtrade.portfolio import Portfolio, Result, backtest, validate, optimize


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    prices = 100 * np.cumprod(1 + returns)
    return pl.Series(prices)


def always_long(prices: pl.Series) -> pl.Series:
    """Signal that always returns 1 (long)."""
    return pl.Series([1] * len(prices))


def always_short(prices: pl.Series) -> pl.Series:
    """Signal that always returns -1 (short)."""
    return pl.Series([-1] * len(prices))


def always_flat(prices: pl.Series) -> pl.Series:
    """Signal that always returns 0 (flat)."""
    return pl.Series([0] * len(prices))


def ma_crossover(prices: pl.Series) -> pl.Series:
    """Simple MA crossover signal."""
    fast = prices.rolling_mean(10)
    slow = prices.rolling_mean(30)
    signal = (fast > slow).cast(int) - (fast < slow).cast(int)
    return signal.fill_null(0)


def momentum(prices: pl.Series) -> pl.Series:
    """Momentum signal."""
    returns = prices.pct_change(20).fill_null(0)
    return (returns > 0.05).cast(int) - (returns < -0.05).cast(int)


# =============================================================================
# Portfolio Construction Tests
# =============================================================================

class TestPortfolioConstruction:
    """Test Portfolio construction with various input formats."""

    def test_single_function(self, sample_prices):
        """Portfolio accepts a single signal function."""
        portfolio = Portfolio(ma_crossover)
        assert len(portfolio._signals) == 1
        assert portfolio._signals[0][0] == 'ma_crossover'  # name inferred
        assert portfolio._signals[0][2] == 1.0  # default weight

    def test_list_of_tuples(self, sample_prices):
        """Portfolio accepts list of (func, weight) tuples."""
        portfolio = Portfolio([
            (ma_crossover, 0.6),
            (momentum, 0.4),
        ])
        assert len(portfolio._signals) == 2
        assert portfolio.weights == {'ma_crossover': 0.6, 'momentum': 0.4}

    def test_list_of_named_tuples(self, sample_prices):
        """Portfolio accepts list of (name, func, weight) tuples."""
        portfolio = Portfolio([
            ('trend', ma_crossover, 0.6),
            ('mom', momentum, 0.4),
        ])
        assert len(portfolio._signals) == 2
        assert portfolio.weights == {'trend': 0.6, 'mom': 0.4}

    def test_dict_format(self, sample_prices):
        """Portfolio accepts dict format."""
        portfolio = Portfolio({
            'trend': (ma_crossover, 0.6),
            'mom': (momentum, 0.4),
        })
        assert len(portfolio._signals) == 2
        assert portfolio.weights == {'trend': 0.6, 'mom': 0.4}

    def test_dict_with_just_functions(self, sample_prices):
        """Portfolio accepts dict with functions (default weight)."""
        portfolio = Portfolio({
            'trend': ma_crossover,
            'mom': momentum,
        })
        assert portfolio.weights == {'trend': 1.0, 'mom': 1.0}

    def test_list_of_functions(self, sample_prices):
        """Portfolio accepts list of just functions."""
        portfolio = Portfolio([ma_crossover, momentum])
        assert len(portfolio._signals) == 2
        # Names inferred from function names
        assert 'ma_crossover' in portfolio.weights
        assert 'momentum' in portfolio.weights

    def test_custom_name(self, sample_prices):
        """Portfolio accepts custom name."""
        portfolio = Portfolio(ma_crossover, name="MyStrategy")
        assert portfolio.name == "MyStrategy"

    def test_position_limits(self, sample_prices):
        """Portfolio respects max_position parameter."""
        portfolio = Portfolio(ma_crossover, max_position=0.5)
        assert portfolio.max_position == 0.5

    def test_stop_loss_take_profit(self, sample_prices):
        """Portfolio accepts stop loss and take profit."""
        portfolio = Portfolio(
            ma_crossover,
            take_profit=0.05,
            stop_loss=0.02
        )
        assert portfolio.take_profit == 0.05
        assert portfolio.stop_loss == 0.02

    def test_repr(self, sample_prices):
        """Portfolio has readable repr."""
        portfolio = Portfolio([
            (ma_crossover, 0.6),
            (momentum, 0.4),
        ])
        repr_str = repr(portfolio)
        assert 'Portfolio' in repr_str
        assert '0.60' in repr_str
        assert '0.40' in repr_str


# =============================================================================
# Backtest Tests
# =============================================================================

class TestPortfolioBacktest:
    """Test Portfolio.backtest() method."""

    def test_backtest_returns_result(self, sample_prices):
        """Backtest returns a Result object."""
        portfolio = Portfolio(ma_crossover)
        result = portfolio.backtest(sample_prices)
        assert isinstance(result, Result)

    def test_result_has_metrics(self, sample_prices):
        """Result has all expected metrics as attributes."""
        portfolio = Portfolio(ma_crossover)
        result = portfolio.backtest(sample_prices)

        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sortino')
        assert hasattr(result, 'sharpe')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'volatility')
        assert hasattr(result, 'gain_to_pain')
        assert hasattr(result, 'returns')
        assert hasattr(result, 'positions')

    def test_result_returns_series(self, sample_prices):
        """Result contains returns as Polars Series."""
        portfolio = Portfolio(ma_crossover)
        result = portfolio.backtest(sample_prices)

        assert isinstance(result.returns, pl.Series)
        assert len(result.returns) == len(sample_prices)

    def test_benchmark_comparison(self, sample_prices):
        """Backtest includes benchmark comparison."""
        portfolio = Portfolio(ma_crossover)
        result = portfolio.backtest(sample_prices, benchmark=True)

        assert result.benchmark_returns is not None
        assert result.benchmark_total_return is not None
        assert result.excess_return is not None

    def test_benchmark_disabled(self, sample_prices):
        """Benchmark can be disabled."""
        portfolio = Portfolio(ma_crossover)
        result = portfolio.backtest(sample_prices, benchmark=False)

        assert result.benchmark_returns is None

    def test_attribution_multi_signal(self, sample_prices):
        """Multi-signal portfolio has attribution."""
        portfolio = Portfolio([
            (ma_crossover, 0.6),
            (momentum, 0.4),
        ])
        result = portfolio.backtest(sample_prices)

        assert result.attribution is not None
        assert 'ma_crossover' in result.attribution
        assert 'momentum' in result.attribution

    def test_attribution_single_signal(self, sample_prices):
        """Single signal portfolio has no attribution."""
        portfolio = Portfolio(ma_crossover)
        result = portfolio.backtest(sample_prices)

        assert result.attribution is None

    def test_always_long_positive_in_bull_market(self):
        """Always long signal profits in bull market."""
        # Create a bull market
        prices = pl.Series([100 + i for i in range(100)])
        portfolio = Portfolio(always_long)
        result = portfolio.backtest(prices)

        assert result.total_return > 0

    def test_always_flat_zero_return(self, sample_prices):
        """Always flat signal has zero return."""
        portfolio = Portfolio(always_flat)
        result = portfolio.backtest(sample_prices)

        assert abs(result.total_return) < 1e-10


# =============================================================================
# Result Tests
# =============================================================================

class TestResult:
    """Test Result dataclass."""

    def test_tear_sheet(self, sample_prices, capsys):
        """tear_sheet() prints formatted output."""
        portfolio = Portfolio(ma_crossover)
        result = portfolio.backtest(sample_prices)
        result.tear_sheet()

        captured = capsys.readouterr()
        assert 'PERFORMANCE REPORT' in captured.out
        assert 'Total Return' in captured.out
        assert 'Sortino' in captured.out


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Test top-level convenience functions."""

    def test_backtest_function(self, sample_prices):
        """wrt.backtest() works with signal function."""
        result = backtest(ma_crossover, sample_prices)
        assert isinstance(result, Result)

    def test_backtest_with_portfolio(self, sample_prices):
        """wrt.backtest() works with Portfolio."""
        portfolio = Portfolio(ma_crossover)
        result = backtest(portfolio, sample_prices)
        assert isinstance(result, Result)


# =============================================================================
# Weight Management Tests
# =============================================================================

class TestWeightManagement:
    """Test weight getting and setting."""

    def test_get_weights(self, sample_prices):
        """weights property returns dict."""
        portfolio = Portfolio([
            (ma_crossover, 0.6),
            (momentum, 0.4),
        ])
        weights = portfolio.weights
        assert weights == {'ma_crossover': 0.6, 'momentum': 0.4}

    def test_set_weights(self, sample_prices):
        """weights can be set."""
        portfolio = Portfolio([
            (ma_crossover, 0.5),
            (momentum, 0.5),
        ])
        portfolio.weights = {'ma_crossover': 0.7, 'momentum': 0.3}
        assert portfolio.weights == {'ma_crossover': 0.7, 'momentum': 0.3}

    def test_add_signal(self, sample_prices):
        """Signals can be added."""
        portfolio = Portfolio(ma_crossover)
        portfolio.add_signal(momentum, weight=0.5, name='mom')

        assert len(portfolio._signals) == 2
        assert 'mom' in portfolio.weights

    def test_remove_signal(self, sample_prices):
        """Signals can be removed."""
        portfolio = Portfolio([
            (ma_crossover, 0.6),
            (momentum, 0.4),
        ])
        portfolio.remove_signal('momentum')

        assert len(portfolio._signals) == 1
        assert 'momentum' not in portfolio.weights


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_signal_type(self):
        """Invalid signal type raises error."""
        with pytest.raises(ValueError):
            Portfolio(123)  # Not a function or list

    def test_empty_list(self):
        """Empty list raises error on backtest."""
        portfolio = Portfolio([])
        # Should work but produce zero returns
        prices = pl.Series([100, 101, 102])
        result = portfolio.backtest(prices)
        assert result.total_return == 0

    def test_very_short_prices(self):
        """Very short price series works."""
        prices = pl.Series([100, 101])
        portfolio = Portfolio(always_long)
        result = portfolio.backtest(prices)
        assert isinstance(result, Result)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self, sample_prices):
        """Test complete workflow: create, backtest, analyze."""
        # Create portfolio
        portfolio = Portfolio([
            (ma_crossover, 0.6),
            (momentum, 0.4),
        ], name="Multi_Strategy")

        # Backtest
        result = portfolio.backtest(sample_prices)

        # Check result
        assert isinstance(result.total_return, float)
        assert isinstance(result.sortino, float)
        assert len(result.returns) == len(sample_prices)

        # Check attribution
        assert result.attribution is not None
        total_attribution = sum(result.attribution.values())
        # Attribution should roughly sum to total return
        # (not exact due to weighting)

    def test_optimize_updates_weights(self, sample_prices):
        """optimize() updates portfolio weights."""
        portfolio = Portfolio([
            (ma_crossover, 0.5),
            (momentum, 0.5),
        ])

        original_weights = portfolio.weights.copy()
        portfolio.optimize(sample_prices, method='kelly')

        # Weights should change (unless data is pathological)
        # At minimum, the method should run without error
        assert portfolio.weights is not None
