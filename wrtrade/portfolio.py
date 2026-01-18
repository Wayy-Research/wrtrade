"""
Core Portfolio data type for wrtrade.

Design philosophy: Portfolio is a data type, like numpy.ndarray or torch.Tensor.
Simple construction, fast operations, intuitive API.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field


@dataclass
class Result:
    """
    Backtest result with metrics as attributes.

    Example:
        result = portfolio.backtest(prices)
        print(result.sortino)
        print(result.max_drawdown)
        result.tear_sheet()
    """
    returns: pl.Series
    total_return: float
    sortino: float
    sharpe: float
    max_drawdown: float
    volatility: float
    gain_to_pain: float

    # Optional detailed data
    positions: Optional[pl.Series] = None
    benchmark_returns: Optional[pl.Series] = None
    benchmark_total_return: Optional[float] = None
    excess_return: Optional[float] = None

    # Component-level attribution (for multi-signal portfolios)
    attribution: Optional[Dict[str, float]] = None

    def tear_sheet(self) -> None:
        """Print formatted performance report."""
        print("=" * 50)
        print("           PERFORMANCE REPORT")
        print("=" * 50)
        print()
        print(f"  Total Return:        {self.total_return:>10.2%}")
        print(f"  Volatility (Annual): {self.volatility:>10.2%}")
        print(f"  Sortino Ratio:       {self.sortino:>10.2f}")
        print(f"  Sharpe Ratio:        {self.sharpe:>10.2f}")
        print(f"  Max Drawdown:        {self.max_drawdown:>10.2%}")
        print(f"  Gain to Pain:        {self.gain_to_pain:>10.2f}")

        if self.excess_return is not None:
            print()
            print(f"  Benchmark Return:    {self.benchmark_total_return:>10.2%}")
            print(f"  Excess Return:       {self.excess_return:>10.2%}")

        if self.attribution:
            print()
            print("  Attribution:")
            for name, contrib in self.attribution.items():
                print(f"    {name}: {contrib:>10.4f}")

        print()
        print("=" * 50)

    def plot(self, interactive: bool = True):
        """
        Plot backtest results.

        Args:
            interactive: Use wrchart (True) or matplotlib (False)
        """
        if interactive:
            return self._plot_interactive()
        else:
            return self._plot_matplotlib()

    def _plot_interactive(self):
        """Plot using wrchart."""
        try:
            import wrchart as wrc
        except ImportError:
            print("wrchart not installed. Install with: pip install wrchart")
            print("Falling back to matplotlib...")
            return self._plot_matplotlib()

        cumulative = self.returns.cum_sum()
        n = len(self.returns)

        df = pl.DataFrame({
            'time': list(range(n)),
            'cumulative_returns': cumulative,
        })

        chart = wrc.Chart(width=800, height=400, title='Portfolio Returns')
        chart.add_area(df, time_col='time', value_col='cumulative_returns')

        return chart

    def _plot_matplotlib(self):
        """Plot using matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")

        cumulative = self.returns.cum_sum().to_numpy()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cumulative, label='Portfolio')
        ax.set_ylabel('Cumulative Returns')
        ax.set_xlabel('Time')
        ax.set_title('Portfolio Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return fig


# Type alias for signal functions
SignalFunc = Callable[[pl.Series], pl.Series]

# Type for signal specification: function, or (function, weight), or (name, function, weight)
SignalSpec = Union[
    SignalFunc,
    Tuple[SignalFunc, float],
    Tuple[str, SignalFunc, float],
]


class Portfolio:
    """
    Core portfolio data type.

    A Portfolio holds one or more trading signals with weights.
    It can backtest, validate, and optimize itself.

    Examples:
        # Single signal
        portfolio = Portfolio(my_signal)

        # Multiple signals with weights
        portfolio = Portfolio([
            (trend_signal, 0.6),
            (momentum_signal, 0.4),
        ])

        # Named signals
        portfolio = Portfolio({
            'trend': (trend_signal, 0.6),
            'momentum': (momentum_signal, 0.4),
        })

        # Backtest
        result = portfolio.backtest(prices)
        print(result.sortino)

        # Validate
        p_value = portfolio.validate(prices)

        # Optimize
        portfolio.optimize(prices)
    """

    def __init__(
        self,
        signals: Union[SignalFunc, List[SignalSpec], Dict[str, SignalSpec]],
        name: Optional[str] = None,
        max_position: float = 1.0,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ):
        """
        Create a portfolio from signals.

        Args:
            signals: One of:
                - A signal function: def signal(prices) -> pl.Series
                - A list: [(signal_func, weight), ...] or [(name, signal_func, weight), ...]
                - A dict: {'name': signal_func} or {'name': (signal_func, weight)}
            name: Portfolio name (auto-generated if not provided)
            max_position: Maximum position size (1.0 = 100%)
            take_profit: Take profit threshold (e.g., 0.05 for 5%)
            stop_loss: Stop loss threshold (e.g., 0.02 for 2%)
        """
        self.name = name or "Portfolio"
        self.max_position = max_position
        self.take_profit = take_profit
        self.stop_loss = stop_loss

        # Parse signals into normalized format: [(name, func, weight), ...]
        self._signals = self._parse_signals(signals)

        # Cache for computed values
        self._last_prices = None
        self._last_returns = None
        self._last_positions = None

    def _parse_signals(
        self,
        signals: Union[SignalFunc, List, Dict]
    ) -> List[Tuple[str, SignalFunc, float]]:
        """Parse various signal formats into normalized list."""

        # Single function
        if callable(signals):
            name = getattr(signals, '__name__', 'signal')
            return [(name, signals, 1.0)]

        # Dictionary format
        if isinstance(signals, dict):
            result = []
            for name, spec in signals.items():
                if callable(spec):
                    result.append((name, spec, 1.0))
                elif isinstance(spec, tuple) and len(spec) == 2:
                    func, weight = spec
                    result.append((name, func, weight))
                else:
                    raise ValueError(f"Invalid signal spec for '{name}': {spec}")
            return result

        # List format
        if isinstance(signals, list):
            result = []
            for i, spec in enumerate(signals):
                if callable(spec):
                    name = getattr(spec, '__name__', f'signal_{i}')
                    result.append((name, spec, 1.0))
                elif isinstance(spec, tuple):
                    if len(spec) == 2:
                        func, weight = spec
                        name = getattr(func, '__name__', f'signal_{i}')
                        result.append((name, func, weight))
                    elif len(spec) == 3:
                        name, func, weight = spec
                        result.append((name, func, weight))
                    else:
                        raise ValueError(f"Invalid signal tuple: {spec}")
                else:
                    raise ValueError(f"Invalid signal spec: {spec}")
            return result

        raise ValueError(f"signals must be a function, list, or dict, got {type(signals)}")

    @property
    def weights(self) -> Dict[str, float]:
        """Get current signal weights."""
        return {name: weight for name, _, weight in self._signals}

    @weights.setter
    def weights(self, new_weights: Dict[str, float]) -> None:
        """Set signal weights."""
        self._signals = [
            (name, func, new_weights.get(name, weight))
            for name, func, weight in self._signals
        ]
        # Clear cache
        self._last_returns = None
        self._last_positions = None

    def __repr__(self) -> str:
        signals_str = ", ".join(f"{name}={weight:.2f}" for name, _, weight in self._signals)
        return f"Portfolio({signals_str})"

    def _generate_signals(self, prices: pl.Series) -> pl.Series:
        """Generate combined signal from all components."""
        if len(self._signals) == 1:
            _, func, _ = self._signals[0]
            return func(prices)

        # Weighted combination of signals
        total_signal = pl.Series(values=[0.0] * len(prices))
        total_weight = 0.0

        for name, func, weight in self._signals:
            signal = func(prices)
            total_signal = total_signal + (signal.cast(pl.Float64) * weight)
            total_weight += abs(weight)

        if total_weight > 0:
            total_signal = total_signal / total_weight

        return total_signal.clip(-1, 1)

    def _calculate_positions(self, signals: pl.Series) -> pl.Series:
        """Convert signals to positions."""
        positions = signals.clip(-self.max_position, self.max_position)
        return positions

    def _calculate_returns(self, prices: pl.Series, positions: pl.Series) -> pl.Series:
        """Calculate strategy returns from positions."""
        price_returns = prices.pct_change().fill_null(0)
        strategy_returns = positions.shift(1).fill_null(0) * price_returns
        return strategy_returns

    def _apply_stops(
        self,
        prices: pl.Series,
        positions: pl.Series,
        signals: pl.Series
    ) -> pl.Series:
        """Apply take profit and stop loss."""
        if self.take_profit is None and self.stop_loss is None:
            return positions

        positions_np = positions.to_numpy().copy()
        prices_np = prices.to_numpy()
        signals_np = signals.to_numpy()

        entry_price = None
        in_position = False

        for i in range(len(positions_np)):
            if not in_position and positions_np[i] != 0:
                # Entering position
                entry_price = prices_np[i]
                in_position = True
            elif in_position:
                if entry_price is not None:
                    pnl = (prices_np[i] - entry_price) / entry_price * np.sign(positions_np[i-1])

                    # Check take profit
                    if self.take_profit and pnl >= self.take_profit:
                        positions_np[i] = 0
                        in_position = False
                        entry_price = None
                    # Check stop loss
                    elif self.stop_loss and pnl <= -self.stop_loss:
                        positions_np[i] = 0
                        in_position = False
                        entry_price = None
                    # Check if signal changed
                    elif positions_np[i] == 0:
                        in_position = False
                        entry_price = None

        return pl.Series(positions_np)

    def backtest(
        self,
        prices: pl.Series,
        benchmark: bool = True,
        risk_free_rate: float = 0.0,
    ) -> Result:
        """
        Run backtest on price data.

        Args:
            prices: Price series
            benchmark: Include buy-and-hold benchmark comparison
            risk_free_rate: Annual risk-free rate for Sharpe calculation

        Returns:
            Result object with all metrics

        Example:
            result = portfolio.backtest(prices)
            print(f"Sortino: {result.sortino:.2f}")
            result.tear_sheet()
        """
        # Generate signals and positions
        signals = self._generate_signals(prices)
        positions = self._calculate_positions(signals)

        # Apply stops if configured
        if self.take_profit is not None or self.stop_loss is not None:
            positions = self._apply_stops(prices, positions, signals)

        # Calculate returns
        returns = self._calculate_returns(prices, positions)

        # Cache for potential reuse
        self._last_prices = prices
        self._last_returns = returns
        self._last_positions = positions

        # Calculate metrics
        metrics = self._calculate_metrics(returns, risk_free_rate)

        # Benchmark
        benchmark_returns = None
        benchmark_total = None
        excess = None

        if benchmark:
            benchmark_returns = prices.pct_change().fill_null(0)
            benchmark_total = float(benchmark_returns.sum())
            excess = metrics['total_return'] - benchmark_total

        # Attribution for multi-signal portfolios
        attribution = None
        if len(self._signals) > 1:
            attribution = self._calculate_attribution(prices)

        return Result(
            returns=returns,
            total_return=metrics['total_return'],
            sortino=metrics['sortino'],
            sharpe=metrics['sharpe'],
            max_drawdown=metrics['max_drawdown'],
            volatility=metrics['volatility'],
            gain_to_pain=metrics['gain_to_pain'],
            positions=positions,
            benchmark_returns=benchmark_returns,
            benchmark_total_return=benchmark_total,
            excess_return=excess,
            attribution=attribution,
        )

    def _calculate_metrics(
        self,
        returns: pl.Series,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """Calculate all performance metrics."""
        returns_np = returns.to_numpy()

        # Total return
        total_return = float(returns.sum())

        # Volatility (annualized)
        std = returns.std()
        volatility = float(std * np.sqrt(252)) if std else 0.0

        # Sortino ratio
        daily_rf = risk_free_rate / 252
        excess_returns = returns_np - daily_rf
        downside = excess_returns[excess_returns < 0]

        if len(downside) > 0 and np.std(downside) > 0:
            downside_std = np.std(downside) * np.sqrt(252)
            sortino = (np.mean(excess_returns) * 252) / downside_std
        else:
            sortino = float('inf') if np.mean(excess_returns) > 0 else 0.0

        # Sharpe ratio
        if volatility > 0:
            annual_return = np.mean(returns_np) * 252
            sharpe = (annual_return - risk_free_rate) / volatility
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = returns.cum_sum()
        running_max = cumulative.cum_max()
        drawdown = cumulative - running_max
        max_drawdown = float(drawdown.min())

        # Gain to pain ratio
        gains = returns.filter(returns > 0).sum()
        losses = returns.filter(returns < 0).sum()
        gains = gains if gains is not None else 0.0
        losses = losses if losses is not None else 0.0

        if losses != 0:
            gain_to_pain = float(gains / abs(losses))
        else:
            gain_to_pain = float('inf') if gains > 0 else 0.0

        return {
            'total_return': total_return,
            'volatility': volatility,
            'sortino': float(sortino),
            'sharpe': float(sharpe),
            'max_drawdown': max_drawdown,
            'gain_to_pain': gain_to_pain,
        }

    def _calculate_attribution(self, prices: pl.Series) -> Dict[str, float]:
        """Calculate return attribution for each signal."""
        attribution = {}

        for name, func, weight in self._signals:
            signal = func(prices)
            positions = self._calculate_positions(signal)
            returns = self._calculate_returns(prices, positions)
            attribution[name] = float(returns.sum() * weight)

        return attribution

    def validate(
        self,
        prices: pl.Series,
        n_permutations: int = 1000,
        metric: str = 'sortino',
        parallel: bool = True,
        train_window: Optional[int] = None,
    ) -> float:
        """
        Statistical validation using permutation testing.

        Tests whether the strategy's performance is statistically significant
        or could be due to random chance.

        Args:
            prices: Price series
            n_permutations: Number of permutations (more = more accurate)
            metric: Metric to test ('sortino', 'sharpe', 'total_return')
            parallel: Use parallel processing
            train_window: If set, use walk-forward test with this training window

        Returns:
            p-value (lower is better, <0.05 is typically significant)

        Example:
            p_value = portfolio.validate(prices)
            if p_value < 0.05:
                print("Strategy is statistically significant!")
        """
        from wrtrade.permutation import PermutationTester, PermutationConfig

        config = PermutationConfig(
            n_permutations=n_permutations,
            parallel=parallel,
        )
        tester = PermutationTester(config)

        # Map simplified metric names to internal names
        metric_map = {
            'sortino': 'sortino_ratio',
            'sharpe': 'sortino_ratio',  # Use sortino as proxy
            'total_return': 'gain_to_pain_ratio',
            'gain_to_pain': 'gain_to_pain_ratio',
        }
        internal_metric = metric_map.get(metric, metric)

        # Strategy function that returns this portfolio
        def strategy_func(p):
            return self

        if train_window:
            results = tester.run_walkforward_test(
                prices, strategy_func, train_window, internal_metric
            )
        else:
            results = tester.run_insample_test(
                prices, strategy_func, internal_metric
            )

        return results['p_value']

    def optimize(
        self,
        prices: pl.Series,
        method: str = 'kelly',
        lookback_window: int = 252,
        max_leverage: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, float]:
        """
        Optimize signal weights.

        Args:
            prices: Price series for optimization
            method: Optimization method ('kelly' or 'equal')
            lookback_window: Historical window for statistics
            max_leverage: Maximum total leverage
            min_weight: Minimum weight per signal
            max_weight: Maximum weight per signal
            risk_free_rate: Annual risk-free rate

        Returns:
            Dict of optimized weights

        Example:
            weights = portfolio.optimize(prices)
            print(f"Optimized weights: {weights}")
        """
        if len(self._signals) == 1:
            return {self._signals[0][0]: 1.0}

        if method == 'equal':
            n = len(self._signals)
            new_weights = {name: 1.0 / n for name, _, _ in self._signals}
            self.weights = new_weights
            return new_weights

        elif method == 'kelly':
            from wrtrade.kelly import KellyOptimizer, KellyConfig

            config = KellyConfig(
                lookback_window=lookback_window,
                max_leverage=max_leverage,
                min_weight=min_weight,
                max_weight=max_weight,
                risk_free_rate=risk_free_rate,
            )
            optimizer = KellyOptimizer(config)

            # Calculate returns for each signal
            returns_list = []
            names = []

            for name, func, _ in self._signals:
                signal = func(prices)
                positions = self._calculate_positions(signal)
                returns = self._calculate_returns(prices, positions)
                returns_list.append(returns.to_numpy())
                names.append(name)

            returns_matrix = np.column_stack(returns_list)

            # Get Kelly weights
            kelly_weights = optimizer.calculate_portfolio_kelly(
                returns_matrix, names, risk_free_rate
            )

            # Update portfolio weights
            self.weights = kelly_weights
            return kelly_weights

        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def add_signal(
        self,
        signal: SignalFunc,
        weight: float = 1.0,
        name: Optional[str] = None
    ) -> 'Portfolio':
        """
        Add a signal to the portfolio.

        Args:
            signal: Signal function
            weight: Signal weight
            name: Signal name (optional)

        Returns:
            Self for method chaining
        """
        name = name or getattr(signal, '__name__', f'signal_{len(self._signals)}')
        self._signals.append((name, signal, weight))
        self._last_returns = None  # Clear cache
        return self

    def remove_signal(self, name: str) -> 'Portfolio':
        """
        Remove a signal by name.

        Args:
            name: Signal name to remove

        Returns:
            Self for method chaining
        """
        self._signals = [(n, f, w) for n, f, w in self._signals if n != name]
        self._last_returns = None  # Clear cache
        return self


# =============================================================================
# Convenience functions for one-line usage
# =============================================================================

def backtest(
    signal: Union[SignalFunc, Portfolio],
    prices: pl.Series,
    **kwargs
) -> Result:
    """
    One-line backtest.

    Args:
        signal: Signal function or Portfolio
        prices: Price series
        **kwargs: Passed to Portfolio.backtest()

    Returns:
        Result object

    Example:
        result = wrt.backtest(my_signal, prices)
        print(result.sortino)
    """
    if isinstance(signal, Portfolio):
        return signal.backtest(prices, **kwargs)
    else:
        portfolio = Portfolio(signal)
        return portfolio.backtest(prices, **kwargs)


def validate(
    signal: Union[SignalFunc, Portfolio],
    prices: pl.Series,
    n_permutations: int = 1000,
    **kwargs
) -> float:
    """
    One-line statistical validation.

    Args:
        signal: Signal function or Portfolio
        prices: Price series
        n_permutations: Number of permutations
        **kwargs: Passed to Portfolio.validate()

    Returns:
        p-value (lower is better, <0.05 is typically significant)

    Example:
        p_value = wrt.validate(my_signal, prices)
        if p_value < 0.05:
            print("Strategy is significant!")
    """
    if isinstance(signal, Portfolio):
        return signal.validate(prices, n_permutations=n_permutations, **kwargs)
    else:
        portfolio = Portfolio(signal)
        return portfolio.validate(prices, n_permutations=n_permutations, **kwargs)


def optimize(
    portfolio: Portfolio,
    prices: pl.Series,
    **kwargs
) -> Dict[str, float]:
    """
    Optimize portfolio weights.

    Args:
        portfolio: Portfolio to optimize
        prices: Price series
        **kwargs: Passed to Portfolio.optimize()

    Returns:
        Dict of optimized weights

    Example:
        weights = wrt.optimize(portfolio, prices)
    """
    return portfolio.optimize(prices, **kwargs)
