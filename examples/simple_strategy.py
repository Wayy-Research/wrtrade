"""
Simple Moving Average Strategy Example

This strategy uses a simple moving average crossover approach.
Deploy with:
    wrtrade strategy deploy examples/simple_strategy.py \
        --name simple_ma \
        --broker alpaca \
        --symbols AAPL,TSLA
"""

import wrtrade as wrt
import polars as pl


def ma_crossover_signal(prices: pl.Series) -> pl.Series:
    """
    Simple moving average crossover signal.
    Buy when fast MA > slow MA, sell when fast MA < slow MA.
    """
    fast_ma = prices.rolling_mean(10)
    slow_ma = prices.rolling_mean(30)

    # Generate signals: 1 for buy, -1 for sell, 0 for hold
    signals = (fast_ma > slow_ma).cast(int) - (fast_ma < slow_ma).cast(int)
    return signals


def build_portfolio():
    """
    Build the trading portfolio.
    This function is called by the deployment system.
    """
    builder = wrt.NDimensionalPortfolioBuilder()

    # Create signal component
    ma_component = builder.create_signal_component(
        "MA_Crossover",
        ma_crossover_signal,
        weight=1.0
    )

    # Build portfolio
    portfolio = builder.create_portfolio(
        "Simple_MA_Strategy",
        [ma_component]
    )

    return portfolio


if __name__ == "__main__":
    # Backtest example
    import numpy as np

    # Generate sample prices
    np.random.seed(42)
    prices = pl.Series([100 * (1 + r) for r in np.cumsum(np.random.normal(0.001, 0.02, 252))])

    # Build portfolio
    portfolio = build_portfolio()

    # Backtest
    manager = wrt.AdvancedPortfolioManager(portfolio)
    results = manager.backtest(prices)

    print("=== Backtest Results ===")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sortino Ratio: {results['portfolio_metrics']['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {results['portfolio_metrics']['max_drawdown']:.2%}")

    # Validate with permutation testing
    print("\n=== Statistical Validation ===")
    tester = wrt.PermutationTester(wrt.PermutationConfig(n_permutations=100, parallel=False))
    perm_results = tester.run_insample_test(prices, lambda p: portfolio, 'sortino_ratio')

    print(f"P-value: {perm_results['p_value']:.4f}")
    if perm_results['p_value'] < 0.05:
        print("✓ Strategy is statistically significant")
        print("\nReady to deploy with CLI:")
        print("  wrtrade strategy deploy examples/simple_strategy.py \\")
        print("      --name simple_ma \\")
        print("      --broker alpaca \\")
        print("      --symbols AAPL,TSLA")
    else:
        print("✗ Strategy not statistically significant - needs improvement")
