"""
Multi-Strategy Portfolio Example

Combines trend following and momentum strategies.
Deploy with:
    wrtrade strategy deploy examples/multi_strategy.py \
        --name multi_strat \
        --broker alpaca \
        --symbols AAPL,TSLA,MSFT
"""

import wrtrade as wrt
import polars as pl


def trend_signal(prices: pl.Series) -> pl.Series:
    """Trend following using moving averages."""
    fast = prices.rolling_mean(10)
    slow = prices.rolling_mean(30)
    return (fast > slow).cast(int) - (fast < slow).cast(int)


def momentum_signal(prices: pl.Series) -> pl.Series:
    """Momentum based on 20-day returns."""
    returns = prices.pct_change(20)
    return (returns > 0.05).cast(int) - (returns < -0.05).cast(int)


def build_portfolio():
    """
    Build a multi-strategy portfolio.
    Combines trend following (60%) and momentum (40%).
    """
    builder = wrt.NDimensionalPortfolioBuilder()

    # Create components
    trend = builder.create_signal_component("Trend", trend_signal, weight=0.6)
    momentum = builder.create_signal_component("Momentum", momentum_signal, weight=0.4)

    # Build portfolio
    portfolio = builder.create_portfolio(
        "Multi_Strategy",
        [trend, momentum]
    )

    return portfolio


if __name__ == "__main__":
    import numpy as np

    # Generate sample prices
    np.random.seed(42)
    prices = pl.Series([100 * (1 + r) for r in np.cumsum(np.random.normal(0.001, 0.02, 252))])

    # Build and backtest
    portfolio = build_portfolio()
    manager = wrt.AdvancedPortfolioManager(portfolio)
    results = manager.backtest(prices)

    print("=== Multi-Strategy Backtest ===")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sortino Ratio: {results['portfolio_metrics']['sortino_ratio']:.2f}")

    # Show portfolio structure
    print("\n=== Portfolio Structure ===")
    manager.print_structure()

    # Optimize with Kelly
    print("\n=== Kelly Optimization ===")
    optimizer = wrt.HierarchicalKellyOptimizer()
    kelly_results = optimizer.optimize_portfolio(portfolio, prices, rebalance=True)

    for level in kelly_results['level_optimizations']:
        print(f"{level['portfolio_name']}: {level['kelly_weights']}")

    print("\nDeploy with:")
    print("  wrtrade strategy deploy examples/multi_strategy.py \\")
    print("      --name multi_strat \\")
    print("      --broker alpaca \\")
    print("      --symbols AAPL,TSLA,MSFT")
