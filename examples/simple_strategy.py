"""
Simple Moving Average Strategy Example

Demonstrates the wrtrade API:
    1. One-line backtest
    2. Portfolio construction
    3. Statistical validation
    4. Kelly optimization

Deploy with:
    wrtrade strategy deploy examples/simple_strategy.py \
        --name simple_ma \
        --broker alpaca \
        --symbols AAPL,TSLA
"""

import wrtrade as wrt
import polars as pl
import numpy as np


def ma_crossover(prices: pl.Series) -> pl.Series:
    """
    Simple moving average crossover signal.
    Buy when fast MA > slow MA, sell when fast MA < slow MA.
    """
    fast_ma = prices.rolling_mean(10)
    slow_ma = prices.rolling_mean(30)
    return (fast_ma > slow_ma).cast(int) - (fast_ma < slow_ma).cast(int)


def momentum(prices: pl.Series) -> pl.Series:
    """
    Momentum signal.
    Buy when 20-day return > 5%, sell when < -5%.
    """
    returns = prices.pct_change(20).fill_null(0)
    return (returns > 0.05).cast(int) - (returns < -0.05).cast(int)


def build_portfolio():
    """
    Build the trading portfolio.
    This function is called by the deployment system.
    """
    return wrt.Portfolio(ma_crossover, name="Simple_MA_Strategy")


if __name__ == "__main__":
    # Generate sample prices
    np.random.seed(42)
    prices = pl.Series([100 * (1 + r) for r in np.cumsum(np.random.normal(0.001, 0.02, 252))])

    print("=" * 60)
    print("  wrtrade API Demo")
    print("=" * 60)

    # =========================================================================
    # 1. ONE-LINE BACKTEST
    # =========================================================================
    print("\n1. One-line backtest:")
    print("-" * 40)

    result = wrt.backtest(ma_crossover, prices)
    print(f"   Sortino:      {result.sortino:.2f}")
    print(f"   Total Return: {result.total_return:.2%}")
    print(f"   Max Drawdown: {result.max_drawdown:.2%}")

    # =========================================================================
    # 2. PORTFOLIO WITH SINGLE SIGNAL
    # =========================================================================
    print("\n2. Portfolio with single signal:")
    print("-" * 40)

    portfolio = wrt.Portfolio(ma_crossover)
    result = portfolio.backtest(prices)
    print(f"   {portfolio}")
    print(f"   Sortino: {result.sortino:.2f}")

    # =========================================================================
    # 3. MULTI-SIGNAL PORTFOLIO
    # =========================================================================
    print("\n3. Multi-signal portfolio:")
    print("-" * 40)

    portfolio = wrt.Portfolio([
        (ma_crossover, 0.6),
        (momentum, 0.4),
    ])
    result = portfolio.backtest(prices)
    print(f"   {portfolio}")
    print(f"   Sortino: {result.sortino:.2f}")

    if result.attribution:
        print("   Attribution:")
        for name, contrib in result.attribution.items():
            print(f"     {name}: {contrib:.4f}")

    # =========================================================================
    # 4. NAMED SIGNALS (dict format)
    # =========================================================================
    print("\n4. Named signals (dict format):")
    print("-" * 40)

    portfolio = wrt.Portfolio({
        'trend': (ma_crossover, 0.6),
        'momentum': (momentum, 0.4),
    })
    print(f"   {portfolio}")
    print(f"   Weights: {portfolio.weights}")

    # =========================================================================
    # 5. TEAR SHEET
    # =========================================================================
    print("\n5. Performance tear sheet:")
    print("-" * 40)
    result.tear_sheet()

    # =========================================================================
    # 6. STATISTICAL VALIDATION
    # =========================================================================
    print("\n6. Statistical validation (permutation test):")
    print("-" * 40)

    p_value = wrt.validate(ma_crossover, prices, n_permutations=100, parallel=False)
    print(f"   P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("   Result: Strategy is statistically significant!")
    else:
        print("   Result: Strategy may be due to chance")

    # =========================================================================
    # 7. KELLY OPTIMIZATION
    # =========================================================================
    print("\n7. Kelly optimization:")
    print("-" * 40)

    portfolio = wrt.Portfolio([
        (ma_crossover, 0.5),
        (momentum, 0.5),
    ])
    print(f"   Before: {portfolio.weights}")

    weights = portfolio.optimize(prices, method='kelly')
    print(f"   After:  {portfolio.weights}")

    # =========================================================================
    # 8. DEPLOYMENT INFO
    # =========================================================================
    print("\n8. Deployment:")
    print("-" * 40)
    print("   To deploy this strategy:")
    print()
    print("   wrtrade strategy deploy examples/simple_strategy.py \\")
    print("       --name simple_ma \\")
    print("       --broker alpaca \\")
    print("       --symbols AAPL,TSLA")
    print()
    print("=" * 60)
