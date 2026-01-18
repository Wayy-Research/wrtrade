# wrtrade

Ultra-fast backtesting and trading framework. Built with Polars for speed, designed for simplicity.

## Install

```bash
pip install wrtrade
```

## Quick Start

```python
import wrtrade as wrt
import polars as pl

# Define your signal
def trend(prices):
    fast = prices.rolling_mean(10)
    slow = prices.rolling_mean(30)
    return (fast > slow).cast(int) - (fast < slow).cast(int)

# One-line backtest
result = wrt.backtest(trend, prices)
print(f"Sortino: {result.sortino:.2f}")
```

That's it. No builders, no managers, no factories.

## The API

### Portfolio

`Portfolio` is the core data type. It holds trading signals.

```python
# Single signal
portfolio = wrt.Portfolio(trend)

# Multiple signals with weights
portfolio = wrt.Portfolio([
    (trend, 0.6),
    (momentum, 0.4),
])

# Named signals
portfolio = wrt.Portfolio({
    'trend': (trend, 0.6),
    'momentum': (momentum, 0.4),
})
```

### Backtest

```python
# From a signal function
result = wrt.backtest(trend, prices)

# From a portfolio
result = portfolio.backtest(prices)

# Access metrics
result.sortino        # Sortino ratio
result.sharpe         # Sharpe ratio
result.max_drawdown   # Maximum drawdown
result.total_return   # Total return
result.volatility     # Annualized volatility
result.returns        # Returns series

# Print formatted report
result.tear_sheet()
```

### Validate

Statistical validation using permutation testing:

```python
# One-line validation
p_value = wrt.validate(trend, prices)

if p_value < 0.05:
    print("Strategy is statistically significant!")

# With more control
p_value = portfolio.validate(prices, n_permutations=1000, parallel=True)
```

### Optimize

Kelly criterion optimization for position sizing:

```python
# Optimize weights
portfolio.optimize(prices)

# With parameters
portfolio.optimize(prices, method='kelly', max_leverage=1.0)
```

## Complete Example

```python
import wrtrade as wrt
import polars as pl
import numpy as np

# 1. Define signals
def trend(prices):
    fast = prices.rolling_mean(10)
    slow = prices.rolling_mean(30)
    return (fast > slow).cast(int) - (fast < slow).cast(int)

def momentum(prices):
    returns = prices.pct_change(20).fill_null(0)
    return (returns > 0.05).cast(int) - (returns < -0.05).cast(int)

# 2. Create portfolio
portfolio = wrt.Portfolio([
    (trend, 0.6),
    (momentum, 0.4),
])

# 3. Backtest
result = portfolio.backtest(prices)
print(f"Sortino: {result.sortino:.2f}")
print(f"Return: {result.total_return:.2%}")

# 4. Validate
p_value = portfolio.validate(prices)
if p_value < 0.05:
    print("Strategy is significant!")

# 5. Optimize
portfolio.optimize(prices)
print(f"Optimized weights: {portfolio.weights}")

# 6. View report
result.tear_sheet()
```

## Deployment

Create a strategy file with a `build_portfolio()` function:

```python
# my_strategy.py
import wrtrade as wrt

def trend(prices):
    fast = prices.rolling_mean(10)
    slow = prices.rolling_mean(30)
    return (fast > slow).cast(int) - (fast < slow).cast(int)

def build_portfolio():
    return wrt.Portfolio(trend, name="Trend_Strategy")
```

Deploy:

```bash
wrtrade strategy deploy my_strategy.py \
    --name my_strategy \
    --broker alpaca \
    --symbols AAPL,TSLA

wrtrade strategy start my_strategy
```

## Signal Functions

Signal functions take a price series and return signals:

```python
def my_signal(prices: pl.Series) -> pl.Series:
    """
    Args:
        prices: Polars Series of prices

    Returns:
        Polars Series with values:
        -1 = short
         0 = flat
         1 = long
    """
    # Your logic here
    return signals
```

## Advanced Usage

All advanced features remain accessible:

```python
# Permutation testing with full control
from wrtrade import PermutationTester, PermutationConfig

config = PermutationConfig(
    n_permutations=5000,
    parallel=True,
    preserve_gaps=True,
)
tester = PermutationTester(config)
results = tester.run_walkforward_test(prices, strategy_func, train_window=252)

# Kelly optimization with parameters
from wrtrade import KellyOptimizer, KellyConfig

config = KellyConfig(
    lookback_window=252,
    max_leverage=1.0,
    min_weight=0.0,
    max_weight=1.0,
)
optimizer = KellyOptimizer(config)
```

## CLI Reference

```bash
wrtrade strategy deploy <file>  # Deploy strategy
wrtrade strategy start <name>   # Start trading
wrtrade strategy stop <name>    # Stop trading
wrtrade strategy status         # Check status
wrtrade strategy logs <name>    # View logs
```

## Why wrtrade?

- **Fast**: Built on Polars, 10-50x faster than pandas
- **Simple**: `Portfolio` is a data type, not a framework
- **Validated**: Built-in permutation testing catches overfitting
- **Production-ready**: CLI deployment with monitoring

## License

MIT
