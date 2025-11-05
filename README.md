# wrtrade

Ultra-fast Python backtesting and trading framework. Built with Polars for speed, designed for simplicity.

## Philosophy

**Simple is better than complex.** wrtrade provides the essential tools for systematic trading:
- Fast backtesting with Polars (10-50x faster than pandas)
- Composable portfolio architecture
- Statistical validation (permutation testing)
- Kelly criterion optimization
- CLI-based deployment to paper/live trading

## Install

```bash
git clone https://github.com/Wayy-Research/wrtrade
cd wrtrade
pip install -e .
```

## Quick Start

### Basic Backtest

```python
import wrtrade as wrt
import polars as pl
import numpy as np

# Generate sample prices
np.random.seed(42)
prices = pl.Series([100 * (1 + r) for r in np.cumsum(np.random.normal(0.001, 0.02, 252))])

# Simple buy signal
signals = pl.Series([1] * len(prices))

# Backtest
portfolio = wrt.Portfolio(prices, signals)
portfolio.calculate_performance()

# View results
wrt.tear_sheet(portfolio.returns)
```

### Multi-Strategy Portfolio

```python
# Define signals
def trend_following(prices):
    fast = prices.rolling_mean(10)
    slow = prices.rolling_mean(30)
    return (fast > slow).cast(int) - (fast < slow).cast(int)

def momentum(prices):
    ret = prices.pct_change(20)
    return (ret > 0.05).cast(int) - (ret < -0.05).cast(int)

# Build portfolio
builder = wrt.NDimensionalPortfolioBuilder()

trend_comp = builder.create_signal_component("Trend", trend_following, weight=0.6)
mom_comp = builder.create_signal_component("Momentum", momentum, weight=0.4)

portfolio = builder.create_portfolio("MyStrategy", [trend_comp, mom_comp])

# Backtest
manager = wrt.AdvancedPortfolioManager(portfolio)
results = manager.backtest(prices)

print(f"Sortino Ratio: {results['portfolio_metrics']['sortino_ratio']:.2f}")
print(f"Total Return: {results['total_return']:.2%}")
```

## Statistical Validation

Avoid overfitting with permutation testing:

```python
tester = wrt.PermutationTester(wrt.PermutationConfig(n_permutations=1000))

results = tester.run_insample_test(
    prices,
    lambda p: portfolio,
    metric='sortino_ratio'
)

print(f"P-value: {results['p_value']:.4f}")
if results['p_value'] < 0.05:
    print("✓ Strategy is statistically significant")
```

## Kelly Optimization

Optimize position sizing for maximum geometric growth:

```python
optimizer = wrt.HierarchicalKellyOptimizer(
    wrt.KellyConfig(lookback_window=252, max_leverage=1.0)
)

# Optimize entire portfolio hierarchy
results = optimizer.optimize_portfolio(portfolio, prices, rebalance=True)

for level in results['level_optimizations']:
    print(f"{level['portfolio_name']}: {level['kelly_weights']}")
```

## Deployment

### 1. Create a Strategy File

Create a Python file with a `build_portfolio()` function:

```python
# my_strategy.py
import wrtrade as wrt
import polars as pl

def ma_crossover_signal(prices: pl.Series) -> pl.Series:
    """Moving average crossover signal."""
    fast = prices.rolling_mean(10)
    slow = prices.rolling_mean(30)
    return (fast > slow).cast(int) - (fast < slow).cast(int)

def build_portfolio():
    """Build the trading portfolio."""
    builder = wrt.NDimensionalPortfolioBuilder()

    ma_component = builder.create_signal_component(
        "MA_Crossover",
        ma_crossover_signal
    )

    return builder.create_portfolio("MA_Strategy", [ma_component])

if __name__ == "__main__":
    # Backtest code here for testing
    import numpy as np
    np.random.seed(42)
    prices = pl.Series([100 * (1 + r) for r in np.cumsum(np.random.normal(0.001, 0.02, 252))])

    portfolio = build_portfolio()
    manager = wrt.AdvancedPortfolioManager(portfolio)
    results = manager.backtest(prices)

    print(f"Sortino: {results['portfolio_metrics']['sortino_ratio']:.2f}")
```

### 2. Test Locally

Run the strategy file to backtest:

```bash
python my_strategy.py
```

### 3. Set Up Credentials

```bash
# Set environment variables
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"

# Or create .env file
echo "ALPACA_API_KEY=your_api_key" > .env
echo "ALPACA_SECRET_KEY=your_secret_key" >> .env
```

### 4. Deploy with CLI

```bash
# Deploy the strategy
wrtrade strategy deploy my_strategy.py \
    --name my_strategy \
    --broker alpaca \
    --symbols AAPL,TSLA,MSFT

# Start trading
wrtrade strategy start my_strategy

# Monitor
wrtrade strategy status my_strategy
wrtrade strategy logs my_strategy --follow

# Stop when done
wrtrade strategy stop my_strategy
```

The CLI manages your strategy as a background process with automatic restarts, logging, and monitoring.

## Example Strategies

See the `examples/` directory for complete strategy examples:

### Simple Strategy
```bash
# Test it
python examples/simple_strategy.py

# Deploy it
wrtrade strategy deploy examples/simple_strategy.py \
    --name simple_ma \
    --broker alpaca \
    --symbols AAPL

wrtrade strategy start simple_ma
```

### Multi-Strategy Portfolio
```bash
# Test it
python examples/multi_strategy.py

# Deploy it
wrtrade strategy deploy examples/multi_strategy.py \
    --name multi_strat \
    --broker alpaca \
    --symbols AAPL,TSLA,MSFT

wrtrade strategy start multi_strat
```

## Complete Workflow

End-to-end development to deployment:

```python
# 1. Define strategy (my_strategy.py)
import wrtrade as wrt
import polars as pl

def my_signal(prices):
    sma = prices.rolling_mean(20)
    return (prices > sma).cast(int) - (prices < sma).cast(int)

def build_portfolio():
    builder = wrt.NDimensionalPortfolioBuilder()
    component = builder.create_signal_component("SMA", my_signal)
    return builder.create_portfolio("SMA_Strategy", [component])

# 2. Backtest
if __name__ == "__main__":
    import numpy as np
    prices = pl.Series([...])  # Your data

    portfolio = build_portfolio()
    manager = wrt.AdvancedPortfolioManager(portfolio)
    results = manager.backtest(prices)
    print(f"Sortino: {results['portfolio_metrics']['sortino_ratio']:.2f}")

    # 3. Validate
    tester = wrt.PermutationTester(wrt.PermutationConfig(n_permutations=1000))
    perm = tester.run_insample_test(prices, lambda p: portfolio, 'sortino_ratio')

    if perm['p_value'] < 0.05:
        print("✓ Ready to deploy!")
    else:
        print("✗ Needs improvement")
```

Then deploy via CLI:
```bash
wrtrade strategy deploy my_strategy.py --name sma_strat --broker alpaca --symbols AAPL
wrtrade strategy start sma_strat
```

## API Reference

### Core Components

```python
# Basic backtesting
portfolio = wrt.Portfolio(prices, signals, max_position=float('inf'))
portfolio.calculate_performance()

# Component-based portfolios
component = wrt.SignalComponent(name, signal_func, weight=1.0)
portfolio = wrt.CompositePortfolio(name, components)

# Portfolio builder (recommended)
builder = wrt.NDimensionalPortfolioBuilder()
component = builder.create_signal_component(name, signal_func, weight=1.0)
portfolio = builder.create_portfolio(name, [components])

# Backtesting
manager = wrt.AdvancedPortfolioManager(portfolio)
results = manager.backtest(prices)
```

### Optimization

```python
# Kelly criterion
config = wrt.KellyConfig(lookback_window=252, max_leverage=1.0)
optimizer = wrt.KellyOptimizer(config)
weights = optimizer.calculate_portfolio_kelly(returns_matrix, names)

# Hierarchical Kelly (for composite portfolios)
optimizer = wrt.HierarchicalKellyOptimizer(config)
results = optimizer.optimize_portfolio(portfolio, prices, rebalance=True)
```

### Validation

```python
# Permutation testing
config = wrt.PermutationConfig(n_permutations=1000, parallel=True)
tester = wrt.PermutationTester(config)
results = tester.run_insample_test(prices, strategy_func, 'sortino_ratio')

# Walk-forward testing
results = tester.run_walkforward_test(
    prices,
    strategy_func,
    train_window=252,
    metric='sortino_ratio'
)
```

### Metrics

```python
# Performance metrics
wrt.tear_sheet(returns)  # Print formatted report
metrics = wrt.calculate_all_metrics(returns)
rolling = wrt.calculate_all_rolling_metrics(returns, window=252)

# Available metrics:
# - volatility (annualized)
# - sortino_ratio
# - gain_to_pain_ratio
# - max_drawdown
# - sharpe_ratio
```

## CLI Reference

```bash
# Initialize workspace
wrtrade init

# Strategy deployment and management
wrtrade strategy deploy <file> --name <name> --broker <broker> --symbols <symbols>
wrtrade strategy start <name>
wrtrade strategy stop <name>
wrtrade strategy restart <name>
wrtrade strategy status [name]
wrtrade strategy logs <name> [--follow] [--lines N]
wrtrade strategy remove <name> [--remove-data]

# Deployment options
wrtrade strategy deploy my_strategy.py \
    --name my_strat \
    --broker alpaca \
    --symbols AAPL,TSLA \
    --max-position 1000 \
    --risk-per-trade 0.02 \
    --cycle-minutes 5 \
    --kelly-optimization \
    --auto-restart

# Broker info
wrtrade broker list

# Version
wrtrade version
```

## Hierarchical Portfolios

Build complex nested strategies:

```python
builder = wrt.NDimensionalPortfolioBuilder()

# Level 1: Individual signals
ma_fast = builder.create_signal_component("MA_Fast", fast_signal, weight=0.5)
ma_slow = builder.create_signal_component("MA_Slow", slow_signal, weight=0.5)
rsi = builder.create_signal_component("RSI", rsi_signal, weight=1.0)

# Level 2: Strategy groups
trend = builder.create_portfolio("Trend", [ma_fast, ma_slow], weight=0.6)
mean_reversion = builder.create_portfolio("MeanRev", [rsi], weight=0.4)

# Level 3: Master portfolio
master = builder.create_portfolio("Master", [trend, mean_reversion])

# View structure
manager = wrt.AdvancedPortfolioManager(master)
manager.print_structure()

# Output:
# === Portfolio Structure: Master ===
#   Trend (weight: 0.600)
#     MA_Fast (weight: 0.500)
#     MA_Slow (weight: 0.500)
#   MeanRev (weight: 0.400)
#     RSI (weight: 1.000)
```

## Strategy File Requirements

Your strategy file must have a `build_portfolio()` function that returns a portfolio:

```python
import wrtrade as wrt

def build_portfolio():
    """
    Required function for deployment.
    Must return a CompositePortfolio.
    """
    builder = wrt.NDimensionalPortfolioBuilder()
    # ... build your portfolio
    return portfolio
```

The CLI will:
1. Import your file
2. Call `build_portfolio()`
3. Deploy the returned portfolio with your specified symbols

## Performance

Compared to pandas-based frameworks:

- **10-50x faster** on large datasets
- **Lower memory usage** with Polars columnar format
- **Parallel permutation testing** with multiprocessing
- **Vectorized operations** eliminate Python loops

Perfect for:
- High-frequency strategy development
- Large-scale backtesting (millions of data points)
- Parameter optimization
- Statistical validation

## Design Principles

1. **Simple is better than complex** - Minimal API surface
2. **Fast by default** - Built on Polars
3. **Composable** - Build complex strategies from simple parts
4. **Validated** - Statistical testing built-in
5. **Production-ready** - CLI deployment for real trading

## Testing

Run tests:
```bash
pytest tests/ -v
```

Current coverage: **113 tests passing**

## Directory Structure

```
wrtrade/
├── examples/
│   ├── simple_strategy.py      # Basic MA crossover
│   └── multi_strategy.py       # Multi-strategy portfolio
├── wrtrade/
│   ├── __init__.py
│   ├── portfolio.py            # Core backtesting
│   ├── components.py           # Portfolio components
│   ├── ndimensional_portfolio.py  # Portfolio builder
│   ├── kelly.py                # Kelly optimization
│   ├── permutation.py          # Statistical validation
│   ├── metrics.py              # Performance metrics
│   └── cli.py                  # CLI commands
└── tests/                      # Test suite
```

## Contributing

Contributions welcome! Please:
1. Keep it simple
2. Add tests
3. Follow existing style
4. Update docs

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: https://github.com/wayy-research/wrtrade/issues
- **Examples**: See `examples/` directory
- **Questions**: Open an issue with the "question" label
