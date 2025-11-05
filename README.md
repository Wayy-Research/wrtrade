# wrtrade

Ultra-fast Python backtesting and trading framework. Built with Polars for speed, designed for simplicity.

## For Quantitative Traders

You have a trading idea. You want to test it, validate it's not overfit, optimize position sizing, and deploy it to live trading. wrtrade makes this workflow simple and fast.

**The Journey:**
1. **Research** - Develop your strategy and backtest it
2. **Validate** - Use permutation testing to ensure it's statistically significant
3. **Optimize** - Apply Kelly criterion for optimal position sizing
4. **Paper Trade** - Deploy to paper trading to verify in live markets
5. **Go Live** - Deploy to live trading with confidence

## Install

```bash
git clone https://github.com/Wayy-Research/wrtrade
cd wrtrade
pip install -e .
```

## Step 1: Research & Backtest

Start with an idea. Let's say you think a simple moving average crossover might work.

### Your First Backtest

```python
import wrtrade as wrt
import polars as pl
import numpy as np

# Get some price data (or use your own)
np.random.seed(42)
prices = pl.Series([100 * (1 + r) for r in np.cumsum(np.random.normal(0.001, 0.02, 252))])

# Define your signal
def ma_crossover(prices):
    fast = prices.rolling_mean(10)
    slow = prices.rolling_mean(30)
    return (fast > slow).cast(int) - (fast < slow).cast(int)

# Build portfolio
builder = wrt.NDimensionalPortfolioBuilder()
component = builder.create_signal_component("MA_Cross", ma_crossover)
portfolio = builder.create_portfolio("MA_Strategy", [component])

# Backtest
manager = wrt.AdvancedPortfolioManager(portfolio)
results = manager.backtest(prices)

# See the results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sortino Ratio: {results['portfolio_metrics']['sortino_ratio']:.2f}")
print(f"Max Drawdown: {results['portfolio_metrics']['max_drawdown']:.2%}")

# Detailed performance report
wrt.tear_sheet(results['returns'])
```

**Output:**
```
Total Return: 15.3%
Sortino Ratio: 1.45
Max Drawdown: -8.2%

=== Performance Metrics ===
Volatility (Annual): 18.5%
Sortino Ratio: 1.45
Gain-to-Pain: 2.31
Max Drawdown: -8.2%
```

Great! Your strategy looks promising. But is it real or just luck?

## Step 2: Validate with Permutation Testing

**The Problem:** Many strategies look great in backtests but fail in live trading. Why? Overfitting.

**The Solution:** Permutation testing. This scrambles your price data thousands of times while preserving statistical properties. If your strategy only works on the exact historical sequence, it's probably overfit.

```python
# Test if your strategy is statistically significant
tester = wrt.PermutationTester(wrt.PermutationConfig(
    n_permutations=1000,
    parallel=False  # Set to True for faster testing
))

results = tester.run_insample_test(
    prices,
    lambda p: portfolio,
    metric='sortino_ratio'
)

print(f"Your Strategy: {results['real_performance']:.2f}")
print(f"Random Average: {results['permutation_mean']:.2f}")
print(f"P-value: {results['p_value']:.4f}")

if results['p_value'] < 0.05:
    print("✓ Strategy is statistically significant!")
    print("  Your edge is unlikely due to chance.")
else:
    print("✗ Strategy may be overfit")
    print("  Performance could be due to luck.")
```

**What does this mean?**
- **P-value < 0.05**: Your strategy performs significantly better than random chance
- **P-value > 0.05**: Your strategy might just be lucky - keep researching!

### Walk-Forward Testing

Even better - test your strategy on data it's never seen:

```python
# Split data into train/test periods
results = tester.run_walkforward_test(
    prices,
    lambda p: portfolio,
    train_window=252,  # Train on 1 year
    metric='sortino_ratio'
)

print(f"Out-of-Sample P-value: {results['p_value']:.4f}")
```

If your strategy passes both in-sample AND out-of-sample tests, you have a robust edge.

## Step 3: Optimize Position Sizing

You have a validated strategy. Now optimize how much to bet using the Kelly criterion.

**The Kelly Criterion** tells you the optimal fraction of capital to allocate for maximum long-term growth.

```python
# Apply Kelly optimization
optimizer = wrt.HierarchicalKellyOptimizer(wrt.KellyConfig(
    lookback_window=252,      # Use 1 year of data
    max_leverage=1.0,         # No leverage (conservative)
    rebalance_frequency=21,   # Rebalance monthly
    risk_free_rate=0.02       # 2% risk-free rate
))

kelly_results = optimizer.optimize_portfolio(portfolio, prices, rebalance=True)

print("=== Kelly Optimization ===")
for level in kelly_results['level_optimizations']:
    print(f"Strategy: {level['portfolio_name']}")
    print(f"  Optimal Allocation: {level['kelly_weights']}")
    print(f"  Expected Sharpe: {level['metrics']['sharpe_ratio']:.2f}")
```

**What changed?**
Your portfolio now uses Kelly-optimal position sizing instead of equal weights. This maximizes long-term growth while controlling risk.

## Step 4: Combine Multiple Strategies

Don't put all your eggs in one basket. Combine multiple strategies for diversification.

```python
# Define multiple signals
def trend_following(prices):
    fast = prices.rolling_mean(10)
    slow = prices.rolling_mean(30)
    return (fast > slow).cast(int) - (fast < slow).cast(int)

def momentum(prices):
    returns = prices.pct_change(20)
    return (returns > 0.05).cast(int) - (returns < -0.05).cast(int)

def mean_reversion(prices):
    sma = prices.rolling_mean(20)
    std = prices.rolling_std(20)
    z_score = (prices - sma) / std
    # Buy when oversold, sell when overbought
    return (z_score < -2).cast(int) - (z_score > 2).cast(int)

# Build hierarchical portfolio
builder = wrt.NDimensionalPortfolioBuilder()

# Create individual components
trend = builder.create_signal_component("Trend", trend_following, weight=0.4)
mom = builder.create_signal_component("Momentum", momentum, weight=0.3)
mean_rev = builder.create_signal_component("MeanRev", mean_reversion, weight=0.3)

# Combine into master portfolio
portfolio = builder.create_portfolio("Multi_Strategy", [trend, mom, mean_rev])

# View structure
manager = wrt.AdvancedPortfolioManager(portfolio)
manager.print_structure()
```

**Output:**
```
=== Portfolio Structure: Multi_Strategy ===
Multi_Strategy (weight: 1.000)
  Trend (weight: 0.400)
  Momentum (weight: 0.300)
  MeanRev (weight: 0.300)
```

Now validate and optimize this multi-strategy portfolio the same way!

## Step 5: Deploy to Paper Trading

You've researched, validated, and optimized. Time to test in live markets - but safely.

### Create a Deployable Strategy File

```python
# my_strategy.py
import wrtrade as wrt
import polars as pl

def trend_signal(prices: pl.Series) -> pl.Series:
    """Your validated trend following signal."""
    fast = prices.rolling_mean(10)
    slow = prices.rolling_mean(30)
    return (fast > slow).cast(int) - (fast < slow).cast(int)

def momentum_signal(prices: pl.Series) -> pl.Series:
    """Your validated momentum signal."""
    returns = prices.pct_change(20)
    return (returns > 0.05).cast(int) - (returns < -0.05).cast(int)

def build_portfolio():
    """
    Required function for deployment.
    The CLI calls this to get your portfolio.
    """
    builder = wrt.NDimensionalPortfolioBuilder()

    trend = builder.create_signal_component("Trend", trend_signal, weight=0.6)
    mom = builder.create_signal_component("Momentum", momentum_signal, weight=0.4)

    return builder.create_portfolio("My_Strategy", [trend, mom])

# Backtest when run directly
if __name__ == "__main__":
    import numpy as np

    # Load your data
    prices = pl.Series([...])  # Your historical prices

    portfolio = build_portfolio()
    manager = wrt.AdvancedPortfolioManager(portfolio)
    results = manager.backtest(prices)

    print(f"Sortino: {results['portfolio_metrics']['sortino_ratio']:.2f}")

    # Validate
    tester = wrt.PermutationTester(wrt.PermutationConfig(n_permutations=1000))
    perm = tester.run_insample_test(prices, lambda p: portfolio, 'sortino_ratio')

    if perm['p_value'] < 0.05:
        print("✓ Ready to deploy!")
```

### Test Locally First

```bash
python my_strategy.py
```

Make sure everything works before deploying.

### Set Up Broker Credentials

```bash
# Get API keys from Alpaca (alpaca.markets)
# Start with paper trading account (free)

export ALPACA_API_KEY="PK..."
export ALPACA_SECRET_KEY="..."
```

### Deploy to Paper Trading

```bash
# Deploy your strategy
wrtrade strategy deploy my_strategy.py \
    --name my_strategy \
    --broker alpaca \
    --symbols AAPL,TSLA,MSFT \
    --max-position 1000 \
    --risk-per-trade 0.02 \
    --cycle-minutes 5 \
    --kelly-optimization

# Start trading (paper money)
wrtrade strategy start my_strategy
```

**What's happening?**
- wrtrade runs your strategy in the background
- Every 5 minutes, it fetches new prices
- Generates signals using your portfolio
- Places trades with your broker (paper account)
- Automatically restarts if it crashes
- Logs everything

### Monitor Your Paper Trading

```bash
# Check status
wrtrade strategy status my_strategy

# Watch logs in real-time
wrtrade strategy logs my_strategy --follow

# See all strategies
wrtrade strategy status
```

**Output:**
```
🟢 my_strategy: RUNNING (2h 15m)
   Status: RUNNING
   PID: 12345
   Uptime: 2h 15m
   Trades: 12 (P&L: $145.32)
   Broker: alpaca
   Symbols: AAPL, TSLA, MSFT
```

### Let It Run

Paper trade for at least 30-60 days. Watch for:
- Does it perform as expected?
- Are slippage and fees accounted for?
- Does it handle market volatility?
- Any execution issues?

If everything looks good after paper trading, you're ready for live deployment.

## Step 6: Go Live

After successful paper trading:

```bash
# Stop paper trading
wrtrade strategy stop my_strategy

# Update credentials to live account
export ALPACA_API_KEY="your_live_key"
export ALPACA_SECRET_KEY="your_live_secret"

# Deploy to live (same command, but with live credentials)
wrtrade strategy deploy my_strategy.py \
    --name my_strategy_live \
    --broker alpaca \
    --symbols AAPL,TSLA,MSFT \
    --max-position 500 \
    --risk-per-trade 0.01 \
    --kelly-optimization

# Start live trading
wrtrade strategy start my_strategy_live
```

**Start small!** Use lower position sizes and risk limits when going live.

## Examples

### Simple MA Crossover

```bash
# Test the example
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
# Test the example
python examples/multi_strategy.py

# Deploy it
wrtrade strategy deploy examples/multi_strategy.py \
    --name multi_strat \
    --broker alpaca \
    --symbols AAPL,TSLA,MSFT

wrtrade strategy start multi_strat
```

## Complete Workflow Example

Here's the entire journey in one script:

```python
import wrtrade as wrt
import polars as pl
import numpy as np

# 1. RESEARCH - Define your strategy
def my_signal(prices):
    sma = prices.rolling_mean(20)
    return (prices > sma).cast(int) - (prices < sma).cast(int)

# 2. BACKTEST
builder = wrt.NDimensionalPortfolioBuilder()
component = builder.create_signal_component("SMA20", my_signal)
portfolio = builder.create_portfolio("SMA_Strategy", [component])

prices = pl.Series([...])  # Your data
manager = wrt.AdvancedPortfolioManager(portfolio)
results = manager.backtest(prices)

print(f"Backtest Sortino: {results['portfolio_metrics']['sortino_ratio']:.2f}")

# 3. VALIDATE - Permutation testing
tester = wrt.PermutationTester(wrt.PermutationConfig(n_permutations=1000))
perm = tester.run_insample_test(prices, lambda p: portfolio, 'sortino_ratio')

if perm['p_value'] >= 0.05:
    print("✗ Strategy not significant - back to research!")
    exit()

print(f"✓ P-value: {perm['p_value']:.4f} - Strategy is significant!")

# 4. OPTIMIZE - Kelly criterion
optimizer = wrt.HierarchicalKellyOptimizer()
kelly_results = optimizer.optimize_portfolio(portfolio, prices, rebalance=True)
print(f"✓ Kelly optimization complete")

# 5. DEPLOY - Save to file and use CLI
print("\nReady to deploy! Use:")
print("  wrtrade strategy deploy my_strategy.py --name sma20 --broker alpaca --symbols AAPL")
```

## Why wrtrade?

### Fast
Built on Polars - **10-50x faster** than pandas-based frameworks. Backtest millions of data points in seconds.

### Validated
Built-in permutation testing catches overfitting before you lose money.

### Optimized
Kelly criterion optimization for maximum long-term growth.

### Production-Ready
CLI deployment manages strategies as background processes with monitoring, logging, and auto-restart.

### Simple
Clean API. The entire workflow is straightforward:
```python
builder = wrt.NDimensionalPortfolioBuilder()
component = builder.create_signal_component("MySignal", my_func)
portfolio = builder.create_portfolio("MyStrategy", [component])
manager = wrt.AdvancedPortfolioManager(portfolio)
results = manager.backtest(prices)
```

## API Reference

### Core Components

```python
# Basic backtesting
portfolio = wrt.Portfolio(prices, signals, max_position=float('inf'))
portfolio.calculate_performance()

# Component-based portfolios (recommended)
builder = wrt.NDimensionalPortfolioBuilder()
component = builder.create_signal_component(name, signal_func, weight=1.0)
portfolio = builder.create_portfolio(name, [components])

# Backtesting
manager = wrt.AdvancedPortfolioManager(portfolio)
results = manager.backtest(prices)
```

### Validation

```python
# Permutation testing
config = wrt.PermutationConfig(n_permutations=1000, parallel=False)
tester = wrt.PermutationTester(config)

# In-sample test
results = tester.run_insample_test(prices, strategy_func, 'sortino_ratio')

# Walk-forward test (recommended)
results = tester.run_walkforward_test(
    prices, strategy_func, train_window=252, metric='sortino_ratio'
)
```

### Optimization

```python
# Kelly criterion
config = wrt.KellyConfig(lookback_window=252, max_leverage=1.0)
optimizer = wrt.HierarchicalKellyOptimizer(config)
results = optimizer.optimize_portfolio(portfolio, prices, rebalance=True)
```

### Metrics

```python
# Performance metrics
wrt.tear_sheet(returns)  # Detailed report
metrics = wrt.calculate_all_metrics(returns)
rolling = wrt.calculate_all_rolling_metrics(returns, window=252)

# Available metrics:
# - volatility, sortino_ratio, sharpe_ratio
# - gain_to_pain_ratio, max_drawdown
```

## CLI Reference

```bash
# Initialize workspace
wrtrade init

# Deploy strategy
wrtrade strategy deploy <file> \
    --name <name> \
    --broker alpaca \
    --symbols AAPL,TSLA \
    --max-position 1000 \
    --risk-per-trade 0.02 \
    --cycle-minutes 5 \
    --kelly-optimization \
    --auto-restart

# Manage strategies
wrtrade strategy start <name>
wrtrade strategy stop <name>
wrtrade strategy status [name]
wrtrade strategy logs <name> [--follow]
wrtrade strategy remove <name>

# Broker info
wrtrade broker list
```

## Strategy File Requirements

Your strategy file must have a `build_portfolio()` function:

```python
import wrtrade as wrt

def build_portfolio():
    """
    Required function - CLI calls this to get your portfolio.
    Must return a CompositePortfolio.
    """
    builder = wrt.NDimensionalPortfolioBuilder()
    # ... build your portfolio
    return portfolio

if __name__ == "__main__":
    # Optional: backtest when run directly
    pass
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

**113 tests passing** - Full coverage of backtesting, validation, optimization, and deployment.

## Best Practices

### 1. Always Validate
Don't trust a backtest alone. Use permutation testing to verify statistical significance.

### 2. Walk-Forward Test
In-sample tests can still overfit. Walk-forward testing simulates real trading.

### 3. Start with Paper Trading
Always paper trade for at least 30 days before going live.

### 4. Use Kelly Optimization
Kelly criterion maximizes long-term growth while controlling risk.

### 5. Diversify
Combine multiple strategies with different return profiles.

### 6. Start Small
When going live, use smaller positions than in paper trading.

### 7. Monitor Everything
Watch logs, check performance daily, especially in first weeks of live trading.

## Common Pitfalls

❌ **Overfitting**: Testing too many parameters until something works
✓ **Solution**: Use permutation testing and walk-forward validation

❌ **Survivorship Bias**: Only testing on stocks that survived
✓ **Solution**: Include delisted stocks in your data

❌ **Look-Ahead Bias**: Using future information
✓ **Solution**: Careful with rolling calculations, ensure proper alignment

❌ **Ignoring Costs**: Not accounting for slippage and fees
✓ **Solution**: Paper trade first, be conservative in backtests

❌ **Over-Leverage**: Taking too much risk
✓ **Solution**: Use Kelly criterion, start with max_leverage=1.0

## Performance

- **10-50x faster** than pandas-based frameworks
- **Parallel permutation testing** for faster validation
- **Memory efficient** with Polars columnar format
- **Scales** to millions of data points

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
- **Questions**: Open an issue

---

**Remember**: Past performance doesn't guarantee future results. Always start with paper trading. Never risk more than you can afford to lose.
